"""
timur/symbolic/router.py
═════════════════════════════════════════════════════════════════════════════
TIMUR XAI — Sembolik Yönlendirici

FAZ 1 — ÖN EĞİTİM / KEŞİF (Pre-training Discovery)
    Ham veriyi alıp arama uzayını taran ve en optimal analitik denklemi çıkaran
    sembolik yönlendiriciyi çalıştırır.

    Adımlar:
        1. Gatekeeper analizi → nonlineerlik skoru + yönlendirme kararı
        2. Sembolik ön eğitim → taslak denklem (ham, büyük arama uzayı)
        3. İyileştirme eğitimi → taslak denklem üzerinde rafine → nihai φ(x)

FAZ 2 — DONDURMA (Freeze)
    Bulunan fiziksel/matematiksel denklemi mutlak bir gerçeklik çapası
    olarak sabitler. Bu modül denklemi + katsayıları döndürür;
    timur/pinn/loss.py bu denklemi kayıp fonksiyonuna bağlar.

Kullanım:
    from timur.symbolic.router import TIMURSymbolicRouter

    router = TIMURSymbolicRouter(verbose=True)
    result = router.discover(X_train, y_train)
    print(result.equation_str)       # "y = 2.718·x0² + sin(x1)"
    print(result.frozen_fn)          # callable: torch.Tensor → torch.Tensor
"""

from __future__ import annotations

import importlib
import logging
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import ElasticNetCV, LassoCV, Ridge, RidgeCV
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

_log = logging.getLogger(__name__)

# ─── Sabitler ─────────────────────────────────────────────────────────────────

COMPONENT_WEIGHTS: Dict[str, float] = {
    "linear_residual": 0.35,
    "correlation_gap": 0.30,
    "pca_complexity":  0.15,
    "mutual_info":     0.20,
}

RoutingDecision = Literal["linear", "poly", "pysr"]

# ─── Veri yapıları ────────────────────────────────────────────────────────────

@dataclass
class FeatureProfile:
    index    : int
    name     : str
    pearson  : float
    spearman : float
    gap      : float
    relevance: float
    mi_score : float = 0.0
    selected : bool  = True

    @property
    def is_nonlinear(self) -> bool:
        return (self.gap > 0.05 and self.relevance > 0.10) or \
               (self.mi_score > 0.15 and self.relevance > 0.05)


@dataclass
class GatekeeperReport:
    nonlinearity_score    : float
    routing_decision      : RoutingDecision
    linear_r2             : float
    linear_residual_score : float
    correlation_gap_score : float
    pca_dim_score         : float
    mi_score              : float
    feature_profiles      : List[FeatureProfile]
    selected_feature_idx  : List[int]
    n_samples             : int
    n_features            : int
    n_selected            : int
    adaptive_linear_thr   : float
    adaptive_pysr_thr     : float
    weights               : Dict[str, float] = field(
        default_factory=lambda: dict(COMPONENT_WEIGHTS)
    )

    @property
    def linearity_score(self) -> float:
        return round(1.0 - self.nonlinearity_score, 4)

    @property
    def recommended_engine(self) -> str:
        return {
            "linear": "Ridge / Lasso  (Lineer Uzman)",
            "poly"  : "ElasticNet + Polinom  (Nonlineer Uzman)",
            "pysr"  : "PySR  — Genetik Sembolik Regresyon",
        }[self.routing_decision]

    @property
    def nonlinear_features(self) -> List[FeatureProfile]:
        return [fp for fp in self.feature_profiles if fp.is_nonlinear]

    @staticmethod
    def _gauge(val: float, width: int = 24) -> str:
        filled = round(max(0.0, min(1.0, val)) * width)
        return "█" * filled + "░" * (width - filled)

    def summary(self) -> str:
        nl  = self.nonlinearity_score
        lin = self.linearity_score
        w   = self.weights
        sep = "  " + "═" * 60
        dim = "  " + "─" * 60
        lines = [
            "",
            sep,
            "  TIMUR XAI  |  SEMBOLİK YÖNLENDİRİCİ  v1.0",
            sep,
            f"  NonLineerlik  [{self._gauge(nl)}]  {nl:>5.1%}",
            f"  Doğrusallık   [{self._gauge(lin)}]  {lin:>5.1%}",
            dim,
            "  Bileşen Puanları:",
            f"  +-- Ridge R²              :  {self.linear_r2:.4f}",
            f"  +-- Lineer Kalıntı        :  {self.linear_residual_score:.4f}"
            f"   (ağırlık: {w['linear_residual']:.0%})",
            f"  +-- Korelasyon Uçurumu    :  {self.correlation_gap_score:.4f}"
            f"   (ağırlık: {w['correlation_gap']:.0%})",
            f"  +-- PCA Boyutsallığı      :  {self.pca_dim_score:.4f}"
            f"   (ağırlık: {w['pca_complexity']:.0%})",
            f"  +-- Mutual Information    :  {self.mi_score:.4f}"
            f"   (ağırlık: {w['mutual_info']:.0%})",
            dim,
            f"  Veri          : {self.n_samples} örnek  x  {self.n_features} özellik",
            f"  Seçilen (F0)  : {self.n_selected} özellik",
            f"  Adaptif Eşik  : linear < {self.adaptive_linear_thr:.2f}"
            f"  |  pysr >= {self.adaptive_pysr_thr:.2f}",
            sep,
            f"  >> YÖNLENDİRME  ->  {self.routing_decision.upper()}",
            f"     Motor       ->  {self.recommended_engine}",
            sep,
            "",
        ]
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


@dataclass
class DiscoveryResult:
    """
    TIMURSymbolicRouter.discover() tam sonucu.

    equation_str : İnsan-okunabilir denklem (XAI çıktısı)
    frozen_fn    : PyTorch-uyumlu callable φ(x: Tensor) → Tensor
                   timur/pinn/loss.py bu fonksiyonu kayıp olarak kullanır.
    coef_        : Katsayı dizisi (numpy)
    intercept_   : Sabit terim
    r2_pretrain  : Ön eğitim R² skoru
    r2_refine    : Rafine eğitim R² skoru (Faz 2 iyileştirmesi)
    routing      : Kullanılan motor kararı
    gatekeeper   : GatekeeperReport
    fit_time_s   : Toplam fit süresi (saniye)
    metadata     : Ek bilgiler
    """
    equation_str : str
    frozen_fn    : Callable[[torch.Tensor], torch.Tensor]
    coef_        : np.ndarray
    intercept_   : float
    r2_pretrain  : float
    r2_refine    : float
    routing      : RoutingDecision
    gatekeeper   : GatekeeperReport
    fit_time_s   : float
    metadata     : Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        sep  = "━" * 50
        thin = "─" * 50
        return "\n".join([
            "",
            sep,
            "  TIMUR XAI — SEMBOLİK KEŞİF RAPORU",
            sep,
            f"  Denklem     :  {self.equation_str}",
            thin,
            f"  Ön Eğitim R²:  {self.r2_pretrain:.4f}",
            f"  Rafine R²    :  {self.r2_refine:.4f}",
            f"  Motor        :  {self.routing.upper()}",
            f"  Süre         :  {self.fit_time_s:.2f}s",
            sep,
            "",
        ])


# ─── Gatekeeper ───────────────────────────────────────────────────────────────

class Gatekeeper:
    """Faz 0 özellik eleme + Faz 1 dört bileşenli veri fizik analizi."""

    def __init__(
        self,
        linear_threshold : float = 0.40,
        pysr_threshold   : float = 0.65,
        variance_target  : float = 0.95,
        top_k_features   : int   = 20,
        feature_names    : Optional[List[str]] = None,
        random_state     : int   = 42,
    ) -> None:
        self.linear_threshold = linear_threshold
        self.pysr_threshold   = pysr_threshold
        self.variance_target  = variance_target
        self.top_k_features   = top_k_features
        self.feature_names    = feature_names
        self.random_state     = random_state
        self._scaler          = StandardScaler()

    def analyze(self, X: np.ndarray, y: np.ndarray) -> GatekeeperReport:
        X, y = self._validate(X, y)
        n, p = X.shape

        # Mutlak Sabit Kaçış Kapsülü (0 Özellik Paradoksu Düzeltmesi)
        if p == 0:
            lin_thr, pysr_thr = self._adaptive_thresholds(n)
            return GatekeeperReport(
                nonlinearity_score=0.0,
                routing_decision="linear",
                linear_r2=1.0,
                linear_residual_score=0.0,
                correlation_gap_score=0.0,
                pca_dim_score=0.0,
                mi_score=0.0,
                feature_profiles=[],
                selected_feature_idx=[],
                n_samples=n,
                n_features=0,
                n_selected=0,
                adaptive_linear_thr=lin_thr,
                adaptive_pysr_thr=pysr_thr,
                weights=dict(COMPONENT_WEIGHTS)
            )

        names_all = self._get_names(p)

        # Faz 0: MI bazlı özellik eleme
        sel_idx, mi_all = self._phase0(X, y, p)
        X_sel     = X[:, sel_idx]
        n_sel     = len(sel_idx)
        names_sel = [names_all[i] for i in sel_idx]
        X_sc      = self._scaler.fit_transform(X_sel)
        
    

        # Faz 1: Dört bileşen
        r2, s_res = self._score_linear_residual(X_sc, y)
        s_gap, profiles_sel = self._score_correlation_gap(
            X_sel, y, n_sel, names_sel,
            mi_scores_sel=[float(mi_all[i]) for i in sel_idx]
        )
        s_pca = self._score_pca(X_sc, n_sel)
        s_mi  = self._score_mi([float(mi_all[i]) for i in sel_idx])

        profiles_all = self._build_profiles(X, y, p, names_all, mi_all, sel_idx, profiles_sel)

        w = COMPONENT_WEIGHTS
        nl = float(np.clip(
            w["linear_residual"] * s_res
            + w["correlation_gap"] * s_gap
            + w["pca_complexity"]  * s_pca
            + w["mutual_info"]     * s_mi,
            0.0, 1.0
        ))

        lin_thr, pysr_thr = self._adaptive_thresholds(n)
        routing = self._route(nl, lin_thr, pysr_thr)

        return GatekeeperReport(
            nonlinearity_score    = nl,
            routing_decision      = routing,
            linear_r2             = r2,
            linear_residual_score = s_res,
            correlation_gap_score = s_gap,
            pca_dim_score         = s_pca,
            mi_score              = s_mi,
            feature_profiles      = profiles_all,
            selected_feature_idx  = sel_idx,
            n_samples             = n,
            n_features            = p,
            n_selected            = n_sel,
            adaptive_linear_thr   = lin_thr,
            adaptive_pysr_thr     = pysr_thr,
            weights               = dict(w),
        )

    def _phase0(self, X, y, p):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mi = mutual_info_regression(X, y, random_state=self.random_state)
        if p <= self.top_k_features:
            return list(range(p)), mi
        top = sorted(np.argsort(mi)[::-1][:self.top_k_features].tolist())
        return top, mi

    def _score_linear_residual(self, X_sc, y):
        m = Ridge(alpha=1.0).fit(X_sc, y)
        r2 = float(np.clip(r2_score(y, m.predict(X_sc)), 0.0, 1.0))
        return r2, 1.0 - r2

    def _score_correlation_gap(self, X, y, p, names, mi_scores_sel):
        profiles, gaps, weights = [], [], []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i in range(p):
                xi = X[:, i]
                pr, _ = stats.pearsonr(xi, y)
                sr, _ = stats.spearmanr(xi, y)
                apr, asr = abs(float(pr)), abs(float(sr))
                gap = asr - apr
                rel = max(apr, asr)
                profiles.append(FeatureProfile(
                    index=i, name=names[i],
                    pearson=round(float(pr), 6), spearman=round(float(sr), 6),
                    gap=round(gap, 6), relevance=round(rel, 6),
                    mi_score=round(float(mi_scores_sel[i]), 6), selected=True,
                ))
                gaps.append(max(0.0, gap))
                weights.append(rel + 1e-9)
        w = np.array(weights)
        w /= w.sum()
        score = float(np.clip(np.dot(w, gaps) / 0.5, 0.0, 1.0))
        return score, profiles

    def _score_pca(self, X_sc, p):
        if p <= 1:
            return 0.0
        pca = PCA(random_state=self.random_state).fit(X_sc)
        cum = np.cumsum(pca.explained_variance_ratio_)
        n_eff = min(int(np.searchsorted(cum, self.variance_target)) + 1, p)
        return float((n_eff - 1) / max(p - 1, 1))

    def _score_mi(self, mi_scores):
        if not mi_scores:
            return 0.0
        arr  = np.array(mi_scores, dtype=float)
        norm = arr / (arr + 0.5)
        sig  = norm[arr > 0.05]
        return float(np.clip(np.mean(sig), 0.0, 1.0)) if len(sig) > 0 else 0.0

    def _adaptive_thresholds(self, n):
        conf  = min(1.0, n / 1000.0)
        a_lin = min(0.60, self.linear_threshold + 0.10 * (1.0 - conf))
        a_psr = min(0.90, self.pysr_threshold   + 0.08 * (1.0 - conf))
        return round(a_lin, 3), round(a_psr, 3)

    def _route(self, score, lin, pysr):
        if score < lin:  return "linear"
        if score < pysr: return "poly"
        return "pysr"

    def _build_profiles(self, X, y, p, names, mi_all, sel_idx, sel_profiles):
        sel_map = {sel_idx[j]: sel_profiles[j] for j in range(len(sel_idx))}
        result  = []
        for i in range(p):
            if i in sel_map:
                fp = sel_map[i]
                result.append(FeatureProfile(
                    index=i, name=names[i], pearson=fp.pearson,
                    spearman=fp.spearman, gap=fp.gap, relevance=fp.relevance,
                    mi_score=round(float(mi_all[i]), 6), selected=True,
                ))
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        pr, _ = stats.pearsonr(X[:, i], y)
                        sr, _ = stats.spearmanr(X[:, i], y)
                        apr, asr = abs(float(pr)), abs(float(sr))
                    except Exception:
                        pr = sr = apr = asr = 0.0
                result.append(FeatureProfile(
                    index=i, name=names[i],
                    pearson=round(float(pr), 6), spearman=round(float(sr), 6),
                    gap=round(asr - apr, 6), relevance=round(max(apr, asr), 6),
                    mi_score=round(float(mi_all[i]), 6), selected=False,
                ))
        return result

    @staticmethod
    def _validate(X, y):
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float).ravel()
        if X.ndim == 1: X = X.reshape(-1, 1)
        if len(X) != len(y):
            raise ValueError(f"X ve y uzunlukları eşleşmiyor: {len(X)} vs {len(y)}")
        if len(X) < 10:
            raise ValueError(f"En az 10 örnek gerekli. Verilen: {len(X)}")
        return X, y

    def _get_names(self, n):
        if self.feature_names and len(self.feature_names) == n:
            return list(self.feature_names)
        return [f"x{i}" for i in range(n)]


# ─── Sembolik Motorlar ─────────────────────────────────────────────────────────

class _LinearEngine:
    """Ridge/Lasso — lineer denklem keşfi."""

    def __init__(self, feature_names=None):
        self.feature_names = feature_names
        self._coef = None
        self._intercept = 0.0
        self._r2 = None
        self._alpha = None
        self._chosen = None
        self._scaler = StandardScaler()

    def fit(self, X, y):
        Xs = self._scaler.fit_transform(X)
        # Lasso dene
        alphas = np.logspace(-4, 4, 50)
        lasso  = LassoCV(alphas=alphas, cv=5, max_iter=10_000).fit(Xs, y)
        zero_frac = float(np.mean(np.abs(lasso.coef_) < 1e-10))
        if zero_frac > 0.30:
            model = lasso
            self._chosen = "Lasso"
            self._alpha  = float(lasso.alpha_)
        else:
            model = RidgeCV(alphas=alphas, cv=5).fit(Xs, y)
            self._chosen = "Ridge"
            self._alpha  = float(model.alpha_)

        scale_ = self._scaler.scale_
        mean_  = self._scaler.mean_
        self._coef      = model.coef_ / scale_
        self._intercept = float(model.intercept_ - np.dot(model.coef_ / scale_, mean_))
        self._r2        = float(r2_score(y, model.predict(Xs)))
        return self

    def predict(self, X):
        return X @ self._coef + self._intercept

    def equation_str(self, names):
        terms = [f"{c:.4f}·{n}" for n, c in zip(names, self._coef) if abs(c) > 1e-10]
        bias  = f"+ {self._intercept:.4f}" if self._intercept >= 0 else f"- {abs(self._intercept):.4f}"
        return "y = " + " + ".join(terms) + " " + bias

    def frozen_fn(self) -> Callable[[torch.Tensor], torch.Tensor]:
        """Dondurulmuş lineer denklem → differentiable PyTorch fonksiyonu."""
        coef_t = torch.tensor(self._coef, dtype=torch.float32)
        bias_t = torch.tensor(self._intercept, dtype=torch.float32)

        def _fn(x: torch.Tensor) -> torch.Tensor:
            return x @ coef_t + bias_t

        return _fn


class _PolyEngine:
    """
    İki aşamalı polinomik sembolik motor.
    
    Ön Eğitim (pre-training):  Geniş polinom uzayı (derece 3) taranır,
                                ElasticNetCV ile seyrek katsayılar bulunur.
    Rafine Eğitim (refine):    Sıfır-olmayan terimler alınır, Ridge ile
                               daha hassas katsayılar hesaplanır.
    """

    def __init__(self, feature_names=None, pretrain_degree=3, refine_degree=3):
        self.feature_names    = feature_names
        self.pretrain_degree  = pretrain_degree
        self.refine_degree    = refine_degree
        self._coef_pretrain   = None
        self._r2_pretrain     = None
        self._coef_refine     = None
        self._intercept       = 0.0
        self._r2_refine       = None
        self._poly_pretrain   = None
        self._poly_refine     = None
        self._active_mask     = None
        self._term_names      = None
        self._scaler          = StandardScaler()

    def fit_pretrain(self, X, y):
        """Faz 1a: Geniş arama. Y için de içsel ölçekleme (Target Scaling) eklenmiştir."""
        from sklearn.preprocessing import StandardScaler
        
        self._poly_pretrain = PolynomialFeatures(degree=self.pretrain_degree, include_bias=False)
        Xp = self._poly_pretrain.fit_transform(X)

        self._term_names = self._poly_pretrain.get_feature_names_out(self.feature_names)
        
        # Özellikler için kalkan
        self._scaler = StandardScaler()
        Xp_scaled = self._scaler.fit_transform(Xp)
        
        # HEDEF (y) için kalkan
        self._y_scaler = StandardScaler()
        y_scaled = self._y_scaler.fit_transform(y.reshape(-1, 1)).flatten()
        
        # Optimize edilmiş ölçekli uzayda eğitim (Hatalar burada ölür)
        model = ElasticNetCV(cv=5, l1_ratio=0.5, max_iter=10000, tol=1e-3, n_jobs=-1)
        model.fit(Xp_scaled, y_scaled)
        
        # Katsayıları ham SI uzayına geri döndürme matematiği
        # w_raw = (w_scaled * sigma_y) / sigma_x
        w_raw = (model.coef_ * self._y_scaler.scale_[0]) / self._scaler.scale_
        
        # b_raw = mu_y + sigma_y * b_scaled - sum(w_raw * mu_x)
        b_raw = self._y_scaler.mean_[0] + (self._y_scaler.scale_[0] * model.intercept_) - np.dot(w_raw, self._scaler.mean_)
        
        self._coef_pretrain = w_raw
        self._intercept = float(b_raw)
        
        # Maske belirleme
        self._active_mask = np.abs(self._coef_pretrain) > 1e-15
        
        y_pred = Xp @ self._coef_pretrain + self._intercept
        self._r2_pretrain = float(r2_score(y, y_pred))
        return self

    def fit_refine(self, X, y):
        """
        Rafine aşama: aktif terimler üzerinde Ridge ile ince ayar.
        Tam Otonom SI Uyumluluğu: Matris çökmesini engellemek için içsel ölçekleme yapılır
        ve katsayılar tekrar orijinal SI uzayına geri yansıtılır.
        """
        if self._poly_pretrain is None:
            raise RuntimeError("Önce fit_pretrain() çağırılmalı.")

        Xp   = self._poly_pretrain.transform(X)
        Xact = Xp[:, self._active_mask]

        if Xact.shape[1] == 0:
            _log.warning("Rafine: aktif terim bulunamadı, ön eğitim sonucu kullanılıyor.")
            self._r2_refine = self._r2_pretrain
            return self

        # 1. İÇSEL KALKAN: Ridge için özellikleri norm uzayına çek (Matrislerin çökmesini engeller)
        refine_scaler = StandardScaler()
        Xact_scaled = refine_scaler.fit_transform(Xact)

        # 2. OPTİMİZASYON: Güvenli uzayda modeli eğit
        alphas = np.logspace(-4, 3, 40)
        model  = RidgeCV(alphas=alphas, cv=5).fit(Xact_scaled, y)

        # 3. TERSİNE MÜHENDİSLİK: Katsayıları tekrar ham SI uzayına (orijinal gerçekliğe) projete et
        # w_orijinal = w_model / sigma
        # b_orijinal = b_model - sum(w_model * mu / sigma)
        scale_ = refine_scaler.scale_
        mean_  = refine_scaler.mean_
        
        w_raw = model.coef_ / scale_
        b_raw = float(model.intercept_ - np.dot(model.coef_ / scale_, mean_))

        # 4. NİHAİ KATSAYILARI KAYDET (Dış dünya artık saf SI denklemini görecek)
        coef_full = np.zeros(len(self._active_mask))
        coef_full[self._active_mask] = w_raw
        self._coef_refine  = coef_full
        self._intercept    = b_raw
        
        # R² skorunu ham Xp ve ham katsayılar ile orijinal y üzerinden hesapla
        y_pred          = Xp @ coef_full + self._intercept
        self._r2_refine = float(r2_score(y, y_pred))

        _log.info("Rafine eğitim tamamlandı: R²=%.4f (Δ=%.4f)",
                  self._r2_refine, self._r2_refine - self._r2_pretrain)
        return self

    def predict(self, X):
        Xp   = self._poly_pretrain.transform(X)
        coef = self._coef_refine if self._coef_refine is not None else self._coef_pretrain
        return Xp @ coef + self._intercept

    def equation_str(self) -> str:
        coef = self._coef_refine if self._coef_refine is not None else self._coef_pretrain
        active_terms = []
        for name, c in zip(self._term_names, coef):
            if abs(c) > 1e-8:
                pretty = name.replace("^", "^").replace(" ", "·")
                active_terms.append(f"{c:.4f}·{pretty}")
        if not active_terms:
            return f"y = {self._intercept:.4f}  [sabit]"
        bias = f"+ {self._intercept:.4f}" if self._intercept >= 0 else f"- {abs(self._intercept):.4f}"
        return "y = " + " + ".join(active_terms[:8]) + \
               (f" + ... [{len(active_terms) - 8} terim daha]" if len(active_terms) > 8 else "") + \
               " " + bias

    def frozen_fn(self) -> Callable[[torch.Tensor], torch.Tensor]:
        """
        Dondurulmuş polinomik denklem → tam türevlenebilir PyTorch fonksiyonu.
        String ayrıştırma (parsing) yerine, doğrudan matrisin saf üs uzayı (powers_) kullanılır.
        Bu sayede hem her türlü isimlendirme (Π_1 vb.) sorunsuz çalışır, hem de tensor işlemi ışık hızındadır.
        """
        coef = (self._coef_refine if self._coef_refine is not None
                else self._coef_pretrain)
        
        # Aktif terimlerin üs matrisini ve katsayılarını çek (Saf lineer cebir)
        active_powers = self._poly_pretrain.powers_[self._active_mask]
        active_coefs  = coef[self._active_mask]
        intercept     = self._intercept

        # Dondurulmuş çapalar olarak tensorlere çevir
        powers_t = torch.tensor(active_powers, dtype=torch.float32)
        coefs_t  = torch.tensor(active_coefs, dtype=torch.float32)

        def _fn(x: torch.Tensor) -> torch.Tensor:
            # x ve katsayıların cihazlarını eşitle (CPU/GPU güvenliği)
            p_t = powers_t.to(x.device)
            c_t = coefs_t.to(x.device)
            
            # x'i genişlet: (Batch, 1, Özellik_Sayısı)
            x_expanded = x.unsqueeze(1)
            
            # Üsleri al: torch.pow(0, 0) matematiksel olarak 1.0 döner (tam istediğimiz şey)
            # terms boyutu: (Batch, Aktif_Terim_Sayısı, Özellik_Sayısı)
            terms = torch.pow(x_expanded, p_t)
            
            # Değişkenleri kendi içinde çarp (Örn: x0^2 * x1^1)
            # terms_prod boyutu: (Batch, Aktif_Terim_Sayısı)
            terms_prod = torch.prod(terms, dim=2)
            
            # Aktif katsayılarla matris çarpımı yapıp sabiti ekle
            return torch.matmul(terms_prod, c_t) + intercept

        return _fn



class _PySREngine:
    """
    Genetik Sembolik Regresyon Motoru (PySR).
    Sıfırıncı Bakış Açısı: Polinomik taklitler yerine mutlak formülü evrimsel olarak keşfeder.
    """
    def __init__(self, feature_names=None):
        self.feature_names = feature_names
        self.model = None
        self._r2 = None

    def fit(self, X, y):
        # Sadece ihtiyaç anında import edilir
        from pysr import PySRRegressor
        
        # Güncellenmiş, temiz PySR API'si
        self.model = PySRRegressor(
            niterations=1000,      # İsteğin doğrultusunda 1000'e çıkardık
            population_size=40,    # Daha fazla çeşitlilik (evrim için kritik)
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["exp", "log", "sin", "inv"],
            model_selection="best",
            # loss yerine elementwise_loss kullanıyoruz
            elementwise_loss="loss(prediction, target) = (prediction - target)^2",
            # enable_autodiff=True satırını KALDIRDIK (artık otomatik)
            verbosity=0
        )
        
        # PySR'ı çalıştır
        self.model.fit(X, y, variable_names=self.feature_names)
        
        y_pred = self.model.predict(X)
        self._r2 = float(r2_score(y, y_pred))
        return self

    def predict(self, X):
        return self.model.predict(X)

    def equation_str(self) -> str:
        # En iyi denklemi SymPy formatında döndürür
        return f"y = {self.model.sympy()}"

    def frozen_fn(self) -> Callable[[torch.Tensor], torch.Tensor]:
        """
        PySR'ın en vahşi yeteneği: Bulduğu denklemi anında PyTorch Neural Network
        katmanına (differentiable node) çevirir. Geriye yayılım zinciri kopmaz.
        """
        pt_model = self.model.pytorch()
        
        def _fn(x: torch.Tensor) -> torch.Tensor:
            # x'in cihazını (CPU/GPU) modele uygula ve tahmin al
            pt_model.to(x.device)
            return pt_model(x).squeeze()
            
        return _fn

# ─── Ana Yönlendirici ─────────────────────────────────────────────────────────

class TIMURSymbolicRouter:
    """
    TIMUR XAI Sembolik Yönlendirici — Tek giriş noktası.

    Faz 1 (Ön Eğitim):  Ham veriyi analiz eder, sembolik denklem çıkarır.
    Faz 2 (Dondurma):   Denklemi sabitler; frozen_fn döndürür.
                         Bu fonksiyon timur/pinn/loss.py'ye kayıp terimi
                         olarak geçirilir.

    Parameters
    ----------
    linear_threshold : float  Lineer eşik. Varsayılan: 0.40
    pysr_threshold   : float  PySR eşiği.  Varsayılan: 0.65
    feature_names    : list[str] | None
    pretrain_degree  : int    Ön eğitim polinom derecesi. Varsayılan: 3
    refine_degree    : int    Rafine eğitim derecesi. Varsayılan: 3
    random_state     : int
    verbose          : bool

    Örnek
    -----
    >>> router = TIMURSymbolicRouter(verbose=True)
    >>> result = router.discover(X_train, y_train)
    >>> print(result.equation_str)
    >>> frozen_phi = result.frozen_fn   # PyTorch'a hazır
    """

    def __init__(
        self,
        linear_threshold : float = 0.40,
        pysr_threshold   : float = 0.65,
        feature_names    : Optional[List[str]] = None,
        pretrain_degree  : int   = 3,
        refine_degree    : int   = 3,
        random_state     : int   = 42,
        verbose          : bool  = False,
    ) -> None:
        self.linear_threshold = linear_threshold
        self.pysr_threshold   = pysr_threshold
        self.feature_names    = feature_names
        self.pretrain_degree  = pretrain_degree
        self.refine_degree    = refine_degree
        self.random_state     = random_state
        self.verbose          = verbose

        self._gatekeeper = Gatekeeper(
            linear_threshold = linear_threshold,
            pysr_threshold   = pysr_threshold,
            feature_names    = feature_names,
            random_state     = random_state,
        )
        self._result: Optional[DiscoveryResult] = None

    def discover(self, X: np.ndarray, y: np.ndarray) -> DiscoveryResult:
        """
        Tam keşif süreci: Gatekeeper → Ön Eğitim → Rafine → Dondurma.

        Returns
        -------
        DiscoveryResult
            equation_str, frozen_fn, r2_pretrain, r2_refine, ...
        """
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float).ravel()
        t0 = time.perf_counter()

        # ── Adım 1: Gatekeeper ──────────────────────────────────────────
        if self.verbose:
            print("\n[TIMUR] ─── Faz 0+1: Gatekeeper Analizi ───────────────")
        gk_report = self._gatekeeper.analyze(X, y)
        routing   = gk_report.routing_decision
        nl_pct    = gk_report.nonlinearity_score

        if self.verbose:
            print(f"  NonLineerlik: {nl_pct:.1%}  →  Yönlendirme: {routing.upper()}")

        names = self.feature_names or [f"x{i}" for i in range(X.shape[1])]

        # ── Adım 2+3: Motor seçimi + Ön Eğitim + Rafine ─────────────────

        # ── Adım 2+3: Motor seçimi + Ön Eğitim + Rafine ─────────────────
        if routing == "linear":
            result = self._run_linear(X, y, names, gk_report, t0)
        elif routing == "pysr":
            result = self._run_pysr(X, y, names, gk_report, t0)
        else:
            result = self._run_poly(X, y, names, gk_report, routing, t0)
            
        self._result = result

        if self.verbose:
            print(f"\n[TIMUR] ─── Keşif Tamamlandı ──────────────────────────")
            print(f"  Denklem     : {result.equation_str}")
            print(f"  Ön Eğitim R²: {result.r2_pretrain:.4f}")
            print(f"  Rafine R²   : {result.r2_refine:.4f}")
            print(f"  Süre        : {result.fit_time_s:.2f}s")

        return result

    def _run_linear(self, X: np.ndarray, y: np.ndarray, names: list,
                    gk_report, t0: float):
        n_samples, n_features_original = X.shape

        if n_features_original == 0:
            X_fit = np.ones((n_samples, 1))
            names_fit = ["Mutlak_Sabit"]
        else:
            X_fit = X
            names_fit = names

        from sklearn.linear_model import LinearRegression
        import torch
        
        # Ridge(alpha=1.0) yerine saf Doğrusal Regresyon (Cezasız, Ölçek Bağımsız)
        model = LinearRegression()
        model.fit(X_fit, y)
        
        if n_features_original == 0:
            eq_str = f"y = {model.intercept_:.4f}  [Saf Sabit]"
            bias_t = float(model.intercept_)
            def frozen_fn(x: torch.Tensor) -> torch.Tensor:
                return torch.full((x.shape[0],), bias_t, device=x.device, dtype=torch.float32)
        else:
            # 1e-25 gibi astronomik derecede küçük katsayıları bile koru
            terms = [f"{coef:.4g}*{name}" for coef, name in zip(model.coef_, names_fit) if abs(coef) > 1e-35]
            eq_str = "y = " + " + ".join(terms) + f" + {model.intercept_:.4g}"
            coef_t = torch.tensor(model.coef_, dtype=torch.float32)
            intercept_t = torch.tensor(model.intercept_, dtype=torch.float32)
            def frozen_fn(x: torch.Tensor) -> torch.Tensor:
                return torch.matmul(x, coef_t) + intercept_t

        r2 = float(model.score(X_fit, y))

        return DiscoveryResult(
            equation_str = eq_str,
            frozen_fn    = frozen_fn,
            coef_        = model.coef_,
            intercept_   = float(model.intercept_),
            r2_pretrain  = r2,
            r2_refine    = r2,
            routing      = "linear",
            gatekeeper   = gk_report,
            fit_time_s   = time.perf_counter() - t0,
        )
    
    
    def _run_poly(self, X, y, names, gk_report, routing, t0) -> DiscoveryResult:
        if self.verbose:
            print(f"\n[TIMUR] ─── Faz 1 Ön Eğitim: Polinom (derece={self.pretrain_degree}) ─")

        eng = _PolyEngine(feature_names=names,
                          pretrain_degree=self.pretrain_degree,
                          refine_degree=self.refine_degree)

        # Faz 1a: Ön eğitim — geniş arama
        eng.fit_pretrain(X, y)
        r2_pre = eng._r2_pretrain

        if self.verbose:
            active_n = int(eng._active_mask.sum())
            total_n  = len(eng._active_mask)
            print(f"  Ön eğitim R²: {r2_pre:.4f}   [{active_n}/{total_n} terim aktif]")
            print(f"\n[TIMUR] ─── Faz 2 Rafine Eğitim: Ridge ince ayar ──────")

        # Faz 1b: Rafine — aktif terimler üzerinde hassas iyileştirme
        eng.fit_refine(X, y)
        r2_ref = eng._r2_refine

        if self.verbose:
            print(f"  Rafine R²   : {r2_ref:.4f}")

        eq = eng.equation_str()

        return DiscoveryResult(
            equation_str = eq,
            frozen_fn    = eng.frozen_fn(),
            coef_        = eng._coef_refine if eng._coef_refine is not None else eng._coef_pretrain,
            intercept_   = eng._intercept,
            r2_pretrain  = r2_pre,
            r2_refine    = r2_ref,
            routing      = routing,
            gatekeeper   = gk_report,
            fit_time_s   = time.perf_counter() - t0,
            metadata     = {
                "active_terms": int(eng._active_mask.sum()),
                "total_terms" : len(eng._active_mask),
                "term_names"  : [n for n, m in zip(eng._term_names, eng._active_mask) if m],
            },
        )
    
    def _run_pysr(self, X, y, names, gk_report, t0) -> DiscoveryResult:
        if self.verbose:
            print(f"\n[TIMUR] ─── Faz 1 Ön Eğitim: PySR (Genetik Motor) ──────────")
            
        eng = _PySREngine(feature_names=names)
        eng.fit(X, y)
        
        eq = eng.equation_str()
        r2 = eng._r2
        
        if self.verbose:
            print(f"  PySR Keşif R² : {r2:.4f}  [Evrimsel arama tamamlandı]")
            
        return DiscoveryResult(
            equation_str = eq,
            frozen_fn    = eng.frozen_fn(),
            coef_        = np.array([]), # PySR katsayıları denkleme gömülüdür
            intercept_   = 0.0,
            r2_pretrain  = r2,
            r2_refine    = r2,       # PySR'da rafine aşaması yoktur, tek atışta bulur
            routing      = "pysr",
            gatekeeper   = gk_report,
            fit_time_s   = time.perf_counter() - t0,
        )

    @property
    def last_result(self) -> Optional[DiscoveryResult]:
        return self._result