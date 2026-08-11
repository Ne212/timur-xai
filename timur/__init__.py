"""
timur — TIMUR XAI Çerçevesi
═══════════════════════════════════════════════════════════════════════════════
Açıklanabilir ve Fizik-Kısıtlı Makine Öğrenmesi

TİMUR (Teorik İlkelerle Makine Usulü Regresyon) üç aşamalı çalışır:

  1. Sembolik Keşif    → Veriyi analiz et, denklem bul, dondur
  2. PINN Entegrasyonu → Dondurulmuş denklemi sinir ağı kaybına bağla
  3. Hibrit Eğitim     → L_total = L_data + λ · L_symbolic

Hızlı Başlangıç:
    from timur import TIMURModel

    model = TIMURModel(lambda_sym=0.1, verbose=True)
    model.fit(X_train, y_train)
    print(model.equation)
    y_pred = model.predict(X_test)

Versiyon: 1.0.0
Yazar   : Neşet
"""

from timur.symbolic.router import (
    TIMURSymbolicRouter,
    Gatekeeper,
    GatekeeperReport,
    FeatureProfile,
    DiscoveryResult,
)
from timur.pinn.loss import (
    TIMURLoss,
    TIMURNet,
    timur_fit,
)
from timur.symbolic.dimensions import DimensionalAnalyzer

__version__ = "1.2.0"
__author__  = "Neşet Koçkar"

import numpy as np
import torch
from sklearn.metrics import r2_score as _r2_score
from typing import Optional, List


class TIMURModel:
    """
    TIMUR XAI — Tek Giriş Noktası (Scikit-learn uyumlu API).

    fit() ile:
        1. Sembolik denklem keşfedilir ve dondurulur (Faz 1)
        2. Hibrit sinir ağı eğitilir (Faz 2+3)

    Parameters
    ----------
    lambda_sym      : float    Sembolik kısıtlama kuvveti. Varsayılan: 0.1
    linear_threshold: float    Lineer eşik. Varsayılan: 0.40
    pysr_threshold  : float    PySR eşiği. Varsayılan: 0.65
    hidden_dims     : list     Ağ gizli katmanları. Varsayılan: [64, 32]
    epochs          : int      Eğitim epoch. Varsayılan: 500
    lr              : float    Öğrenme hızı. Varsayılan: 1e-3
    feature_names   : list     Özellik isimleri
    early_stop_patience: int|None
        Verilirse VE fit() çağrısında X_val/y_val sağlanırsa: val R² bu kadar
        epoch boyunca iyileşmezse eğitim erken durdurulur ve ağ en iyi val
        R²'ye sahip ağırlıklara geri yüklenir. None (varsayılan) ile,
        X_val/y_val verilse de erken durdurma YAPILMAZ ama en iyi ağırlıklar
        yine de sonunda geri yüklenir (overfitting'e karşı "ücretsiz" koruma).
    verbose         : bool     Ekrana yaz. Varsayılan: False
    """

    def __init__(
        self,
        lambda_sym       : float = 0.1,
        linear_threshold : float = 0.40,
        pysr_threshold   : float = 0.65,
        hidden_dims      : Optional[List[int]] = None,
        epochs           : int   = 500,
        lr               : float = 1e-3,
        feature_names    : Optional[List[str]] = None,
        feature_dims     : Optional[List[dict]] = None,
        target_dim       : Optional[dict] = None,
        constants        : Optional[dict] = None,
        activation       : str   = "tanh",
        data_loss        : str   = "mse",
        early_stop_patience: Optional[int] = None,
        verbose          : bool  = False,
    ) -> None:
        self.lambda_sym       = lambda_sym
        self.feature_dims = feature_dims
        self.target_dim   = target_dim
        self.constants    = constants or {}
        self.linear_threshold = linear_threshold
        self.pysr_threshold   = pysr_threshold
        self.hidden_dims      = hidden_dims or [64, 32]
        self.epochs           = epochs
        self.lr               = lr
        self.feature_names    = feature_names
        self.activation       = activation
        self.data_loss        = data_loss
        self.early_stop_patience = early_stop_patience
        self.verbose          = verbose

        self._router                = None
        self._result                = None
        self._net                   = None
        self._history               = None
        self._fitted                = False
        self._n_feat                = None
        self._physical_equation_str = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "TIMURModel":
        """
        X_val, y_val verilirse (opsiyonel): PINN eğitimi (Faz 3) sırasında
        her epoch'ta bağımsız bir val R² hesaplanır, en iyi val R²'ye sahip
        ağırlıklar saklanır ve eğitim sonunda ağ o duruma geri yüklenir
        (bkz. early_stop_patience). Sembolik keşif (Faz 1) HER ZAMAN sadece
        X/y (train) üzerinde çalışır - val seti sızıntısı yoktur.
        """
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32).ravel()
        self._n_feat = X.shape[1]
        # Orijinal feature isimlerini Pi dönüşümünden önce sakla
        self._orig_feature_names = list(self.feature_names) if self.feature_names else None
        # SI uzayındaki ham veriyi sakla — fit sonu gerçek R² logu için
        self._X_si_backup = X.copy()
        self._y_si_backup = y.copy()

        # YENİ: Boyutsuzlaştırma Katmanı (Buckingham Pi)
        if self.feature_dims and self.target_dim:
            if self.verbose:
                print("\n[TIMUR] ─── Boyutsuzlaştırma (Buckingham Pi) Aktif ───")
            
            const_dims = [c[1] for c in self.constants.values()]
            const_vals = {k: c[0] for k, c in self.constants.items()}
            
            self._dim_analyzer = DimensionalAnalyzer(
                feature_dims=self.feature_dims, 
                target_dim=self.target_dim, 
                constant_dims=const_dims
            )
            
            # X ve y artık boyutsuz uzayda (Pi matrisi)
            # display_names: rapor/grafik için; pysr_names: sembolik keşif motoru için
            X, y, self.feature_names = self._dim_analyzer.transform_to_pi(
                X, y, const_vals, feature_names=self.feature_names
            )
            self._n_feat = X.shape[1]

        # Router'a PySR uyumlu isimleri ver; yoksa display isimleri kullan
        _router_feature_names = (
            getattr(self._dim_analyzer, '_pysr_feature_names', None)
            if hasattr(self, '_dim_analyzer') else None
        ) or self.feature_names

        self._router = TIMURSymbolicRouter(
            linear_threshold = self.linear_threshold,
            pysr_threshold   = self.pysr_threshold,
            feature_names    = _router_feature_names,
            verbose          = self.verbose,
        )
        self._result = self._router.discover(X, y)

        # Denklem string'indeki PySR-safe isimleri display isimleriyle değiştir.
        # Örnek: "Pi_dalga_boyu_sicaklik_h_n1_c_n1_kB" → "Pi(dalga_boyu·sicaklik·h⁻¹·c⁻¹·kB)"
        if hasattr(self, '_dim_analyzer'):
            pysr_names    = getattr(self._dim_analyzer, '_pysr_feature_names', []) or []
            display_names = self.feature_names or []
            eq_pysr_safe  = self._result.equation_str  # PySR-safe isimleri sakla

            # Fiziksel denklemi (F=ma tarzı) geri-dönüştür — display'den ÖNCE
            const_vals_for_phys = {k: c[0] for k, c in self.constants.items()}
            _, self._physical_equation_str = self._dim_analyzer.get_physical_equation(
                eq_pysr_safe,
                feature_names   = self._orig_feature_names,
                constants_dict  = const_vals_for_phys,
            )

            eq = eq_pysr_safe
            # Uzun isimden kısaya sırayla replace — prefix çakışmasını önler
            for pysr_n, disp_n in sorted(
                zip(pysr_names, display_names),
                key=lambda p: len(p[0]), reverse=True
            ):
                eq = eq.replace(pysr_n, disp_n)
            self._result.equation_str = eq

            if self.verbose:
                print(f"\n[TIMUR] ─── Denklem (boyutsuz Pi → display) ─────────────────")
                print(f"  {eq}")
                print(f"\n[TIMUR] ─── Fiziksel Denklem (F=ma tarzı) ───────────────────")
                print(f"  {self._physical_equation_str}")

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)

        # ── Saf-Sabit Durumu (Buckingham Pi tamamen belirledi) ──────────────
        # X_pi boşsa (in_features=0), sistem saf-sabit anlamına gelir:
        # φ(X) = c → y = c · dimensional_factor
        # PINN fazı anlamsız; frozen_fn'i direkt döndür.
        if self._n_feat == 0:
            if self.verbose:
                print("\n[TIMUR] ─── Saf-Sabit Sistem: PINN fazı atlandı ───────────────")
                print("  Pi teoremi sistemi tamamen belirledi → ağ eğitimine gerek yok.")

            # Saf-sabit durumda "y = 0.9997 [Saf Sabit]" yerine anlamlı SI denklemi üret.
            # Pi_target = y / (değişken kombinasyonu) = c
            # → y = c * (değişken kombinasyonu)
            # dim_analyzer'dan hedef Pi grubunun çarpan yapısını çekeriz.
            if hasattr(self, '_dim_analyzer'):
                self._result.equation_str = self._build_si_equation_str(
                    self._result.equation_str
                )
                if self.verbose:
                    print(f"  SI Denklemi: {self._result.equation_str}")

            self._net = None
            self._history = {"L_total": [], "L_data": [], "L_symbolic": []}
            self._fitted = True
            return self
        # ────────────────────────────────────────────────────────────────────

        # Olcekleme istatistikleri (egitim verisinden, z-skoru) - TIMURNet
        # icinde bunlari kullanip Tanh doygunlugu/yavas yakinsama sorununu
        # onler. X/y burada hangi uzaydaysa (Buckingham Pi aktifse Pi uzayi,
        # degilse ham SI/dogal uzay) o uzayin istatistikleri hesaplanir -
        # frozen_fn/TIMURLoss/predict ile tutarli (onlar da ayni X uzayini kullanir).
        x_mean = X.mean(axis=0)
        x_std = X.std(axis=0)
        y_mean = float(y.mean())
        y_std = float(y.std())

        self._net = TIMURNet(
            n_features  = self._n_feat,
            frozen_fn   = self._result.frozen_fn,
            hidden_dims = self.hidden_dims,
            activation  = self.activation,
            x_mean      = x_mean,
            x_std       = x_std,
            y_mean      = y_mean,
            y_std       = y_std,
        )

        # ── Doğrulama (validation) seti — varsa aynı uzaya (Pi aktifse Pi
        # uzayına, değilse ham uzaya) dönüştürülür ve timur_fit'e geçirilir.
        X_val_t = None
        y_val_t = None
        if X_val is not None and y_val is not None:
            X_val_arr = np.array(X_val, dtype=np.float32)
            y_val_arr = np.array(y_val, dtype=np.float32).ravel()
            if hasattr(self, '_dim_analyzer'):
                const_vals = {k: c[0] for k, c in self.constants.items()}
                X_val_arr, y_val_arr, _ = self._dim_analyzer.transform_to_pi(
                    X_val_arr, y_val_arr, const_vals,
                    feature_names=self._orig_feature_names,
                )
            X_val_t = torch.tensor(X_val_arr, dtype=torch.float32)
            y_val_t = torch.tensor(y_val_arr, dtype=torch.float32)

        self._history = timur_fit(
            net        = self._net,
            X_train    = X_t,
            y_train    = y_t,
            frozen_fn  = self._result.frozen_fn,
            lambda_sym = self.lambda_sym,
            epochs     = self.epochs,
            lr         = self.lr,
            data_loss  = self.data_loss,
            verbose    = self.verbose,
            X_val      = X_val_t,
            y_val      = y_val_t,
            early_stop_patience = self.early_stop_patience,
        )

        # timur_fit'in son R² logu Pi uzayında hesaplanır ve yanıltıcıdır.
        # Boyutsuzlaştırma aktifse, SI uzayındaki gerçek R²'yi yeniden hesapla ve yazdır.
        if self.verbose and hasattr(self, '_dim_analyzer') and self._dim_analyzer is not None:
            self._fitted = True  # predict için geçici olarak işaretle
            from sklearn.metrics import r2_score as _r2_tmp
            _X_orig = np.array(X_t.numpy(), dtype=np.float32)
            # X_t Pi uzayında — inverse transform için orijinal X lazım, _X_orig_si'yi sakladık mı?
            # Saklamadık; ama y_true Pi uzayında y_t var, frozen_fn de Pi uzayında çalışıyor.
            # Gerçek SI R²'yi predict_symbolic ile hesaplamak için _X_orig_si gerekiyor.
            # fit() başında sakla — aşağıda _X_si_backup olarak erişeceğiz.
            if hasattr(self, '_X_si_backup') and hasattr(self, '_y_si_backup'):
                y_si_pred = self.predict_symbolic(self._X_si_backup)
                r2_si = float(_r2_tmp(self._y_si_backup, y_si_pred))
                print(f"\n  [SI Uzayı R²] {r2_si:.4f}  ← Pi normalizasyonu çıkarıldı")

        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Ham veriyi (X) eğitilmiş boyutsuz Pi uzayına dönüştürür."""
        self._check_fitted()
        if hasattr(self, '_dim_analyzer') and self._dim_analyzer is not None:
            const_vals = {k: c[0] for k, c in self.constants.items()}
            dummy_y = np.ones(X.shape[0])
            # fit() sırasında feature_names zaten Pi isimlerine dönüşmüş olur,
            # transform() orijinal isimleri değil Pi isimlerini korur.
            # Burada DimensionalAnalyzer'ın orijinal feature boyutunu kullanmak için
            # all_dims'ten geri hesaplıyoruz.
            n_orig_features = len(self._dim_analyzer.feature_dims)
            orig_feat_names = self._orig_feature_names  # fit'te saklanır
            X_pi, _, _ = self._dim_analyzer.transform_to_pi(
                X, dummy_y, const_vals, feature_names=orig_feat_names
            )
            return X_pi
        return X

    def predict(self, X: np.ndarray) -> np.ndarray:
        """PINN ağı üzerinden tahmin yapar ve sonucu klasik SI birimlerine geri yansıtır.
        Saf-sabit sistemlerde (in_features=0) sembolik denklem direkt kullanılır."""
        self._check_fitted()
        X_transformed = self.transform(X)
        X_t = torch.tensor(np.array(X_transformed, dtype=np.float32), dtype=torch.float32)

        if self._net is None:
            # Saf-sabit sistem: ağ yok, frozen_fn'i kullan
            with torch.no_grad():
                y_pi_pred = self._result.frozen_fn(X_t).numpy().flatten()
        else:
            self._net.eval()
            with torch.no_grad():
                y_pi_pred = self._net(X_t).numpy().flatten()

        if hasattr(self, '_dim_analyzer') and self._dim_analyzer is not None:
            const_vals = {k: c[0] for k, c in self.constants.items()}
            return self._dim_analyzer.inverse_transform_target(X, y_pi_pred, const_vals)

        return y_pi_pred

    def predict_symbolic(self, X: np.ndarray) -> np.ndarray:
        """Sembolik motorun bulduğu denklem üzerinden tahmin yapar ve SI birimlerine geri yansıtır."""
        self._check_fitted()
        X_transformed = self.transform(X)
        X_t = torch.tensor(np.array(X_transformed, dtype=np.float32), dtype=torch.float32)

        with torch.no_grad():
            y_pi_pred = self._result.frozen_fn(X_t).numpy().flatten()

        if hasattr(self, '_dim_analyzer') and self._dim_analyzer is not None:
            const_vals = {k: c[0] for k, c in self.constants.items()}
            return self._dim_analyzer.inverse_transform_target(X, y_pi_pred, const_vals)

        return y_pi_pred

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(_r2_score(y, self.predict(X)))

    def score_symbolic(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(_r2_score(y, self.predict_symbolic(X)))

    @property
    def equation(self) -> str:
        self._check_fitted()
        return self._result.equation_str

    @property
    def discovery_result(self) -> "DiscoveryResult":
        self._check_fitted()
        return self._result

    @property
    def gatekeeper_report(self) -> "GatekeeperReport":
        self._check_fitted()
        return self._result.gatekeeper

    @property
    def training_history(self):
        self._check_fitted()
        return self._history

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def get_xai_report(self) -> str:
        """Tam XAI raporu."""
        self._check_fitted()
        r    = self._result
        sep  = "━" * 56
        thin = "─" * 56
        lines = [
            "",
            sep,
            "  TIMUR XAI — TAM ANALİZ RAPORU  v1.0",
            sep,
            "  SEMBOLİK KEŞİF (boyutsuz Pi uzayı):",
            f"  {r.equation_str}",
        ]

        # Fiziksel denklem (F=ma tarzı) — boyutsuzlaştırma aktifse
        if self._physical_equation_str:
            lines += [
                thin,
                "  FİZİKSEL DENKLEM (orijinal değişkenler):",
                *[f"  {ln}" for ln in self._physical_equation_str.splitlines()],
            ]

        lines += [
            thin,
            f"  Ön Eğitim R²     : {r.r2_pretrain:.4f}",
            f"  Rafine R²        : {r.r2_refine:.4f}",
            f"  Motor            : {r.routing.upper()}",
            f"  Keşif Süresi     : {r.fit_time_s:.2f}s",
            thin,
            "  GATEKEEPER ANALİZİ:",
            f"  NonLineerlik     : {r.gatekeeper.nonlinearity_score:.1%}",
            f"  Yönlendirme      : {r.routing.upper()}",
            f"  Veri             : {r.gatekeeper.n_samples} örnek × {r.gatekeeper.n_features} özellik",
            thin,
            "  KAYIP YAPISI:",
            (
                "  Saf-sabit sistem: PINN fazı atlandı (in_features=0)"
                if self._net is None
                else f"  L_total = L_data + {self.lambda_sym:.3f} · L_symbolic"
            ),
            sep,
            "",
        ]
        return "\n".join(lines)

    def _build_si_equation_str(self, pi_eq_str: str) -> str:
        """
        Saf-sabit Pi denklemini ("y = 0.9997 [Saf Sabit]") SI uzayında
        anlamlı bir denklem string'ine çevirir.

        Pi_target = y / (σ·T⁴) = 0.9997
        → y = 0.9997 · σ · T⁴
        → çıktı: "y ≈ sigma · sicaklik⁴"  (üs=4 → sicaklik⁴)

        Katsayıyı (örn. 0.9997) string'den alır,
        dim_analyzer.pi_exponents'tan hedef Pi grubunun çarpan yapısını üretir.
        """
        import re
        # Katsayıyı Pi string'inden al ("y = 18.8441 [Saf Sabit]" → 18.8441)
        m = re.search(r'y\s*=\s*([\d.eE+\-]+)', pi_eq_str)
        coef = float(m.group(1)) if m else 1.0

        da   = self._dim_analyzer
        # Hedef Pi grubunun sütun indeksini bul
        target_row = len(da.all_dims) - 1
        target_col = -1
        for j in range(da.pi_exponents.shape[1]):
            if abs(da.pi_exponents[target_row, j]) > 1e-5:
                target_col = j
                break
        if target_col == -1:
            return pi_eq_str  # bulunamazsa orijinalini döndür

        # Tüm değişken isimlerini oluştur (orig features + sabitler)
        feat_names  = self._orig_feature_names or [f"x{i}" for i in range(len(da.feature_dims))]
        const_names = list(self.constants.keys())
        all_names   = feat_names + const_names  # hedef (y) hariç

        superscripts = {2: "²", 3: "³", 4: "⁴", -1: "⁻¹", -2: "⁻²", -3: "⁻³", -4: "⁻⁴"}

        parts = []
        for i, name in enumerate(all_names):
            p = da.pi_exponents[i, target_col]
            if abs(p) < 1e-5:
                continue
            # y üssü = 1 normalize edilmiş, diğerleri ters işaretli
            exp = -p  # Pi = y / (X^p) → y = Pi * X^p → X ile üs +p
            exp_r = round(exp, 6)
            exp_i = int(round(exp_r))
            if abs(exp_r - 1.0) < 1e-5:
                parts.append(name)
            elif abs(exp_r - exp_i) < 1e-5 and exp_i in superscripts:
                parts.append(f"{name}{superscripts[exp_i]}")
            else:
                parts.append(f"{name}^{exp_r:.3g}")

        if not parts:
            return pi_eq_str

        coef_str = f"{coef:.4g}" if abs(coef - 1.0) > 1e-3 else ""
        rhs = " · ".join(parts)
        if coef_str:
            rhs = f"{coef_str} · {rhs}"
        return f"y = {rhs}"

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("Model henüz eğitilmedi. Önce .fit(X, y) çağırın.")


__all__ = [
    "TIMURModel",
    "TIMURSymbolicRouter",
    "Gatekeeper",
    "GatekeeperReport",
    "FeatureProfile",
    "DiscoveryResult",
    "TIMURLoss",
    "TIMURNet",
    "timur_fit",
]