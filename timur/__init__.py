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

__version__ = "1.0.0"
__author__  = "Timur"

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
        self.verbose          = verbose

        self._router   = None
        self._result   = None
        self._net      = None
        self._history  = None
        self._fitted   = False
        self._n_feat   = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TIMURModel":
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32).ravel()
        self._n_feat = X.shape[1]


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
            X, y, self.feature_names = self._dim_analyzer.transform_to_pi(X, y, const_vals)
            self._n_feat = X.shape[1]

        self._router = TIMURSymbolicRouter(
            linear_threshold = self.linear_threshold,
            pysr_threshold   = self.pysr_threshold,
            feature_names    = self.feature_names,
            verbose          = self.verbose,
        )
        self._result = self._router.discover(X, y)

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)

        self._net = TIMURNet(
            n_features  = self._n_feat,
            frozen_fn   = self._result.frozen_fn,
            hidden_dims = self.hidden_dims,
            activation  = self.activation,
        )

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
        )
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X_t = torch.tensor(np.array(X, dtype=np.float32), dtype=torch.float32)
        self._net.eval()
        with torch.no_grad():
            return self._net(X_t).numpy()

    def predict_symbolic(self, X: np.ndarray) -> np.ndarray:
        """Sadece dondurulmuş sembolik denklem ile tahmin."""
        self._check_fitted()
        X_t = torch.tensor(np.array(X, dtype=np.float32), dtype=torch.float32)
        with torch.no_grad():
            return self._result.frozen_fn(X_t).numpy()

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
        r = self._result
        sep  = "━" * 56
        thin = "─" * 56
        lines = [
            "",
            sep,
            "  TIMUR XAI — TAM ANALİZ RAPORU  v1.0",
            sep,
            "  SEMBOLİK KEŞİF:",
            f"  {r.equation_str}",
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
            f"  L_total = L_data + {self.lambda_sym:.3f} · L_symbolic",
            sep,
            "",
        ]
        return "\n".join(lines)

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