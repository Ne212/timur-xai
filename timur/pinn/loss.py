"""
timur/pinn/loss.py
═════════════════════════════════════════════════════════════════════════════
TIMUR XAI — Dondurulmuş Sembolik Denklem Kayıp Fonksiyonu

FAZ 2 — DONDURMA SİSTEMİ (Freeze System)
    TIMURSymbolicRouter.discover() tarafından bulunan fiziksel/matematiksel
    denklemi mutlak bir gerçeklik çapası olarak sabitler.
    Model ağırlıklarının bu temel denklemin formunu bozmasına izin vermez.

FAZ 3 — ENTEGRASYON
    Dondurulan bu denklemi ana sinir ağının kayıp fonksiyonuna bir
    regülarizasyon (kısıtlama) terimi olarak dahil eder:

        L_total = L_data + λ · L_symbolic

    Buradaki λ (ceza katsayısı), kullanıcının ağın veriye mi yoksa
    fiziksel formüle mi daha çok itaat edeceğini ayarlayabilmesi için
    dışarıdan parametrik olarak verilebilir.

Kullanım:
    from timur.pinn.loss import TIMURLoss, TIMURNet

    loss_fn = TIMURLoss(frozen_fn=result.frozen_fn, lambda_sym=0.1)
    model   = TIMURNet(n_features=2, frozen_fn=result.frozen_fn)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(500):
        y_pred = model(X_t)
        loss   = loss_fn(y_pred, y_true=y_t, X=X_t)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn

_log = logging.getLogger(__name__)


# ─── Kayıp Fonksiyonu ─────────────────────────────────────────────────────────

class TIMURLoss(nn.Module):
    """
    TIMUR XAI Hibrit Kayıp Fonksiyonu.

        L_total = L_data + λ · L_symbolic

    Tam türevlenebilir: frozen_fn içindeki tüm PyTorch operasyonları
    geriye yayılım (backpropagation) zincirine dahildir.

    Parameters
    ----------
    frozen_fn   : Callable[[Tensor], Tensor]
        TIMURSymbolicRouter.discover() → result.frozen_fn
        Dondurulmuş sembolik denklem φ(x) → ŷ_sym
    lambda_sym  : float
        Sembolik regülarizasyon kuvveti.
        0.0 → saf veri kaybı (L_data)
        1.0 → veri ve sembolik eşit ağırlıklı
        >1.0 → sembolik denklem baskın
        Varsayılan: 0.1
    data_loss   : 'mse' | 'mae' | 'huber'
        Veri kayıp türü. Varsayılan: 'mse'
    sym_loss    : 'mse' | 'mae'
        Sembolik regülarizasyon kayıp türü. Varsayılan: 'mse'
    huber_delta : float
        Huber kaybı için delta. Varsayılan: 1.0

    Örnek
    -----
    >>> loss_fn = TIMURLoss(frozen_fn=result.frozen_fn, lambda_sym=0.2)
    >>> y_pred  = model(X_tensor)
    >>> L       = loss_fn(y_pred, y_true=y_tensor, X=X_tensor)
    >>> L.backward()
    """

    def __init__(
        self,
        frozen_fn  : Callable[[torch.Tensor], torch.Tensor],
        lambda_sym : float = 0.1,
        data_loss  : str   = "mse",
        sym_loss   : str   = "mse",
        huber_delta: float = 1.0,
    ) -> None:
        super().__init__()
        self.frozen_fn   = frozen_fn
        self.lambda_sym  = lambda_sym
        self.data_loss   = data_loss
        self.sym_loss    = sym_loss
        self.huber_delta = huber_delta

        # Veri kaybı seçimi
        if data_loss == "mse":
            self._data_loss_fn = nn.MSELoss()
        elif data_loss == "mae":
            self._data_loss_fn = nn.L1Loss()
        elif data_loss == "huber":
            self._data_loss_fn = nn.HuberLoss(delta=huber_delta)
        else:
            raise ValueError(f"Bilinmeyen data_loss: '{data_loss}'. 'mse', 'mae', 'huber' kullanın.")

        # Sembolik kayıp seçimi
        if sym_loss == "mse":
            self._sym_loss_fn = nn.MSELoss()
        elif sym_loss == "mae":
            self._sym_loss_fn = nn.L1Loss()
        else:
            raise ValueError(f"Bilinmeyen sym_loss: '{sym_loss}'. 'mse', 'mae' kullanın.")

    def forward(
        self,
        y_pred : torch.Tensor,
        y_true : torch.Tensor,
        X      : torch.Tensor,
    ) -> torch.Tensor:
        """
        Toplam kayıp: L_total = L_data + λ · L_symbolic

        Parameters
        ----------
        y_pred : (batch,) — ağın tahmini
        y_true : (batch,) — gerçek değer
        X      : (batch, n_features) — giriş özellikleri

        Returns
        -------
        loss : scalar Tensor (gradyan akışlı)
        """
        # L_data: Ağ tahmini ile gerçek değer arasındaki kayıp
        L_data = self._data_loss_fn(y_pred.squeeze(), y_true.squeeze())

        if self.lambda_sym == 0.0:
            return L_data

        # Sembolik denklem tahmini — gradyan DIŞI (frozen)
        # Dondurulan φ bir gerçeklik çapası; onun gradyanını hesaplamıyoruz.
        with torch.no_grad():
            y_sym = self.frozen_fn(X).squeeze()

        # L_symbolic: Ağ tahmini ile sembolik denklem arasındaki fark
        # Bu terim ağı, keşfedilen denkleme sadık kalmaya zorlar.
        L_sym = self._sym_loss_fn(y_pred.squeeze(), y_sym)

        return L_data + self.lambda_sym * L_sym

    def component_losses(
        self,
        y_pred : torch.Tensor,
        y_true : torch.Tensor,
        X      : torch.Tensor,
    ) -> Dict[str, float]:
        """
        Her kaybı ayrı ayrı döndürür (diagnoz/debug için).

        Returns
        -------
        dict : {'L_data': ..., 'L_symbolic': ..., 'L_total': ..., 'lambda': ...}
        """
        with torch.no_grad():
            L_data  = self._data_loss_fn(y_pred.squeeze(), y_true.squeeze())
            y_sym   = self.frozen_fn(X).squeeze()
            L_sym   = self._sym_loss_fn(y_pred.squeeze(), y_sym)
            L_total = L_data + self.lambda_sym * L_sym
            return {
                "L_data"    : float(L_data.item()),
                "L_symbolic": float(L_sym.item()),
                "L_total"   : float(L_total.item()),
                "lambda"    : self.lambda_sym,
            }

    def extra_repr(self) -> str:
        return (
            f"lambda_sym={self.lambda_sym}, "
            f"data_loss='{self.data_loss}', "
            f"sym_loss='{self.sym_loss}'"
        )


# ─── Sinir Ağı ────────────────────────────────────────────────────────────────

class TIMURNet(nn.Module):
    """
    TIMUR XAI Hibrit Sinir Ağı.

    Yapı:
        [Girdi] → [Gizli Katmanlar] → [Çıktı]

    Eğitim sırasında TIMURLoss ile birlikte kullanılır:
        L_total = MSE(y_pred, y_true) + λ · MSE(y_pred, φ(x))

    Böylece ağ hem veriye uyar hem de sembolik denklemi "takip eder".

    Parameters
    ----------
    n_features  : int          Giriş boyutu
    frozen_fn   : callable     Dondurulmuş sembolik denklem (salt bilgi için)
    hidden_dims : list[int]    Gizli katman boyutları. Varsayılan: [64, 32]
    activation  : str          'relu' | 'tanh' | 'silu'. Varsayılan: 'tanh'
    dropout     : float        Dropout oranı. Varsayılan: 0.0

    Örnek
    -----
    >>> net = TIMURNet(n_features=2, frozen_fn=result.frozen_fn)
    >>> y   = net(X_tensor)     # (batch, 1)
    """

    def __init__(
        self,
        n_features  : int,
        frozen_fn   : Callable[[torch.Tensor], torch.Tensor],
        hidden_dims : list = None,
        activation  : str  = "tanh",
        dropout     : float = 0.0,
    ) -> None:
        super().__init__()
        self.frozen_fn  = frozen_fn
        hidden_dims     = hidden_dims or [64, 32]

        activations = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "silu": nn.SiLU,
        }
        if activation not in activations:
            raise ValueError(f"Bilinmeyen aktivasyon: '{activation}'. "
                             f"'relu', 'tanh', 'silu' kullanın.")
        act_cls = activations[activation]

        layers = []
        in_dim = n_features
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(act_cls())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, n_features) → y_pred: (batch,)
        """
        return self.net(x).squeeze(-1)

    def symbolic_residual(self, x: torch.Tensor) -> torch.Tensor:
        """
        Ağın tahmini ile dondurulmuş denklem arasındaki fark.
        Eğitim sonrası ne kadar saptığını ölçmek için kullanılır.
        """
        with torch.no_grad():
            y_net = self.forward(x)
            y_sym = self.frozen_fn(x).squeeze()
            return y_net - y_sym


# ─── Hızlı Eğitim Yardımcısı ─────────────────────────────────────────────────

def timur_fit(
    net        : TIMURNet,
    X_train    : torch.Tensor,
    y_train    : torch.Tensor,
    frozen_fn  : Callable[[torch.Tensor], torch.Tensor],
    lambda_sym : float = 0.1,
    epochs     : int   = 500,
    lr         : float = 1e-3,
    data_loss  : str   = "mse",
    verbose    : bool  = True,
    log_every  : int   = 50,
) -> Dict[str, list]:
    """
    TIMURNet için tam eğitim döngüsü.

    Parameters
    ----------
    net        : TIMURNet
    X_train    : (n, p) Tensor
    y_train    : (n,)   Tensor
    frozen_fn  : Dondurulmuş denklem
    lambda_sym : Sembolik regülarizasyon kuvveti
    epochs     : Eğitim epoch sayısı
    lr         : Öğrenme hızı
    data_loss  : 'mse' | 'mae' | 'huber'
    verbose    : Ekrana yazdır
    log_every  : Kaç epoch'ta bir loglansın

    Returns
    -------
    history : {'L_total': [...], 'L_data': [...], 'L_symbolic': [...]}
    """
    loss_fn   = TIMURLoss(frozen_fn=frozen_fn, lambda_sym=lambda_sym,
                          data_loss=data_loss)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"L_total": [], "L_data": [], "L_symbolic": []}

    if verbose:
        print(f"\n[TIMUR] ─── Faz 3 Entegrasyon Eğitimi ──────────────────")
        print(f"  Ağ mimarisi  : {net.net}")
        print(f"  Kayıp        : L_data + {lambda_sym:.3f} · L_symbolic")
        print(f"  Epoch sayısı : {epochs}")
        print(f"  Öğrenme hızı : {lr}")
        print()

    for epoch in range(1, epochs + 1):
        net.train()
        y_pred = net(X_train)
        loss   = loss_fn(y_pred, y_train, X_train)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        if epoch % log_every == 0 or epoch == 1:
            net.eval()
            with torch.no_grad():
                comps = loss_fn.component_losses(net(X_train), y_train, X_train)
            history["L_total"].append(comps["L_total"])
            history["L_data"].append(comps["L_data"])
            history["L_symbolic"].append(comps["L_symbolic"])

            if verbose:
                print(
                    f"  Epoch {epoch:>4d}/{epochs}  "
                    f"L_total={comps['L_total']:.4f}  "
                    f"L_data={comps['L_data']:.4f}  "
                    f"L_sym={comps['L_symbolic']:.4f}"
                )

    if verbose:
        net.eval()
        with torch.no_grad():
            y_pred_np = net(X_train).numpy()
            y_true_np = y_train.numpy()
        from sklearn.metrics import r2_score
        final_r2 = r2_score(y_true_np, y_pred_np)
        print(f"\n  [Tamamlandı] Son R² = {final_r2:.4f}")

    return history