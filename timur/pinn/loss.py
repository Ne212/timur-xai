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

import copy
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
        # weighted_mse / normalized_weighted_mse:
        #   L = mean( w_i * (y_pred_i - y_true_i)^2 )
        #   w_i = (1/|y_true_i|²) / mean(1/|y_true|²)
        #
        #   Normalize etmek iki şeyi garanti eder:
        #   (1) Kayıp değeri ölçeğinden bağımsız → standart MSE gibi yorumlanabilir
        #   (2) Gradyan patlaması yok (w_i = y_true bağımlı, tahmin değil)
        #   Geniş dinamik aralıklı hedeflerde (10⁻¹³ → 10⁰ gibi) tüm örnekler
        #   eşit göreli önem alır — MAPE'yi yaklaşık minimize eder.
        if data_loss == "mse":
            self._data_loss_fn = nn.MSELoss()
        elif data_loss == "mae":
            self._data_loss_fn = nn.L1Loss()
        elif data_loss == "huber":
            self._data_loss_fn = nn.HuberLoss(delta=huber_delta)
        elif data_loss in ("weighted_mse", "normalized_weighted_mse"):
            self._data_loss_fn = None  # forward() içinde elle hesaplanır
        else:
            raise ValueError(
                f"Bilinmeyen data_loss: '{data_loss}'. "
                f"'mse', 'mae', 'huber', 'weighted_mse' kullanın."
            )

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
        if self.data_loss == "weighted_mse":
            L_data = self._weighted_mse(y_pred.squeeze(), y_true.squeeze())
        else:
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

    @staticmethod
    def _weighted_mse(y_pred: torch.Tensor, y_true: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        Normalize Edilmiş Ağırlıklı MSE:
            L = mean( w_i * (y_pred_i - y_true_i)² )
            w_i = (1/|y_true_i|²) / mean(1/|y_true|²)

        Normalizasyon iki kritik özellik sağlar:
        (1) Kayıp ölçeği standart MSE ile karşılaştırılabilir — ~1.0 civarı
        (2) Gradyan patlaması yok: w_i y_true'dan gelir (sabit), y_pred'den değil

        Ham (normalize edilmemiş) formda küçük y_true değerleri w→∞ üretir
        ve eğitim başında ağ 0 civarı çıktı üretirken kayıp ~10¹⁰ olabilir.
        Normalize formda bu problem yoktur.

        eps: w hesabında sıfıra bölmeyi önler
        """
        w = 1.0 / (y_true.abs() + eps) ** 2
        w = w / w.mean()  # normalize: sum(w)/N = 1
        return (w * (y_pred - y_true) ** 2).mean()

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
            if self.data_loss == "weighted_mse":
                L_data = self._weighted_mse(y_pred.squeeze(), y_true.squeeze())
            else:
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
    x_mean, x_std : array-like | None
        Girdi z-skoru standardizasyonu için ortalama/std (egitim verisinden
        hesaplanır). None ise kimlik (no-op) ölçekleme kullanılır - eski
        davranışla geriye dönük uyumluluk için.
    y_mean, y_std : float
        Hedef z-skoru standardizasyonu için ortalama/std. Varsayılan: 0.0/1.0
        (no-op).

    NEDEN OLCEKLEME GEREKLI (bug-fix):
        Ham ölçekli girdiler (örn. kafes sabitleri ~3-5 Å, açılar ~90-100,
        hedef ~0-20) Tanh aktivasyonunu erken doygunluğa sürüklüyor ve ağ
        yüzlerce epoch boyunca neredeyse hiç yakınsamıyordu (gözlemlenen
        örnekte: saf PySR denklemi R²=0.93 iken, ÜZERİNE bina edilmesi
        gereken PINN ağı R²=0.22 ile DAHA KÖTÜ sonuç veriyordu). Çözüm:
        ağın İÇİNDE (forward'ın başında/sonunda) sabit (eğitilemez) z-skoru
        normalize/de-normalize katmanı - dışarıya (frozen_fn, TIMURLoss,
        TIMURModel.predict) hâlâ HAM ölçekli x/y arayüzü sunulur, hiçbir
        çağıran kod değişmek zorunda kalmaz.

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
        x_mean      = None,
        x_std       = None,
        y_mean      : float = 0.0,
        y_std       : float = 1.0,
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

        # Sabit (eğitilemez) z-skoru ölçekleme tamponları. x_mean/x_std
        # verilmezse kimlik ölçekleme (mean=0, std=1) kullanılır - eski
        # davranışla geriye dönük uyumlu.
        if x_mean is None:
            x_mean_t = torch.zeros(n_features, dtype=torch.float32)
        else:
            x_mean_t = torch.as_tensor(x_mean, dtype=torch.float32)
        if x_std is None:
            x_std_t = torch.ones(n_features, dtype=torch.float32)
        else:
            x_std_t = torch.as_tensor(x_std, dtype=torch.float32).clone()
            x_std_t[x_std_t < 1e-8] = 1.0  # sabit sütunda sıfıra bölmeyi engelle

        self.register_buffer("x_mean", x_mean_t)
        self.register_buffer("x_std", x_std_t)
        self.register_buffer("y_mean", torch.tensor(float(y_mean), dtype=torch.float32))
        self.register_buffer("y_std", torch.tensor(float(y_std) if y_std > 1e-8 else 1.0, dtype=torch.float32))

    def _init_weights(self):
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, n_features) [HAM olcek] → y_pred: (batch,) [HAM olcek]

        Icte z-skoru normalize/de-normalize uygulanir; disaridan
        bakildiginda arayuz HAM olcekte kalir (frozen_fn/TIMURLoss/predict
        degismez).
        """
        x_norm = (x - self.x_mean) / self.x_std
        y_norm = self.net(x_norm).squeeze(-1)
        return y_norm * self.y_std + self.y_mean

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
    X_val      : Optional[torch.Tensor] = None,
    y_val      : Optional[torch.Tensor] = None,
    early_stop_patience: Optional[int] = None,
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
    X_val, y_val : Optional[Tensor]
        Verilirse, HER epoch'ta (eğitim setinden bağımsız) val R² hesaplanır
        ve en iyi val R²'ye sahip ağırlıklar (deep-copy state_dict) saklanır;
        eğitim sonunda ağ bu EN İYİ duruma geri yüklenir. Bu, gerçek (kör
        tahmin olmayan) bir erken-durdurma sağlar — önceden `lambda_sym`
        yükseltilerek overfitting'e karşı dolaylı/kaba bir önlem alınıyordu;
        bu artık val R²'ye bakarak doğru durma noktasını bulur.
        Verilmezse (None, varsayılan), eski davranış DEĞİŞMEZ.
    early_stop_patience : Optional[int]
        X_val/y_val verildiyse: val R² bu kadar epoch boyunca iyileşmezse
        eğitim erken durdurulur (en iyi ağırlıklar zaten yüklenecek). None
        ise erken durdurma YAPILMAZ (tam `epochs` kadar çalışır), ama en iyi
        val R²'ye sahip ağırlıklar yine de sonunda geri yüklenir.

    Returns
    -------
    history : {'L_total': [...], 'L_data': [...], 'L_symbolic': [...], 'epoch': [...],
               (X_val verildiyse ek olarak) 'val_epoch': [...], 'val_r2': [...],
               'train_r2': [...], 'best_epoch': int, 'best_val_r2': float}
    """
    loss_fn   = TIMURLoss(frozen_fn=frozen_fn, lambda_sym=lambda_sym,
                          data_loss=data_loss)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"L_total": [], "L_data": [], "L_symbolic": [], "epoch": []}
    val_takibi = X_val is not None and y_val is not None
    if val_takibi:
        from sklearn.metrics import r2_score as _r2_fn
        history["val_epoch"] = []
        history["val_r2"] = []
        history["train_r2"] = []

    if verbose:
        print(f"\n[TIMUR] ─── Faz 3 Entegrasyon Eğitimi ──────────────────")
        print(f"  Ağ mimarisi  : {net.net}")
        print(f"  Kayıp        : L_data + {lambda_sym:.3f} · L_symbolic")
        print(f"  Epoch sayısı : {epochs}")
        print(f"  Öğrenme hızı : {lr}")
        if val_takibi:
            ek = f", erken-durdurma patience={early_stop_patience}" if early_stop_patience else ", erken durdurma YOK (sadece en iyi ağırlık saklanır)"
            print(f"  Val takibi   : AKTİF ({len(y_val)} örnek){ek}")
        print()

    best_state = None
    best_val_r2 = -float("inf")
    best_epoch = None
    epochs_since_improve = 0

    for epoch in range(1, epochs + 1):
        net.train()
        y_pred = net(X_train)
        loss   = loss_fn(y_pred, y_train, X_train)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        val_r2 = train_r2_chk = None
        if val_takibi:
            net.eval()
            with torch.no_grad():
                y_val_pred = net(X_val).numpy()
                y_train_pred_chk = net(X_train).numpy()
            val_r2 = float(_r2_fn(y_val.numpy(), y_val_pred))
            train_r2_chk = float(_r2_fn(y_train.numpy(), y_train_pred_chk))
            history["val_epoch"].append(epoch)
            history["val_r2"].append(val_r2)
            history["train_r2"].append(train_r2_chk)

            if val_r2 > best_val_r2:
                best_val_r2 = val_r2
                best_epoch = epoch
                best_state = copy.deepcopy(net.state_dict())
                epochs_since_improve = 0
            else:
                epochs_since_improve += 1

        if epoch % log_every == 0 or epoch == 1:
            net.eval()
            with torch.no_grad():
                comps = loss_fn.component_losses(net(X_train), y_train, X_train)
            history["L_total"].append(comps["L_total"])
            history["L_data"].append(comps["L_data"])
            history["L_symbolic"].append(comps["L_symbolic"])
            history["epoch"].append(epoch)

            if verbose:
                ek = ""
                if val_takibi:
                    ek = f"  train_R2={train_r2_chk:.4f}  val_R2={val_r2:.4f}"
                    if best_epoch == epoch:
                        ek += "  (en iyi!)"
                print(
                    f"  Epoch {epoch:>4d}/{epochs}  "
                    f"L_total={comps['L_total']:.4f}  "
                    f"L_data={comps['L_data']:.4f}  "
                    f"L_sym={comps['L_symbolic']:.4f}{ek}"
                )

        if val_takibi and early_stop_patience is not None and epochs_since_improve >= early_stop_patience:
            if verbose:
                print(f"\n  Erken durdurma: val_R² {early_stop_patience} epoch boyunca "
                      f"iyileşmedi (epoch {epoch}, en iyi epoch {best_epoch}, "
                      f"en iyi val_R²={best_val_r2:.4f}).")
            break

    if val_takibi and best_state is not None:
        net.load_state_dict(best_state)
        history["best_epoch"] = best_epoch
        history["best_val_r2"] = best_val_r2
        if verbose:
            print(f"\n  [En iyi duruma dönüş] Epoch {best_epoch} ağırlıkları "
                  f"geri yüklendi (val_R²={best_val_r2:.4f}).")

    if verbose:
        net.eval()
        with torch.no_grad():
            y_pred_np = net(X_train).numpy()
            y_true_np = y_train.numpy()
        from sklearn.metrics import r2_score
        final_r2 = r2_score(y_true_np, y_pred_np)
        print(f"\n  [Tamamlandı] Son R² (train) = {final_r2:.4f}")
        if val_takibi:
            with torch.no_grad():
                y_val_pred_np = net(X_val).numpy()
            final_val_r2 = r2_score(y_val.numpy(), y_val_pred_np)
            print(f"  [Tamamlandı] Son R² (val)   = {final_val_r2:.4f}")

    return history