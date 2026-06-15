import pysr
import numpy as np
import scipy.constants as const
from timur import TIMURModel


def planck_law(wav, T):
    """Kara cisim ışıması (Spectral Radiance) SI Birimlerinde (W / m^3 / sr)"""
    h = const.h
    c = const.c
    k = const.k
    
    a = 2.0 * h * c**2
    b = h * c / (wav * k * T)
    return a / ( (wav**5) * (np.exp(b) - 1.0) )

np.random.seed(42)

# Ham SI birimlerinde veriler (Metre ve Kelvin)
wavelengths = np.random.uniform(100e-9, 3000e-9, 1000)
temperatures = np.random.uniform(3000, 8000, 1000)

X = np.column_stack((wavelengths, temperatures))
y_true = planck_law(X[:, 0], X[:, 1])
# y_noisy = y_true + np.random.normal(...) satırını SİL, yerine şunu yaz:
y_noisy = y_true * np.random.normal(1.0, 0.01, size=y_true.shape)

print("Saf SI Veri seti oluşturuldu.")
print("\nTIMUR Boyutsuzlaştırma (Buckingham Pi) Motoru Ateşleniyor...\n")

# İŞTE YENİ KURUMSAL API: Dışarıdan sadece birimleri veriyoruz
model = TIMURModel(
    feature_names=["dalga_boyu", "sicaklik"],
    feature_dims=[{"m": 1}, {"K": 1}],
    target_dim={"kg": 1, "m": -1, "s": -3},  # Işıma Şiddeti SI birimi (kg / (m * s^3))
    constants={
        "h":  (const.h, {"kg": 1, "m": 2, "s": -1}),
        "c":  (const.c, {"m": 1, "s": -1}),
        "kB": (const.k, {"kg": 1, "m": 2, "s": -2, "K": -1})
    },
    lambda_sym=0.5,
    linear_threshold=0.15,
    pysr_threshold=0.20,
    verbose=True
)

# Eğitimi başlat (Gatekeeper + Boyutsuzlaştırma + Sembolik Regresyon + PINN)
model.fit(X, y_noisy)

# Çıktı Raporu
print(model.get_xai_report())