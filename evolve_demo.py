"""
evolve_demo.py
═════════════════════════════════════════════════════════════════════════════
timur/evolve/ (MAP-Elites özyinelemeli keşif katmanı) için uçtan-uca demo.

Veri seti: Stefan-Boltzmann Toplam Güç Yoğunluğu — j*(T) = σ·T⁴
(timur_benchmark.py'deki DS2 ile aynı üretim mantığı, bkz. make_ds2()).

Bu veri seti bilinçli olarak seçildi: tek özellik (T) + tek sabit (σ) +
hedef arasındaki Buckingham Pi analizi tam olarak SIFIR serbest Pi-grubu
bırakıyor (in_features=0 → "saf-sabit" hızlı yol, bkz. timur/__init__.py
fit()). Bu sayede demo, PySR/Julia kurulu olmasa bile HIZLI ve GÜVENİLİR
şekilde uçtan uca koşar. Sonuç: her SEED/mutasyon çağrısı yapısal olarak
aynı denklem biçimine (tek sabit × sıcaklık⁴) yakınsayacağından arşiv
kapsamı (coverage) tek hücrede kalması BEKLENEN bir durumdur — bu demo
MAP-Elites çeşitliliğini değil, SEED→LOOP→ARŞİV bağlantısının uçtan uca
doğru çalıştığını göstermeyi amaçlar. Gerçek çeşitlilik (farklı complexity/
operatör ailesi hücreleri), PySR'a yönlendirilen doğrusal-olmayan veri
setlerinde (ör. DS1 Planck) ortaya çıkar.
"""
import numpy as np
import scipy.constants as const

from timur.evolve.loop import evolve

SEED = 42
N = 800
NOISE_STD = 0.01


def make_stefan_boltzmann_dataset():
    rng = np.random.default_rng(SEED)
    sigma = const.sigma
    T = rng.uniform(300, 6000, N)
    X = T.reshape(-1, 1)

    y_true = sigma * T**4
    y_noisy = y_true * rng.normal(1.0, NOISE_STD, size=y_true.shape)
    return X, y_noisy


if __name__ == "__main__":
    X, y = make_stefan_boltzmann_dataset()

    # Basit train/val bölmesi
    n_val = int(0.2 * len(y))
    X_val, y_val = X[:n_val], y[:n_val]
    X_train, y_train = X[n_val:], y[n_val:]

    model_kwargs = dict(
        constants={"sigma": (const.sigma, {"kg": 1, "s": -3, "K": -4})},
        lambda_sym=0.5,
        linear_threshold=0.15,
        pysr_threshold=0.20,
        verbose=False,
    )

    archive = evolve(
        X_train, y_train, X_val, y_val,
        feature_names=["sicaklik"],
        feature_dims=[{"K": 1}],
        target_dim={"kg": 1, "s": -3},
        n_iterations=8,
        seed_runs=3,
        model_kwargs=model_kwargs,
        patience=4,
        save_path="./evolve_demo_archive.json",
        verbose=True,
    )

    print(f"\nArşiv kapsamı (coverage): {archive.coverage()} / 64 hücre")
    best = archive.best()
    if best:
        print(f"En iyi aday: fitness={best.fitness:.4f}  {best.expr_str}")
