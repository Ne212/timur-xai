"""
timur_benchmark_5datasets.py
============================
5 Farklı Teorik Fizik Yasası üzerinde TIMURModel Kıyaslama Scripti

Her veri seti için:
  - Veriler internet kaynaklı sabitler kullanılarak sentetik üretilir
    (gerçek ölçüm aralıkları referans alınır)
  - TIMURModel çalıştırılır
  - Sonuçlar JSON + TXT olarak kaydedilir
  - 3 grafik kaydedilir: (1) y_true vs y_pred scatter,
                          (2) residual dağılımı,
                          (3) boyutsuz Pi grupları korelasyon ısı haritası

Veri Setleri:
  DS1 - Planck Kara Cisim Işıması       : B(λ,T) = 2hc²/λ⁵ · 1/(exp(hc/λkT)-1)
  DS2 - Stefan-Boltzmann Toplam Güç     : j*(T) = σ·T⁴
  DS3 - Stokes Viskozite Kuvveti        : F = 6π·η·r·v
  DS4 - Gravitasyonel Potansiyel Enerji : U = G·M·m / r
  DS5 - Wien Deplasman (λ_max·T = b)    : λ_max = b / T

Çalıştırma:
  python timur_benchmark_5datasets.py

Çıktı dizini: ./timur_benchmark_results/
"""

import os
import json
import time
import traceback
import warnings

import numpy as np
import scipy.constants as const
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from timur import TIMURModel

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# ÇIKTI DİZİNİ
# ──────────────────────────────────────────────────────────────────────────────
OUTPUT_DIR = "./timur_benchmark_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = 42
np.random.seed(SEED)
N = 800          # Her veri seti için örnek sayısı
NOISE_STD = 0.01  # Çarpımsal gürültü oranı (%1)


# ──────────────────────────────────────────────────────────────────────────────
# YARDIMCI FONKSİYONLAR
# ──────────────────────────────────────────────────────────────────────────────

def add_multiplicative_noise(y, std=NOISE_STD, rng=None):
    """Çarpımsal Gaussian gürültü: y_noisy = y * N(1, std²)"""
    if rng is None:
        rng = np.random.default_rng(SEED)
    return y * rng.normal(1.0, std, size=y.shape)


def compute_metrics(y_true, y_pred):
    """R², RMSE, MAE ve MAPE hesapla"""
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true != 0)
    yt, yp = y_true[mask], y_pred[mask]
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmse = np.sqrt(np.mean((yt - yp) ** 2))
    mae = np.mean(np.abs(yt - yp))
    mape = np.mean(np.abs((yt - yp) / yt)) * 100
    return {"R2": float(r2), "RMSE": float(rmse), "MAE": float(mae), "MAPE_%": float(mape)}


def save_plots(ds_id, ds_name, X, y_true, y_pred, feature_names, output_dir):
    """3 panel grafik kaydet"""
    fig = plt.figure(figsize=(18, 5))
    fig.suptitle(f"DS{ds_id}: {ds_name}", fontsize=14, fontweight="bold", y=1.02)
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    # --- Panel 1: Gerçek vs Tahmin scatter ---
    ax1 = fig.add_subplot(gs[0])
    lim_min = min(y_true.min(), y_pred.min())
    lim_max = max(y_true.max(), y_pred.max())
    ax1.scatter(y_true, y_pred, alpha=0.35, s=12, color="#2563EB", edgecolors="none")
    ax1.plot([lim_min, lim_max], [lim_min, lim_max], "r--", lw=1.5, label="Mükemmel uyum")
    ax1.set_xlabel("y_gerçek")
    ax1.set_ylabel("y_tahmin")
    ax1.set_title("Gerçek vs Tahmin")
    ax1.legend(fontsize=8)
    ax1.ticklabel_format(style="sci", axis="both", scilimits=(0, 0))

    # --- Panel 2: Artık (Residual) dağılımı ---
    ax2 = fig.add_subplot(gs[1])
    residuals = y_pred - y_true
    ax2.hist(residuals, bins=40, color="#16A34A", edgecolor="white", alpha=0.85)
    ax2.axvline(0, color="red", lw=1.5, linestyle="--")
    ax2.set_xlabel("Artık (y_tahmin − y_gerçek)")
    ax2.set_ylabel("Frekans")
    ax2.set_title("Artık Dağılımı")
    ax2.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

    # --- Panel 3: Özellik korelasyon ısı haritası ---
    ax3 = fig.add_subplot(gs[2])
    # X sütunları + y_true birleştir, korelasyon hesapla
    combined = np.column_stack([X, y_true.reshape(-1, 1)])
    col_names = feature_names + ["y_true"]
    corr = np.corrcoef(combined.T)
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        xticklabels=col_names,
        yticklabels=col_names,
        cmap="coolwarm",
        center=0,
        ax=ax3,
        cbar_kws={"shrink": 0.8},
        annot_kws={"size": 8},
    )
    ax3.set_title("Özellik Korelasyon Matrisi")
    ax3.tick_params(axis="x", rotation=30, labelsize=8)
    ax3.tick_params(axis="y", rotation=0, labelsize=8)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"DS{ds_id}_grafik.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [✓] Grafik kaydedildi → {plot_path}")
    return plot_path


def log_result(ds_id, ds_name, formula, metrics, xai_report, elapsed_s,
               plot_path, output_dir, error=None):
    """Sonuçları JSON ve TXT olarak kaydet"""
    record = {
        "ds_id": ds_id,
        "ds_name": ds_name,
        "formula": formula,
        "elapsed_seconds": round(elapsed_s, 2),
        "metrics": metrics,
        "xai_report_snippet": (xai_report[:1500] if xai_report else None),
        "plot_path": plot_path,
        "error": error,
    }
    json_path = os.path.join(output_dir, f"DS{ds_id}_sonuc.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

    txt_path = os.path.join(output_dir, f"DS{ds_id}_sonuc.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"{'='*70}\n")
        f.write(f"DS{ds_id}: {ds_name}\n")
        f.write(f"Formül : {formula}\n")
        f.write(f"Süre   : {elapsed_s:.2f} s\n")
        f.write(f"{'='*70}\n\n")
        if metrics:
            f.write("METRİKLER:\n")
            for k, v in metrics.items():
                f.write(f"  {k:10s}: {v:.6g}\n")
        if error:
            f.write(f"\nHATA:\n{error}\n")
        if xai_report:
            f.write(f"\nXAI RAPORU:\n{xai_report}\n")

    print(f"  [✓] Sonuçlar kaydedildi → {json_path}")
    return json_path, txt_path


# ──────────────────────────────────────────────────────────────────────────────
# VERİ SETİ TANIMLARI
# ──────────────────────────────────────────────────────────────────────────────

def make_ds1():
    """
    DS1 — Planck Kara Cisim Işıması
    B(λ, T) = 2hc²/λ⁵ · 1/(exp(hc/λkT) − 1)
    Birimler: W · m⁻³ · sr⁻¹  →  kg · m⁻¹ · s⁻³
    Özellikler: λ [m], T [K]
    """
    h, c, k = const.h, const.c, const.k
    wav = np.random.uniform(100e-9, 3000e-9, N)
    T   = np.random.uniform(3000, 8000, N)
    X   = np.column_stack([wav, T])

    def fn(wav, T):
        b = h * c / (wav * k * T)
        return (2.0 * h * c**2) / (wav**5 * (np.exp(b) - 1.0))

    y_true  = fn(X[:, 0], X[:, 1])
    y_noisy = add_multiplicative_noise(y_true)

    model_kwargs = dict(
        feature_names=["dalga_boyu", "sicaklik"],
        feature_dims=[{"m": 1}, {"K": 1}],
        target_dim={"kg": 1, "m": -1, "s": -3},
        constants={
            "h":  (const.h, {"kg": 1, "m": 2, "s": -1}),
            "c":  (const.c, {"m": 1, "s": -1}),
            "kB": (const.k, {"kg": 1, "m": 2, "s": -2, "K": -1}),
        },
        lambda_sym=0.5,
        linear_threshold=0.0,
        pysr_threshold=0.0,
        verbose=True,
    )
    return (
        X, y_true, y_noisy,
        ["dalga_boyu_m", "sicaklik_K"],
        model_kwargs,
        "B(λ,T) = 2hc²/λ⁵ · 1/(exp(hc/λkT)−1)",
    )


def make_ds2():
    """
    DS2 — Stefan-Boltzmann Toplam Güç Yoğunluğu
    j*(T) = σ · T⁴      →  σ = 5.6704e-8 W·m⁻²·K⁻⁴
    Birimler: W/m²  →  kg · s⁻³
    Özellikler: T [K]
    """
    sigma = const.sigma  # Stefan-Boltzmann sabiti
    T     = np.random.uniform(300, 6000, N)
    X     = T.reshape(-1, 1)

    y_true  = sigma * T**4
    y_noisy = add_multiplicative_noise(y_true)

    model_kwargs = dict(
        feature_names=["sicaklik"],
        feature_dims=[{"K": 1}],
        target_dim={"kg": 1, "s": -3},
        constants={
            "sigma": (sigma, {"kg": 1, "s": -3, "K": -4}),
        },
        lambda_sym=0.5,
        linear_threshold=0.15,
        pysr_threshold=0.20,
        verbose=True,
    )
    return (
        X, y_true, y_noisy,
        ["sicaklik_K"],
        model_kwargs,
        "j*(T) = σ·T⁴",
    )


def make_ds3():
    """
    DS3 — Stokes Viskozite Kuvveti
    F = 6π · η · r · v
    Birimler: N  →  kg · m · s⁻²
    Özellikler: η [Pa·s = kg·m⁻¹·s⁻¹], r [m], v [m·s⁻¹]
    Referans: Su ≈ 1e-3 Pa·s, hava ≈ 1.8e-5 Pa·s
    """
    eta = np.random.uniform(1e-4, 1e-2, N)   # dinamik viskozite [Pa·s]
    r   = np.random.uniform(1e-6, 1e-3, N)   # parçacık yarıçapı [m]
    v   = np.random.uniform(1e-4, 1e-1, N)   # hız [m/s]
    X   = np.column_stack([eta, r, v])

    y_true  = 6 * np.pi * eta * r * v
    y_noisy = add_multiplicative_noise(y_true)

    model_kwargs = dict(
        feature_names=["viskozite", "yaricap", "hiz"],
        feature_dims=[
            {"kg": 1, "m": -1, "s": -1},   # η: Pa·s
            {"m": 1},                        # r: m
            {"m": 1, "s": -1},              # v: m/s
        ],
        target_dim={"kg": 1, "m": 1, "s": -2},  # N
        constants={},
        lambda_sym=0.5,
        linear_threshold=0.15,
        pysr_threshold=0.20,
        verbose=True,
    )
    return (
        X, y_true, y_noisy,
        ["eta_Pa_s", "r_m", "v_m_s"],
        model_kwargs,
        "F = 6π·η·r·v",
    )


def make_ds4():
    """
    DS4 — Gravitasyonel Potansiyel Enerji
    U = G · M · m / r
    Birimler: J  →  kg · m² · s⁻²
    Özellikler: M [kg], m [kg], r [m]
    Referans: Gezegenler arası ölçek
    """
    G = const.G
    M = np.random.uniform(1e24, 2e30, N)   # büyük kütle [kg]  (Dünya → Güneş arası)
    m = np.random.uniform(1e0,  1e5,  N)   # küçük kütle [kg]
    r = np.random.uniform(1e7,  1e12, N)   # mesafe [m]
    X = np.column_stack([M, m, r])

    y_true  = G * M * m / r
    y_noisy = add_multiplicative_noise(y_true)

    model_kwargs = dict(
        feature_names=["kitle_buyuk", "kitle_kucuk", "mesafe"],
        feature_dims=[
            {"kg": 1},
            {"kg": 1},
            {"m": 1},
        ],
        target_dim={"kg": 1, "m": 2, "s": -2},  # J
        constants={
            "G": (G, {"m": 3, "kg": -1, "s": -2}),
        },
        lambda_sym=0.5,
        linear_threshold=0.15,
        pysr_threshold=0.20,
        verbose=True,
    )
    return (
        X, y_true, y_noisy,
        ["M_kg", "m_kg", "r_m"],
        model_kwargs,
        "U = G·M·m/r",
    )


def make_ds5():
    """
    DS5 — Wien Deplasman Yasası
    λ_max = b / T      b = 2.897771955e-3 m·K
    Birimler: m
    Özellikler: T [K]
    """
    b_wien = const.Wien    # 2.897771955e-3 m·K
    T      = np.random.uniform(300, 30000, N)
    X      = T.reshape(-1, 1)

    y_true  = b_wien / T
    y_noisy = add_multiplicative_noise(y_true)

    model_kwargs = dict(
        feature_names=["sicaklik"],
        feature_dims=[{"K": 1}],
        target_dim={"m": 1},
        constants={
            "b_wien": (b_wien, {"m": 1, "K": 1}),
        },
        lambda_sym=0.5,
        linear_threshold=0.15,
        pysr_threshold=0.20,
        verbose=True,
    )
    return (
        X, y_true, y_noisy,
        ["sicaklik_K"],
        model_kwargs,
        "λ_max = b/T  (b=2.898e-3 m·K)",
    )


# ──────────────────────────────────────────────────────────────────────────────
# ANA DÖNGÜ
# ──────────────────────────────────────────────────────────────────────────────

DATASETS = [
    (1, "Planck Kara Cisim Işıması",       make_ds1),
    (2, "Stefan-Boltzmann Güç Yoğunluğu",  make_ds2),
    (3, "Stokes Viskozite Kuvveti",         make_ds3),
    (4, "Gravitasyonel Potansiyel Enerji",  make_ds4),
    (5, "Wien Deplasman Yasası",            make_ds5),
]

summary_rows = []

for ds_id, ds_name, make_fn in DATASETS:
    sep = "═" * 70
    print(f"\n{sep}")
    print(f"  VERİ SETİ {ds_id}/5 : {ds_name}")
    print(f"{sep}")

    try:
        X, y_true, y_noisy, feature_labels, model_kwargs, formula = make_fn()
        print(f"  Formül    : {formula}")
        print(f"  Örnekler  : {len(y_true)}")
        print(f"  y aralığı : [{y_true.min():.3e}, {y_true.max():.3e}]\n")

        model = TIMURModel(**model_kwargs)

        t0 = time.time()
        model.fit(X, y_noisy)
        elapsed = time.time() - t0

        # Tahmin al
        try:
            y_pred = model.predict_symbolic(X)
        except AttributeError:
            # predict yoksa transform dene
            y_pred = model.transform(X)

        metrics = compute_metrics(y_true, y_pred)

        print(f"\n  ─── Metrikler ───")
        for k, v in metrics.items():
            print(f"    {k:12s}: {v:.6g}")

        xai_report = model.get_xai_report()
        print(f"\n  ─── XAI Raporu (ilk 500 karakter) ───")
        print(f"  {xai_report[:500]}")

        plot_path = save_plots(
            ds_id, ds_name, X, y_true, y_pred, feature_labels, OUTPUT_DIR
        )
        json_path, txt_path = log_result(
            ds_id, ds_name, formula, metrics, xai_report, elapsed, plot_path, OUTPUT_DIR
        )

        summary_rows.append({
            "DS": f"DS{ds_id}",
            "Ad": ds_name,
            "R²": f"{metrics['R2']:.4f}",
            "RMSE": f"{metrics['RMSE']:.3e}",
            "MAPE_%": f"{metrics['MAPE_%']:.2f}",
            "Süre_s": f"{elapsed:.1f}",
            "Durum": "✓ BAŞARILI",
        })

    except Exception as exc:
        elapsed = time.time() - t0 if "t0" in dir() else 0.0
        err_msg = traceback.format_exc()
        print(f"\n  [!] HATA:\n{err_msg}")
        log_result(
            ds_id, ds_name,
            formula if "formula" in dir() else "?",
            None, None, elapsed, None, OUTPUT_DIR,
            error=err_msg,
        )
        summary_rows.append({
            "DS": f"DS{ds_id}",
            "Ad": ds_name,
            "R²": "—",
            "RMSE": "—",
            "MAPE_%": "—",
            "Süre_s": f"{elapsed:.1f}",
            "Durum": "✗ HATA",
        })


# ──────────────────────────────────────────────────────────────────────────────
# ÖZET TABLO
# ──────────────────────────────────────────────────────────────────────────────

print(f"\n\n{'═'*70}")
print("  GENEL ÖZET")
print(f"{'═'*70}")
header = f"{'DS':<6} {'Ad':<38} {'R²':>8} {'RMSE':>12} {'MAPE_%':>8} {'Süre_s':>8} {'Durum':>12}"
print(header)
print("─" * 70)
for row in summary_rows:
    print(
        f"{row['DS']:<6} {row['Ad']:<38} {row['R²']:>8} "
        f"{row['RMSE']:>12} {row['MAPE_%']:>8} {row['Süre_s']:>8} {row['Durum']:>12}"
    )
print(f"{'─'*70}")

# Özet tablosunu da kaydet
summary_path = os.path.join(OUTPUT_DIR, "OZET_TABLO.txt")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("TIMUR KIYASLAMA ÖZET TABLOSU\n")
    f.write(f"{'='*70}\n")
    f.write(header + "\n")
    f.write("─" * 70 + "\n")
    for row in summary_rows:
        f.write(
            f"{row['DS']:<6} {row['Ad']:<38} {row['R²']:>8} "
            f"{row['RMSE']:>12} {row['MAPE_%']:>8} {row['Süre_s']:>8} {row['Durum']:>12}\n"
        )

# Tüm özet JSON
summary_json = os.path.join(OUTPUT_DIR, "OZET_TABLO.json")
with open(summary_json, "w", encoding="utf-8") as f:
    json.dump(summary_rows, f, ensure_ascii=False, indent=2)

print(f"\n[✓] Tüm çıktılar → {OUTPUT_DIR}/")
print(f"[✓] Özet tablo   → {summary_path}")
print(f"[✓] Özet JSON    → {summary_json}")