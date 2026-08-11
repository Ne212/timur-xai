"""
evolve_benchmark.py
═════════════════════════════════════════════════════════════════════════════
5 fiziksel yasa (timur_benchmark.py ile aynı veri üretim mantığı) üzerinde
evolve() MAP-Elites döngüsünü koşar — tek-geçişli TIMURModel.fit() yerine.
timur/__init__.py, symbolic/, pinn/ HİÇ DEĞİŞTİRİLMEDİ; bu script sadece
timur/evolve/'i kara-kutu TIMURModel üzerinden tekrar tekrar çağırır.

NOT: DS2 (Stefan-Boltzmann), DS3 (Stokes) ve DS5 (Wien) Buckingham Pi
analizinde tam olarak SIFIR serbest Pi-grubu bırakır (in_features=0
"saf-sabit" yolu) — bu üç sette PySR hiç çalışmaz, MAP-Elites arşivi tek
hücrede kalır. Bu BİR HATA DEĞİL: bu veri setlerinin fiziksel olarak
tam-belirlenmiş (Pi teoremi tarafından tamamen çözülmüş) olmasının doğrudan
sonucudur — bkz. evolve_demo.py'deki ilk tanı notu ve evolve_demo_planck.py
tanı raporu. DS1 (Planck) in_features=1 ile gerçek PySR aramasını tetikler;
DS4 (Gravitasyon) için a priori kesin değil (M/m oranı teknik olarak serbest
bir Pi-grubu bırakıyor ama gerçek fizik bu değişkenden bağımsız — Gatekeeper
bunu linear/poly/pysr'dan hangisine yönlendirir, koşuda gözlemlenecek).

Yargıç istatistiği (kaç aday üretildi / kaçı rejected_by_judge) için bu
script'e özel, kaynak kodu değiştirmeyen bir gözlem (monkeypatch) eklenmiştir
— evolve_demo_planck.py'de kullanılan yöntemin aynısı.
"""
import os
import sys
import time

import numpy as np
import scipy.constants as const

import timur.evolve.loop as loop_mod
from timur.evolve.judge import LimitConstraint, SymmetryConstraint
from timur.evolve.loop import evolve

# ─── TEŞHİS AMAÇLI GÖZLEM (yalnızca bu script içinde, kaynak koda dokunmaz) ───

_orig_evaluate = loop_mod.evaluate
_judge_stats = {"total": 0, "rejected": 0, "reject_reasons": []}


def _patched_evaluate(candidate, X_val, y_val, rng=None, **kwargs):
    _judge_stats["total"] += 1
    result = _orig_evaluate(candidate, X_val, y_val, rng=rng, **kwargs)
    if not result:
        _judge_stats["rejected"] += 1
        report = candidate.metadata.get("judge_report", {})
        failed = []
        if not report.get("r2_passed", True):
            failed.append(f"r2={report.get('r2'):.4f}<{report.get('r2_threshold')}")
        if not report.get("finiteness", {}).get("passed", True):
            failed.append("finiteness")
        if not report.get("limit_behavior", {}).get("passed", True):
            failed.append("limit_behavior")
        if not report.get("symmetry", {}).get("passed", True):
            failed.append("symmetry")
        if not report.get("conservation", {}).get("passed", True):
            failed.append("conservation")
        _judge_stats["reject_reasons"].append({
            "expr": candidate.expr_str, "descriptor": tuple(candidate.descriptor),
            "failed": failed,
        })
    return result


loop_mod.evaluate = _patched_evaluate

# ─── Ortak veri üretimi (timur_benchmark.py ile aynı sabitler/mantık) ────────

SEED = 42
N = 800


def _add_noise(y, rng, std=0.01):
    return y * rng.normal(1.0, std, size=y.shape)


def make_ds1_planck():
    rng = np.random.default_rng(SEED)
    h, c, k = const.h, const.c, const.k
    wav = rng.uniform(100e-9, 3000e-9, N)
    T = rng.uniform(3000, 8000, N)
    X = np.column_stack([wav, T])

    def fn(wav, T):
        b = h * c / (wav * k * T)
        return (2.0 * h * c**2) / (wav**5 * (np.exp(b) - 1.0))

    y = _add_noise(fn(X[:, 0], X[:, 1]), rng)

    model_kwargs = dict(
        constants={
            "h": (h, {"kg": 1, "m": 2, "s": -1}),
            "c": (c, {"m": 1, "s": -1}),
            "kB": (k, {"kg": 1, "m": 2, "s": -2, "K": -1}),
        },
        lambda_sym=0.5, linear_threshold=0.0, pysr_threshold=0.0,
        data_loss="weighted_mse", epochs=20, verbose=False,
    )
    evolve_kwargs = dict(
        feature_names=["dalga_boyu", "sicaklik"],
        feature_dims=[{"m": 1}, {"K": 1}],
        target_dim={"kg": 1, "m": -1, "s": -3},
        limit_constraints=[
            LimitConstraint(feature_index=1, behavior="monotonic_increasing"),
            LimitConstraint(feature_index=0, behavior="finite_at_zero"),
            LimitConstraint(feature_index=0, behavior="finite_at_inf"),
        ],
    )
    return (X, y, model_kwargs, evolve_kwargs,
            "B(λ,T) = 2hc²/λ⁵ · 1/(exp(hc/λkT)-1)", "pysr bekleniyor (in_features=1)")


def make_ds2_stefan_boltzmann():
    rng = np.random.default_rng(SEED)
    sigma = const.sigma
    T = rng.uniform(300, 6000, N)
    X = T.reshape(-1, 1)
    y = _add_noise(sigma * T**4, rng)

    model_kwargs = dict(
        constants={"sigma": (sigma, {"kg": 1, "s": -3, "K": -4})},
        lambda_sym=0.5, linear_threshold=0.15, pysr_threshold=0.20, verbose=False,
    )
    evolve_kwargs = dict(
        feature_names=["sicaklik"],
        feature_dims=[{"K": 1}],
        target_dim={"kg": 1, "s": -3},
        limit_constraints=[LimitConstraint(feature_index=0, behavior="monotonic_increasing")],
    )
    return (X, y, model_kwargs, evolve_kwargs,
            "j*(T) = σ·T⁴", "saf-sabit bekleniyor (in_features=0)")


def make_ds3_stokes():
    rng = np.random.default_rng(SEED)
    eta = rng.uniform(1e-4, 1e-2, N)
    r = rng.uniform(1e-6, 1e-3, N)
    v = rng.uniform(1e-4, 1e-1, N)
    X = np.column_stack([eta, r, v])
    y = _add_noise(6 * np.pi * eta * r * v, rng)

    model_kwargs = dict(
        constants={}, lambda_sym=0.5, linear_threshold=0.15, pysr_threshold=0.20, verbose=False,
    )
    evolve_kwargs = dict(
        feature_names=["viskozite", "yaricap", "hiz"],
        feature_dims=[{"kg": 1, "m": -1, "s": -1}, {"m": 1}, {"m": 1, "s": -1}],
        target_dim={"kg": 1, "m": 1, "s": -2},
        # F=6*pi*eta*r*v viskozitede derece-1 homojendir: F(a*eta,r,v)=a*F(eta,r,v)
        symmetry_constraints=[SymmetryConstraint(kind="scale", feature_index=0, scale_exponent=1.0)],
    )
    return (X, y, model_kwargs, evolve_kwargs,
            "F = 6π·η·r·v", "saf-sabit bekleniyor (in_features=0)")


def make_ds4_gravity():
    rng = np.random.default_rng(SEED)
    G = const.G
    M = rng.uniform(1e24, 2e30, N)
    m = rng.uniform(1e0, 1e5, N)
    r = rng.uniform(1e7, 1e12, N)
    X = np.column_stack([M, m, r])
    y = _add_noise(G * M * m / r, rng)

    model_kwargs = dict(
        constants={"G": (G, {"m": 3, "kg": -1, "s": -2})},
        lambda_sym=0.5, linear_threshold=0.15, pysr_threshold=0.20, epochs=20, verbose=False,
    )
    evolve_kwargs = dict(
        feature_names=["kitle_buyuk", "kitle_kucuk", "mesafe"],
        feature_dims=[{"kg": 1}, {"kg": 1}, {"m": 1}],
        target_dim={"kg": 1, "m": 2, "s": -2},
        # U=G*M*m/r büyük kütlede derece-1 homojendir
        symmetry_constraints=[SymmetryConstraint(kind="scale", feature_index=0, scale_exponent=1.0)],
    )
    return (X, y, model_kwargs, evolve_kwargs,
            "U = G·M·m/r", "belirsiz — koşuda gözlenecek (M/m oranı teknik olarak serbest Pi-grubu)")


def make_ds5_wien():
    rng = np.random.default_rng(SEED)
    b_wien = const.Wien
    T = rng.uniform(300, 30000, N)
    X = T.reshape(-1, 1)
    y = _add_noise(b_wien / T, rng)

    model_kwargs = dict(
        constants={"b_wien": (b_wien, {"m": 1, "K": 1})},
        lambda_sym=0.5, linear_threshold=0.15, pysr_threshold=0.20, verbose=False,
    )
    evolve_kwargs = dict(
        feature_names=["sicaklik"],
        feature_dims=[{"K": 1}],
        target_dim={"m": 1},
        limit_constraints=[LimitConstraint(feature_index=0, behavior="monotonic_decreasing")],
    )
    return (X, y, model_kwargs, evolve_kwargs,
            "λ_max = b/T", "saf-sabit bekleniyor (in_features=0)")


DATASETS = [
    ("DS1_Planck", make_ds1_planck),
    ("DS2_StefanBoltzmann", make_ds2_stefan_boltzmann),
    ("DS3_Stokes", make_ds3_stokes),
    ("DS4_Gravity", make_ds4_gravity),
    ("DS5_Wien", make_ds5_wien),
]

OUTPUT_DIR = "./evolve_benchmark_results"


def run_dataset(name, make_fn, n_iterations, seed_runs):
    X, y, model_kwargs, evolve_kwargs, formula, expectation = make_fn()
    n_val = int(0.2 * len(y))
    X_val, y_val = X[:n_val], y[:n_val]
    X_train, y_train = X[n_val:], y[n_val:]

    print(f"\n{'='*70}\n{name}: {formula}\n  [beklenti: {expectation}]\n{'='*70}")

    _judge_stats["total"] = 0
    _judge_stats["rejected"] = 0
    _judge_stats["reject_reasons"] = []

    t0 = time.time()
    archive = evolve(
        X_train, y_train, X_val, y_val,
        n_iterations=n_iterations, seed_runs=seed_runs,
        model_kwargs=model_kwargs,
        patience=n_iterations + seed_runs + 1,   # bu koşuda erken durdurma istemiyoruz
        save_path=f"{OUTPUT_DIR}/{name}_archive.json",
        verbose=True,
        **evolve_kwargs,
    )
    elapsed = time.time() - t0

    all_cands = archive.all_candidates()
    top3 = sorted(all_cands, key=lambda c: c.fitness, reverse=True)[:3]
    return {
        "name": name, "formula": formula, "expectation": expectation,
        "elapsed_s": elapsed,
        "n_candidates": _judge_stats["total"],
        "n_rejected": _judge_stats["rejected"],
        "reject_reasons": list(_judge_stats["reject_reasons"]),
        "coverage": archive.coverage(),
        "best": archive.best(),
        "top3": top3,
    }


def print_summary(results):
    print(f"\n\n{'#'*78}\nÖZET TABLO\n{'#'*78}")
    header = (f"{'Dataset':<22} {'coverage':>9} {'best_fit':>9} "
              f"{'aday':>5} {'red':>4} {'süre_s':>8}")
    print(header)
    print("-" * len(header))
    for r in results:
        best = r["best"]
        best_fit = best.fitness if best else float("nan")
        print(f"{r['name']:<22} {r['coverage']:>6}/64 {best_fit:>9.4f} "
              f"{r['n_candidates']:>5} {r['n_rejected']:>4} {r['elapsed_s']:>8.1f}")

    for r in results:
        print(f"\n--- {r['name']} ({r['formula']}) — en iyi 3 aday ---")
        for i, c in enumerate(r["top3"], 1):
            print(f"  #{i} fitness={c.fitness:.4f} descriptor={c.descriptor} [{c.source}]  {c.expr_str}")
        if r["reject_reasons"]:
            print(f"  reddedilenler ({len(r['reject_reasons'])}):")
            for rr in r["reject_reasons"]:
                print(f"    descriptor={rr['descriptor']} kalan_testler={rr['failed']}  {rr['expr'][:60]}")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    mode = sys.argv[1] if len(sys.argv) > 1 else "full"
    n_iterations = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    seed_runs = int(sys.argv[3]) if len(sys.argv) > 3 else 3

    if mode == "calibrate":
        selected = [d for d in DATASETS if d[0] in ("DS1_Planck", "DS2_StefanBoltzmann")]
    else:
        selected = DATASETS

    results = []
    t_total0 = time.time()
    for name, make_fn in selected:
        results.append(run_dataset(name, make_fn, n_iterations, seed_runs))
    t_total = time.time() - t_total0

    print_summary(results)
    print(f"\nTOPLAM SÜRE ({mode}): {t_total:.1f}s ({t_total/60:.1f} dk)")
