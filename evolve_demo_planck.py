"""
evolve_demo_planck.py
═════════════════════════════════════════════════════════════════════════════
TANI / DOĞRULAMA KOŞUSU — evolve() döngüsünün gerçekten çeşitlilik ürettiğini
PySR yoluna giden bir veri setinde (DS1 Planck Kara Cisim Işıması) test eder.

evolve_demo.py, Stefan-Boltzmann (DS2) kullanıyordu; o set Buckingham Pi
analizinde in_features=0 "saf-sabit" yoluna düşüp PySR'ı HİÇ çalıştırmadığı
için arşiv zorunlu olarak tek hücrede kalıyordu. Bu script DS1 Planck'ı
kullanıyor (bkz. timur_benchmark.py make_ds1) — bu set in_features=1 ile
gerçek PySR aramasını (exp/sin gibi doğrusal-olmayan operatörler) tetikliyor.

Bu script timur/evolve/'e HİÇBİR yeni özellik EKLEMİYOR — evolve() birebir
olduğu gibi çağrılıyor (kara-kutu). Yalnızca TEŞHİS amaçlı, bu dosyaya özel
geçici bir gözlem (monkeypatch) eklenmiştir: Archive.add ve loop.evaluate
sarmalanarak her adayın "yeni_hücre / iyileşti / mevcut_elite_daha_iyi /
yargıç_reddetti" sınıflandırması, düştüğü descriptor hücresi VE (Faz 2)
yargıcın hangi testten geçip hangisinden kaldığı (judge_report) loglanır.
Bu sarmalama evolve()'un iç mantığını DEĞİŞTİRMEZ, sadece dışarıdan izler.

Faz 2 güncellemesi: Görev A (Pi(...) tekilleştirme düzeltmesi) ve Görev B
(cardinality → nesting depth ekseni) uygulandıktan sonraki ızgara/yargıç ile
koşuluyor — Faz 1 koşusuyla (accept_threshold=0.0, cardinality ekseni,
dedup hatası) DOĞRUDAN KARŞILAŞTIRILAMAZ; bu bilinçli bir yeniden-temel
(re-baseline) koşusudur.
"""
import time

import numpy as np
import scipy.constants as const

import timur.evolve.loop as loop_mod
from timur.evolve.archive import Archive
from timur.evolve.judge import LimitConstraint
from timur.evolve.loop import evolve

# ─── TEŞHİS AMAÇLI GÖZLEM (yalnızca bu script içinde, kaynak koda dokunmaz) ───

stats = {"new_cell": 0, "improved": 0, "discarded_by_archive": 0, "rejected_by_judge": 0}
event_log = []

_orig_add = Archive.add
_orig_evaluate = loop_mod.evaluate


def _patched_add(self, candidate):
    cell = self._clip(candidate.descriptor)
    existed = cell in self._cells
    changed = _orig_add(self, candidate)
    if changed:
        kind = "improved" if existed else "new_cell"
    else:
        kind = "discarded_by_archive"
    stats[kind] += 1
    event_log.append({"cell": cell, "kind": kind, "fitness": candidate.fitness,
                       "source": candidate.source})
    print(f"    [İZLEME] descriptor={cell}  sonuç={kind:<22} fitness={candidate.fitness:.4f}")
    return changed


def _patched_evaluate(candidate, X_val, y_val, rng=None, **kwargs):
    result = _orig_evaluate(candidate, X_val, y_val, rng=rng, **kwargs)
    if not result:
        stats["rejected_by_judge"] += 1
        report = candidate.metadata.get("judge_report", {})
        failed = [k for k in ("r2_passed",) if not report.get(k, True)]
        if not report.get("finiteness", {}).get("passed", True):
            failed.append("finiteness")
        if not report.get("limit_behavior", {}).get("passed", True):
            failed.append("limit_behavior")
        event_log.append({"cell": tuple(candidate.descriptor), "kind": "rejected_by_judge",
                           "fitness": candidate.fitness, "source": candidate.source,
                           "failed_tests": failed})
        print(f"    [İZLEME] descriptor={tuple(candidate.descriptor)}  "
              f"sonuç=rejected_by_judge (kalan: {failed})  fitness={candidate.fitness:.4f}")
    return result


Archive.add = _patched_add
loop_mod.evaluate = _patched_evaluate

# ─── DS1 Planck veri seti (timur_benchmark.py make_ds1 ile aynı üretim) ───────

SEED = 42
N = 800


def make_planck_dataset():
    rng = np.random.default_rng(SEED)
    h, c, k = const.h, const.c, const.k
    wav = rng.uniform(100e-9, 3000e-9, N)
    T = rng.uniform(3000, 8000, N)
    X = np.column_stack([wav, T])

    def fn(wav, T):
        b = h * c / (wav * k * T)
        return (2.0 * h * c**2) / (wav**5 * (np.exp(b) - 1.0))

    y_true = fn(X[:, 0], X[:, 1])
    y_noisy = y_true * rng.normal(1.0, 0.01, size=y_true.shape)
    return X, y_noisy


if __name__ == "__main__":
    X, y = make_planck_dataset()

    n_val = int(0.2 * len(y))
    X_val, y_val = X[:n_val], y[:n_val]
    X_train, y_train = X[n_val:], y[n_val:]

    h, c, k = const.h, const.c, const.k
    model_kwargs = dict(
        constants={
            "h": (h, {"kg": 1, "m": 2, "s": -1}),
            "c": (c, {"m": 1, "s": -1}),
            "kB": (k, {"kg": 1, "m": 2, "s": -2, "K": -1}),
        },
        lambda_sym=0.5,
        linear_threshold=0.0,
        pysr_threshold=0.0,
        data_loss="weighted_mse",
        epochs=20,          # PINN fazını kısalt — PySR zaten dominant maliyet (~80s/çağrı)
        verbose=False,
    )

    # Fiziksel olarak motive edilmiş limit kısıtları (Görev C demosu):
    #   - sicaklik (sütun 1) arttıkça B(λ,T) monoton artmalı (Planck yasası,
    #     sabit λ'da ∂B/∂T > 0 her zaman doğrudur)
    #   - dalga_boyu (sütun 0) 0'a ve ∞'a yakın uçlarda (gözlenen aralığın
    #     dışına taşan ekstrapolasyonda) hâlâ sonlu çıktı vermeli
    limit_constraints = [
        LimitConstraint(feature_index=1, behavior="monotonic_increasing"),
        LimitConstraint(feature_index=0, behavior="finite_at_zero"),
        LimitConstraint(feature_index=0, behavior="finite_at_inf"),
    ]

    t0 = time.time()
    archive = evolve(
        X_train, y_train, X_val, y_val,
        feature_names=["dalga_boyu", "sicaklik"],
        feature_dims=[{"m": 1}, {"K": 1}],
        target_dim={"kg": 1, "m": -1, "s": -3},
        n_iterations=8,
        seed_runs=3,
        model_kwargs=model_kwargs,
        patience=20,        # bu tanı koşusunda erken durdurma istemiyoruz
        r2_threshold=0.5,   # Faz 2 varsayılanı (Faz 1'de 0.0'dı)
        limit_constraints=limit_constraints,
        check_finiteness=True,
        save_path="./evolve_demo_planck_archive.json",
        verbose=True,
    )
    elapsed = time.time() - t0

    print(f"\n{'='*70}")
    print(f"TOPLAM SÜRE: {elapsed:.1f}s")
    print(f"OLAY İSTATİSTİKLERİ: {stats}")
    print(f"FİNAL COVERAGE: {archive.coverage()} / 64")
    print(f"{'='*70}")

    print("\nEN İYİ 5 ADAY:")
    top5 = sorted(archive.all_candidates(), key=lambda cnd: cnd.fitness, reverse=True)[:5]
    for rank, cnd in enumerate(top5, 1):
        jr = cnd.metadata.get("judge_report", {})
        print(f"  #{rank}  fitness={cnd.fitness:.4f}  descriptor={cnd.descriptor}  "
              f"[{cnd.source}]\n        {cnd.expr_str}"
              f"\n        judge: r2_ok={jr.get('r2_passed')} "
              f"finiteness_ok={jr.get('finiteness', {}).get('passed')} "
              f"limit_ok={jr.get('limit_behavior', {}).get('passed')}"
              f" detay={jr.get('limit_behavior')}")

    print("\nDOLU HÜCRELER (descriptor -> fitness):")
    for cell, cnd in sorted(archive._cells.items()):
        print(f"  {cell}  fitness={cnd.fitness:.4f}  [{cnd.source}]  {cnd.expr_str[:70]}")
