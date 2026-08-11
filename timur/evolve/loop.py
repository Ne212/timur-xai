"""
timur/evolve/loop.py
═════════════════════════════════════════════════════════════════════════════
Ana özyinelemeli döngü: SEED → (sample → mutate → judge → archive) LOOP.

TIMURModel.discover()'ın tek-geçişli davranışını, çoklu-aday üreten,
arşivlenen ve fitness'a göre geri-beslenen bir arama sürecine dönüştüren üst
katman. Çekirdek kod (timur/__init__.py, timur/symbolic/, timur/pinn/) HİÇ
DEĞİŞTİRİLMEZ — yalnızca import edilip tekrar tekrar çağrılır.

Faz 2: judge.score()+judge.accept() ayrı ayrı çağrılmak yerine tek bir
judge.evaluate() akışına birleştirildi — R² eşiği + sonluluk testi + (varsa)
limit davranışı testleri artık her adayda birlikte değerlendiriliyor.
Faz 3: evaluate() artık ayrıca opsiyonel simetri/korunum testlerini de
(symmetry_constraints/conservation_constraints, None-safe) destekliyor.
"""
from __future__ import annotations

import logging
import random
import time
from typing import Any, Dict, List, Optional

import numpy as np

from timur import TIMURModel
from timur.evolve.archive import Archive, Candidate
from timur.evolve.features import extract_descriptor
from timur.evolve.judge import (
    DEFAULT_ACCEPT_THRESHOLD,
    ConservationConstraint,
    LimitConstraint,
    SymmetryConstraint,
    evaluate,
)
from timur.evolve.mutate import mutate

_log = logging.getLogger(__name__)

DEFAULT_PATIENCE = 10       # plato kontrolü: son P iterasyonda iyileşme yoksa dur
DEFAULT_EPSILON = 1e-4      # "iyileşme" sayılmak için gereken min. fitness artışı


def _make_candidate(model: TIMURModel, feature_names: Optional[List[str]],
                     source: str) -> Candidate:
    result = model.discovery_result
    descriptor = extract_descriptor(result.equation_str, feature_names)
    return Candidate(
        expr_str=result.equation_str,
        descriptor=descriptor,
        fitness=float("nan"),
        frozen_fn=result.frozen_fn,
        source=source,
        metadata={
            "model": model,
            "routing": result.routing,
            "r2_pretrain": result.r2_pretrain,
            "r2_refine": result.r2_refine,
        },
    )


def _seed_archive(
    archive: Archive,
    X: np.ndarray,
    y: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_kwargs: Dict[str, Any],
    feature_names: Optional[List[str]],
    seed_runs: int,
    judge_kwargs: Dict[str, Any],
    rng: np.random.Generator,
    verbose: bool,
) -> None:
    for i in range(seed_runs):
        # Her seed koşusunda satırları bootstrap ile yeniden örnekle: PySR'ın
        # kendi iç rastgeleliğiyle birlikte farklı başlangıç adayları üretir.
        n = X.shape[0]
        idx = rng.integers(0, n, size=n)

        model = TIMURModel(**model_kwargs)
        model.fit(X[idx], y[idx])

        cand = _make_candidate(model, feature_names, source="seed")
        accepted = evaluate(cand, X_val, y_val, rng=rng, **judge_kwargs)
        if accepted:
            archive.add(cand)

        if verbose:
            print(f"  [SEED {i + 1}/{seed_runs}] fitness={cand.fitness:.4f} "
                  f"descriptor={cand.descriptor} kabul={accepted}")


def evolve(
    X: np.ndarray,
    y: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: Optional[List[str]] = None,
    feature_dims: Optional[List[dict]] = None,
    target_dim: Optional[dict] = None,
    n_iterations: int = 50,
    seed_runs: int = 5,
    model_kwargs: Optional[Dict[str, Any]] = None,
    patience: int = DEFAULT_PATIENCE,
    epsilon: float = DEFAULT_EPSILON,
    r2_threshold: float = DEFAULT_ACCEPT_THRESHOLD,
    limit_constraints: Optional[List[LimitConstraint]] = None,
    symmetry_constraints: Optional[List[SymmetryConstraint]] = None,
    conservation_constraints: Optional[List[ConservationConstraint]] = None,
    check_finiteness: bool = True,
    seed: int = 42,
    save_path: Optional[str] = None,
    verbose: bool = True,
) -> Archive:
    """SEED + LOOP MAP-Elites döngüsü.

    1. SEED: TIMURModel'i `seed_runs` kez (bootstrap yeniden-örneklemeyle)
       çağırır → ilk adaylar üretilir, davranış tanımlayıcısı çıkarılır,
       judge.evaluate() ile puanlanır + test edilir, kabul edilirse arşive
       eklenir.
    2. LOOP (n_iterations): arşivden bir elite örneklenir → mutate() ile yeni
       aday üretilir → judge.evaluate() (R² eşiği + sonluluk + varsa
       limit/simetri/korunum testleri) kabul ederse archive.add() ile arşive
       yerleştirilir.
    3. Her iterasyonda iterasyon no, arşiv coverage'ı ve en iyi fitness loglanır.
    4. DURMA KRİTERİ: `n_iterations` dolması VEYA son `patience` iterasyonda en
       iyi fitness'taki iyileşme `epsilon`'dan az ise (plato).

    r2_threshold             : judge.evaluate()'e geçilen R² eşiği (varsayılan 0.5)
    limit_constraints        : judge.LimitConstraint listesi — verilmezse atlanır (None-safe)
    symmetry_constraints     : judge.SymmetryConstraint listesi — verilmezse atlanır (None-safe)
    conservation_constraints : judge.ConservationConstraint listesi — verilmezse atlanır (None-safe)
    check_finiteness         : sonluluk testini aç/kapat

    TIMURModel/DiscoveryResult kara-kutu olarak kullanılır; çekirdek kod hiç
    değiştirilmez. `model_kwargs` içine örn. `epochs=50` vererek her adayın
    PINN eğitim süresini kısaltıp döngüyü hızlandırabilirsiniz. Sonluluk/limit/
    simetri/korunum testleri için örnekleme alanı olarak X (X_val değil, daha
    geniş gözlenen aralık) kullanılır.
    """
    np_rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)

    base_kwargs: Dict[str, Any] = dict(model_kwargs or {})
    base_kwargs.setdefault("feature_names", feature_names)
    base_kwargs.setdefault("feature_dims", feature_dims)
    base_kwargs.setdefault("target_dim", target_dim)

    judge_kwargs: Dict[str, Any] = dict(
        X_domain=X,
        r2_threshold=r2_threshold,
        limit_constraints=limit_constraints,
        symmetry_constraints=symmetry_constraints,
        conservation_constraints=conservation_constraints,
        check_finiteness=check_finiteness,
    )

    archive = Archive()

    if verbose:
        print(f"\n[EVOLVE] ─── SEED aşaması ({seed_runs} koşu) ───")
    _seed_archive(
        archive, X, y, X_val, y_val, base_kwargs, feature_names,
        seed_runs, judge_kwargs, np_rng, verbose,
    )
    if verbose:
        best0 = archive.best()
        best0_fit = best0.fitness if best0 else float("nan")
        print(f"[EVOLVE] Seed tamamlandı. coverage={archive.coverage()} "
              f"best_fitness={best0_fit:.4f}")

    best_fitness_history: List[float] = []
    t0 = time.perf_counter()

    for it in range(1, n_iterations + 1):
        elites = archive.sample_elites(1, rng=py_rng)
        if not elites:
            _log.warning("Arşiv boş — iterasyon %d atlanıyor (hiçbir seed kabul edilmedi).", it)
            break
        elite = elites[0]

        improved = False
        try:
            cand = mutate(elite, X, y, base_kwargs, rng=np_rng)
            if evaluate(cand, X_val, y_val, rng=np_rng, **judge_kwargs):
                improved = archive.add(cand)
        except Exception as exc:
            _log.warning("iterasyon %d: mutasyon başarısız (%s)", it, exc)

        best = archive.best()
        best_fitness = best.fitness if best else float("-inf")
        best_fitness_history.append(best_fitness)

        if verbose:
            print(f"[EVOLVE] iter {it:>4d}/{n_iterations}  "
                  f"coverage={archive.coverage():>3d}  best_fitness={best_fitness:.4f}"
                  f"{'  (yeni elite!)' if improved else ''}")

        # ── Durma kriteri: plato ──────────────────────────────────────────
        if len(best_fitness_history) > patience:
            window = best_fitness_history[-patience:]
            if max(window) - min(window) < epsilon:
                if verbose:
                    print(f"\n[EVOLVE] Durduruldu: son {patience} iterasyonda "
                          f"iyileşme < {epsilon} (plato).")
                break

    elapsed = time.perf_counter() - t0
    if verbose:
        print(f"\n[EVOLVE] ─── Tamamlandı ({elapsed:.1f}s, coverage={archive.coverage()}) ───")
        top5 = sorted(archive.all_candidates(), key=lambda c: c.fitness, reverse=True)[:5]
        for rank, c in enumerate(top5, 1):
            print(f"  #{rank}  fitness={c.fitness:.4f}  descriptor={c.descriptor}  "
                  f"[{c.source}]  {c.expr_str}")

    if save_path:
        archive.save(save_path)
        if verbose:
            print(f"[EVOLVE] Arşiv kaydedildi → {save_path}")

    return archive
