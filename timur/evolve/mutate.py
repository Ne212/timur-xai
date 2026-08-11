"""
timur/evolve/mutate.py
═════════════════════════════════════════════════════════════════════════════
Varyasyon üreteci.

Bir elite Candidate'tan yola çıkarak TIMURModel'i (kara-kutu olarak) yeniden
çağırıp yeni bir aday üretir — MAP-Elites'in "varyasyon operatörü" adımı.
TIMURModel harici bir random_state parametresi almadığından (bkz.
timur/__init__.py), Faz 1 stratejisi stokastikliği veri düzeyinde (bootstrap
yeniden-örnekleme) ve eşik pertürbasyonuyla sağlar; bu PySR'ın kendi iç
rastgeleliğiyle birleşince her çağrıda farklı bir denklem adayına yol açar.

`strategy` parametresi ile değiştirilebilir (pluggable) tasarlanmıştır — Faz
2'de burada bir LLM-tabanlı mutasyon stratejisi takılabilir.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from timur import TIMURModel
from timur.evolve.archive import Candidate
from timur.evolve.features import extract_descriptor

MutationStrategy = Callable[
    [Optional[Candidate], np.ndarray, np.ndarray, Dict[str, Any], np.random.Generator],
    Tuple[np.ndarray, np.ndarray, Dict[str, Any]],
]

# linear_threshold/pysr_threshold'a uygulanan +/- pertürbasyon genliği
THRESHOLD_JITTER = 0.05


def bootstrap_jitter_strategy(
    elite: Optional[Candidate],
    X: np.ndarray,
    y: np.ndarray,
    model_kwargs: Dict[str, Any],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Faz 1 varsayılan strateji: satırları bootstrap ile yeniden örnekle +
    yönlendirme eşiklerini hafifçe pertürbe et. `elite` şu an kullanılmıyor
    (Faz 2'de örn. elite'in kullandığı özellik alt-kümesine göre
    koşullandırma eklenebilir)."""
    n = X.shape[0]
    idx = rng.integers(0, n, size=n)
    X_mut, y_mut = X[idx], y[idx]

    kwargs = dict(model_kwargs)
    for key in ("linear_threshold", "pysr_threshold"):
        if kwargs.get(key) is not None:
            jitter = float(rng.uniform(-THRESHOLD_JITTER, THRESHOLD_JITTER))
            kwargs[key] = float(np.clip(kwargs[key] + jitter, 0.0, 1.0))
    return X_mut, y_mut, kwargs


def mutate(
    elite: Optional[Candidate],
    X: np.ndarray,
    y: np.ndarray,
    model_kwargs: Dict[str, Any],
    rng: Optional[np.random.Generator] = None,
    strategy: MutationStrategy = bootstrap_jitter_strategy,
) -> Candidate:
    """elite'i temel alarak TIMURModel'i yeniden çalıştırır ve fitness'ı henüz
    hesaplanmamış (NaN) yeni bir Candidate döndürür — fitness'ı judge.score()
    ile ayrıca hesaplayıp doldurmak çağıranın sorumluluğundadır."""
    rng = rng or np.random.default_rng()
    X_mut, y_mut, kwargs = strategy(elite, X, y, model_kwargs, rng)

    model = TIMURModel(**kwargs)
    model.fit(X_mut, y_mut)
    result = model.discovery_result

    descriptor = extract_descriptor(result.equation_str, kwargs.get("feature_names"))

    return Candidate(
        expr_str=result.equation_str,
        descriptor=descriptor,
        fitness=float("nan"),
        frozen_fn=result.frozen_fn,
        source="mutation",
        metadata={
            "model": model,  # judge.score() SI-uzayı predict_symbolic() için kullanır
            "routing": result.routing,
            "r2_pretrain": result.r2_pretrain,
            "r2_refine": result.r2_refine,
        },
    )
