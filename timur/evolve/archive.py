"""
timur/evolve/archive.py
═════════════════════════════════════════════════════════════════════════════
MAP-Elites arşivi.

Candidate: tek bir aday denklem + davranış tanımlayıcısı + fitness'ı taşıyan
veri yapısı. Archive: features.py'nin ürettiği 3 eksenli (complexity ×
depth × operator_family) ızgarada, her hücrede o davranış bölgesinin
EN İYİ (en yüksek fitness) tek adayını tutan popülasyon deposu — TIMUR'un
"tek-geçişli discover() → tek sonuç" davranışının yerini alan çeşitlilik
koruyan bellek katmanı budur.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

# features.py'deki COMPLEXITY_BINS/DEPTH_BINS ile eşleşir (her biri 4 bin).
GRID_SHAPE: Tuple[int, int, int] = (4, 4, 4)


@dataclass
class Candidate:
    expr_str: str
    descriptor: Tuple[int, int, int]
    fitness: float
    frozen_fn: Optional[Callable] = None
    source: str = "seed"                 # 'seed' | 'mutation'
    metadata: Dict[str, Any] = field(default_factory=dict)


class Archive:
    """MAP-Elites ızgarası: her hücrede o davranış bölgesindeki en iyi tek aday."""

    def __init__(self, grid_shape: Tuple[int, int, int] = GRID_SHAPE) -> None:
        self.grid_shape = grid_shape
        self._cells: Dict[Tuple[int, int, int], Candidate] = {}

    def _clip(self, descriptor: Tuple[int, int, int]) -> Tuple[int, int, int]:
        return tuple(
            max(0, min(d, self.grid_shape[i] - 1)) for i, d in enumerate(descriptor)
        )

    def add(self, candidate: Candidate) -> bool:
        """Hedef hücrede mevcut elite'ten daha fit ise yerleştirir/değiştirir.
        Arşiv değiştiyse True döner (yeni hücre ya da iyileşme)."""
        cell = self._clip(candidate.descriptor)
        current = self._cells.get(cell)
        if current is None or candidate.fitness > current.fitness:
            self._cells[cell] = candidate
            return True
        return False

    def best(self) -> Optional[Candidate]:
        if not self._cells:
            return None
        return max(self._cells.values(), key=lambda c: c.fitness)

    def all_candidates(self) -> List[Candidate]:
        return list(self._cells.values())

    def sample_elites(
        self, n: int, rng: Optional[random.Random] = None
    ) -> List[Candidate]:
        """Dolu hücrelerden rastgele n elite döndürür (boş hücreleri atlar)."""
        rng = rng or random
        pool = self.all_candidates()
        if not pool:
            return []
        n = min(n, len(pool))
        return rng.sample(pool, n)

    def coverage(self) -> int:
        return len(self._cells)

    def to_dict(self) -> Dict[str, Any]:
        def _json_safe(v: Any) -> Any:
            try:
                json.dumps(v)
                return v
            except TypeError:
                return str(v)

        return {
            "grid_shape": list(self.grid_shape),
            "coverage": self.coverage(),
            "cells": [
                {
                    "cell": list(cell),
                    "expr_str": c.expr_str,
                    "descriptor": list(c.descriptor),
                    "fitness": c.fitness,
                    "source": c.source,
                    "metadata": {k: _json_safe(v) for k, v in c.metadata.items()},
                }
                for cell, c in self._cells.items()
            ],
        }

    def save(self, path: str) -> None:
        """JSON serileştirme — frozen_fn hariç (yalnızca expr/descriptor/fitness)."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
