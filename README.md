# TIMUR XAI (v1.2.0)
**Autonomous Dimensionless Physics-Informed Symbolic Regression Engine**

TIMUR (Theoretical Inference and Multiphysics Universal Regressor) is an advanced, autonomous Explainable AI (XAI) architecture that bridges the gap between raw dimensional data, universal physical constants, and deep learning. 

Unlike standard machine learning models that blindly fit curves to scale-dependent data, TIMUR utilizes the **Buckingham $\pi$ Theorem** to autonomously project variables into a scale-invariant dimensionless space. It then leverages evolutionary genetic algorithms (PySR) and purely analytical engines to discover the exact underlying fundamental physical equations. Finally, it integrates these absolute truths into a Physics-Informed Neural Network (PINN).

### 🚀 What's New in v1.1.0
* **Autonomous Inverse Transformation:** Discovered dimensionless invariants ($\Pi$-space) are now autonomously mapped back into exact Standard International (SI) units, bridging the gap between symbolic logic and real-world physical reality.
* **Zero-Dimensional PyTorch Shields:** Analytically resolves absolute constants (e.g., $6\pi$ in Stokes' Law) directly into the tensor space, bypassing standard neural network matrix limitations.
* **Pure Analytic Scaling:** Removed L2 regularization penalties from the dimensional engine, allowing the discovery of astronomical scaling ratios (e.g., $10^{-27}$ in Universal Gravitation) without vanishing coefficient errors.

## Features
* **Zero-Prior Dimensional Analysis:** Pass your features, target, and physical constants. TIMUR dynamically computes the Null Space matrix and transforms the chaotic data into scale-invariant dimensionless Pi groups.
* **Evolutionary Symbolic Discovery:** Escapes polynomial approximations. Genetically evolves complex fractional, exponential, and trigonometric truths (e.g., Planck's Law of Black-Body Radiation) underlying the data.
* **PINN Integration:** Converts the discovered physical law into a fully differentiable, frozen PyTorch tensor space without string-parsing overhead.
* **The Neuro-Symbolic Gatekeeper:** Analyzes data non-linearity to autonomously route the logic between Analytical Linear solvers or Evolutionary Genetic (PySR) engines.

## Installation
```bash
pip install --upgrade timur-xai
python -c "import pysr; pysr.install()"
```

## Quick Start
```python
from timur import TIMURModel
import scipy.constants as const

# Initialize the engine with dimensional awareness (Example: Planck's Law)
model = TIMURModel(
    feature_names=["wavelength", "temperature"],
    feature_dims=[{"m": 1}, {"K": 1}],
    target_dim={"kg": 1, "m": -1, "s": -3},
    constants={
        "h":  (const.h, {"kg": 1, "m": 2, "s": -1}),
        "c":  (const.c, {"m": 1, "s": -1}),
        "kB": (const.k, {"kg": 1, "m": 2, "s": -2, "K": -1})
    },
    linear_threshold=0.15,
    pysr_threshold=0.20, # Set to 0.0 to force evolutionary genetic discovery
    verbose=True
)

# TIMUR will autonomously discover the dimensionless Pi law and train the PINN
model.fit(X, y)

# Output the symbolic XAI discovery report
print(model.get_xai_report())
```

### 🚀 What's New in v1.2.0: Autonomous Discovery via MAP-Elites

`TIMURModel.fit()` is single-pass: it runs `discover()` once and returns one
result. The new `timur/evolve/` layer sits on top of that — without changing
any core `timur/`, `timur/symbolic/`, or `timur/pinn/` code — and turns TIMUR
into a **black-box candidate generator** that is called repeatedly, scored,
and archived for diversity instead of just accuracy.

* **MAP-Elites archive:** candidates are mapped into a 3-axis behavioral grid
  — `(complexity, depth, operator_family)` — and only the single best (highest
  fitness) candidate per cell is kept, preserving a diverse population of
  equations instead of collapsing to one.
* **Evolutionary loop:** `evolve()` seeds the archive with N bootstrap-resampled
  TIMUR runs, then iterates M times — sample an elite, mutate it, re-evaluate,
  and add it back if it's accepted — until `n_iterations` is reached or fitness
  plateaus for `patience` iterations.
* **Multi-layered physical judge (`judge.py`):** every candidate must clear up
  to five independent tests before entering the archive:
  1. **R² threshold** on a held-out validation split (default 0.5)
  2. **Finiteness** — no NaN/Inf across a sampled input domain
  3. **Limit / monotonicity behavior** (`LimitConstraint`) — e.g. finite at
     zero/infinity, monotonic increasing/decreasing
  4. **Symmetry** (`SymmetryConstraint`) — even/odd/scale invariance,
     e.g. `f(a·x) = a^k·f(x)`
  5. **Conservation** (`ConservationConstraint`) — a trapezoidal-integral
     check against an expected constant or axis-independence

  Dimensional consistency is not a separate test: since every candidate is
  already generated in Buckingham Pi space, it's structurally guaranteed.
  All constraint types are optional and None-safe — omitting them just skips
  that test.

### Five-Law Benchmark Validation

`evolve_benchmark.py` runs the full evolutionary loop against five physical
laws (same data-generation logic as `timur_benchmark.py`):

| Law | R² | Status |
|---|---|---|
| DS1 — Planck Blackbody Radiation | 0.7756 | ✓ Passed |
| DS2 — Stefan-Boltzmann Power Density | 1.0000 | ✓ Passed |
| DS3 — Stokes' Drag Force | 1.0000 | ✓ Passed |
| DS4 — Gravitational Potential Energy | 0.9974 | ✓ Passed |
| DS5 — Wien's Displacement Law | 1.0000 | ✓ Passed |

Archived candidates for each law are in `evolve_benchmark_results/`. See
`timur/evolve/README.md` for architecture details and `evolve_demo.py` /
`evolve_demo_planck.py` for end-to-end runnable examples.

## License

TIMUR-XAI is released under the **MIT License** — free for any use, including commercial, closed-source, and research. No permission needed. Just keep the copyright notice.

See the [LICENSE](LICENSE) file for details.
