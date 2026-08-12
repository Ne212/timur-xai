# TIMUR-XAI

**Autonomous dimensionless physics-informed symbolic regression.**

TIMUR (Theoretical Inference and Multiphysics Universal Regressor) takes raw
dimensional data plus the relevant physical constants and tries to recover the
underlying physical law as a compact symbolic expression — not a black-box fit.

The core idea: instead of fitting curves to scale-dependent data, TIMUR uses the
**Buckingham π theorem** to project the variables into a dimensionless space
first, discovers the law there (analytically when it can, with genetic symbolic
regression when it can't), and maps the result back into SI units. A physical
judge then filters candidates that fit well numerically but violate physical
constraints.

> **Scope, honestly.** This is a tool for the *"given reasonably clean data,
> what's the compact law"* step — recovering structure from calibrated data or
> simulation output. It is **not** a tool for pulling real laws out of raw,
> systematics-dominated experimental measurements; separating true physics from
> systematic error is a genuinely harder problem and TIMUR does not claim to
> solve it. Benchmarks below use synthetic data generated from known laws (the
> standard way symbolic-regression methods are validated), with added noise.

---

## Installation

```bash
pip install --upgrade timur-xai
```

The core engine (dimensional analysis + analytic discovery + PINN) needs only
NumPy, SciPy, SymPy, scikit-learn and PyTorch.

**Optional — genetic symbolic regression.** Laws that don't reduce to a simple
analytic form (e.g. Planck) use [PySR](https://github.com/MilesCranmer/PySR),
which installs a Julia backend on first import. This needs internet and can take
a few minutes the first time. You only need it for the evolutionary examples; the
Quick Start below runs without it.

---

## Quick Start

Runs in seconds, no PySR/Julia required. TIMUR recovers the Stefan–Boltzmann law
`j = σ·T⁴` from noisy data:

```python
import numpy as np
import scipy.constants as const
from timur import TIMURModel

# Generate data from j = σ·T⁴, with 1% noise
np.random.seed(0)
T = np.random.uniform(100, 2000, 500)
X = T.reshape(-1, 1)
y = const.sigma * T**4
y *= 1 + np.random.normal(0, 0.01, size=y.shape)

model = TIMURModel(
    feature_names=["temperature"],
    feature_dims=[{"K": 1}],
    target_dim={"kg": 1, "s": -3},
    constants={"sigma": (const.sigma, {"kg": 1, "s": -3, "K": -4})},
    verbose=False,
)
model.fit(X, y)
print(model.get_xai_report())
# Discovered:  y = temperature⁴ · sigma
```

The report prints the recovered dimensionless law and its SI form. (Note: the
`Pretrain R²` / `Refine R²` fields in the report refer to the optional PINN
refinement stage and read 0.0 for laws solved purely analytically — the fit
quality for those is the R² reported by the evolutionary benchmark, below.)

---

## How it works

1. **Dimensional reduction.** Given features, target, and constants with their
   dimensions, TIMUR computes the null space of the dimensional matrix and forms
   the dimensionless Buckingham π groups automatically — no manual group-picking.

2. **Discovery in π-space.** A gatekeeper checks how nonlinear the reduced target
   is and routes it: a linear/analytic solver when the dimensionless law is
   simple, or genetic symbolic regression (PySR) when it needs exponential,
   fractional or trigonometric structure.

3. **Inverse transform.** The discovered dimensionless invariant is mapped back
   into exact SI units.

4. **PINN refinement (optional).** When the π group is a non-trivial function
   rather than a constant, the result can be refined as a differentiable
   PyTorch model.

5. **Physical judge.** Candidates are filtered by up to five independent checks
   before being accepted (see below).

---

## The physical judge

High R² does not mean physically correct. Each candidate must clear up to five
independent, toggleable tests:

1. **R² threshold** on a held-out validation split (default 0.5).
2. **Finiteness** — no NaN/Inf across a sampled input domain.
3. **Limit / monotonicity** — e.g. finite as `x→0` or `x→∞`, monotonic in a
   given variable (`LimitConstraint`).
4. **Symmetry** — even/odd or scale invariance, e.g. `f(a·x) = aᵏ·f(x)`
   (`SymmetryConstraint`).
5. **Conservation** — a trapezoidal-integral check against an expected constant
   or axis-independence (`ConservationConstraint`).

**Important:** the symmetry and conservation constraints are ones *you encode*
for a given problem — the judge enforces the physical priors you supply, it does
not infer them on its own. Dimensional consistency is the one automatic
guarantee, since every candidate is generated in π-space to begin with. All
constraint types are optional and None-safe; omitting one just skips that test.

---

## Benchmark

`evolve_benchmark.py` runs the full evolutionary loop (MAP-Elites archive +
judge) against five laws, using synthetic data with added noise. Best recovered
candidate per law:

| Law | R² | Recovered form |
|---|---|---|
| Planck black-body radiation | 0.9999 | `2 / (exp(1/Π) − 1)` structure in `Π = λkT/hc` |
| Stefan–Boltzmann | 0.9999 | `T⁴ · σ` |
| Stokes' drag | 0.9999 | `≈6π · η·r·v`  (recovered coefficient 18.86 vs 6π ≈ 18.85) |
| Gravitational potential energy | 0.9999 | `Π(m₁⁻¹·m₂)` |
| Wien's displacement | 0.9998 | `T⁻¹ · b` |

The Planck run is the interesting one: dimensional analysis alone only tells you
`Π₁ = f(Π₂)` — it never gives you `f`. TIMUR found the dimensionless group on its
own *and* recovered the `exp`/fraction structure of `f`, with constants landing
almost exactly on their true values. Archived candidates for every law are in
`evolve_benchmark_results/`.

See `timur/evolve/README.md` for the MAP-Elites architecture, and
`evolve_demo.py` / `evolve_demo_planck.py` for runnable end-to-end examples.

---

## MAP-Elites archive

`TIMURModel.fit()` is single-pass. The `timur/evolve/` layer sits on top without
touching the core engine and turns TIMUR into a candidate generator that's called
repeatedly, scored, and archived for *diversity* rather than just accuracy:
candidates are placed in a 3-axis grid — `(complexity, depth, operator_family)` —
keeping the best expression per cell. This exposes the accuracy-vs-simplicity
tradeoff instead of collapsing to a single winner.

---

## License

MIT — free for any use, including commercial and closed-source. Just keep the
copyright notice. See [LICENSE](LICENSE).
