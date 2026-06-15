# TIMUR XAI (v1.1.0)
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
Quick Start
Python
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
Licensing & Commercial Use
Dual-License Strategy

TIMUR XAI is released under the GNU General Public License v3.0 (GPLv3).

Academic & Open Source: Free to use, modify, and distribute strictly for non-commercial, open-source academic research.

Commercial / Enterprise: If you intend to use TIMUR XAI within a closed-source, proprietary, or commercial product (e.g., industrial R&D, corporate AI models), a Commercial License is strictly required. Contact the author directly to acquire commercial rights.
