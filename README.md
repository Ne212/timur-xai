# TIMUR XAI (v1.0.0)
**Dimensionless Physics-Informed Symbolic Regression Engine**

TIMUR is an advanced, autonomous Explainable AI (XAI) architecture that bridges the gap between raw data, universal physical constants, and deep learning. 

Unlike standard machine learning models that blindly fit curves, TIMUR utilizes the **Buckingham Pi Theorem** to autonomously project variables into a dimensionless space. It then leverages evolutionary genetic algorithms (PySR) to discover the underlying fundamental physics equation, which is seamlessly integrated into a neural network as a Physics-Informed Neural Network (PINN) loss function.

## Features
* **Autonomous Dimensional Analysis:** Pass your features, target, and SI constants. TIMUR handles the Null Space matrix operations and transforms the space into dimensionless Pi groups.
* **Evolutionary Symbolic Discovery:** Escapes polynomial approximations. Finds complex fractional, exponential, and trigonometric truths underlying the data.
* **PINN Integration:** Converts the discovered physical law into a fully differentiable PyTorch tensor space without string-parsing overhead.
* **The Gatekeeper:** Analyzes data non-linearity to autonomously route between Ridge/Lasso, Polynomial, or Genetic Evolutionary engines.

## Installation
```bash
pip install timur-xai
python -c "import pysr; pysr.install()"

## Quick Start
```python
from timur import TIMURModel
import scipy.constants as const

# Initialize the engine with dimensional awareness
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
    pysr_threshold=0.20,
    verbose=True
)

# TIMUR will autonomously discover the dimensionless Pi law and train the PINN
model.fit(X, y)
print(model.get_xai_report())




Licensing & Commercial Use
Dual-License Strategy

TIMUR XAI is released under the GNU General Public License v3.0 (GPLv3).

Academic & Open Source: Free to use, modify, and distribute for non-commercial, open-source academic research.

Commercial / Enterprise: If you intend to use TIMUR XAI within a closed-source, proprietary, or commercial product, a Commercial License is strictly required. Contact the author directly.


