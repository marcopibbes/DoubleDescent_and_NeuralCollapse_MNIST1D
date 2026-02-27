# 🧠 Double Descent · Neural Collapse · CKA
### An Experimental Analysis on MNIST-1D

<p align="center">
  <img src="neural_collapse_cka_results.png" alt="Results Overview" width="900"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/MNIST--1D-Dataset-4B8BBE?style=flat-square"/>
  <img src="https://img.shields.io/badge/License-MIT-22c55e?style=flat-square"/>
</p>

---

## Overview

This project investigates three fundamental phenomena in modern deep learning, using **MNIST-1D** as a controlled testbed with **20% label noise**:

| Phenomenon | What we measure | Key finding |
|---|---|---|
| **Double Descent** | Test error vs. model width | Error peaks at interpolation threshold, then drops again |
| **Neural Collapse** | NC1–NC4 geometry metrics | Features collapse to ETF simplex as capacity grows |
| **CKA** | Layer-wise representation similarity | Deep layers drift more; representations are seed-invariant |

---

## Experiments

### Exp 1 — Model-Wise (Double Descent + NC)
Trains **ResNet-1D** and **CNN-1D** at 11 different widths `w ∈ {2, 4, 6, 8, 10, 12, 16, 24, 32, 48, 64}` with both **SGD** and **Adam**, measuring:
- Test error (double descent curve)
- NC1: within/between-class scatter ratio
- NC2: deviation from ideal ETF simplex
- NC3: weight–mean self-duality
- NC4: NCC vs. linear classifier mismatch rate

### Exp 2 — Epoch-Wise Dynamics
Tracks the evolution of test error and NC1 over **15 000 epochs** for:
- `Critical model` (w=12, near interpolation peak)
- `Large model` (w=64, overparameterized regime)

### Exp 3 — Generalized NC (Feature Dimension)
Fixes width at 64 and varies the **feature space dimension** `d ∈ {2, 3, 5, 8, 9, 10, 12, 16}` via a linear projector, probing the theoretical threshold **d = K−1 = 9** for ETF realizability.

### Exp 4 — Centered Kernel Alignment (CKA)

Three sub-experiments using `linear_CKA` computed on 500-sample subsets:

#### 4A — Cross-Architecture
Compares internal representations of a **small** (w=8) and a **large** (w=64) model via a 4×4 layer-pair CKA heatmap.

```
             Large L0   L1   L2   L3
Small L0   [  ·    ·    ·    ·  ]
      L1   [  ·    ·    ·    ·  ]
      L2   [  ·    ·    ·    ·  ]
      L3   [  ·    ·    ·    ·  ]
```

High diagonal → architectures of different width learn similar per-level abstractions.

#### 4B — Epoch-Wise Drift
Measures how much each layer's representations change relative to **epoch 1**, at checkpoints `{1, 50, 200, 500, 1000, 2000}`.  
Deep layers show greater drift; surface layers stabilize early.

#### 4C — Cross-Seed Reproducibility
Three models (w=32) trained with seeds `{42, 123, 456}`. CKA across all three pairs, per layer.  
High CKA → representations are **functionally determined by the data**, not the random init.

---

## Architecture

```
ResNet-1D (width w)
───────────────────────────────────────
Input (1, 40)
  └─ Conv1D(1→w, k=3) + BN + ReLU         ← Layer 0 / Stem
  └─ ResBlock(w): Conv→BN→ReLU→Conv→BN    ← Layer 1
     + skip connection + ReLU
  └─ ResBlock(w): same                    ← Layer 2
  └─ GlobalAvgPool → h ∈ R^w             ← Layer 3 / Features
  └─ Linear(w→10, no bias)               ← Classifier W
```

```
CNN-1D (width w)
───────────────────────────────────────
Conv(1→w)+BN+ReLU+MaxPool  ×2           (w channels)
Conv(w→2w)+BN+ReLU         ×2           (2w channels)
GlobalAvgPool → Linear(2w→10)
```

---

## NC Metrics — Quick Reference

$$\text{NC1} = \frac{\text{Tr}(\Sigma_W)}{\text{Tr}(\Sigma_B) + \varepsilon}$$

$$\text{NC2} = \left\| G_\text{emp} - G_\text{ideal} \right\|_F, \quad G_\text{ideal} = \frac{K}{K-1}\!\left(I_K - \tfrac{1}{K}\mathbf{1}\mathbf{1}^\top\right)$$

$$\text{NC3} = \left\| \hat{W} - \hat{M} \right\|_F$$

$$\text{NC4} = \frac{1}{N}\sum_i \mathbf{1}[\text{NCC}(h_i) \neq \arg\max \, W h_i]$$

$$\text{CKA}(X, Y) = \frac{\|Y^\top X\|_F^2}{\|X^\top X\|_F \cdot \|Y^\top Y\|_F}$$

---

## Installation

```bash
# Clone the repo
git clone https://github.com/your-username/double-descent-nc-cka.git
cd double-descent-nc-cka

# Create environment
conda create -n dd_nc python=3.10
conda activate dd_nc

# Install dependencies
pip install torch torchvision matplotlib numpy mnist1d
```

---

## Usage

```bash
python neural_collapse_cka.py
```

The script will:
1. Download MNIST-1D automatically on first run (`mnist1d_data.pkl`)
2. Run all 4 experiments sequentially (~hours on CPU, ~30 min on GPU)
3. Save the full results figure as `neural_collapse_cka_results.png`

> **Tip:** To run individual experiments, call `run_model_wise_complete()`, `run_epoch_wise_dynamics()`, `run_generalized_nc()`, or `run_cka_experiment()` directly.

### Estimated runtimes (NVIDIA RTX 3080)

| Experiment | Approx. time |
|---|---|
| Exp 1 — Model-Wise (4 configs × 11 widths) | ~90 min |
| Exp 2 — Epoch-Wise (2 models × 15k epochs) | ~45 min |
| Exp 3 — Generalized NC (8 dims) | ~20 min |
| Exp 4 — CKA (3 sub-exps) | ~30 min |

---

## Configuration

All key hyperparameters are defined at the top of `neural_collapse_cka.py`:

```python
SEED            = 1100
LABEL_NOISE     = 0.20       # fraction of corrupted labels
BATCH_SIZE      = 128

EPOCHS_MODEL_WISE = 3000
EPOCHS_EPOCH_WISE = 15000
EPOCHS_GEN_NC     = 3000
EPOCHS_CKA        = 2000

WIDTHS = [2, 4, 6, 8, 10, 12, 16, 24, 32, 48, 64]
```

---

## Project Structure

```
.
├── neural_collapse_cka.py        # Main experiment script
├── mnist1d_data.pkl              # Auto-downloaded dataset cache
├── neural_collapse_cka_results.png  # Output figure (generated)
└── README.md
```

---

## References

- Papyan et al. — *Prevalence of Neural Collapse during the Terminal Phase of Deep Learning Training*, PNAS 2020
- Kornblith et al. — *Similarity of Neural Network Representations Revisited*, ICML 2019
- Belkin et al. — *Reconciling Modern Machine-Learning Practice and the Classical Bias–Variance Trade-off*, PNAS 2019
- Nakkiran et al. — *Deep Double Descent: Where Bigger Models and More Data Hurt*, JSMTE 2021
- Greydanus — *Scaling Down Deep Learning*, arXiv 2020

---

## License

MIT