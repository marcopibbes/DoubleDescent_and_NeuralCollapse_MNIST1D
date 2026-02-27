#  Double Descent · Neural Collapse · CKA
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

| Phenomenon | What we measure | Key finding (observed) |
|---|---|---|
| **Double Descent** | Test error vs. model width | Interpolation peak ~10³ params; ResNet+Adam reaches **~8% error** at w=64 |
| **Neural Collapse** | NC1–NC4 geometry metrics | NC1 drops >2 orders of magnitude; NC4 hits **exactly 0** for large models; NC3 plateaus due to label noise |
| **CKA** | Layer-wise representation similarity | Stem CKA **0.97** cross-seed; Feature layer drift to **~0.25** vs epoch 1 |

---

## Experiments

### Exp 1 — Model-Wise (Double Descent + NC)

Trains **ResNet-1D** and **CNN-1D** at 11 widths `w ∈ {2,4,6,8,10,12,16,24,32,48,64}` with both **SGD** and **Adam**.

**Observed results:**

**Test error** shows a clear double descent across all 4 configs. The interpolation peak appears around 10³ parameters, reaching ~0.50–0.60 error. ResNet+Adam achieves the lowest final error (~8%) at w=64; CNN+SGD remains the highest (~13%).

**NC1** shows a bell-shaped curve with the peak at the interpolation threshold — the model memorizes noisy labels without organizing features geometrically. In the overparameterized regime NC1 drops by **more than two orders of magnitude** (from ~10⁰ to ~10⁻²). ResNet+Adam achieves the cleanest collapse.

**NC2** decreases monotonically without the intermediate peak seen in NC1, converging to ~0.5–1.0 (not zero) even at w=64 — a direct consequence of 20% label noise polluting the class means. All four configurations reach similar final values, confirming NC2 is architecture/optimizer-agnostic.

**NC3** behaves anomalously: it rises at the interpolation peak (like NC1), then decreases only partially, plateauing at **~2.5–3.5** far from the theoretical zero. Label noise systematically misaligns the weight matrix W from the class means M. ResNet+Adam shows NC3 ~0.5–1.0 units lower than CNN+SGD throughout.

**NC4** shows the most dramatic behavior, dropping from ~0.5–0.6 for small models to **exactly 0** for large ResNet+Adam and CNN+SGD models. The nearest-class-center classifier becomes fully equivalent to the learned linear classifier — the feature geometry alone is sufficient for classification.

> **Key insight:** NC4 reaches 0 while NC3 stays at ~3.0. Functional equivalence (NC4) is achieved without full metric alignment (NC3) — a decoupling caused by label noise.

---

### Exp 2 — Epoch-Wise Dynamics

Tracks test error and NC1 over **15,000 epochs** for a critical model (w=12) and a large model (w=64).

**Observed results:**

The **critical model (w=12)** exhibits non-monotonic test error: it descends, rises near the interpolation threshold with visible oscillations (epoch ~100–200), then slowly stabilizes around 0.20–0.25. NC1 decreases slowly and remains relatively high. Full Neural Collapse does **not emerge** even after 15k epochs.

The **large model (w=64)** descends monotonically to ~0.07–0.10 with an epoch-wise double descent bump visible around epoch 100. NC1 collapses rapidly and completely within ~2,000 epochs.

Neural Collapse is a **capacity-dependent phenomenon**: it emerges quickly and cleanly only in overparameterized models.

---

### Exp 3 — Generalized NC (Feature Dimension)

Fixes width at 64 and varies feature space dimension `d ∈ {2,3,5,8,9,10,12,16}` via a linear projector. Theoretical threshold: **d = K−1 = 9**.

**Observed results:**

**NC1 vs d** is non-monotonic and counterintuitive. NC1 is low for small d (classes compressed together — both intra- and inter-class distances are small), peaks sharply at **d=9**, then drops for d≥10. The peak at the theoretical threshold occurs because the model attempts to open the ETF geometry but the structure is not yet stable at exactly d=K−1.

**NC2 vs d** decreases monotonically with a sharp knee at **d=9**. NC2≈6.5 for d=2 (ETF geometrically impossible in ℝ²), drops to ~1.0 at d=9, and stabilizes at ~1.0–1.5 for d≥10. The theoretical threshold is **clearly confirmed**. NC2 does not reach zero even for large d due to label noise.

---

### Exp 4 — Centered Kernel Alignment (CKA)

Three sub-experiments using `linear_CKA` on 500-sample subsets.

#### 4A — Cross-Architecture (Small w=8 vs Large w=64)

```
              Large
         L0(Stem) L1(Bl1) L2(Bl2) L3(Feat)
Small L0 [  0.94   0.61    0.24    0.24  ]
      L1 [  0.62   0.57    0.27    0.27  ]
      L2 [  0.28   0.33    0.34    0.34  ]
      L3 [  0.26   0.33    0.34    0.34  ]
```

The diagonal is strong at shallow layers (Stem-to-Stem = **0.94**) and degrades with depth (Feature-to-Feature = **0.34**). Off-diagonal values are uniformly low (~0.24–0.34) with no layer-shift effect. Notably, L2 and L3 of the small model show nearly identical CKA profiles, indicating the small model saturates representational capacity at Block 2 and cannot differentiate deeper levels.

#### 4B — Epoch-Wise Drift (w=32, reference = epoch 1)

| Layer | CKA at epoch 2000 | Behavior |
|---|---|---|
| L0 — Stem | **~0.97** | Stable throughout — generic feature extractor |
| L1 — Block 1 | **~0.80** | Gradual moderate drift |
| L2 — Block 2 | **~0.55** | Stronger drift — task adaptation begins here |
| L3 — Feature | **~0.25** | Rapid and dramatic reorganization |

Training proceeds **bottom-up in stability**: deep layers change fast, shallow layers stabilize early. Consistent with feature reuse in residual networks.

#### 4C — Cross-Seed Reproducibility (w=32, seeds {42, 123, 456})

| Layer | CKA (mean ± std) |
|---|---|
| L0 — Stem | **0.97 ± ~0.01** |
| L1 — Block 1 | **0.81 ± ~0.02** |
| L2 — Block 2 | **0.59 ± ~0.03** |
| L3 — Feature | **0.59 ± ~0.03** |

The sharpest transition is between L1 and L2. Once variability enters at Block 2, it propagates unchanged to the final features — L2 and L3 share the same CKA value (~0.59). Low-level features are essentially **data-determined**; high-level features are influenced by the random initialization and amplified by label noise.

---

## Architecture

```
ResNet-1D (width w)
───────────────────────────────────────
Input (1, 40)
  └─ Conv1D(1→w, k=3) + BN + ReLU         <- Layer 0 / Stem
  └─ ResBlock(w): Conv->BN->ReLU->Conv->BN <- Layer 1 (+ skip)
  └─ ResBlock(w): same                    <- Layer 2 (+ skip)
  └─ GlobalAvgPool -> h in R^w            <- Layer 3 / Features
  └─ Linear(w->10, no bias)               <- Classifier W

CNN-1D (width w)
───────────────────────────────────────
Conv(1->w)+BN+ReLU+MaxPool  x2           (w channels)
Conv(w->2w)+BN+ReLU         x2           (2w channels)
GlobalAvgPool -> Linear(2w->10)
```

---

## NC Metrics — Quick Reference

$$\text{NC1} = \frac{\text{Tr}(\Sigma_W)}{\text{Tr}(\Sigma_B) + \varepsilon}$$

$$\text{NC2} = \left\| G_\text{emp} - G_\text{ideal} \right\|_F, \quad G_\text{ideal} = \frac{K}{K-1}\left(I_K - \frac{1}{K}\mathbf{1}\mathbf{1}^\top\right)$$

$$\text{NC3} = \left\| \hat{W} - \hat{M} \right\|_F \qquad \text{NC4} = \frac{1}{N}\sum_i \mathbf{1}[\text{NCC}(h_i) \neq \arg\max\, W h_i]$$

$$\text{CKA}(X, Y) = \frac{\|Y^\top X\|_F^2}{\|X^\top X\|_F \cdot \|Y^\top Y\|_F}$$

---

## Key Findings Summary

| Metric | Small models | Peak | Large models (w=64) |
|---|---|---|---|
| Test Error | ~0.40–0.50 | ~0.50–0.60 | **~0.08–0.13** |
| NC1 | ~10⁰ | ~10⁰ (max) | **~10⁻²** |
| NC2 | ~4–7 | — | **~0.5–1.0** |
| NC3 | ~3.5–4.5 | — | **~2.5–3.5** (plateau, noise effect) |
| NC4 | ~0.4–0.6 | — | **0.0** (exact) |
| CKA Stem cross-seed | — | — | **0.97** |
| CKA Feature cross-seed | — | — | **0.59** |
| NC1 peak (Gen NC, d) | at d=9 | max | drops for d≥10 |
| NC2 knee (Gen NC, d) | ~6.5 at d=2 | — | ~1.0 at d=9 |

---

## Installation

```bash
git clone https://github.com/marcopinners/double-descent-nc-cka.git
cd double-descent-nc-cka
conda create -n dd_nc python=3.10
conda activate dd_nc
pip install torch torchvision matplotlib numpy mnist1d
```

## Usage

```bash
python neural_collapse_cka.py
```

The script downloads MNIST-1D automatically, runs all 4 experiments, and saves `neural_collapse_cka_results.png`.

### Estimated runtimes (NVIDIA RTX 3080)

| Experiment | Approx. time |
|---|---|
| Exp 1 — Model-Wise (4 configs × 11 widths) | ~90 min |
| Exp 2 — Epoch-Wise (2 models × 15k epochs) | ~45 min |
| Exp 3 — Generalized NC (8 dims) | ~20 min |
| Exp 4 — CKA (3 sub-exps) | ~30 min |

---

## Project Structure

```
.
├── neural_collapse_cka.py           # Main experiment script
├── mnist1d_data.pkl                 # Auto-downloaded dataset cache
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