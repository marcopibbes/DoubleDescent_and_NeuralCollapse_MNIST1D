# Deep Double Descent & Neural Collapse on MNIST1D

This repository contains a PyTorch implementation to reproduce and analyze the **Deep Double Descent** phenomenon and its correlation with **Neural Collapse (NC)** using the **MNIST1D** dataset.

The project investigates how over-parameterization affects generalization and geometric structure of the feature space in deep learning models, bridging the gap between two seminal papers:
1. **Deep Double Descent** (Nakkiran et al., 2019)
2. **Neural Collapse** (Papyan et al., 2020)

## Key Experiments

The code (`doubledescent.py`) performs three distinct experiments in a single run:

### 1. Model-wise Double Descent
*   **Goal:** Observe the test error peak at the interpolation threshold ($N_{params} \approx N_{samples}$) and the subsequent descent.
*   **Setup:** Compares **ResNet1D (Adam)** vs **Standard CNN (SGD)** with varying widths ($k$).
*   **Metrics:** Test Error, Train Error, NC1 (Variability), NC2 (Simplex ETF), NC3, NC4.

### 2. Epoch-wise Double Descent
*   **Goal:** Observe "Benign Overfitting" over long training times (20k epochs).
*   **Setup:** Compares a **Critical Model** (width near interpolation threshold) vs a **Large Model** (highly over-parameterized).
*   **Observation:** The large model improves over time (Test Error $\downarrow$, NC1 $\downarrow$), while the critical model overfits (Test Error $\uparrow$, NC1 $\uparrow$).

### 3. Generalized Neural Collapse
*   **Goal:** Verify geometric constraints when the feature dimension $d$ is smaller than the number of classes $K$.
*   **Setup:** Fixed width over-parameterized model, varying penultimate layer dimension $d$.
*   **Observation:** NC2 (Simplex ETF) fails to converge if $d < K-1$.

## 🛠️ Requirements

The code relies on `mnist1d`, a lightweight dataset designed to demonstrate deep learning phenomena efficiently.

```bash
pip install torch numpy matplotlib mnist1d
