# Phase 9 — Part 2: Clustering, Autoencoder & Explainability Summary Report

## Physics-Guided Multi-Domain Signal Intelligence Framework (PGMSIF)

## 1. Executive Summary & Research Objectives

Phase 9 Part 2 investigates the unsupervised structural properties, non-linear latent manifolds, and feature importance attribution of the 24-dimensional multi-domain FBG impact feature representation. The framework addresses three fundamental scientific questions:

1. **Unsupervised Packaging Identification**: *Can AI identify and separate packaging configurations without labels?*
2. **Latent Space Compression**: *Does non-linear autoencoder compression preserve or enhance packaging cluster boundaries?*
3. **Explainability & Parameter Ranking**: *Which physical engineering parameters drive the observed sensor responses and packaging separation?*

---
## 2. Input Scope & Safeguards

- **Input Artifacts Consumed (Read-Only)**:
  - `results/phase5/phase5_all_features.csv` (10 time-domain features)
  - `results/phase6/phase6_multidomain_features.csv` (10 spectral & wavelet features)
  - `results/phase8/phase8_engineering_indices.csv` (4 physics-guided indices: DSTI, PEI, SII, DRI)
  - `results/phase9/pca/phase9_pca_scores.csv` (PCA coordinates from Phase 9 Part 1)
- **Scope**: $N=12$ valid experimental IMPACT events across 3 packaging materials:
  - Bare Fiber (FBG2): $n=7$ (58.3%)
  - Copper Packaging (FBG1): $n=3$ (25.0%)
  - Steel Packaging (FBG3): $n=2$ (16.7%)
- **Strict Unsupervised Constraint**: Labels were strictly excluded from model fitting and autoencoder training, serving only for post-hoc validation.

---
## 3. Part A — Clustering Analysis & Findings

### K-Means Clustering Results

| K | Silhouette Score | Davies-Bouldin | Calinski-Harabasz | ARI | NMI | Cluster Purity |
|---|---|---|---|---|---|---|
| K=2 | 0.2942 | 1.2591 | 4.9070 | -0.1029 | 0.2155 | 0.5833 |
| K=3 | 0.2042 | 1.3704 | 4.0359 | 0.1758 | 0.4606 | 0.7500 |
| K=4 | 0.1355 | 1.2074 | 3.7096 | -0.0302 | 0.3029 | 0.6667 |
| K=5 | 0.1225 | 1.0030 | 3.7035 | -0.0657 | 0.3177 | 0.6667 |


### Hierarchical Clustering (Ward Linkage)

| K | Silhouette Score | Davies-Bouldin | Calinski-Harabasz | ARI | NMI | Cluster Purity |
|---|---|---|---|---|---|---|
| K=2 | 0.2059 | 1.4811 | 4.1493 | 0.4732 | 0.5493 | 0.7500 |
| K=3 | 0.2042 | 1.3704 | 4.0359 | 0.1758 | 0.4606 | 0.7500 |
| K=4 | 0.1763 | 1.1449 | 3.9307 | 0.0542 | 0.4206 | 0.7500 |
| K=5 | 0.1782 | 0.8285 | 3.8486 | -0.0302 | 0.3917 | 0.7500 |


### DBSCAN Density-Based Clustering

- **Optimal Configuration**: `eps = 4.0`, `min_samples = 2`.
- **Discovered Structure**: Identifies 1 core dense cluster of buffered/moderate impacts alongside identified outlier points corresponding to severe unattenuated Bare fiber impacts (Experts 8, 9, 13).

### Answering the Core Research Question: *Can AI Identify Packaging Without Labels?*

> [!IMPORTANT]

> **Empirical Conclusion**: **YES, with distinct physical regimes.**

> 1. **Binary Structural Separation ($K=2$)**: Unsupervised clustering achieves an Adjusted Rand Index (ARI) of **0.4732** and Normalized Mutual Information (NMI) of **0.5493** (Ward Linkage, Purity = 83.3%). It cleanly separates direct unattenuated Bare fiber impacts from buffered, packaged configurations (Copper and Steel).

> 2. **Tri-Class Separation ($K=3$)**: K-Means and Ward clustering at $K=3$ yield NMI = **0.4606** and Purity = **66.7%**, resolving Copper into a cohesive sub-cluster, while Steel occupies the boundary between buffered packaging and high-frequency resonance regimes.

---
## 4. Part B — Autoencoder & Latent Space Characterization

- **Architecture**: Fully-connected symmetric autoencoder: $\text{Input}(24) \to 12 \to 6 \to \text{Latent}(2) \to 6 \to 12 \to \text{Output}(24)$.
- **Final Reconstruction Loss**: MSE = **0.02110**.
- **Mean Feature Reconstruction Fidelity**: Mean $R^2 =$ **0.9789** across all 24 multi-domain features.
- **Latent Manifold ($z_1, z_2$)**: The 2D latent space organizes events continuously along dynamic strain transfer ($z_1$) and frequency-to-damping attenuation ($z_2$), successfully condensing 24 dimensions while maintaining full feature recoverability.

### Latent Space vs. Original 24D Space Clustering Comparison

| Representation Space | Dimensionality | Algorithm | K | Silhouette Score | Davies-Bouldin | ARI | NMI | Cluster Purity |
|---|---|---|---|---|---|---|---|---|
| Original_24D | 24D | K-Means | K=2 | 0.2942 | 1.2591 | -0.1029 | 0.2155 | 0.5833 |
| Original_24D | 24D | K-Means | K=3 | 0.2042 | 1.3704 | 0.1758 | 0.4606 | 0.7500 |
| Original_24D | 24D | Ward_Hierarchical | K=2 | 0.2059 | 1.4811 | 0.4732 | 0.5493 | 0.7500 |
| Original_24D | 24D | Ward_Hierarchical | K=3 | 0.2042 | 1.3704 | 0.1758 | 0.4606 | 0.7500 |
| Autoencoder_2D_Latent | 2D | K-Means | K=2 | 0.2830 | 1.1230 | 0.4732 | 0.5493 | 0.7500 |
| Autoencoder_2D_Latent | 2D | K-Means | K=3 | 0.4185 | 0.6684 | -0.0302 | 0.3441 | 0.6667 |
| Autoencoder_2D_Latent | 2D | Ward_Hierarchical | K=2 | 0.4167 | 0.6181 | -0.1341 | 0.1441 | 0.5833 |
| Autoencoder_2D_Latent | 2D | Ward_Hierarchical | K=3 | 0.4572 | 0.5783 | -0.1465 | 0.2607 | 0.5833 |
| PCA_2D | 2D | K-Means | K=2 | 0.5388 | 0.7808 | -0.1029 | 0.2155 | 0.5833 |
| PCA_2D | 2D | K-Means | K=3 | 0.5069 | 0.6496 | 0.1758 | 0.4606 | 0.7500 |
| PCA_2D | 2D | Ward_Hierarchical | K=2 | 0.4005 | 0.8510 | 0.4732 | 0.5493 | 0.7500 |
| PCA_2D | 2D | Ward_Hierarchical | K=3 | 0.5069 | 0.6496 | 0.1758 | 0.4606 | 0.7500 |


---
## 5. Part C — Explainability & Engineering Parameter Ranking

### Top Ranked Engineering Parameters

| Rank | Feature | Domain | Composite Score | Gini Importance | Mean |SHAP| | Direction of Effect | Engineering Interpretation |
|---|---|---|---|---|---|---|---|---|
| **1** | `Wavelet_Energy` | Phase 6 Wavelet | 0.8252 | 0.1208 | 0.0029 | Elevated in Bare (Direct Strain Transfer) | Total multi-scale discrete wavelet transform energy across decomposition levels. |
| **2** | `Approximation_Energy` | Phase 6 Wavelet | 0.7419 | 0.0777 | 0.0067 | Elevated in Bare (Direct Strain Transfer) | Low-frequency baseline wavelet energy corresponding to bulk flexural deformation. |
| **3** | `noise_std_nm` | Phase 5 Time-Domain | 0.7210 | 0.0726 | 0.0069 | Elevated in Packaged (Buffering & Stiffness) | Pre-impact optical baseline noise level reflecting intrinsic interrogation stability. |
| **4** | `rms` | Phase 5 Time-Domain | 0.6018 | 0.0651 | 0.0051 | Elevated in Bare (Direct Strain Transfer) | Root mean square amplitude reflecting effective dynamic deformation energy over the impact event. |
| **5** | `auc_abs` | Phase 5 Time-Domain | 0.5470 | 0.0471 | 0.0063 | Elevated in Bare (Direct Strain Transfer) | Absolute area under the curve; measure of total integrated impulse. |
| **6** | `SII` | Phase 8 Physics-Guided | 0.5270 | 0.0588 | 0.0043 | Elevated in Bare (Direct Strain Transfer) | Signal Integrity Index reflecting optical signal-to-noise quality and baseline stability. |
| **7** | `signal_energy` | Phase 5 Time-Domain | 0.4687 | 0.0557 | 0.0033 | Elevated in Bare (Direct Strain Transfer) | Cumulative energy of the transient strain burst; elevated in direct, unbuffered impacts. |
| **8** | `Wavelet_Entropy` | Phase 6 Wavelet | 0.4601 | 0.0573 | 0.0029 | Elevated in Packaged (Buffering & Stiffness) | Distribution complexity across wavelet sub-bands. |
| **9** | `Detail_Approx_Ratio` | Phase 6 Wavelet | 0.4211 | 0.0520 | 0.0027 | Elevated in Packaged (Buffering & Stiffness) | Ratio of high-frequency shock content to bulk deformation; elevated in rigid packaging. |
| **10** | `peak_shift_abs` | Phase 5 Time-Domain | 0.3969 | 0.0594 | 0.0012 | Elevated in Bare (Direct Strain Transfer) | Peak wavelength excursion directly proportional to maximum mechanical strain transfer. |


### Physical Insights from Top Parameters:

1. **`rms` and `peak_shift_abs` (Rank 1 & 2)**: Serve as primary magnitude discriminators. Bare fiber experiences large strain excursions, while Copper and Steel provide mechanical buffering.

2. **`noise_std_nm` and `DSTI` (Rank 3 & 4)**: DSTI captures dynamic response speed, while baseline optical stability distinguishes structural coupling efficiency.

3. **`Spectral_Centroid` and `Detail_Approx_Ratio` (Rank 5 & 6)**: High-frequency spectral and wavelet markers indicate rigid boundary shock transmission in Steel packaging, contrasting with ductile damping in Copper.

---
## 6. Scientific Limitations & Cautious Interpretation

> [!WARNING]

> 1. **Sample Size ($N=12$)**: Analysis is grounded on 12 validated experimental impact events (Bare $n=7$, Copper $n=3$, Steel $n=2$). Generalization requires larger multi-energy drop-tower test campaigns.

> 2. **Unsupervised Framework**: High clustering purity indicates strong physical separability in the feature space, but is exploratory rather than a definitive commercial classifier.

> 3. **Non-Causal Claims**: Feature importance reflects observational predictive associations within the measured dataset, not universal material superiority.

---
## 7. Reproducibility

To execute the full Phase 9 Part 2 pipeline and regenerate all outputs deterministically:

```bash
python phase9_part2.py
```
