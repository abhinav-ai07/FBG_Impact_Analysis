# Phase 9 — Part 2: Clustering, Autoencoder & Explainability Summary Report

## Physics-Guided Multi-Domain Signal Intelligence Framework (PGMSIF)

> [!CAUTION]
> **Dataset Size Limitation**: This entire Phase 9 Part 2 analysis is based on **N=12** validated
> IMPACT events (Bare n=7, Copper n=3, Steel n=2). All ML and exploratory findings must be
> interpreted with extreme caution. No result should be claimed as definitive, optimal,
> or generalisable without validation on a larger independent dataset.

## 1. Executive Summary & Research Objectives

Phase 9 Part 2 investigates the unsupervised structural properties, non-linear latent manifolds, and feature importance attribution of the 24-dimensional multi-domain FBG impact feature representation. **N=12 is the principal limitation throughout this analysis.** The framework addresses three exploratory scientific questions:

1. **Exploratory Clustering**: *Does the 24D multi-domain feature space reveal partial structure associated with packaging configurations?*
2. **Latent Space Compression**: *Can a non-linear autoencoder compress and reconstruct 24 features in-sample?*
3. **Supervised Exploratory Attribution**: *Within this dataset, which features receive the greatest attribution in a supervised exploratory model?*

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
- **Unsupervised Constraint**: Material labels were strictly excluded from clustering and autoencoder training.
  Labels are used only for post-hoc metric computation (ARI, NMI, Purity) and visualisation.
- **Feature Redundancy Note**: Some multidomain variables are physically or mathematically correlated,
  and Phase 8 indices are composites derived from underlying signal features. Therefore, model-based
  importance values should not be interpreted as independent causal contributions.

---
## 3. Part A — Clustering Analysis & Findings

> [!NOTE]
> Clustering is strictly unsupervised: labels are excluded from model fitting.
> ARI and NMI are computed post-hoc for comparison only, not used in clustering.
> Silhouette score is an internal clustering quality metric (no labels required).
> All findings are exploratory given N=12.

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


### DBSCAN Density-Based Clustering (Exploratory Configuration)

- **Selected Exploratory DBSCAN Configuration**: `eps = 4.0`, `min_samples = 2`.
  This configuration was selected for exploratory inspection; it is not claimed as 'optimal'.
- **Result**: 1 cluster(s) identified; 7 noise point(s) (58.3% of N=12).
  The presence of 7 noise point(s) limits the usefulness of this DBSCAN configuration for this small dataset.

### Exploratory Clustering Interpretation

Exploratory clustering reveals partial structure associated with packaging configurations in the 24D multi-domain feature space.
- **Binary structure (K=2)**: Ward hierarchical clustering shows ARI=0.4732, NMI=0.5493, Purity=0.7500.
  This is consistent with a broad separation between the Bare fiber regime and the packaged configurations,
  but does not constitute definitive material classification.
- **Three-group structure (K=3)**: Ward and K-Means clustering at K=3 yield NMI=0.4606, Purity=0.7500.
  Partial recovery of the three packaging types is observed, but given N=12 these metrics must be interpreted with caution.

> [!WARNING]
> No clustering result is treated as 'proof', 'optimal', or 'definitive material classification'.
> Silhouette scores in the range 0.12–0.30 indicate weak-to-moderate cluster separation, which is
> expected given the small sample size and class imbalance.

---
## 4. Part B — Autoencoder & Latent Space Characterisation

- **Architecture**: Fully-connected symmetric autoencoder: $\text{Input}(24) \to 12 \to 6 \to \text{Latent}(2) \to 6 \to 12 \to \text{Output}(24)$.
- **In-Sample Reconstruction Loss**: Mean MSE per feature = **0.02110**.
- **In-Sample Reconstruction Fidelity**: Mean $R^2 =$ **0.9789** across all 24 multi-domain features.

> [!IMPORTANT]
> The reconstruction metric above is an **IN-SAMPLE** fidelity measure.
> The autoencoder is trained and evaluated on the same 12 events.
> **The reconstruction metric is NOT a held-out test score** because the available dataset
> is too small for the current implementation to provide a reliable independent evaluation.
> This result demonstrates in-sample nonlinear representation and reconstruction capability,
> not generalisation to unseen data.

- **Latent Manifold ($z_1, z_2$)**: The 2D latent space organises events continuously along two learned axes,
  successfully condensing 24 dimensions while maintaining in-sample feature recoverability.

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
## 5. Part C — Supervised Exploratory Feature Attribution

> [!IMPORTANT]
> **The Random Forest model is SUPERVISED** — material labels (Y) are used to train it.
> This is NOT unsupervised learning.
> With N=12, the Random Forest achieves 100% in-sample training accuracy, which
> likely reflects memorisation/overfitting rather than genuine predictive generalisation.
> Feature importance and Shapley-style attribution are exploratory and dataset-specific;
> they do not constitute proof of which features universally govern material separation.
> The Shapley-style values are computed via a **custom sampling approximation**,
> not the standard SHAP library.

### Top Ranked Engineering Parameters (derived from actual CSV output)

| Rank | Feature | Domain | Composite Score | Gini Importance | Mean |Shapley| | Direction of Effect |
|---|---|---|---|---|---|---|
| **1** | `Wavelet_Energy` | Phase 6 Wavelet | 0.8252 | 0.1208 | 0.0029 | Elevated in Bare (Direct Strain Transfer) |
| **2** | `Approximation_Energy` | Phase 6 Wavelet | 0.7419 | 0.0777 | 0.0067 | Elevated in Bare (Direct Strain Transfer) |
| **3** | `noise_std_nm` | Phase 5 Time-Domain | 0.7210 | 0.0726 | 0.0069 | Elevated in Packaged (Buffering & Stiffness) |
| **4** | `rms` | Phase 5 Time-Domain | 0.6018 | 0.0651 | 0.0051 | Elevated in Bare (Direct Strain Transfer) |
| **5** | `auc_abs` | Phase 5 Time-Domain | 0.5470 | 0.0471 | 0.0063 | Elevated in Bare (Direct Strain Transfer) |
| **6** | `SII` | Phase 8 Physics-Guided | 0.5270 | 0.0588 | 0.0043 | Elevated in Bare (Direct Strain Transfer) |
| **7** | `signal_energy` | Phase 5 Time-Domain | 0.4687 | 0.0557 | 0.0033 | Elevated in Bare (Direct Strain Transfer) |
| **8** | `Wavelet_Entropy` | Phase 6 Wavelet | 0.4601 | 0.0573 | 0.0029 | Elevated in Packaged (Buffering & Stiffness) |
| **9** | `Detail_Approx_Ratio` | Phase 6 Wavelet | 0.4211 | 0.0520 | 0.0027 | Elevated in Packaged (Buffering & Stiffness) |
| **10** | `peak_shift_abs` | Phase 5 Time-Domain | 0.3969 | 0.0594 | 0.0012 | Elevated in Bare (Direct Strain Transfer) |


### Exploratory Attribution Insights (derived from actual ranking output):

The following insights are derived directly from the computed ranking and must be treated as exploratory,
dataset-specific observations within this N=12 analysis. They are not causal claims.

1. **`Wavelet_Energy` (Rank 1, Score=0.8252)** — Phase 6 Wavelet: Total multi-scale discrete wavelet transform energy across decomposition levels.
   Direction: Elevated in Bare (Direct Strain Transfer).
2. **`Approximation_Energy` (Rank 2, Score=0.7419)** — Phase 6 Wavelet: Low-frequency baseline wavelet energy corresponding to bulk flexural deformation.
   Direction: Elevated in Bare (Direct Strain Transfer).
3. **`noise_std_nm` (Rank 3, Score=0.7210)** — Phase 5 Time-Domain: Pre-impact optical baseline noise level reflecting intrinsic interrogation stability.
   Direction: Elevated in Packaged (Buffering & Stiffness).
4. **`rms` (Rank 4, Score=0.6018)** — Phase 5 Time-Domain: Root mean square amplitude reflecting effective dynamic deformation energy over the impact event.
   Direction: Elevated in Bare (Direct Strain Transfer).
5. **`auc_abs` (Rank 5, Score=0.5470)** — Phase 5 Time-Domain: Absolute area under the curve; measure of total integrated impulse.
   Direction: Elevated in Bare (Direct Strain Transfer).

---
## 6. Scientific Limitations & Cautious Interpretation

> [!CAUTION]
> **CRITICAL LIMITATION — N=12**: The entire Phase 9 Part 2 analysis rests on 12 validated
> experimental impact events with severe class imbalance (Bare n=7, Copper n=3, Steel n=2).
> This is the primary limitation. No finding should be extrapolated beyond this dataset.

> [!WARNING]
> 1. **Clustering**: Results are exploratory. No clustering result is treated as 'proof',
>    'optimal', or 'definitive material classification'. Silhouette scores in the 0.12–0.30
>    range indicate weak-to-moderate separation, consistent with a small imbalanced dataset.
> 2. **Autoencoder**: Demonstrates in-sample nonlinear compression and reconstruction only.
>    The R² metric is IN-SAMPLE, not a held-out generalisation score.
> 3. **Random Forest**: Is a supervised exploratory model, NOT a validated predictive model.
>    100% in-sample training accuracy likely reflects memorisation given N=12.
> 4. **Feature Attribution**: Gini importance, permutation importance, and approximate
>    Shapley-style values are exploratory and dataset-specific. They reflect the fitted
>    model's attributions, not causal or universal feature importance.
> 5. **Feature Redundancy**: Some multidomain variables are physically or mathematically
>    correlated, and Phase 8 indices are composites. Model-based importance values should
>    not be interpreted as independent causal contributions.
> 6. **DBSCAN**: The eps=4.0, min_samples=2 configuration is a selected exploratory
>    configuration, not a statistically validated 'optimal' parameter set.
> 7. **Phase 9 Part 2 prepares the feature space and ML methodology for future predictive
>    modelling, but does NOT claim that a reliable predictive model has already been
>    established. Future work requires a larger, balanced experimental dataset.

---
## 7. Final Scientific Conclusions

- Exploratory clustering provides evidence of partial structure in the multidomain feature space consistent with packaging configurations.
- No clustering result is treated as definitive material classification.
- The autoencoder demonstrates in-sample nonlinear representation and reconstruction capability (Mean R² = 0.9789), not generalisation to unseen data.
- The Random Forest is a supervised exploratory analysis with 100% in-sample training accuracy, not a validated predictive model.
- Approximate Shapley-style attribution and feature importance are exploratory and dataset-specific.
- **N=12 is the major limitation.** Phase 9 Part 2 demonstrates analytical methodology and prepares the
  feature space for future predictive modelling once a larger dataset becomes available.

---
## 8. Reproducibility

To execute the full Phase 9 Part 2 pipeline and regenerate all outputs deterministically:

```bash
python phase9_part2.py
```
