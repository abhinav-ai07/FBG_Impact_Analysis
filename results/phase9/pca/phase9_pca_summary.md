# Phase 9 — Principal Component Analysis (PCA) Summary Report

## 1. Scientific Objective
The objective of Phase 9 PCA is to determine whether the multi-domain feature space—combining transient engineering features (Phase 5), spectral and wavelet features (Phase 6), and physics-guided indices (Phase 8)—contains natural structure or separation associated with the **Bare**, **Copper**, and **Steel** FBG packaging configurations in an unsupervised manner.

---

## 2. Input Datasets & Scope
- **Phase 5 Source**: `results/phase5/phase5_all_features.csv` (Read-only)
- **Phase 6 Source**: `results/phase6/phase6_multidomain_features.csv` (Read-only)
- **Phase 8 Source**: `results/phase8/phase8_engineering_indices.csv` (Read-only)
- **Event-Level Representation**: 1 Row = 1 Valid Impact Event.
- **Sample Distribution**:
  - Total sensor events: 21
  - Excluded NO IMPACT events: 9
  - **Analyzed Valid IMPACT events: 12**
    - Bare (FBG2): $n = 7$ (58.3%)
    - Copper (FBG1): $n = 3$ (25.0%)
    - Steel (FBG3): $n = 2$ (16.7%)

---

## 3. Feature Selection & Standardization
- **Total PCA Features**: 24 multi-domain features across 4 distinct domains:
  1. **Phase 5 Time-Domain Engineering Features (10)**: `peak_shift_abs`, `rise_time_seconds`, `signal_energy`, `rms`, `peak_to_peak`, `std_dev`, `entropy`, `max_slope_abs`, `auc_abs`, `noise_std_nm`.
  2. **Phase 6 Spectral FFT Features (5)**: `Dominant_Frequency`, `Spectral_Energy`, `Spectral_Entropy`, `Spectral_Centroid`, `Bandwidth`.
  3. **Phase 6 Wavelet Multi-Resolution Features (5)**: `Approximation_Energy`, `Detail_Energy`, `Wavelet_Energy`, `Wavelet_Entropy`, `Detail_Approx_Ratio`.
  4. **Phase 8 Physics-Guided Indices (4)**: `DSTI`, `PEI`, `SII`, `DRI`.

### Feature Scaling:
- **Standardization**: Standard z-score scaling ($z = (x - \mu) / \sigma$) via `StandardScaler` was applied prior to PCA to ensure all features contribute equally regardless of physical units.

### Missing-Value Handling:
- Sparse raw columns with significant missingness (`peak_width_seconds` with 8/12 missing, `recovery_time_seconds` with 3/12 missing, `residual_shift_abs` with 3/12 missing) were excluded from raw $X$.
- Dynamic recovery and baseline preservation mechanics are fully and systematically represented without missing values via the validated Phase 8 bounded indices (`PEI`, `SII`, `DRI`).
- Zero samples were discarded or fabricated.

---

## 4. Explained Variance Breakdown

| Principal Component | Eigenvalue | Individual Explained Variance (%) | Cumulative Explained Variance (%) |
|---|---|---|---|
| **PC1** | 9.9946 | **38.17%** | **38.17%** |
| **PC2** | 4.6172 | **17.64%** | **55.81%** |
| **PC3** | 3.3717 | 12.88% | 68.69% |
| **PC4** | 2.6482 | 10.11% | 78.80% |
| **PC5** | 1.7186 | 6.56% | 85.37% |
| **PC6** | 1.2489 | 4.77% | 90.14% |

- **PC1 + PC2 Total Explained Variance**: **55.81%**
- **First 3 Components (PC1–PC3)**: **68.69%**

---

## 5. Dominant PCA Loadings & Physical Interpretation

### Principal Component 1 (38.17% Variance):
- **Dominant Positive Loadings**:
  - `rms`: +0.3117
  - `peak_shift_abs`: +0.3108
  - `DSTI`: +0.2965
- **Dominant Negative Loadings**:
  - `Wavelet_Entropy`: -0.3014
  - `Detail_Approx_Ratio`: -0.2051
  - `noise_std_nm`: -0.1548
- **Physical Interpretation**: PC1 represents the **overall dynamic transient energy and strain magnitude axis**. Positive scores align with high peak strain transfer, elevated signal RMS, high DSTI, and strong signal integrity, balanced against wavelet entropy and detail ratio on the negative axis.

### Principal Component 2 (17.64% Variance):
- **Dominant Positive Loadings**:
  - `Spectral_Centroid`: +0.3628
  - `Dominant_Frequency`: +0.3371
  - `Spectral_Entropy`: +0.3194
- **Dominant Negative Loadings**:
  - `peak_to_peak`: -0.3437
  - `std_dev`: -0.3286
  - `Spectral_Energy`: -0.3161
- **Physical Interpretation**: PC2 represents the **spectral frequency distribution and frequency centering axis**. Positive scores reflect high spectral centroid and dominant frequencies alongside approximation energy, contrasted against high peak-to-peak amplitude excursion and spectral energy.

---

## 6. Structural Observation & Packaging Configuration Separation
In the unsupervised 2D PC1–PC2 projection (55.81% total variance):
1. **Copper Packaging (FBG1)**: Forms a relatively localized cluster in the negative PC1 region ($\\text{PC1} \\in [-3.43, -1.69]$), reflecting moderate, consistent strain transfer and controlled transient energy.
2. **Steel Packaging (FBG3)**: Occupies the negative PC1, positive PC2 quadrant ($\\text{PC1} \\in [-2.93, -2.13]$, $\\text{PC2} \\in [0.66, 1.23]$), characterized by high dynamic response indices, high frequency preservation, and lower raw strain transfer due to heavy structural stiffening.
3. **Bare Fiber (FBG2)**: Exhibits a wider dispersion across PC1 and PC2 ($\\text{PC1} \\in [-2.67, 5.39]$), spanning from moderate-intensity impacts to severe high-strain impacts (e.g., Experts 8, 9, 13) where direct mechanical contact generates extreme strain transfer and large transient RMS.

**Conclusion on Separation**:
The multi-domain feature space exhibits **Case 2: Partial Separation with Distinct Sub-Regions**. While Copper and Steel form localized envelopes corresponding to structured attenuation and mechanical buffering, Bare fiber spans a broader operational envelope reflecting its direct, unattenuated mechanical coupling across differing impact severities.

---

## 7. Explicit Scientific Limitations
1. **Small Sample Size**: The analysis is conducted on $N=12$ valid experimental impact events (Bare $n=7$, Copper $n=3$, Steel $n=2$).
2. **Exploratory Scope**: PCA visualization indicates observable clustering trends but does not constitute supervised proof of material classification boundaries.
3. **Absence of Overfitting**: PCA is strictly linear and unsupervised; no class labels or hyperparameter optimizations were used.
4. **Generalization Cautiousness**: Findings describe the physical response characteristics within this experimental setup and should not be generalized as universal material superiority claims.

---

## 8. Artifacts Generated

- `phase9_pca_scores.csv`: Event-level principal component scores (PC1–PC12) with associated metadata.
- `phase9_pca_loadings.csv`: Complete component loadings across all 24 multi-domain features.
- `phase9_pca_explained_variance.csv`: Eigenvalues, explained variance ratios, and cumulative percentages.
- `phase9_pca_feature_metadata.json`: Feature definitions, domain allocations, and exclusion documentation.
- `plots/phase9_pca_explained_variance.png`: Scree plot showing individual and cumulative explained variance.
- `plots/phase9_pca_pc1_pc2.png`: 2D scatter plot of PC1 vs PC2 with material markers and event annotations.
