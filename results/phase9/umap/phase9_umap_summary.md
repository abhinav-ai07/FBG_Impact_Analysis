# Phase 9 — Uniform Manifold Approximation and Projection (UMAP) Summary Report

## 1. Executive Summary & Scientific Objective
Phase 9 UMAP performs unsupervised non-linear manifold exploration to investigate the geometric structure of Fiber Bragg Grating (FBG) impact signals across **Bare (FBG2)**, **Copper (FBG1)**, and **Steel (FBG3)** packaging configurations.

Unlike supervised classifiers or biased post-hoc optimizations, this implementation follows strict scientific protocol:
1. **Primary Analysis**: Uses the **Complete 24-Feature Multidomain Representation** (integrating Phase 5 Engineering, Phase 6 FFT, Phase 6 Wavelet, and Phase 8 Physics-Guided Indices).
2. **Feature Preservation**: No features were selected or removed to artificially enhance visual cluster separation or maximize silhouette scores.
3. **Unsupervised Embedding**: UMAP was fitted strictly without material labels or metadata.
4. **Transparent Multi-Representation Comparison**: The complete representation is compared against three secondary reduced representations to answer distinct engineering questions.

---

## 2. Input Datasets & Scope
- **Phase 5 Source**: `results/phase5/phase5_all_features.csv` (Read-only)
- **Phase 6 Source**: `results/phase6/phase6_multidomain_features.csv` (Read-only)
- **Phase 8 Source**: `results/phase8/phase8_engineering_indices.csv` (Read-only)
- **PCA Alignment Source**: `results/phase9/pca/phase9_pca_scores.csv` (Read-only verification)
- **Event-Level Representation**: 1 Row = 1 Valid Impact Event.
- **Event Distribution**:
  - Total sensor events: 21
  - Excluded NO IMPACT events: 9
  - **Analyzed Valid IMPACT events: 12**
    - Bare (FBG2): $n = 7$ (58.3%)
    - Copper (FBG1): $n = 3$ (25.0%)
    - Steel (FBG3): $n = 2$ (16.7%)

---

## 3. Systematic Multi-Representation Overview

| Representation | Role | Feature Count | Features Included | Post-hoc Silhouette Score | Core Scientific Question |
|---|---|:---:|---|:---:|---|
| **Complete Multidomain** | **PRIMARY** | **24** | Phase 5 (10) + Phase 6 FFT (5) + Phase 6 Wavelet (5) + Phase 8 (4) | **0.1693** | What manifold structure appears when the full transient, spectral (FFT), wavelet, and physics-guided information is considered? |
| **Traditional + PGMSIF** | Comparative | **14** | Phase 5 Engineering (10) + Phase 8 Physics-Guided Indices (4) | **0.2685** | How does augmenting conventional transient engineering features with physics-guided indices alter non-linear manifold geometry? |
| **PGMSIF Only** | Comparative | **4** | DSTI, PEI, SII, DRI | **0.2483** | What intrinsic geometric structure exists purely within the 4 dimensionless physics-guided engineering indices? |
| **Traditional Only** | Comparative | **10** | Phase 5 Time-Domain Engineering Features (10) | **0.2725** | What non-linear structure exists solely within conventional time-domain transient engineering features? |

---

## 4. UMAP Hyperparameters & Reproducibility
- **Neighbor Size (`n_neighbors`)**: 5 (Appropriate for $N=12$ small-sample manifold topology)
- **Minimum Distance (`min_dist`)**: 0.1
- **Embedding Dimensions (`n_components`)**: 2
- **Distance Metric (`metric`)**: Euclidean
- **Random State (`random_state`)**: 42 (Fixed deterministic seed)
- **Scaling**: Standard z-score standardization ($z = (x - \mu) / \sigma$) applied prior to UMAP fitting.

---

## 5. Manifold Structure & Comparative Observations

### A. Complete Multidomain (Primary Analysis, 24 Features)
- **Manifold Structure**: Exhibits **exploratory partial separation with distinct operational envelopes** across the 12 impact events.
- **Copper ($n=3$)**: Forms an apparent cluster (Exp 8, 9, 10), reflecting consistent dynamic attenuation and mechanical damping.
- **Steel ($n=2$)**: Groups in a distinct region (Exp 9, 13), consistent with preserved high-frequency resonant modes and dynamic response indices (DRI).
- **Bare Fiber ($n=7$)**: Spans across the manifold, dividing into moderate-impact responses (Exp 7, 10, 11, 12) and severe, high-strain impact excursions (Exp 8, 9, 13) due to direct, unattenuated mechanical contact.
- **Silhouette Context**: The post-hoc silhouette score (0.169) reflects the broad intra-class operational dispersion of Bare fiber across differing impact intensities when spectral and wavelet details are fully present.

### B. Traditional + PGMSIF (14 Features)
- Integrates time-domain strain and rate metrics with the 4 dimensionless indices. Shows apparent groupings between Copper and Steel with intermediate Bare transitions (Silhouette: 0.268).

### C. PGMSIF Only (4 Features)
- Maps the pure 4D physics-guided index space (DSTI, PEI, SII, DRI). Shows apparent grouping of the packaging configurations in the reduced PGMSIF feature space (Silhouette: 0.248).

### D. Traditional Only (10 Features)
- Time-domain strain features show structure associated with variation in the conventional engineering features, but lack the frequency-domain packaging insights provided by spectral and wavelet analysis (Silhouette: 0.273).

---

## 6. Methodological & Scientific Limitations
1. **Sample Size & Exploratory Scope**: Experimental data comprises only $N=12$ valid impact events (Bare $n=7$, Copper $n=3$, Steel $n=2$). All UMAP separation observations are strictly exploratory and should not be interpreted as definitive classification or universal material behavior.
2. **Exploratory Manifold Learning**: UMAP is a non-linear visualization technique. Manifold clustering indicates apparent structural tendencies but does not constitute statistical proof of generalized material classification boundaries.
3. **Unsupervised Integrity**: No material labels were used during UMAP optimization; all silhouette scores are strictly exploratory post-hoc evaluations.
4. **No Feature Cherry-Picking**: Higher silhouette scores in reduced feature sets reflect lower intra-class variance in restricted feature spaces, but do not imply that reduced sets are scientifically superior to the complete multidomain representation.

---

## 7. Artifacts Summary
- `phase9_umap_complete_multidomain_coordinates.csv`: Coordinates for Primary 24-feature UMAP.
- `phase9_umap_traditional_pgmsif_coordinates.csv`: Coordinates for Traditional + PGMSIF (14 features).
- `phase9_umap_pgmsif_coordinates.csv`: Coordinates for PGMSIF Only (4 features).
- `phase9_umap_traditional_coordinates.csv`: Coordinates for Traditional Only (10 features).
- `phase9_umap_feature_metadata.json`: Feature definitions and parameter records.
- `plots/phase9_umap_complete_multidomain.png`: Publication plot of Primary UMAP.
- `plots/phase9_umap_traditional_pgmsif.png`: Publication plot of Traditional + PGMSIF.
- `plots/phase9_umap_pgmsif_only.png`: Publication plot of PGMSIF Only.
- `plots/phase9_umap_traditional_only.png`: Publication plot of Traditional Only.
- `plots/phase9_umap_comparison.png`: 2x2 comparison grid of all 4 representations.
