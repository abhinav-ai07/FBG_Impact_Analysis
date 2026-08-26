# Phase 5 – Engineering Signal Characterization Report

## 1. Objective
Phase 5 calculates 13 core engineering signal features for confirmed Phase 4 impact events across FBG sensors prior to Machine Learning.

## 2. Methodology
- **Phase 4 Event Windowing**: Features are calculated strictly within Phase 4 event boundaries.
- **Case Separation**: 12 IMPACT cases analyzed; 9 NO-IMPACT cases separated as non-events (NaN).
- **Material Mapping**: FBG1 -> Copper, FBG2 -> Bare, FBG3 -> Steel.

## 3. Results Summary (Impact Cases Only)

| Feature | Bare (FBG2) | Copper (FBG1) | Steel (FBG3) | Observed Signal Description |
| --- | --- | --- | --- | --- |
| Peak Shift (Absolute) (nm) | 0.019940 (n=7) | 0.007880 (n=3) | 0.002564 (n=2) | Under tested Phase 4 impact events: Bare FBG2 averages 0.019940 nm (n=7), Copper FBG1 averages 0.007880 nm (n=3), and Steel FBG3 averages 0.002564 nm (n=2). |
| Residual Shift (Absolute) (nm) | 0.004239 (n=4) | 0.001090 (n=3) | 0.000469 (n=2) | Under tested Phase 4 impact events: Bare FBG2 averages 0.004239 nm (n=4), Copper FBG1 averages 0.001090 nm (n=3), and Steel FBG3 averages 0.000469 nm (n=2). |
| Rise Time (s) | 0.225994 (n=7) | 1.233187 (n=3) | 0.058054 (n=2) | Under tested Phase 4 impact events: Bare FBG2 averages 0.225994 s (n=7), Copper FBG1 averages 1.233187 s (n=3), and Steel FBG3 averages 0.058054 s (n=2). |
| Recovery Time (s) | 10.794892 (n=4) | 0.040000 (n=3) | 0.020000 (n=2) | Under tested Phase 4 impact events: Bare FBG2 averages 10.794892 s (n=4), Copper FBG1 averages 0.040000 s (n=3), and Steel FBG3 averages 0.020000 s (n=2). |
| Peak Width (FWHM) (s) | 12.409526 (n=3) | N/A | 1.738076 (n=1) | Reported mean across confirmed Phase 4 impact events: Bare FBG2 (12.409526 (n=3)), Copper FBG1 (N/A), Steel FBG3 (1.738076 (n=1)). |
| Maximum Slope (nm/s) | 0.133268 (n=7) | 0.136232 (n=3) | 0.052209 (n=2) | Under tested Phase 4 impact events: Bare FBG2 averages 0.133268 nm/s (n=7), Copper FBG1 averages 0.136232 nm/s (n=3), and Steel FBG3 averages 0.052209 nm/s (n=2). |
| RMS (nm) | 0.015421 (n=7) | 0.003867 (n=3) | 0.000973 (n=2) | Under tested Phase 4 impact events: Bare FBG2 averages 0.015421 nm (n=7), Copper FBG1 averages 0.003867 nm (n=3), and Steel FBG3 averages 0.000973 nm (n=2). |
| Signal Energy (nm²·s) | 0.002503 (n=7) | 0.000012 (n=3) | 0.000002 (n=2) | Under tested Phase 4 impact events: Bare FBG2 averages 0.002503 nm²·s (n=7), Copper FBG1 averages 0.000012 nm²·s (n=3), and Steel FBG3 averages 0.000002 nm²·s (n=2). |
| Peak-to-Peak (nm) | 0.015676 (n=7) | 0.010213 (n=3) | 0.004367 (n=2) | Under tested Phase 4 impact events: Bare FBG2 averages 0.015676 nm (n=7), Copper FBG1 averages 0.010213 nm (n=3), and Steel FBG3 averages 0.004367 nm (n=2). |
| Variance (nm²) | 0.000017 (n=7) | 0.000007 (n=3) | 0.000001 (n=2) | Under tested Phase 4 impact events: Bare FBG2 averages 0.000017 nm² (n=7), Copper FBG1 averages 0.000007 nm² (n=3), and Steel FBG3 averages 0.000001 nm² (n=2). |
| Standard Deviation (nm) | 0.002630 (n=7) | 0.002529 (n=3) | 0.000811 (n=2) | Under tested Phase 4 impact events: Bare FBG2 averages 0.002630 nm (n=7), Copper FBG1 averages 0.002529 nm (n=3), and Steel FBG3 averages 0.000811 nm (n=2). |
| Distributional Entropy (bits) | 3.310566 (n=7) | 1.992344 (n=3) | 3.046085 (n=2) | Under tested Phase 4 impact events: Bare FBG2 averages 3.310566 bits (n=7), Copper FBG1 averages 1.992344 bits (n=3), and Steel FBG3 averages 3.046085 bits (n=2). |
| Area Under Curve (Absolute) (nm·s) | 0.124820 (n=7) | 0.003136 (n=3) | 0.002024 (n=2) | Under tested Phase 4 impact events: Bare FBG2 averages 0.124820 nm·s (n=7), Copper FBG1 averages 0.003136 nm·s (n=3), and Steel FBG3 averages 0.002024 nm·s (n=2). |

## 4. Key Findings (Observed Signal Differences)
1. **Bare FBG (FBG2)** exhibits the largest mean peak wavelength shift (0.0199 nm across 7 impact trials) under the tested conditions.
2. **Copper FBG (FBG1)** exhibits mean peak shift of 0.0079 nm across 3 impact trials.
3. **Steel FBG (FBG3)** exhibits mean peak wavelength shift of 0.0026 nm across 2 impact trials.

## 5. Generated Artifacts
- `phase5_all_features.csv`
- `phase5_feature_summary.csv`
- `phase5_material_comparison.csv`
- `phase5_engineering_explanation.md`
- `phase5_engineering_comparison.md`
- `phase5_beginner_guide.md`
- 13 comparison plots in `plots/`
