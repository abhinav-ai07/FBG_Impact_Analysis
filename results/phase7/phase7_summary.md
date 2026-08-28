# Phase 7 — Statistical Validation Summary

## Material Mapping

| FBG Sensor | Material |
|------------|----------|
| FBG1       | Copper   |
| FBG2       | Bare     |
| FBG3       | Steel    |

## Overview

- **Features Analyzed**: 25
- **Total Statistical Records**: 304
- **Materials Compared**: Bare, Copper, Steel

## Sample Sizes (IMPACT Cases)

| Material | Max Available IMPACT Cases |
|----------|---------------------------|
| Bare | 7 |
| Copper | 3 |
| Steel | 2 |

## Phase 5 Engineering Features (IMPACT Cases Only)

| Feature | Bare Mean ± SD (n) | Copper Mean ± SD (n) | Steel Mean ± SD (n) |
|---------|-------------------|---------------------|---------------------|
| peak_shift_abs | 0.0199397 ± 0.0132716 (n=7) | 0.00788003 ± 0.00198057 (n=3) | 0.00256435 ± 0.000177387 (n=2) |
| residual_shift_abs | 0.00423894 ± 0.00419846 (n=4) | 0.00108967 ± 0.00025986 (n=3) | 0.000469213 ± 6.68949e-05 (n=2) |
| rise_time_seconds | 0.225994 ± 0.308531 (n=7) | 1.23319 ± 1.57501 (n=3) | 0.0580537 ± 0.0821003 (n=2) |
| recovery_time_seconds | 10.7949 ± 9.45194 (n=4) | 0.0399997 ± 5.7735e-07 (n=3) | 0.02 ± 0.0282843 (n=2) |
| peak_width_seconds | 12.4095 ± 10.7429 (n=3) | N/A | 1.73808 (n=1) |
| max_slope_abs | 0.133268 ± 0.117683 (n=7) | 0.136232 ± 0.0212267 (n=3) | 0.0522094 ± 0.00370972 (n=2) |
| rms | 0.0154214 ± 0.0124324 (n=7) | 0.00386737 ± 0.00106121 (n=3) | 0.000973045 ± 0.000112071 (n=2) |
| signal_energy | 0.00250298 ± 0.00279596 (n=7) | 1.21059e-05 ± 1.02978e-05 (n=3) | 2.40832e-06 ± 1.50996e-06 (n=2) |
| peak_to_peak | 0.0156762 ± 0.0127818 (n=7) | 0.0102128 ± 0.00658209 (n=3) | 0.00436739 ± 0.000520556 (n=2) |
| variance | 1.72567e-05 ± 3.96495e-05 (n=7) | 7.4868e-06 ± 6.28126e-06 (n=3) | 6.65401e-07 ± 1.99989e-07 (n=2) |
| std_dev | 0.00263 ± 0.0034732 (n=7) | 0.00252877 ± 0.00127992 (n=3) | 0.000811049 ± 0.00012329 (n=2) |
| entropy | 3.31057 ± 0.707475 (n=7) | 1.99234 ± 1.09747 (n=3) | 3.04608 ± 0.246783 (n=2) |
| auc_abs | 0.12482 ± 0.140755 (n=7) | 0.0031363 ± 0.00329434 (n=3) | 0.00202429 ± 0.001155 (n=2) |

## Phase 5 Baseline Features (All Cases)

| Feature | Bare Mean ± SD (n) | Copper Mean ± SD (n) | Steel Mean ± SD (n) |
|---------|-------------------|---------------------|---------------------|
| baseline_nm | 1.01342e-05 ± 7.1725e-05 (n=7) | -8.86803e-06 ± 0.000185832 (n=7) | -1.02704e-05 ± 0.000108419 (n=7) |
| noise_std_nm | 0.000593874 ± 0.000105839 (n=7) | 0.00193245 ± 0.000150195 (n=7) | 0.000804699 ± 0.000179378 (n=7) |

## Phase 6 Multi-Domain Features (IMPACT Cases Only)

| Feature | Bare Mean ± SD (n) | Copper Mean ± SD (n) | Steel Mean ± SD (n) |
|---------|-------------------|---------------------|---------------------|
| Dominant_Frequency | 1.41444 ± 0.923007 (n=7) | 2.14524 ± 1.24586 (n=3) | 1.98022 ± 0.700113 (n=2) |
| Spectral_Energy | 0.12987 ± 0.180823 (n=7) | 0.0317062 ± 0.00543995 (n=3) | 0.00349947 ± 0.00100841 (n=2) |
| Spectral_Entropy | 4.72714 ± 0.244178 (n=7) | 4.78743 ± 0.0964397 (n=3) | 4.84262 ± 0.0231514 (n=2) |
| Spectral_Centroid | 6.21096 ± 0.662316 (n=7) | 6.56244 ± 0.286412 (n=3) | 6.51516 ± 0.0678584 (n=2) |
| Bandwidth | 6.32796 ± 0.303047 (n=7) | 6.21352 ± 0.260662 (n=3) | 6.25746 ± 0.128178 (n=2) |
| Approximation_Energy | 0.0521884 ± 0.087882 (n=7) | 0.0010372 ± 0.000482914 (n=3) | 3.97503e-05 ± 1.5775e-05 (n=2) |
| Detail_Energy | 0.000785798 ± 0.00146144 (n=7) | 0.000666357 ± 0.000223635 (n=3) | 6.07067e-05 ± 3.0772e-05 (n=2) |
| Wavelet_Energy | 0.0529742 ± 0.0875179 (n=7) | 0.00170355 ± 0.000682402 (n=3) | 0.000100457 ± 1.49971e-05 (n=2) |
| Wavelet_Entropy | 0.646302 ± 0.714719 (n=7) | 1.51264 ± 0.149641 (n=3) | 1.85115 ± 0.183725 (n=2) |
| Detail_Approx_Ratio | 0.217163 ± 0.285877 (n=7) | 0.693613 ± 0.24075 (n=3) | 1.82448 ± 1.49818 (n=2) |

## Coefficient of Variation Summary

CV (%) = (SD / |Mean|) × 100 — indicates relative variability.
NaN indicates undefined CV (mean ≈ 0 or insufficient data).

| Feature | Source | Bare CV (%) | Copper CV (%) | Steel CV (%) |
|---------|--------|-------------|---------------|--------------|
| peak_shift_abs | Phase5_Impact | 66.56 | 25.13 | 6.92 |
| residual_shift_abs | Phase5_Impact | 99.05 | 23.85 | 14.26 |
| rise_time_seconds | Phase5_Impact | 136.52 | 127.72 | 141.42 |
| recovery_time_seconds | Phase5_Impact | 87.56 | 0.00 | 141.42 |
| peak_width_seconds | Phase5_Impact | 86.57 | NaN | NaN |
| max_slope_abs | Phase5_Impact | 88.31 | 15.58 | 7.11 |
| rms | Phase5_Impact | 80.62 | 27.44 | 11.52 |
| signal_energy | Phase5_Impact | 111.71 | 85.06 | 62.70 |
| peak_to_peak | Phase5_Impact | 81.54 | 64.45 | 11.92 |
| variance | Phase5_Impact | 229.76 | 83.90 | 30.06 |
| std_dev | Phase5_Impact | 132.06 | 50.61 | 15.20 |
| entropy | Phase5_Impact | 21.37 | 55.08 | 8.10 |
| auc_abs | Phase5_Impact | 112.77 | 105.04 | 57.06 |
| baseline_nm | Phase5_Baseline | 707.75 | 2095.53 | 1055.65 |
| noise_std_nm | Phase5_Baseline | 17.82 | 7.77 | 22.29 |
| Dominant_Frequency | Phase6_Impact | 65.26 | 58.08 | 35.36 |
| Spectral_Energy | Phase6_Impact | 139.23 | 17.16 | 28.82 |
| Spectral_Entropy | Phase6_Impact | 5.17 | 2.01 | 0.48 |
| Spectral_Centroid | Phase6_Impact | 10.66 | 4.36 | 1.04 |
| Bandwidth | Phase6_Impact | 4.79 | 4.20 | 2.05 |
| Approximation_Energy | Phase6_Impact | 168.39 | 46.56 | 39.69 |
| Detail_Energy | Phase6_Impact | 185.98 | 33.56 | 50.69 |
| Wavelet_Energy | Phase6_Impact | 165.21 | 40.06 | 14.93 |
| Wavelet_Entropy | Phase6_Impact | 110.59 | 9.89 | 9.92 |
| Detail_Approx_Ratio | Phase6_Impact | 131.64 | 34.71 | 82.12 |

## Limitations

- Small sample sizes limit statistical power — Copper has n=3 IMPACT cases, Steel has n=2, Bare has n=7.
- Steel has only n=2 IMPACT cases, so its SD is based on a single degree of freedom and CI is wide.
- Copper has n=0 valid peak_width_seconds observations; statistics are undefined for this feature.
- Some Phase 5 impact features (e.g., recovery_time_seconds, peak_width_seconds) have missing values within IMPACT cases, reducing effective n below the maximum for that material.
- CV is set to NaN where |mean| < 1e-15 to avoid division-by-zero artifacts.
- Phase 6 features for NO IMPACT cases are NaN by design (not computed), so multi-domain statistics are restricted to IMPACT cases only.
- Observed differences between materials reflect signal-level measurement variations — no causal claims about material superiority are made.

## Notes

- All statistics are computed from existing Phase 5 and Phase 6 result CSVs.
- Impact-specific features use only IMPACT cases as classified by Phase 4.
- NO IMPACT cases are excluded from impact-event statistics to avoid mixing.
- Confidence intervals use the t-distribution with n−1 degrees of freedom.
- Differences between materials are reported as observed statistical differences
  in the measured signal features — no causal claims are made.
