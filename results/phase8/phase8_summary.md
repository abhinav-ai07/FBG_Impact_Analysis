# Phase 8 — PGMSIF Core Engineering Indices Summary Report

## Executive Summary
Phase 8 establishes the 4 core indices of the PGMSIF framework evaluated across 21 total sensor events (12 valid IMPACT cases, 9 NO IMPACT cases).

## Material-Level Index Statistics (IMPACT Cases Only)

| Material | Index | n | Mean | Median | SD | CV (%) | Min | Max | 95% CI Lower | 95% CI Upper |
|---|---|---|---|---|---|---|---|---|---|---|
| Bare | DSTI | 7 | 0.5876 | 0.5806 | 0.2203 | 37.50% | 0.2830 | 0.8626 | 0.3838 | 0.7914 |
| Copper | DSTI | 3 | 0.2536 | 0.3317 | 0.1681 | 66.27% | 0.0607 | 0.3684 | -0.1639 | 0.6712 |
| Steel | DSTI | 2 | 0.3335 | 0.3335 | 0.0074 | 2.21% | 0.3283 | 0.3387 | 0.2674 | 0.3997 |
| Bare | PEI | 7 | 0.5220 | 0.3385 | 0.3296 | 63.15% | 0.1957 | 0.9881 | 0.2171 | 0.8269 |
| Copper | PEI | 3 | 0.3961 | 0.3987 | 0.0222 | 5.61% | 0.3727 | 0.4169 | 0.3409 | 0.4513 |
| Steel | PEI | 2 | 0.3449 | 0.3449 | 0.0055 | 1.58% | 0.3411 | 0.3488 | 0.2958 | 0.3940 |
| Bare | SII | 7 | 0.8732 | 0.9304 | 0.1047 | 12.00% | 0.7348 | 0.9723 | 0.7763 | 0.9701 |
| Copper | SII | 3 | 0.5746 | 0.5738 | 0.0218 | 3.80% | 0.5531 | 0.5968 | 0.5203 | 0.6288 |
| Steel | SII | 2 | 0.6683 | 0.6683 | 0.0990 | 14.81% | 0.5983 | 0.7383 | -0.2212 | 1.5578 |
| Bare | DRI | 7 | 0.7510 | 0.8933 | 0.2680 | 35.68% | 0.3280 | 1.0000 | 0.5032 | 0.9988 |
| Copper | DRI | 3 | 0.7942 | 0.8864 | 0.2616 | 32.94% | 0.4990 | 0.9972 | 0.1444 | 1.4440 |
| Steel | DRI | 2 | 0.9765 | 0.9765 | 0.0059 | 0.61% | 0.9723 | 0.9807 | 0.9231 | 1.0300 |

## Traditional vs PGMSIF Comparison

| Material | n | Peak Shift Mean (nm) | DSTI Mean | PEI Mean | SII Mean | DRI Mean | Engineering Insight |
|---|---|---|---|---|---|---|---|
| Bare | 7 | 0.01994 | 0.5876 | 0.5220 | 0.8732 | 0.7510 | Bare shows high sensitivity (DSTI 0.588) with wider recovery variation across severe impacts (DRI 0.751) |
| Copper | 3 | 0.00788 | 0.2536 | 0.3961 | 0.5746 | 0.7942 | Balanced strain transfer (DSTI 0.254) with moderate recovery dynamics (DRI 0.794) |
| Steel | 2 | 0.00256 | 0.3335 | 0.3449 | 0.6683 | 0.9765 | Steel shows the highest DRI (0.977) in this limited dataset, reflecting fast rise and recovery times despite lower peak amplitude |

## Key Engineering Findings & Cautious Interpretation
1. **Dynamic Strain Transfer (DSTI)**: Within this experimental dataset, Bare fiber exhibits the highest mean strain transfer (0.5876) due to direct mechanical coupling, compared to Steel (0.3335) and Copper (0.2536).
2. **Packaging Efficiency (PEI)**: Reflects retained dynamic sensitivity, recovery speed, and linear optical SNR across packaging configurations (Bare: 0.5220, Copper: 0.3961, Steel: 0.3449).
3. **Signal Integrity (SII)**: Bare fiber maintains 0.8732 mean integrity due to a lower optical baseline noise floor; Steel (0.6683) and Copper (0.5746) demonstrate consistent signal preservation without severe residual baseline distortion.
4. **Dynamic Response (DRI)**: Steel shows the highest mean DRI (0.9765) in this limited dataset, reflecting fast rise and short recovery times across its 2 impact events, whereas Bare fiber shows a mean DRI of 0.7510 with greater variation across its 7 events.

## Methodological Limitations
- **Sample Sizes**: Experimental sample sizes are limited (Bare $n=7$, Copper $n=3$, Steel $n=2$). Observations are consistent with physical sensor packaging mechanics but should not be generalized as universal material superiority claims.
- **Confidence Intervals**: Student-$t$ intervals reflect small degrees of freedom ($df = 1$ for Steel) and should be interpreted alongside physical context.

## Readiness for Phase 9
- Phase 8 successfully delivers the complete, audited, and verified 4 core indices.
