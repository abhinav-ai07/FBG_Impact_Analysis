# Phase 8 — Novel Engineering Indices

## Objective

Phase 8 develops deterministic, interpretable engineering indices to transform previously calculated time-domain, frequency-domain, wavelet-domain, and statistical features into unified, physically meaningful characterization measures for Fiber Bragg Grating (FBG) impact response.

## Input Sources

Phase 8 consumes existing result artifacts without rerunning prior phase pipelines:

- `results/phase5/phase5_all_features.csv` (13 time-domain & baseline signal features)
- `results/phase6/phase6_multidomain_features.csv` (10 spectral & wavelet multi-domain features)
- `results/phase7/phase7_statistical_summary.csv` & `phase7_material_comparison.csv` (material-level variability & CI metrics)

## Important Constraint & Methodological Safeguards

> [!IMPORTANT]

> 1. **No Machine Learning**: No PCA, clustering, neural networks, or trained classifiers were used.

> 2. **Deterministic & Interpretable**: All indices use documented physical formulas and normalized composites.

> 3. **No Target Optimization**: No label fitting or parameter optimization was performed.

> 4. **No Pipeline Reruns**: Previous Phase 5, Phase 6, and Phase 7 outputs were consumed strictly as read-only inputs.

> 5. **Safe Missing & NO IMPACT Handling**: NO IMPACT cases are safely set to `NOT_APPLICABLE` (NaN) and excluded from impact response distributions.


## Index Definitions & Mathematical Formulations

### 1. Dynamic Strain Transfer Index (DSTI)

- **Engineering Interpretation**: Measures how efficiently impact-induced strain is transferred into sensor signal response magnitude, rate of deformation, and onset speed.

- **Mathematical Formula**:

  $$\text{DSTI} = \frac{1}{K} \sum_{k \in \text{valid}} C_k$$

  where:

  - $C_1 = \text{norm}(\text{peak\_shift\_abs})$ (Peak wavelength shift component)

  - $C_2 = \text{norm}(\text{max\_slope\_abs})$ (Dynamic slope / rate of change component)

  - $C_3 = 1.0 - \text{norm}(\text{rise\_time\_seconds})$ (Response speed component; faster onset gives higher score)

  - $K = \text{DSTI\_valid\_components}$ (count of non-NaN components)

- **Valid Range**: $[0.0, 1.0]$

- **Interpretation**: Higher DSTI indicates greater strain transfer efficiency and faster dynamic response onset.


### 2. Impact Persistence Index (IPI)

- **Engineering Interpretation**: Measures post-impact strain retention (permanent/residual shift) relative to peak response, combined with signal recovery duration.

- **Mathematical Formula**:

  $$\text{IPI} = \frac{1}{K} \sum_{k \in \text{valid}} C_k$$

  where:

  - $C_1 = \min\left(1.0, \frac{\text{residual\_shift\_abs}}{\text{peak\_shift\_abs}}\right)$ (Residual retention ratio)

  - $C_2 = \text{norm}(\text{recovery\_time\_seconds})$ (Recovery duration component)

- **Valid Range**: $[0.0, 1.0]$

- **Interpretation**: Higher IPI reflects greater permanent residual deformation and/or prolonged recovery time relative to peak magnitude.


### 3. Signal Energy Response Index (SERI)

- **Engineering Interpretation**: Measures total signal energy and impact response magnitude across time-domain energy/amplitude metrics.

- **Mathematical Formula**:

  $$\text{SERI} = \frac{1}{4} \left( \text{norm}(\text{signal\_energy}) + \text{norm}(\text{rms}) + \text{norm}(\text{peak\_to\_peak}) + \text{norm}(\text{auc\_abs}) \right)$$

- **Valid Range**: $[0.0, 1.0]$

- **Interpretation**: Higher SERI indicates greater cumulative energy release and total strain displacement induced by the impact event.


### 4. Response Stability Index (RSI)

- **Engineering Interpretation**: Evaluates material-level repeatability and stability based on Phase 7 statistical Coefficient of Variation (CV%).

- **Mathematical Formula**:

  $$\text{RSI}_{\text{Material}} = \frac{100.0}{100.0 + \overline{\text{CV}}_{\%}}$$

  where $\overline{\text{CV}}_{\%}$ is the mean CV% of key features (`peak_shift_abs`, `max_slope_abs`, `signal_energy`, `rms`, `Dominant_Frequency`, `Spectral_Energy`, `Wavelet_Energy`).

- **Valid Range**: $(0.0, 1.0]$

- **Interpretation**: Higher RSI represents superior repeatability (lower relative variability across trials).


### 5. Multi-Domain Impact Signature Index (MDISI)

- **Engineering Interpretation**: Integrates time-domain, frequency-domain, and wavelet-domain response subscores into a unified multi-domain impact signature score.

- **Mathematical Formula**:

  $$\text{MDISI} = \frac{S_{\text{time}} + S_{\text{freq}} + S_{\text{wavelet}}}{3}$$

  where:

  - $S_{\text{time}} = \text{mean}(\text{norm}(\text{peak\_shift\_abs}, \text{max\_slope\_abs}, \text{signal\_energy}, \text{rms}))$

  - $S_{\text{freq}} = \text{mean}(\text{norm}(\text{Spectral\_Energy}, \text{Spectral\_Centroid}, \text{Dominant\_Frequency}, \text{Bandwidth}))$

  - $S_{\text{wavelet}} = \text{mean}(\text{norm}(\text{Wavelet\_Energy}, \text{Approximation\_Energy}, \text{Detail\_Energy}, \text{Detail\_Approx\_Ratio}))$

- **Valid Range**: $[0.0, 1.0]$

- **Interpretation**: Higher MDISI signifies stronger multi-domain impact energy, frequency concentration, and transient detail content.


## Material-Level Index Summary (IMPACT Cases)


| Material | Index | n | Mean | Median | SD | Min | Max | CV (%) | 95% CI Lower | 95% CI Upper | Missing/Invalid |
|----------|-------|---|------|--------|----|-----|-----|--------|--------------|--------------|-----------------|
| Bare | DSTI | 7 | 0.5814 | 0.6308 | 0.1634 | 0.3194 | 0.8094 | 28.10% | 0.4303 | 0.7325 | 0 |
| Copper | DSTI | 3 | 0.3601 | 0.4536 | 0.1685 | 0.1655 | 0.4612 | 46.80% | -0.0586 | 0.7787 | 0 |
| Steel | DSTI | 2 | 0.3516 | 0.3516 | 0.0035 | 0.3491 | 0.3541 | 1.00% | 0.3201 | 0.3831 | 0 |
| Bare | IPI | 4 | 0.4497 | 0.3998 | 0.2458 | 0.2392 | 0.7600 | 54.66% | 0.0586 | 0.8408 | 3 |
| Copper | IPI | 3 | 0.0747 | 0.0811 | 0.0292 | 0.0429 | 0.1002 | 39.09% | 0.0022 | 0.1473 | 0 |
| Steel | IPI | 2 | 0.0917 | 0.0917 | 0.0060 | 0.0875 | 0.0960 | 6.58% | 0.0375 | 0.1460 | 0 |
| Bare | SERI | 7 | 0.3568 | 0.4013 | 0.2618 | 0.0021 | 0.7371 | 73.37% | 0.1147 | 0.5990 | 0 |
| Copper | SERI | 3 | 0.0763 | 0.0924 | 0.0426 | 0.0280 | 0.1086 | 55.85% | -0.0296 | 0.1822 | 0 |
| Steel | SERI | 2 | 0.0137 | 0.0137 | 0.0020 | 0.0123 | 0.0152 | 14.92% | -0.0047 | 0.0321 | 0 |
| Bare | RSI | 7 | 0.4940 | 0.4940 | 0.0000 | 0.4940 | 0.4940 | 0.00% | 0.4940 | 0.4940 | 0 |
| Copper | RSI | 3 | 0.7228 | 0.7228 | 0.0000 | 0.7228 | 0.7228 | 0.00% | 0.7228 | 0.7228 | 0 |
| Steel | RSI | 2 | 0.8071 | 0.8071 | 0.0000 | 0.8071 | 0.8071 | 0.00% | 0.8071 | 0.8071 | 0 |
| Bare | MDISI | 7 | 0.3339 | 0.3923 | 0.1684 | 0.1020 | 0.5832 | 50.45% | 0.1781 | 0.4896 | 0 |
| Copper | MDISI | 3 | 0.2348 | 0.2408 | 0.0115 | 0.2215 | 0.2421 | 4.90% | 0.2062 | 0.2634 | 0 |
| Steel | MDISI | 2 | 0.2048 | 0.2048 | 0.0713 | 0.1544 | 0.2553 | 34.82% | -0.4360 | 0.8457 | 0 |


## Key Engineering Findings & Observations

1. **Dynamic Strain Transfer (DSTI)**: Bare FBG2 exhibits the highest mean DSTI (0.581), indicating strong strain coupling and sharp dynamic response onset, compared to Copper FBG1 (0.360) and Steel FBG3 (0.352).

2. **Signal Energy Response (SERI)**: Bare FBG2 demonstrates significantly higher overall signal energy response (SERI = 0.357) under impact compared to Copper (0.076) and Steel (0.014).

3. **Response Stability (RSI)**: Steel FBG3 shows the highest repeatability (RSI = 0.807, mean CV = 23.91%), followed by Copper FBG1 (RSI = 0.723, mean CV = 38.36%), whereas Bare FBG2 exhibits lower repeatability (RSI = 0.494, mean CV = 102.41%).

4. **Multi-Domain Impact Signature (MDISI)**: Bare FBG2 produces the highest composite multi-domain signature score (MDISI = 0.334), reflecting high multi-spectral and wavelet energy transfer.


## Limitations

> [!WARNING]

> 1. **Sample Size Constraints**: The dataset contains limited IMPACT events — Bare (n=7), Copper (n=3), and Steel (n=2). High statistical certainty should not be claimed.

> 2. **Steel Sample Size**: Steel has only n=2 IMPACT cases; standard deviations rely on 1 degree of freedom and confidence intervals are wide.

> 3. **Exploratory Nature**: All Phase 8 indices are exploratory engineering constructs designed for relative comparison, requiring validation on larger experimental cohorts.

> 4. **No Causal Superiority Claims**: Higher index values reflect measured sensor signal characteristics, not absolute physical material superiority.


## Reproducibility

To reproduce Phase 8 results and generate all artifacts, execute from the repository root:

```bash
python phase8.py
```
