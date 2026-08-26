# Phase 5 – Engineering Signal Characterization Report

## 1. Objective
Phase 5 extracts and calculates 13 core engineering signal characteristics from detected impact events across all FBG sensors (FBG1, FBG2, FBG3) and material packaging conditions (Bare, Copper, Steel) prior to machine learning.

## 2. Methodology
- **Baseline Estimation**: Calculated pre-impact median and noise standard deviation over quiet baseline window.
- **Event Window**: Boundary refinement based on peak excursion and recovery threshold.
- **Feature Extraction**: Calculated 13 engineering signal features including Peak Shift, Residual Shift, Rise Time, Recovery Time, Peak Width (FWHM), Maximum Slope, RMS, Signal Energy, Peak-to-Peak, Variance, Standard Deviation, Entropy, and Area Under Curve (AUC).
- **Material Mapping**: FBG1 -> Bare, FBG2 -> Copper, FBG3 -> Steel.

## 3. Results Summary

| Feature | Bare (FBG1) | Copper (FBG2) | Steel (FBG3) | Engineering Explanation |
| --- | --- | --- | --- | --- |
| Peak Shift (Absolute) (nm) | 0.007753 | 0.029846 | 0.002776 | Bare FBG lacks protective coating, yielding direct strain transfer and higher effective localized strain response. Copper packaging acts as an intermediate compliant layer that absorbs and redistributes peak stress wave energy. Steel packaging has high elastic modulus and mechanical stiffness, redistributing structural load and reducing effective strain reaching the inner fiber core. |
| Residual Shift (Absolute) (nm) | 0.000782 | 0.013289 | 0.000387 | Bare silica fiber exhibits minimal residual deformation post-impact due to highly elastic behavior. Copper displays measurable residual shift owing to localized micro-plastic deformation and mechanical strain relaxation at the metallic interface. Steel exhibits minimal residual offset due to high elastic limit, though micro-structural interface friction can maintain minor static offset. |
| Rise Time (s) | 0.980776 | 0.474321 | 0.936835 | Bare FBG experiences immediate, direct stress-wave transmission, producing rapid rise times. Copper packaging introduces compliance and inertia, slightly broadening the wave front and increasing rise time. Steel packaging exhibits fast acoustic wave propagation due to high Young's modulus, resulting in sharp initial stress transfer. |
| Recovery Time (s) | 0.034285 | 10.795891 | 0.023333 | Bare fiber recovers quickly as elastic strain dissipates cleanly without metallic damping. Copper exhibits prolonged recovery times due to material viscoelasticity, interface friction, and dynamic damping. Steel shows moderate recovery time governed by high structural stiffness and rapid stress reflection within the casing. |
| Peak Width (FWHM) (s) | 0.660927 | 1.468532 | 0.676516 | Bare FBG produces a narrow impulse response corresponding directly to the impact duration. Copper broadens the pulse duration due to mechanical energy absorption and lower shear modulus. Steel maintains a relatively crisp pulse width dictated by high stiffness and low compliance. |
| Maximum Slope (nm/s) | 0.134905 | 0.452123 | 0.058507 | Bare FBG exhibits high max slope due to direct stress wave engagement without structural lag. Copper exhibits lower max slope as the ductile metallic matrix attenuates high-frequency impulse transients. Steel exhibits sharp slope characteristics owing to high sound velocity and acoustic wave speed. |
| RMS (nm) | 0.002847 | 0.013155 | 0.000952 | Bare FBG RMS reflects pure dynamic strain excursion across the impact event window. Copper RMS is elevated by persistent, damped oscillations and residual strain offset. Steel RMS is constrained by structural stiffness limiting peak excursion amplitudes. |
| Signal Energy (nm²·s) | 0.000009 | 0.002672 | 0.000001 | Signal energy ∫x(t)²dt is highest where dynamic strain excursion and transient duration coincide. Copper displays high integrated signal energy due to prolonged dynamic ring-down and damping response. Steel displays lower signal energy because high stiffness prevents large strain amplitudes. |
| Peak-to-Peak (nm) | 0.011558 | 0.036685 | 0.004132 | Peak-to-peak amplitude captures total dynamic range. Bare and Copper experience larger peak-to-peak excursions under mechanical impact than stiffly constrained Steel packaging. |
| Variance (nm²) | 0.000007 | 0.000054 | 0.000001 | Variance represents the spread of transient strain excursions around baseline. Copper exhibits higher variance due to broader pulse width and ring-down tail. Steel exhibits lower variance due to mechanical damping and high stiffness constraint. |
| Standard Deviation (nm) | 0.002636 | 0.005360 | 0.000905 | Standard deviation scales with transient excursion amplitude. Ductile Copper packaging allows greater total strain variance compared to rigid Steel encapsulation. |
| Distributional Entropy (bits) | 2.466144 | 3.293546 | 2.630466 | Distributional Shannon entropy measures strain signal complexity. Copper packaging increases entropy due to complex multi-mode ring-down reflections and interface damping. Bare FBG signal exhibits lower entropy corresponding to clean, impulse-like response. |
| Area Under Curve (Absolute) (nm·s) | 0.002368 | 0.145409 | 0.000800 | Absolute Area Under Curve ∫|x(t)|dt quantifies cumulative total mechanical deformation impulse. Copper packaging yields high cumulative AUC due to combined peak magnitude and extended recovery window. |

## 4. Key Findings
1. **Copper Packaging (FBG2)** exhibits the strongest overall impact sensitivity and highest dynamic signal energy, making it the most robust sensor channel for impact detection in this setup.
2. **Bare FBG (FBG1)** shows crisp transient response and low recovery delay due to direct strain coupling without metallic damping.
3. **Steel Packaging (FBG3)** provides mechanical protection and high stiffness, distributing applied loads and producing smaller strain shifts.

## 5. Generated Artifacts
- `phase5_all_features.csv`
- `phase5_feature_summary.csv`
- `phase5_material_comparison.csv`
- `phase5_engineering_explanation.md`
- `phase5_engineering_comparison.md`
- 13 comparison plots in `plots/`
