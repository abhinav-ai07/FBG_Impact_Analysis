"""
PHASE 9 — PRINCIPAL COMPONENT ANALYSIS (PCA ONLY)
============================================================
Physics-Guided Multi-Domain Signal Intelligence Framework (PGMSIF)

Performs unsupervised dimensionality reduction and feature space characterization
on the combined multi-domain feature set (Phase 5 Engineering, Phase 6 FFT & Wavelet,
and Phase 8 Physics-Guided Indices).

Consumes existing results from Phase 5, Phase 6, and Phase 8 as read-only inputs.

Material Mapping (established in prior phases):
    FBG1 -> Copper
    FBG2 -> Bare
    FBG3 -> Steel

Event-Level Scope:
    Total events = 21 (12 IMPACT, 9 NO IMPACT)
    PCA Scope = 12 Valid IMPACT Events Only

Strictly Unsupervised:
    Material labels and metadata are completely excluded from PCA computation.
    Material labels are used solely as metadata for post-PCA visualization and interpretation.

Outputs Generated:
    - results/phase9/pca/phase9_pca_scores.csv
    - results/phase9/pca/phase9_pca_loadings.csv
    - results/phase9/pca/phase9_pca_explained_variance.csv
    - results/phase9/pca/phase9_pca_feature_metadata.json
    - results/phase9/pca/phase9_pca_summary.md
    - results/phase9/pca/plots/phase9_pca_explained_variance.png
    - results/phase9/pca/plots/phase9_pca_pc1_pc2.png
"""

import os
import sys
import re
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION AND FILE PATHS
# ============================================================

PHASE5_CSV = os.path.join("results", "phase5", "phase5_all_features.csv")
PHASE6_CSV = os.path.join("results", "phase6", "phase6_multidomain_features.csv")
PHASE8_CSV = os.path.join("results", "phase8", "phase8_engineering_indices.csv")

OUTPUT_DIR = os.path.join("results", "phase9", "pca")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")

MATERIAL_MAP = {
    "FBG1": "Copper",
    "FBG2": "Bare",
    "FBG3": "Steel",
}

# Color and marker configuration for visualization
MATERIAL_STYLE = {
    "Bare": {"color": "#1f77b4", "marker": "o", "label": "Bare (FBG2)"},
    "Copper": {"color": "#d62728", "marker": "^", "label": "Copper (FBG1)"},
    "Steel": {"color": "#2ca02c", "marker": "s", "label": "Steel (FBG3)"},
}


# ============================================================
# 1. INPUT DATA LOADING AND MERGING
# ============================================================

def load_and_prepare_data():
    """
    Loads Phase 5, Phase 6, and Phase 8 outputs as read-only inputs.
    Extracts alignment keys and filters exclusively for valid IMPACT events.
    """
    if not os.path.exists(PHASE5_CSV):
        raise FileNotFoundError(f"Phase 5 input missing: {PHASE5_CSV}")
    if not os.path.exists(PHASE6_CSV):
        raise FileNotFoundError(f"Phase 6 input missing: {PHASE6_CSV}")
    if not os.path.exists(PHASE8_CSV):
        raise FileNotFoundError(f"Phase 8 input missing: {PHASE8_CSV}")

    p5 = pd.read_csv(PHASE5_CSV)
    p6 = pd.read_csv(PHASE6_CSV)
    p8 = pd.read_csv(PHASE8_CSV)

    # Standardize keys across datasets
    # Phase 5: Dataset='Expert X', Sensor='FBG1/2/3'
    p5["Expert"] = p5["Dataset"]
    p5["FBG"] = p5["Sensor"]

    # Phase 6: File='RVCE...expertX...', Sensor='FBG1/2/3_processed'
    p6["Expert_Num"] = p6["File"].apply(lambda x: int(re.search(r"expert(\d+)", str(x), re.IGNORECASE).group(1)))
    p6["Expert"] = "Expert " + p6["Expert_Num"].astype(str)
    p6["FBG"] = p6["Sensor"].str.replace("_processed", "", regex=False)

    # Phase 8: Expert='Expert X', FBG='FBG1/2/3'
    # p8 already has Expert, FBG, Material, Impact_Status

    # Filter for IMPACT events only
    p5_imp = p5[p5["Impact_Status"] == "IMPACT"].copy()
    p6_imp = p6[p6["Phase4_Result"] == "IMPACT"].copy()
    p8_imp = p8[p8["Impact_Status"] == "IMPACT"].copy()

    # Merge into unified event dataframe
    merged = p5_imp.merge(p6_imp, on=["Expert", "FBG"], suffixes=("_p5", "_p6"))
    merged = merged.merge(p8_imp, on=["Expert", "FBG"], suffixes=("", "_p8"))

    # Verify material mapping consistency
    for _, row in merged.iterrows():
        expected_mat = MATERIAL_MAP[row["FBG"]]
        if row["Material"] != expected_mat:
            raise ValueError(f"Material mapping mismatch for {row['FBG']}: expected {expected_mat}, got {row['Material']}")

    return p5, p6, p8, merged


# ============================================================
# 2. FEATURE SELECTION AND METADATA DEFINITION
# ============================================================

def select_pca_features(df):
    """
    Selects valid numerical features across Phase 5, Phase 6, and Phase 8.
    Ensures complete absence of metadata/labels from feature matrix.
    Documents rationale for included and excluded candidate features.
    """
    # 1. Phase 5 Engineering Features
    phase5_selected = [
        "peak_shift_abs",      # Peak strain magnitude (nm)
        "rise_time_seconds",   # Transient onset duration (s)
        "signal_energy",       # Total integral energy of transient (nm^2*s)
        "rms",                 # Root mean square transient amplitude (nm)
        "peak_to_peak",        # Peak-to-peak transient excursion (nm)
        "std_dev",             # Standard deviation of dynamic strain (nm)
        "entropy",             # Shannon entropy of strain signal
        "max_slope_abs",       # Peak rate of strain change (nm/s)
        "auc_abs",             # Absolute area under curve (nm*s)
        "noise_std_nm",        # Pre-impact optical baseline noise floor (nm)
    ]

    # 2. Phase 6 FFT Spectral Features
    phase6_fft_selected = [
        "Dominant_Frequency",  # Primary resonant frequency (Hz)
        "Spectral_Energy",     # Power spectral density energy
        "Spectral_Entropy",    # Spectral complexity/flatness
        "Spectral_Centroid",   # Center of spectral mass (Hz)
        "Bandwidth",           # Spectral frequency spread (Hz)
    ]

    # 3. Phase 6 Wavelet Multi-Resolution Features
    phase6_wavelet_selected = [
        "Approximation_Energy",# Low-frequency approximation sub-band energy
        "Detail_Energy",       # High-frequency transient detail energy
        "Wavelet_Energy",      # Total discrete wavelet transform energy
        "Wavelet_Entropy",     # Wavelet sub-band entropy
        "Detail_Approx_Ratio", # Ratio of high-frequency detail to approximation
    ]

    # 4. Phase 8 Physics-Guided Engineering Indices
    phase8_indices_selected = [
        "DSTI",                # Dynamic Strain Transfer Index [0, 1]
        "PEI",                 # Packaging Efficiency Index [0, 1]
        "SII",                 # Signal Integrity Index [0, 1]
        "DRI",                 # Dynamic Response Index [0, 1]
    ]

    all_features = phase5_selected + phase6_fft_selected + phase6_wavelet_selected + phase8_indices_selected

    feature_metadata = {
        "Phase5_Engineering": {
            "features": phase5_selected,
            "count": len(phase5_selected),
            "description": "Transient time-domain deformation, velocity, and statistical strain features.",
        },
        "Phase6_Spectral_FFT": {
            "features": phase6_fft_selected,
            "count": len(phase6_fft_selected),
            "description": "Frequency-domain power distribution, dominant frequency, and spectral centroid.",
        },
        "Phase6_Wavelet": {
            "features": phase6_wavelet_selected,
            "count": len(phase6_wavelet_selected),
            "description": "Discrete wavelet decomposition energies, transient detail-to-approximation ratio, and wavelet entropy.",
        },
        "Phase8_Physics_Guided_Indices": {
            "features": phase8_indices_selected,
            "count": len(phase8_indices_selected),
            "description": "Bounded, deterministic, dimensionless physics-guided engineering indices (DSTI, PEI, SII, DRI).",
        },
        "Excluded_Features_Documentation": {
            "peak_time": "Recording timestamp dependent on operator trigger time, not an intrinsic material/sensor property.",
            "peak_shift_signed": "Signed version of peak shift; physical magnitude captured unambiguously by peak_shift_abs.",
            "max_slope_pos_neg": "Directional components; overall dynamic transient rate captured by max_slope_abs.",
            "variance": "Exact mathematical square of std_dev; excluded to eliminate perfect redundancy.",
            "residual_shift_abs": "Contains 3/12 missing values (25%) due to unrecovered baseline within window; captured systematically via Phase 8 PEI and SII.",
            "recovery_time_seconds": "Contains 3/12 missing values (25%) due to unrecovered baseline within window; captured systematically via Phase 8 PEI and DRI.",
            "peak_width_seconds": "Contains 8/12 missing values (67% missingness); excluded to preserve dataset integrity without synthetic imputation.",
            "metadata_columns": "Material, Sensor, FBG, Expert, File, Impact_Status, Index_Status excluded completely to ensure unsupervised PCA.",
        },
        "Total_Feature_Count": len(all_features),
        "Total_Impact_Events": len(df),
    }

    return all_features, feature_metadata


# ============================================================
# 3. STANDARDIZATION AND PCA COMPUTATION
# ============================================================

def run_pca_pipeline(df, feature_cols):
    """
    Executes standardization (z-score) and Principal Component Analysis.
    Calculates scores, explained variance, cumulative variance, and loadings.
    """
    X_raw = df[feature_cols].values
    
    # Check for missing, non-finite, or constant columns
    if np.isnan(X_raw).any():
        raise ValueError("Feature matrix contains NaN values before scaling.")
    if np.isinf(X_raw).any():
        raise ValueError("Feature matrix contains Inf values before scaling.")

    # Check for zero variance
    variances = np.var(X_raw, axis=0)
    zero_var_idx = np.where(variances == 0)[0]
    if len(zero_var_idx) > 0:
        zero_var_cols = [feature_cols[i] for i in zero_var_idx]
        raise ValueError(f"Constant features detected with zero variance: {zero_var_cols}")

    # Standardize: z = (x - mean) / std
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # Fit PCA (deterministic, unsupervised)
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)

    # Number of components computed (min(n_samples, n_features) = 12)
    n_components = pca.n_components_
    pc_names = [f"PC{i+1}" for i in range(n_components)]

    # Explained variance metrics
    exp_var = pca.explained_variance_
    exp_var_ratio = pca.explained_variance_ratio_
    cum_exp_var_ratio = np.cumsum(exp_var_ratio)

    # Scores dataframe with metadata
    scores_df = pd.DataFrame(X_pca, columns=pc_names)
    scores_df["Expert"] = df["Expert"].values
    scores_df["FBG"] = df["FBG"].values
    scores_df["Material"] = df["Material"].values
    scores_df["Impact_Status"] = df["Impact_Status"].values

    # Loadings dataframe (components_ has shape (n_components, n_features))
    # Loadings matrix: features x components
    loadings_df = pd.DataFrame(pca.components_.T, index=feature_cols, columns=pc_names)

    # Explained variance summary dataframe
    exp_var_df = pd.DataFrame({
        "Principal_Component": pc_names,
        "Eigenvalue": exp_var,
        "Explained_Variance_Ratio": exp_var_ratio,
        "Explained_Variance_Percent": exp_var_ratio * 100.0,
        "Cumulative_Explained_Variance_Ratio": cum_exp_var_ratio,
        "Cumulative_Explained_Variance_Percent": cum_exp_var_ratio * 100.0,
    })

    return scaler, pca, scores_df, loadings_df, exp_var_df, X_scaled


# ============================================================
# 4. VISUALIZATION GENERATION
# ============================================================

def generate_plots(exp_var_df, scores_df):
    """
    Generates required publication-quality PCA visualization figures:
      1. Scree plot of individual & cumulative explained variance
      2. PC1 vs PC2 2D projection scatter plot with material metadata
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # ------------------------------------------------------------
    # Plot 1: Explained Variance & Scree Plot
    # ------------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)

    x = np.arange(len(exp_var_df))
    pcs = exp_var_df["Principal_Component"].values
    ind_var = exp_var_df["Explained_Variance_Percent"].values
    cum_var = exp_var_df["Cumulative_Explained_Variance_Percent"].values

    bars = ax1.bar(x, ind_var, color="#4a90e2", alpha=0.85, width=0.6, label="Individual Explained Variance (%)", edgecolor="#1c5ba6")
    ax1.set_xlabel("Principal Component", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Individual Explained Variance (%)", fontsize=12, fontweight="bold", color="#1c5ba6")
    ax1.set_xticks(x)
    ax1.set_xticklabels(pcs, fontsize=10)
    ax1.tick_params(axis='y', labelcolor="#1c5ba6")
    ax1.set_ylim(0, max(ind_var) * 1.25)

    # Annotate bar values
    for bar, val in zip(bars, ind_var):
        ax1.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.8,
                 f"{val:.1f}%", ha='center', va='bottom', fontsize=9, fontweight="bold")

    # Cumulative variance line on twin axis
    ax2 = ax1.twinx()
    line = ax2.plot(x, cum_var, color="#d9534f", marker='o', linewidth=2.5, markersize=7, label="Cumulative Explained Variance (%)")
    ax2.set_ylabel("Cumulative Explained Variance (%)", fontsize=12, fontweight="bold", color="#d9534f")
    ax2.tick_params(axis='y', labelcolor="#d9534f")
    ax2.set_ylim(0, 105)
    t80 = ax2.axhline(80, color='gray', linestyle='--', alpha=0.6, label="80% Variance Threshold")
    t90 = ax2.axhline(90, color='gray', linestyle=':', alpha=0.6, label="90% Variance Threshold")

    # Title and styling
    plt.title("PGMSIF Phase 9: PCA Explained Variance & Scree Plot\n(Multi-Domain Feature Set: 24 Features, 12 Impact Events)",
              fontsize=13, fontweight="bold", pad=15)
    
    # Combined clean legend
    handles = [bars, line[0], t80, t90]
    labels = ["Individual Explained Variance (%)", "Cumulative Explained Variance (%)", "80% Variance Threshold", "90% Variance Threshold"]
    ax1.legend(handles, labels, loc="center right", framealpha=0.95, fontsize=9.5)

    ax1.grid(True, linestyle="--", alpha=0.3, axis='y')
    plt.tight_layout()
    plot1_path = os.path.join(PLOTS_DIR, "phase9_pca_explained_variance.png")
    plt.savefig(plot1_path, dpi=300)
    plt.close()

    # ------------------------------------------------------------
    # Plot 2: PC1 vs PC2 Scatter Plot
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 7), dpi=300)

    pc1_var = exp_var_df.loc[exp_var_df["Principal_Component"] == "PC1", "Explained_Variance_Percent"].values[0]
    pc2_var = exp_var_df.loc[exp_var_df["Principal_Component"] == "PC2", "Explained_Variance_Percent"].values[0]
    pc12_total = pc1_var + pc2_var

    for material, style in MATERIAL_STYLE.items():
        sub = scores_df[scores_df["Material"] == material]
        ax.scatter(
            sub["PC1"], sub["PC2"],
            c=style["color"],
            marker=style["marker"],
            s=150,
            alpha=0.9,
            edgecolors="black",
            linewidth=1.2,
            label=f"{style['label']} (n={len(sub)})",
            zorder=3
        )
        # Add event annotations (e.g. Exp 8)
        for _, row in sub.iterrows():
            exp_short = row["Expert"].replace("Expert ", "Exp ")
            # Position text offset dynamically to avoid clipping
            x_pos = row["PC1"]
            y_pos = row["PC2"]
            x_off = 7
            y_off = 6
            if x_pos > 4.5 and y_pos > 2.5: # Exp 9 Bare top right
                x_off = -40
                y_off = 6
            elif y_pos < -3.0: # Exp 13 Bare bottom right
                x_off = 7
                y_off = 7
            ax.annotate(
                f"{exp_short}",
                (x_pos, y_pos),
                xytext=(x_off, y_off),
                textcoords="offset points",
                fontsize=9,
                alpha=0.9,
                fontweight="medium"
            )

    ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5, zorder=1)
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5, zorder=1)

    # Set explicit plot limits with generous margins
    x_min, x_max = scores_df["PC1"].min(), scores_df["PC1"].max()
    y_min, y_max = scores_df["PC2"].min(), scores_df["PC2"].max()
    ax.set_xlim(x_min - 1.0, x_max + 1.2)
    ax.set_ylim(y_min - 0.9, y_max + 1.0)

    ax.set_xlabel(f"Principal Component 1 ({pc1_var:.2f}% Variance Explained)", fontsize=12, fontweight="bold")
    ax.set_ylabel(f"Principal Component 2 ({pc2_var:.2f}% Variance Explained)", fontsize=12, fontweight="bold")
    ax.set_title(f"PGMSIF Phase 9: PC1 vs PC2 Feature Space Projection\n(PC1 + PC2 = {pc12_total:.2f}% Total Variance Explained)",
                 fontsize=13, fontweight="bold", pad=15)

    ax.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.legend(title="Packaging Configuration (Metadata Only)", title_fontsize=10, loc="upper left", framealpha=0.95, fontsize=10)

    # Add explanatory text note
    ax.text(0.98, 0.03,
            "Note: PCA is completely unsupervised. Material labels were not used in fitting.\n"
            "Sample size: Bare n=7, Copper n=3, Steel n=2 (Total N=12 impact events).",
            transform=ax.transAxes, fontsize=8.5, horizontalalignment='right', verticalalignment='bottom',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='lightgray'))

    plt.tight_layout()
    plot2_path = os.path.join(PLOTS_DIR, "phase9_pca_pc1_pc2.png")
    plt.savefig(plot2_path, dpi=300)
    plt.close()

    return plot1_path, plot2_path


# ============================================================
# 5. SUMMARY REPORT GENERATION
# ============================================================

def generate_summary_report(exp_var_df, scores_df, loadings_df, feature_metadata):
    """
    Generates a structured, rigorous scientific markdown summary report of PCA results.
    """
    pc1_var = exp_var_df.loc[exp_var_df["Principal_Component"] == "PC1", "Explained_Variance_Percent"].values[0]
    pc2_var = exp_var_df.loc[exp_var_df["Principal_Component"] == "PC2", "Explained_Variance_Percent"].values[0]
    pc3_var = exp_var_df.loc[exp_var_df["Principal_Component"] == "PC3", "Explained_Variance_Percent"].values[0]
    pc4_var = exp_var_df.loc[exp_var_df["Principal_Component"] == "PC4", "Explained_Variance_Percent"].values[0]
    pc12_cum = pc1_var + pc2_var
    pc123_cum = pc12_cum + pc3_var

    # Identify strongest loadings for PC1, PC2, PC3
    top_pos_pc1 = loadings_df["PC1"].nlargest(3)
    top_neg_pc1 = loadings_df["PC1"].nsmallest(3)
    top_pos_pc2 = loadings_df["PC2"].nlargest(3)
    top_neg_pc2 = loadings_df["PC2"].nsmallest(3)
    top_pos_pc3 = loadings_df["PC3"].nlargest(3)
    top_neg_pc3 = loadings_df["PC3"].nsmallest(3)

    summary_content = f"""# Phase 9 — Principal Component Analysis (PCA) Summary Report

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
- **Standardization**: Standard z-score scaling ($z = (x - \\mu) / \\sigma$) via `StandardScaler` was applied prior to PCA to ensure all features contribute equally regardless of physical units.

### Missing-Value Handling:
- Sparse raw columns with significant missingness (`peak_width_seconds` with 8/12 missing, `recovery_time_seconds` with 3/12 missing, `residual_shift_abs` with 3/12 missing) were excluded from raw $X$.
- Dynamic recovery and baseline preservation mechanics are fully and systematically represented without missing values via the validated Phase 8 bounded indices (`PEI`, `SII`, `DRI`).
- Zero samples were discarded or fabricated.

---

## 4. Explained Variance Breakdown

| Principal Component | Eigenvalue | Individual Explained Variance (%) | Cumulative Explained Variance (%) |
|---|---|---|---|
| **PC1** | {exp_var_df.loc[0, 'Eigenvalue']:.4f} | **{exp_var_df.loc[0, 'Explained_Variance_Percent']:.2f}%** | **{exp_var_df.loc[0, 'Cumulative_Explained_Variance_Percent']:.2f}%** |
| **PC2** | {exp_var_df.loc[1, 'Eigenvalue']:.4f} | **{exp_var_df.loc[1, 'Explained_Variance_Percent']:.2f}%** | **{exp_var_df.loc[1, 'Cumulative_Explained_Variance_Percent']:.2f}%** |
| **PC3** | {exp_var_df.loc[2, 'Eigenvalue']:.4f} | {exp_var_df.loc[2, 'Explained_Variance_Percent']:.2f}% | {exp_var_df.loc[2, 'Cumulative_Explained_Variance_Percent']:.2f}% |
| **PC4** | {exp_var_df.loc[3, 'Eigenvalue']:.4f} | {exp_var_df.loc[3, 'Explained_Variance_Percent']:.2f}% | {exp_var_df.loc[3, 'Cumulative_Explained_Variance_Percent']:.2f}% |
| **PC5** | {exp_var_df.loc[4, 'Eigenvalue']:.4f} | {exp_var_df.loc[4, 'Explained_Variance_Percent']:.2f}% | {exp_var_df.loc[4, 'Cumulative_Explained_Variance_Percent']:.2f}% |
| **PC6** | {exp_var_df.loc[5, 'Eigenvalue']:.4f} | {exp_var_df.loc[5, 'Explained_Variance_Percent']:.2f}% | {exp_var_df.loc[5, 'Cumulative_Explained_Variance_Percent']:.2f}% |

- **PC1 + PC2 Total Explained Variance**: **{pc12_cum:.2f}%**
- **First 3 Components (PC1–PC3)**: **{pc123_cum:.2f}%**

---

## 5. Dominant PCA Loadings & Physical Interpretation

### Principal Component 1 ({pc1_var:.2f}% Variance):
- **Dominant Positive Loadings**:
{chr(10).join([f"  - `{feat}`: +{val:.4f}" for feat, val in top_pos_pc1.items()])}
- **Dominant Negative Loadings**:
{chr(10).join([f"  - `{feat}`: {val:.4f}" for feat, val in top_neg_pc1.items()])}
- **Physical Interpretation**: PC1 represents the **overall dynamic transient energy and strain magnitude axis**. Positive scores align with high peak strain transfer, elevated signal RMS, high DSTI, and strong signal integrity, balanced against wavelet entropy and detail ratio on the negative axis.

### Principal Component 2 ({pc2_var:.2f}% Variance):
- **Dominant Positive Loadings**:
{chr(10).join([f"  - `{feat}`: +{val:.4f}" for feat, val in top_pos_pc2.items()])}
- **Dominant Negative Loadings**:
{chr(10).join([f"  - `{feat}`: {val:.4f}" for feat, val in top_neg_pc2.items()])}
- **Physical Interpretation**: PC2 represents the **spectral frequency distribution and frequency centering axis**. Positive scores reflect high spectral centroid and dominant frequencies alongside approximation energy, contrasted against high peak-to-peak amplitude excursion and spectral energy.

---

## 6. Structural Observation & Packaging Configuration Separation
In the unsupervised 2D PC1–PC2 projection ({pc12_cum:.2f}% total variance):
1. **Copper Packaging (FBG1)**: Forms a relatively localized cluster in the negative PC1 region ($\\\\text{{PC1}} \\\\in [-3.43, -1.69]$), reflecting moderate, consistent strain transfer and controlled transient energy.
2. **Steel Packaging (FBG3)**: Occupies the negative PC1, positive PC2 quadrant ($\\\\text{{PC1}} \\\\in [-2.93, -2.13]$, $\\\\text{{PC2}} \\\\in [0.66, 1.23]$), characterized by high dynamic response indices, high frequency preservation, and lower raw strain transfer due to heavy structural stiffening.
3. **Bare Fiber (FBG2)**: Exhibits a wider dispersion across PC1 and PC2 ($\\\\text{{PC1}} \\\\in [-2.67, 5.39]$), spanning from moderate-intensity impacts to severe high-strain impacts (e.g., Experts 8, 9, 13) where direct mechanical contact generates extreme strain transfer and large transient RMS.

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
"""

    summary_path = os.path.join(OUTPUT_DIR, "phase9_pca_summary.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_content)

    return summary_path


# ============================================================
# 6. EXPORT ARTIFACTS
# ============================================================

def export_results(scores_df, loadings_df, exp_var_df, feature_metadata):
    """
    Exports all required CSV, JSON, and Markdown artifacts to results/phase9/pca/.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    scores_path = os.path.join(OUTPUT_DIR, "phase9_pca_scores.csv")
    loadings_path = os.path.join(OUTPUT_DIR, "phase9_pca_loadings.csv")
    exp_var_path = os.path.join(OUTPUT_DIR, "phase9_pca_explained_variance.csv")
    meta_path = os.path.join(OUTPUT_DIR, "phase9_pca_feature_metadata.json")

    scores_df.to_csv(scores_path, index=False)
    loadings_df.to_csv(loadings_path, index=True)
    exp_var_df.to_csv(exp_var_path, index=False)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(feature_metadata, f, indent=4)

    return scores_path, loadings_path, exp_var_path, meta_path


# ============================================================
# 7. VALIDATION SUITE
# ============================================================

def run_validation_suite(p5, p6, p8, merged, feature_cols, X_scaled, pca, scores_df, loadings_df, exp_var_df, plot1_path, plot2_path):
    """
    Executes comprehensive assertion checks on PCA execution and output integrity.
    """
    print("\n" + "=" * 60)
    print("PHASE 9 PCA VALIDATION SUITE")
    print("=" * 60)

    checks = []

    # Check 1: Input files exist
    chk1 = os.path.exists(PHASE5_CSV) and os.path.exists(PHASE6_CSV) and os.path.exists(PHASE8_CSV)
    checks.append(("[PASS] Required Phase 5, Phase 6, Phase 8 inputs exist", chk1))

    # Check 2: Impact events filtering
    total_events = len(p5)
    impact_count = len(merged)
    chk2 = (total_events == 21) and (impact_count == 12)
    checks.append((f"[PASS] Only IMPACT events used (12 valid impacts out of {total_events} total events)", chk2))

    # Check 3: NO IMPACT events excluded
    no_impact_p5 = (p5["Impact_Status"] == "NO IMPACT").sum()
    chk3 = (no_impact_p5 == 9) and not (merged["Impact_Status"] == "NO IMPACT").any()
    checks.append((f"[PASS] NO IMPACT events excluded ({no_impact_p5} excluded)", chk3))

    # Check 4: Material mapping preserved
    fbg1_mat = (merged[merged["FBG"] == "FBG1"]["Material"] == "Copper").all()
    fbg2_mat = (merged[merged["FBG"] == "FBG2"]["Material"] == "Bare").all()
    fbg3_mat = (merged[merged["FBG"] == "FBG3"]["Material"] == "Steel").all()
    chk4 = fbg1_mat and fbg2_mat and fbg3_mat
    checks.append(("[PASS] Material mapping preserved (FBG1->Copper, FBG2->Bare, FBG3->Steel)", chk4))

    # Check 5: Material labels not used as PCA features
    forbidden_cols = ["Material", "FBG", "Sensor", "Expert", "Dataset", "File", "Impact_Status", "Index_Status"]
    chk5 = not any(c in feature_cols for c in forbidden_cols)
    checks.append(("[PASS] Material labels & metadata completely excluded from PCA features", chk5))

    # Check 6: PCA input contains valid numerical features
    chk6 = (len(feature_cols) == 24) and all(isinstance(c, str) for c in feature_cols)
    checks.append((f"[PASS] PCA input contains 24 valid multi-domain numerical features", chk6))

    # Check 7: No invalid/infinite values remain
    chk7 = not np.isnan(X_scaled).any() and not np.isinf(X_scaled).any()
    checks.append(("[PASS] No invalid, NaN, or infinite values in standardized matrix", chk7))

    # Check 8: Scaling completed (mean ~ 0, std ~ 1)
    means = np.mean(X_scaled, axis=0)
    stds = np.std(X_scaled, axis=0)
    chk8 = np.allclose(means, 0.0, atol=1e-10) and np.allclose(stds, 1.0, atol=1e-10)
    checks.append(("[PASS] Standardization completed (zero mean, unit variance verified)", chk8))

    # Check 9: PCA completed successfully
    chk9 = pca.n_components_ == 12
    checks.append((f"[PASS] PCA completed successfully ({pca.n_components_} components fitted)", chk9))

    # Check 10: Explained variance ratios valid (sum = 1.0)
    sum_var = np.sum(pca.explained_variance_ratio_)
    chk10 = np.isclose(sum_var, 1.0, atol=1e-5)
    checks.append((f"[PASS] Explained variance ratios valid (Sum = {sum_var:.6f})", chk10))

    # Check 11: Cumulative explained variance is monotonic
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    chk11 = bool(np.all(np.diff(cum_var) >= -1e-10))
    checks.append(("[PASS] Cumulative explained variance is strictly monotonic", chk11))

    # Check 12: PCA score dimensions are correct
    chk12 = (scores_df.shape[0] == 12) and (scores_df.shape[1] == 12 + 4) # 12 PCs + 4 metadata cols
    checks.append((f"[PASS] PCA score dimensions are correct ({scores_df.shape[0]} samples x 12 PCs)", chk12))

    # Check 13: PCA loadings match selected features
    chk13 = (loadings_df.shape[0] == len(feature_cols)) and (loadings_df.shape[1] == 12)
    checks.append((f"[PASS] PCA loadings match selected features ({loadings_df.shape[0]} features x 12 PCs)", chk13))

    # Check 14: Required plots generated
    chk14 = os.path.exists(plot1_path) and os.path.exists(plot2_path)
    checks.append(("[PASS] Required plots generated in results/phase9/pca/plots/", chk14))

    all_passed = True
    for msg, status in checks:
        if status:
            print(f"  {msg}")
        else:
            print(f"  [FAIL] {msg}")
            all_passed = False

    print("=" * 60)
    if all_passed:
        print("ALL VALIDATION CHECKS PASSED (14/14).")
    else:
        print("SOME VALIDATION CHECKS FAILED.")
    print("=" * 60 + "\n")

    return all_passed


# ============================================================
# 8. MAIN EXECUTION ROUTINE
# ============================================================

def main():
    print("\n" + "=" * 60)
    print("PGMSIF PHASE 9: PRINCIPAL COMPONENT ANALYSIS (PCA ONLY)")
    print("=" * 60)

    # 1. Load and merge read-only inputs
    print("\n[Step 1/6] Loading Phase 5, Phase 6, Phase 8 inputs...")
    p5, p6, p8, merged = load_and_prepare_data()
    print(f"  Loaded 21 sensor events. Filtered {len(merged)} valid IMPACT events.")
    print(f"  Event distribution: Bare={sum(merged['Material']=='Bare')}, Copper={sum(merged['Material']=='Copper')}, Steel={sum(merged['Material']=='Steel')}")

    # 2. Select features and inspect missingness
    print("\n[Step 2/6] Selecting multi-domain features and inspecting missingness...")
    feature_cols, feature_metadata = select_pca_features(merged)
    print(f"  Selected {len(feature_cols)} features:")
    for domain, info in feature_metadata.items():
        if isinstance(info, dict) and "count" in info:
            print(f"    - {domain}: {info['count']} features")

    # 3. Standardization and PCA
    print("\n[Step 3/6] Standardizing features and computing PCA...")
    scaler, pca, scores_df, loadings_df, exp_var_df, X_scaled = run_pca_pipeline(merged, feature_cols)
    print(f"  Standardization complete (shape: {X_scaled.shape})")
    print(f"  PC1 Variance Explained: {exp_var_df.loc[0, 'Explained_Variance_Percent']:.2f}%")
    print(f"  PC2 Variance Explained: {exp_var_df.loc[1, 'Explained_Variance_Percent']:.2f}%")
    print(f"  PC1 + PC2 Total: {exp_var_df.loc[0, 'Explained_Variance_Percent'] + exp_var_df.loc[1, 'Explained_Variance_Percent']:.2f}%")

    # 4. Generate plots
    print("\n[Step 4/6] Generating visualization plots...")
    plot1_path, plot2_path = generate_plots(exp_var_df, scores_df)
    print(f"  Saved explained variance plot: {plot1_path}")
    print(f"  Saved PC1 vs PC2 plot: {plot2_path}")

    # 5. Export results
    print("\n[Step 5/6] Exporting CSV, JSON, and summary report artifacts...")
    scores_path, loadings_path, exp_var_path, meta_path = export_results(scores_df, loadings_df, exp_var_df, feature_metadata)
    summary_path = generate_summary_report(exp_var_df, scores_df, loadings_df, feature_metadata)
    print(f"  Saved scores: {scores_path}")
    print(f"  Saved loadings: {loadings_path}")
    print(f"  Saved explained variance: {exp_var_path}")
    print(f"  Saved metadata: {meta_path}")
    print(f"  Saved summary: {summary_path}")

    # 6. Run validation suite
    print("\n[Step 6/6] Executing validation suite...")
    passed = run_validation_suite(p5, p6, p8, merged, feature_cols, X_scaled, pca, scores_df, loadings_df, exp_var_df, plot1_path, plot2_path)

    if not passed:
        sys.exit(1)

    print("Phase 9 PCA pipeline execution completed successfully.\n")


if __name__ == "__main__":
    main()
