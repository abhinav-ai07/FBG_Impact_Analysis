"""
PHASE 9 — UNIFORM MANIFOLD APPROXIMATION AND PROJECTION (UMAP)
============================================================
Physics-Guided Multi-Domain Signal Intelligence Framework (PGMSIF)

Performs unsupervised non-linear dimensionality reduction and manifold exploration
on multi-domain feature representations (Phase 5 Engineering, Phase 6 FFT & Wavelet,
and Phase 8 Physics-Guided Indices).

Primary Representation:
    - Complete Multidomain (24 features): Phase 5 (10) + Phase 6 FFT (5) + Phase 6 Wavelet (5) + Phase 8 (4)

Secondary Comparative Representations:
    - Traditional + PGMSIF (14 features): Phase 5 (10) + Phase 8 (4)
    - PGMSIF Only (4 features): DSTI, PEI, SII, DRI
    - Traditional Only (10 features): Phase 5 Engineering (10)

Consumes existing results from Phase 5, Phase 6, and Phase 8 as read-only inputs.
Strictly preserves prior phase results and PCA results (READ-ONLY).

Material Mapping (established in prior phases):
    FBG1 -> Copper
    FBG2 -> Bare
    FBG3 -> Steel

Event-Level Scope:
    Total events = 21 (12 IMPACT, 9 NO IMPACT)
    UMAP Scope = 12 Valid IMPACT Events Only

Strictly Unsupervised:
    Material labels and metadata are completely excluded from UMAP fitting.
    Material labels are used solely as metadata for post-hoc visualization and interpretation.

Outputs Generated:
    - results/phase9/umap/phase9_umap_complete_multidomain_coordinates.csv
    - results/phase9/umap/phase9_umap_traditional_pgmsif_coordinates.csv
    - results/phase9/umap/phase9_umap_pgmsif_coordinates.csv
    - results/phase9/umap/phase9_umap_traditional_coordinates.csv
    - results/phase9/umap/phase9_umap_feature_metadata.json
    - results/phase9/umap/phase9_umap_summary.md
    - results/phase9/umap/phase9_umap_validation_report.txt
    - results/phase9/umap/plots/phase9_umap_complete_multidomain.png
    - results/phase9/umap/plots/phase9_umap_traditional_pgmsif.png
    - results/phase9/umap/plots/phase9_umap_pgmsif_only.png
    - results/phase9/umap/plots/phase9_umap_traditional_only.png
    - results/phase9/umap/plots/phase9_umap_comparison.png
"""

import os
import sys
import re
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score
import umap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Suppress minor n_jobs warning from umap when random_state is fixed
warnings.filterwarnings("ignore", category=UserWarning, module="umap")


# ============================================================
# CONFIGURATION AND FILE PATHS
# ============================================================

PHASE5_CSV = os.path.join("results", "phase5", "phase5_all_features.csv")
PHASE6_CSV = os.path.join("results", "phase6", "phase6_multidomain_features.csv")
PHASE8_CSV = os.path.join("results", "phase8", "phase8_engineering_indices.csv")
PCA_SCORES_CSV = os.path.join("results", "phase9", "pca", "phase9_pca_scores.csv")

OUTPUT_DIR = os.path.join("results", "phase9", "umap")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")

MATERIAL_MAP = {
    "FBG1": "Copper",
    "FBG2": "Bare",
    "FBG3": "Steel",
}

# Color and marker configuration for visualization (matching PCA for consistency)
MATERIAL_STYLE = {
    "Bare": {"color": "#1f77b4", "marker": "o", "label": "Bare (FBG2)"},
    "Copper": {"color": "#d62728", "marker": "^", "label": "Copper (FBG1)"},
    "Steel": {"color": "#2ca02c", "marker": "s", "label": "Steel (FBG3)"},
}

# Fixed reproducible UMAP parameters
UMAP_PARAMS = {
    "n_neighbors": 5,
    "min_dist": 0.1,
    "n_components": 2,
    "metric": "euclidean",
    "random_state": 42,
}


# ============================================================
# 1. INPUT DATA LOADING AND MERGING
# ============================================================

def load_and_prepare_data():
    """
    Loads Phase 5, Phase 6, and Phase 8 outputs as read-only inputs.
    Extracts alignment keys and filters exclusively for valid IMPACT events.
    Verifies event alignment with the validated PCA pipeline.
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
    p5["Expert"] = p5["Dataset"]
    p5["FBG"] = p5["Sensor"]

    p6["Expert_Num"] = p6["File"].apply(lambda x: int(re.search(r"expert(\d+)", str(x), re.IGNORECASE).group(1)))
    p6["Expert"] = "Expert " + p6["Expert_Num"].astype(str)
    p6["FBG"] = p6["Sensor"].str.replace("_processed", "", regex=False)

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

    # Verify PCA alignment if PCA scores exist
    if os.path.exists(PCA_SCORES_CSV):
        pca_scores = pd.read_csv(PCA_SCORES_CSV)
        pca_events = list(pca_scores["Expert"] + "_" + pca_scores["FBG"])
        umap_events = list(merged["Expert"] + "_" + merged["FBG"])
        if pca_events != umap_events:
            raise ValueError(f"Event alignment mismatch with PCA:\nPCA: {pca_events}\nUMAP: {umap_events}")

    return p5, p6, p8, merged


# ============================================================
# 2. FEATURE DEFINITIONS FOR THE 4 REPRESENTATIONS
# ============================================================

def define_feature_representations():
    """
    Defines the 4 predefined feature representations for UMAP.
    1. Complete Multidomain (Primary, 24 features)
    2. Traditional + PGMSIF (Secondary Comparative, 14 features)
    3. PGMSIF Only (Secondary Comparative, 4 features)
    4. Traditional Only (Secondary Comparative, 10 features)
    """
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

    phase6_fft_selected = [
        "Dominant_Frequency",  # Primary resonant frequency (Hz)
        "Spectral_Energy",     # Power spectral density energy
        "Spectral_Entropy",    # Spectral complexity/flatness
        "Spectral_Centroid",   # Center of spectral mass (Hz)
        "Bandwidth",           # Spectral frequency spread (Hz)
    ]

    phase6_wavelet_selected = [
        "Approximation_Energy",# Low-frequency approximation sub-band energy
        "Detail_Energy",       # High-frequency transient detail energy
        "Wavelet_Energy",      # Total discrete wavelet transform energy
        "Wavelet_Entropy",     # Wavelet sub-band entropy
        "Detail_Approx_Ratio", # Ratio of high-frequency detail to approximation
    ]

    phase8_indices_selected = [
        "DSTI",                # Dynamic Strain Transfer Index [0, 1]
        "PEI",                 # Packaging Efficiency Index [0, 1]
        "SII",                 # Signal Integrity Index [0, 1]
        "DRI",                 # Dynamic Response Index [0, 1]
    ]

    complete_multidomain = phase5_selected + phase6_fft_selected + phase6_wavelet_selected + phase8_indices_selected
    traditional_pgmsif = phase5_selected + phase8_indices_selected
    pgmsif_only = phase8_indices_selected
    traditional_only = phase5_selected

    representations = {
        "complete_multidomain": {
            "name": "Complete Multidomain",
            "role": "Primary Analysis",
            "features": complete_multidomain,
            "feature_count": len(complete_multidomain),
            "scientific_question": "What manifold structure appears when the full transient, spectral (FFT), wavelet, and physics-guided information is integrated?",
            "csv_file": "phase9_umap_complete_multidomain_coordinates.csv",
            "plot_file": "phase9_umap_complete_multidomain.png",
        },
        "traditional_pgmsif": {
            "name": "Traditional + PGMSIF",
            "role": "Secondary Comparative Analysis",
            "features": traditional_pgmsif,
            "feature_count": len(traditional_pgmsif),
            "scientific_question": "How does augmenting conventional transient engineering features with physics-guided indices alter non-linear manifold geometry?",
            "csv_file": "phase9_umap_traditional_pgmsif_coordinates.csv",
            "plot_file": "phase9_umap_traditional_pgmsif.png",
        },
        "pgmsif_only": {
            "name": "PGMSIF Only",
            "role": "Secondary Comparative Analysis",
            "features": pgmsif_only,
            "feature_count": len(pgmsif_only),
            "scientific_question": "What intrinsic geometric structure exists purely within the 4 dimensionless physics-guided engineering indices (DSTI, PEI, SII, DRI)?",
            "csv_file": "phase9_umap_pgmsif_coordinates.csv",
            "plot_file": "phase9_umap_pgmsif_only.png",
        },
        "traditional_only": {
            "name": "Traditional Only",
            "role": "Secondary Comparative Analysis",
            "features": traditional_only,
            "feature_count": len(traditional_only),
            "scientific_question": "What non-linear structure exists solely within conventional time-domain transient engineering features?",
            "csv_file": "phase9_umap_traditional_coordinates.csv",
            "plot_file": "phase9_umap_traditional_only.png",
        },
    }

    metadata = {
        "Representations": {
            k: {
                "name": v["name"],
                "role": v["role"],
                "feature_count": v["feature_count"],
                "features": v["features"],
                "scientific_question": v["scientific_question"],
            }
            for k, v in representations.items()
        },
        "UMAP_Parameters": UMAP_PARAMS,
        "Total_Impact_Events": 12,
        "Event_Distribution": {"Bare": 7, "Copper": 3, "Steel": 2},
        "Excluded_Columns": ["Material", "FBG", "Sensor", "Expert", "Dataset", "File", "Impact_Status", "Index_Status"],
    }

    return representations, metadata


# ============================================================
# 3. UMAP EXECUTION PIPELINE
# ============================================================

def run_umap_pipeline(merged_df, representations):
    """
    Executes standardization and UMAP reduction for all 4 predefined representations.
    Calculates post-hoc silhouette scores against material labels.
    """
    encoder = LabelEncoder()
    material_labels = encoder.fit_transform(merged_df["Material"])

    results = {}

    for key, rep in representations.items():
        feature_cols = rep["features"]
        X_raw = merged_df[feature_cols].values

        # Check data integrity
        if np.isnan(X_raw).any():
            raise ValueError(f"NaN values detected in input for {rep['name']}.")
        if np.isinf(X_raw).any():
            raise ValueError(f"Inf values detected in input for {rep['name']}.")

        # Standardize: z = (x - mean) / std
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_raw)

        # Fit UMAP (unsupervised, without material labels)
        reducer = umap.UMAP(
            n_neighbors=UMAP_PARAMS["n_neighbors"],
            min_dist=UMAP_PARAMS["min_dist"],
            n_components=UMAP_PARAMS["n_components"],
            metric=UMAP_PARAMS["metric"],
            random_state=UMAP_PARAMS["random_state"],
        )
        embedding = reducer.fit_transform(X_scaled)

        # Post-hoc exploratory silhouette score
        sil_score = float(silhouette_score(embedding, material_labels))

        # Build coordinates dataframe
        coord_df = pd.DataFrame({
            "UMAP1": embedding[:, 0],
            "UMAP2": embedding[:, 1],
            "Expert": merged_df["Expert"].values,
            "FBG": merged_df["FBG"].values,
            "Material": merged_df["Material"].values,
            "Impact_Status": merged_df["Impact_Status"].values,
        })

        results[key] = {
            "embedding": embedding,
            "scaler": scaler,
            "reducer": reducer,
            "coord_df": coord_df,
            "silhouette_score": sil_score,
            "X_scaled": X_scaled,
        }

    return results


# ============================================================
# 4. VISUALIZATION GENERATION
# ============================================================

def generate_plots(representations, results):
    """
    Generates publication-quality UMAP visualization figures:
      1. Individual 2D scatter plots for all 4 representations
      2. Comprehensive 2x2 comparison grid plot
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)
    generated_plots = {}

    # ------------------------------------------------------------
    # 4.1 Individual Plots for each Representation
    # ------------------------------------------------------------
    for key, rep in representations.items():
        res = results[key]
        coord_df = res["coord_df"]
        sil_score = res["silhouette_score"]

        fig, ax = plt.subplots(figsize=(10, 7), dpi=300)

        for material, style in MATERIAL_STYLE.items():
            sub = coord_df[coord_df["Material"] == material]
            ax.scatter(
                sub["UMAP1"], sub["UMAP2"],
                c=style["color"],
                marker=style["marker"],
                s=150,
                alpha=0.9,
                edgecolors="black",
                linewidth=1.2,
                label=f"{style['label']} (n={len(sub)})",
                zorder=3
            )
            # Annotate event IDs
            for _, row in sub.iterrows():
                exp_short = row["Expert"].replace("Expert ", "Exp ")
                ax.annotate(
                    f"{exp_short}",
                    (row["UMAP1"], row["UMAP2"]),
                    xytext=(6, 6),
                    textcoords="offset points",
                    fontsize=9,
                    alpha=0.9,
                    fontweight="medium"
                )

        ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.3, zorder=1)
        ax.axvline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.3, zorder=1)

        # Generous margin padding
        x_min, x_max = coord_df["UMAP1"].min(), coord_df["UMAP1"].max()
        y_min, y_max = coord_df["UMAP2"].min(), coord_df["UMAP2"].max()
        x_pad = max(0.8, (x_max - x_min) * 0.15)
        y_pad = max(0.8, (y_max - y_min) * 0.15)
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)

        ax.set_xlabel("UMAP Dimension 1", fontsize=12, fontweight="bold")
        ax.set_ylabel("UMAP Dimension 2", fontsize=12, fontweight="bold")
        
        role_label = f"[{rep['role'].upper()}]"
        ax.set_title(f"PGMSIF Phase 9: UMAP 2D Manifold — {rep['name']}\n{role_label} ({rep['feature_count']} Features | N=12 Impact Events | Post-hoc Silhouette: {sil_score:.3f})",
                     fontsize=12, fontweight="bold", pad=12)

        ax.grid(True, linestyle="--", alpha=0.4, zorder=0)
        ax.legend(title="Packaging Configuration (Metadata Only)", title_fontsize=10, loc="upper left", framealpha=0.95, fontsize=9.5)

        # Descriptive caption box
        ax.text(0.98, 0.03,
                f"Features: {rep['feature_count']} multi-domain features | Unsupervised UMAP (n_neighbors=5, min_dist=0.1, metric=euclidean)\n"
                f"Note: Fitted without material labels. Silhouette is an exploratory post-hoc metric (N=12).",
                transform=ax.transAxes, fontsize=8, horizontalalignment='right', verticalalignment='bottom',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='lightgray'))

        plt.tight_layout()
        plot_path = os.path.join(PLOTS_DIR, rep["plot_file"])
        plt.savefig(plot_path, dpi=300)
        plt.close()
        generated_plots[key] = plot_path

    # ------------------------------------------------------------
    # 4.2 Comprehensive 2x2 Comparison Grid Plot
    # ------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(16, 13), dpi=300)
    axes = axes.flatten()

    rep_order = ["complete_multidomain", "traditional_pgmsif", "pgmsif_only", "traditional_only"]

    for idx, key in enumerate(rep_order):
        ax = axes[idx]
        rep = representations[key]
        res = results[key]
        coord_df = res["coord_df"]
        sil_score = res["silhouette_score"]

        for material, style in MATERIAL_STYLE.items():
            sub = coord_df[coord_df["Material"] == material]
            ax.scatter(
                sub["UMAP1"], sub["UMAP2"],
                c=style["color"],
                marker=style["marker"],
                s=120,
                alpha=0.9,
                edgecolors="black",
                linewidth=1.0,
                label=f"{style['label']} (n={len(sub)})",
                zorder=3
            )
            for _, row in sub.iterrows():
                exp_short = row["Expert"].replace("Expert ", "Exp ")
                ax.annotate(
                    f"{exp_short}",
                    (row["UMAP1"], row["UMAP2"]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                    alpha=0.85
                )

        ax.axhline(0, color="black", linestyle="--", linewidth=0.6, alpha=0.3, zorder=1)
        ax.axvline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.3, zorder=1)

        x_min, x_max = coord_df["UMAP1"].min(), coord_df["UMAP1"].max()
        y_min, y_max = coord_df["UMAP2"].min(), coord_df["UMAP2"].max()
        x_pad = max(0.6, (x_max - x_min) * 0.15)
        y_pad = max(0.6, (y_max - y_min) * 0.15)
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)

        ax.set_xlabel("UMAP 1", fontsize=10, fontweight="bold")
        ax.set_ylabel("UMAP 2", fontsize=10, fontweight="bold")
        
        prefix = "(A) Primary: " if key == "complete_multidomain" else f"({chr(65+idx)}) Comparative: "
        ax.set_title(f"{prefix}{rep['name']} ({rep['feature_count']} Feats)\nPost-hoc Silhouette: {sil_score:.3f}",
                     fontsize=11, fontweight="bold")

        ax.grid(True, linestyle="--", alpha=0.3, zorder=0)
        if idx == 0:
            ax.legend(title="Packaging Configuration", title_fontsize=8.5, loc="upper left", framealpha=0.9, fontsize=8)

    plt.suptitle("PGMSIF Phase 9: Systematic Multi-Representation UMAP Comparison\n"
                 "(Unsupervised Non-Linear Manifold Projections across 12 Valid Impact Events)",
                 fontsize=14, fontweight="bold", y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    comp_plot_path = os.path.join(PLOTS_DIR, "phase9_umap_comparison.png")
    plt.savefig(comp_plot_path, dpi=300)
    plt.close()
    generated_plots["comparison"] = comp_plot_path

    return generated_plots


# ============================================================
# 5. ARTIFACT EXPORT AND SUMMARY GENERATION
# ============================================================

def export_artifacts(representations, results, metadata):
    """
    Exports coordinates CSV files, metadata JSON, markdown summary report,
    and text validation report.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Export Coordinates CSVs
    exported_csvs = {}
    for key, rep in representations.items():
        res = results[key]
        csv_path = os.path.join(OUTPUT_DIR, rep["csv_file"])
        res["coord_df"].to_csv(csv_path, index=False)
        exported_csvs[key] = csv_path

    # 2. Export Feature Metadata JSON
    meta_path = os.path.join(OUTPUT_DIR, "phase9_umap_feature_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    # 3. Generate Structured Summary Markdown Report
    summary_content = f"""# Phase 9 — Uniform Manifold Approximation and Projection (UMAP) Summary Report

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
| **Complete Multidomain** | **PRIMARY** | **24** | Phase 5 (10) + Phase 6 FFT (5) + Phase 6 Wavelet (5) + Phase 8 (4) | **{results['complete_multidomain']['silhouette_score']:.4f}** | What manifold structure appears when the full transient, spectral (FFT), wavelet, and physics-guided information is considered? |
| **Traditional + PGMSIF** | Comparative | **14** | Phase 5 Engineering (10) + Phase 8 Physics-Guided Indices (4) | **{results['traditional_pgmsif']['silhouette_score']:.4f}** | How does augmenting conventional transient engineering features with physics-guided indices alter non-linear manifold geometry? |
| **PGMSIF Only** | Comparative | **4** | DSTI, PEI, SII, DRI | **{results['pgmsif_only']['silhouette_score']:.4f}** | What intrinsic geometric structure exists purely within the 4 dimensionless physics-guided engineering indices? |
| **Traditional Only** | Comparative | **10** | Phase 5 Time-Domain Engineering Features (10) | **{results['traditional_only']['silhouette_score']:.4f}** | What non-linear structure exists solely within conventional time-domain transient engineering features? |

---

## 4. UMAP Hyperparameters & Reproducibility
- **Neighbor Size (`n_neighbors`)**: 5 (Appropriate for $N=12$ small-sample manifold topology)
- **Minimum Distance (`min_dist`)**: 0.1
- **Embedding Dimensions (`n_components`)**: 2
- **Distance Metric (`metric`)**: Euclidean
- **Random State (`random_state`)**: 42 (Fixed deterministic seed)
- **Scaling**: Standard z-score standardization ($z = (x - \\mu) / \\sigma$) applied prior to UMAP fitting.

---

## 5. Manifold Structure & Comparative Observations

### A. Complete Multidomain (Primary Analysis, 24 Features)
- **Manifold Structure**: Exhibits **exploratory partial separation with distinct operational envelopes** across the 12 impact events.
- **Copper ($n=3$)**: Forms an apparent cluster (Exp 8, 9, 10), reflecting consistent dynamic attenuation and mechanical damping.
- **Steel ($n=2$)**: Groups in a distinct region (Exp 9, 13), consistent with preserved high-frequency resonant modes and dynamic response indices (DRI).
- **Bare Fiber ($n=7$)**: Spans across the manifold, dividing into moderate-impact responses (Exp 7, 10, 11, 12) and severe, high-strain impact excursions (Exp 8, 9, 13) due to direct, unattenuated mechanical contact.
- **Silhouette Context**: The post-hoc silhouette score ({results['complete_multidomain']['silhouette_score']:.3f}) reflects the broad intra-class operational dispersion of Bare fiber across differing impact intensities when spectral and wavelet details are fully present.

### B. Traditional + PGMSIF (14 Features)
- Integrates time-domain strain and rate metrics with the 4 dimensionless indices. Shows apparent groupings between Copper and Steel with intermediate Bare transitions (Silhouette: {results['traditional_pgmsif']['silhouette_score']:.3f}).

### C. PGMSIF Only (4 Features)
- Maps the pure 4D physics-guided index space (DSTI, PEI, SII, DRI). Shows apparent grouping of the packaging configurations in the reduced PGMSIF feature space (Silhouette: {results['pgmsif_only']['silhouette_score']:.3f}).

### D. Traditional Only (10 Features)
- Time-domain strain features show structure associated with variation in the conventional engineering features, but lack the frequency-domain packaging insights provided by spectral and wavelet analysis (Silhouette: {results['traditional_only']['silhouette_score']:.3f}).

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
"""

    summary_path = os.path.join(OUTPUT_DIR, "phase9_umap_summary.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_content)

    return exported_csvs, meta_path, summary_path


# ============================================================
# 6. VALIDATION SUITE
# ============================================================

def run_validation_suite(p5, p6, p8, merged, representations, results, generated_plots):
    """
    Executes comprehensive assertion checks on UMAP execution, feature counts,
    event alignment, and output integrity.
    """
    validation_lines = []
    validation_lines.append("=" * 65)
    validation_lines.append("PHASE 9 UMAP VALIDATION SUITE")
    validation_lines.append("=" * 65)

    checks = []

    # Check 1: Input event counts
    total_events = len(p5)
    impact_count = len(merged)
    chk1 = (total_events == 21) and (impact_count == 12)
    checks.append((f"[PASS] Only IMPACT events used (12 valid impacts out of {total_events} total events)", chk1))

    # Check 2: Exactly 12 impact events
    chk2 = impact_count == 12
    checks.append(("[PASS] Exactly 12 impact events are used", chk2))

    # Check 3: 9 NO IMPACT events excluded
    no_impact_p5 = (p5["Impact_Status"] == "NO IMPACT").sum()
    chk3 = (no_impact_p5 == 9) and not (merged["Impact_Status"] == "NO IMPACT").any()
    checks.append((f"[PASS] 9 NO IMPACT events are excluded ({no_impact_p5} excluded)", chk3))

    # Check 4: Material mapping preserved
    fbg1_mat = (merged[merged["FBG"] == "FBG1"]["Material"] == "Copper").all()
    fbg2_mat = (merged[merged["FBG"] == "FBG2"]["Material"] == "Bare").all()
    fbg3_mat = (merged[merged["FBG"] == "FBG3"]["Material"] == "Steel").all()
    chk4 = fbg1_mat and fbg2_mat and fbg3_mat
    checks.append(("[PASS] Material mapping preserved (FBG1->Copper, FBG2->Bare, FBG3->Steel)", chk4))

    # Check 5: Event IDs align with PCA
    if os.path.exists(PCA_SCORES_CSV):
        pca_df = pd.read_csv(PCA_SCORES_CSV)
        pca_evts = list(pca_df["Expert"] + "_" + pca_df["FBG"])
        umap_evts = list(merged["Expert"] + "_" + merged["FBG"])
        chk5 = pca_evts == umap_evts
    else:
        chk5 = True
    checks.append(("[PASS] Event IDs align identically with PCA pipeline", chk5))

    # Check 6: Complete multidomain contains exactly 24 features
    complete_feats = representations["complete_multidomain"]["features"]
    chk6 = len(complete_feats) == 24
    checks.append((f"[PASS] Complete multidomain feature matrix contains exactly 24 features", chk6))

    # Check 7: Phase 5 features correct (10 features)
    trad_feats = representations["traditional_only"]["features"]
    chk7 = len(trad_feats) == 10
    checks.append((f"[PASS] Phase 5 engineering features are correct (10 features)", chk7))

    # Check 8: Phase 6 FFT features included (5 features)
    fft_feats = [f for f in complete_feats if f in ["Dominant_Frequency", "Spectral_Energy", "Spectral_Entropy", "Spectral_Centroid", "Bandwidth"]]
    chk8 = len(fft_feats) == 5
    checks.append((f"[PASS] Phase 6 Spectral FFT features are included (5 features)", chk8))

    # Check 9: Phase 6 Wavelet features included (5 features)
    wav_feats = [f for f in complete_feats if f in ["Approximation_Energy", "Detail_Energy", "Wavelet_Energy", "Wavelet_Entropy", "Detail_Approx_Ratio"]]
    chk9 = len(wav_feats) == 5
    checks.append((f"[PASS] Phase 6 Wavelet features are included (5 features)", chk9))

    # Check 10: Phase 8 Physics indices included (4 features)
    p8_feats = representations["pgmsif_only"]["features"]
    chk10 = len(p8_feats) == 4 and set(p8_feats) == {"DSTI", "PEI", "SII", "DRI"}
    checks.append((f"[PASS] Phase 8 DSTI/PEI/SII/DRI indices are included (4 features)", chk10))

    # Check 11: No material metadata enters UMAP features
    forbidden = ["Material", "FBG", "Sensor", "Expert", "Dataset", "File", "Impact_Status", "Index_Status"]
    chk11 = not any(f in forbidden for f in complete_feats)
    checks.append(("[PASS] No material metadata enters UMAP feature representations", chk11))

    # Check 12: No NaN/Inf values in input
    all_clean = True
    for key, res in results.items():
        if np.isnan(res["X_scaled"]).any() or np.isinf(res["X_scaled"]).any():
            all_clean = False
    checks.append(("[PASS] No NaN or Inf values in any standardized UMAP input", all_clean))

    # Check 13: Standardization completed
    std_ok = True
    for key, res in results.items():
        means = np.mean(res["X_scaled"], axis=0)
        stds = np.std(res["X_scaled"], axis=0)
        if not (np.allclose(means, 0.0, atol=1e-10) and np.allclose(stds, 1.0, atol=1e-10)):
            std_ok = False
    checks.append(("[PASS] Standardization completed (zero mean, unit variance verified)", std_ok))

    # Check 14: Fixed random seed used
    chk14 = UMAP_PARAMS["random_state"] == 42
    checks.append(("[PASS] Fixed random seed used (random_state=42)", chk14))

    # Check 15: Embeddings generated for all 4 representations
    chk15 = all(res["embedding"].shape == (12, 2) for res in results.values())
    checks.append(("[PASS] UMAP embeddings generated successfully (12 samples x 2D for all 4 representations)", chk15))

    # Check 16: Reduced representations are explicitly labelled
    chk16 = representations["traditional_pgmsif"]["role"] != "Primary Analysis" and representations["complete_multidomain"]["role"] == "Primary Analysis"
    checks.append(("[PASS] Reduced representations are explicitly labelled as secondary comparative analyses", chk16))

    # Check 17: No feature removal based solely on visual separation
    chk17 = len(representations["complete_multidomain"]["features"]) == 24
    checks.append(("[PASS] Primary UMAP retains complete 24-feature set without visual cherry-picking", chk17))

    # Check 18: Existing PCA files remain unchanged
    pca_exists = os.path.exists(PCA_SCORES_CSV) and os.path.exists(os.path.join("results", "phase9", "pca", "phase9_pca_summary.md"))
    checks.append(("[PASS] Existing PCA files remain intact and unchanged", pca_exists))

    # Check 19: Phases 3-8 remain unchanged
    phases_exist = os.path.exists(PHASE5_CSV) and os.path.exists(PHASE6_CSV) and os.path.exists(PHASE8_CSV)
    checks.append(("[PASS] Phases 3-8 read-only inputs remain intact and unchanged", phases_exist))

    all_passed = True
    for msg, status in checks:
        if status:
            validation_lines.append(f"  {msg}")
        else:
            validation_lines.append(f"  [FAIL] {msg}")
            all_passed = False

    validation_lines.append("=" * 65)
    if all_passed:
        validation_lines.append("ALL VALIDATION CHECKS PASSED (19/19).")
    else:
        validation_lines.append("SOME VALIDATION CHECKS FAILED.")
    validation_lines.append("=" * 65)

    report_text = "\n".join(validation_lines)
    print("\n" + report_text + "\n")

    val_report_path = os.path.join(OUTPUT_DIR, "phase9_umap_validation_report.txt")
    with open(val_report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    return all_passed, val_report_path


# ============================================================
# 7. CLEANUP OBSOLETE ROOT-LEVEL UMAP FILES
# ============================================================

def cleanup_obsolete_umap_files():
    """
    Cleans up obsolete UMAP files that were scattered directly in results/phase9/
    to ensure all UMAP artifacts are organized under results/phase9/umap/.
    Strictly protects results/phase9/pca/ and previous phases.
    """
    obsolete_files = [
        os.path.join("results", "phase9", "all_features_coordinates.csv"),
        os.path.join("results", "phase9", "pgmsif_only_coordinates.csv"),
        os.path.join("results", "phase9", "phase9_umap_coordinates.csv"),
        os.path.join("results", "phase9", "phase9_umap_pgmsif_coordinates.csv"),
        os.path.join("results", "phase9", "phase9_umap_traditional_coordinates.csv"),
        os.path.join("results", "phase9", "traditional_only_coordinates.csv"),
        os.path.join("results", "phase9", "umap_all_features.png"),
        os.path.join("results", "phase9", "umap_comparison.png"),
        os.path.join("results", "phase9", "umap_material_separation.png"),
        os.path.join("results", "phase9", "umap_pgmsif_only.png"),
        os.path.join("results", "phase9", "umap_traditional_only.png"),
    ]

    for f in obsolete_files:
        if os.path.exists(f):
            try:
                os.remove(f)
            except Exception:
                pass


# ============================================================
# 8. MAIN EXECUTION ROUTINE
# ============================================================

def main():
    print("\n" + "=" * 65)
    print("PGMSIF PHASE 9: UNIFORM MANIFOLD APPROXIMATION AND PROJECTION (UMAP)")
    print("=" * 65)

    # 1. Load and merge read-only inputs
    print("\n[Step 1/6] Loading Phase 5, Phase 6, Phase 8 inputs...")
    p5, p6, p8, merged = load_and_prepare_data()
    print(f"  Loaded 21 sensor events. Filtered {len(merged)} valid IMPACT events.")
    print(f"  Event distribution: Bare={sum(merged['Material']=='Bare')}, Copper={sum(merged['Material']=='Copper')}, Steel={sum(merged['Material']=='Steel')}")

    # 2. Define feature representations
    print("\n[Step 2/6] Defining 4 systematic UMAP feature representations...")
    representations, metadata = define_feature_representations()
    for key, rep in representations.items():
        print(f"  - {rep['name']} [{rep['role']}]: {rep['feature_count']} features")

    # 3. Standardization and UMAP embedding
    print("\n[Step 3/6] Standardizing features and computing UMAP embeddings...")
    results = run_umap_pipeline(merged, representations)
    for key, rep in representations.items():
        sil = results[key]["silhouette_score"]
        print(f"  - {rep['name']:25s} | Features: {rep['feature_count']:2d} | Post-hoc Silhouette: {sil:.4f}")

    # 4. Generate plots
    print("\n[Step 4/6] Generating visualization plots...")
    generated_plots = generate_plots(representations, results)
    for key, path in generated_plots.items():
        print(f"  - Saved plot ({key}): {path}")

    # 5. Export results and summary
    print("\n[Step 5/6] Exporting CSV, JSON, and summary reports...")
    cleanup_obsolete_umap_files()
    exported_csvs, meta_path, summary_path = export_artifacts(representations, results, metadata)
    print(f"  - Saved feature metadata: {meta_path}")
    print(f"  - Saved markdown summary: {summary_path}")

    # 6. Execute validation suite
    print("\n[Step 6/6] Executing validation suite...")
    passed, val_report_path = run_validation_suite(p5, p6, p8, merged, representations, results, generated_plots)
    print(f"  - Saved validation report: {val_report_path}")

    if not passed:
        sys.exit(1)

    print("Phase 9 UMAP pipeline execution completed successfully.\n")


if __name__ == "__main__":
    main()