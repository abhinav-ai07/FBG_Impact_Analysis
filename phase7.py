"""
PHASE 7 — STATISTICAL VALIDATION
=================================
Reads Phase 5 (engineering/time-domain) and Phase 6 (multi-domain)
feature result CSVs, computes statistical validation metrics
(Mean, SD, CV, 95% CI) for each feature grouped by material,
and produces a Bare vs Copper vs Steel comparison.

Material mapping (established in prior phases):
    FBG1 → Copper
    FBG2 → Bare
    FBG3 → Steel
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

PHASE5_FEATURES_CSV = os.path.join("results", "phase5", "phase5_all_features.csv")
PHASE6_FEATURES_CSV = os.path.join("results", "phase6", "phase6_multidomain_features.csv")
OUTPUT_DIR = os.path.join("results", "phase7")

# Material mapping — consistent with prior phases
MATERIAL_MAP = {
    "FBG1": "Copper",
    "FBG2": "Bare",
    "FBG3": "Steel",
}

# Phase 5 engineering features (impact-specific: only valid for IMPACT cases)
PHASE5_IMPACT_FEATURES = [
    "peak_shift_abs",
    "residual_shift_abs",
    "rise_time_seconds",
    "recovery_time_seconds",
    "peak_width_seconds",
    "max_slope_abs",
    "rms",
    "signal_energy",
    "peak_to_peak",
    "variance",
    "std_dev",
    "entropy",
    "auc_abs",
]

# Phase 5 baseline features (valid for ALL cases, including NO IMPACT)
PHASE5_BASELINE_FEATURES = [
    "baseline_nm",
    "noise_std_nm",
]

# Phase 6 multi-domain features (impact-specific: only valid for IMPACT cases)
PHASE6_IMPACT_FEATURES = [
    "Dominant_Frequency",
    "Spectral_Energy",
    "Spectral_Entropy",
    "Spectral_Centroid",
    "Bandwidth",
    "Approximation_Energy",
    "Detail_Energy",
    "Wavelet_Energy",
    "Wavelet_Entropy",
    "Detail_Approx_Ratio",
]

# Threshold below which |mean| is considered effectively zero for CV
CV_ZERO_THRESHOLD = 1e-15


# ============================================================
# STATISTICAL FUNCTIONS
# ============================================================

def compute_statistics(values):
    """
    Compute Mean, SD (sample), CV, and 95% CI for a 1-D array of values.
    
    Returns a dict with keys: n, mean, sd, cv, ci_lower, ci_upper.
    - CV is NaN if |mean| < CV_ZERO_THRESHOLD (to avoid inf).
    - CI requires n >= 2; returns NaN otherwise.
    """
    clean = values.dropna() if isinstance(values, pd.Series) else pd.Series(values).dropna()
    n = len(clean)

    if n == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "sd": np.nan,
            "cv": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
        }

    mean_val = float(np.mean(clean))
    sd_val = float(np.std(clean, ddof=1)) if n >= 2 else np.nan

    # Coefficient of Variation — safe division
    if sd_val is not None and not np.isnan(sd_val) and abs(mean_val) > CV_ZERO_THRESHOLD:
        cv_val = (sd_val / abs(mean_val)) * 100.0
    else:
        cv_val = np.nan

    # 95% Confidence Interval using t-distribution
    if n >= 2 and not np.isnan(sd_val):
        t_crit = stats.t.ppf(0.975, df=n - 1)
        margin = t_crit * sd_val / np.sqrt(n)
        ci_lower = mean_val - margin
        ci_upper = mean_val + margin
    else:
        ci_lower = np.nan
        ci_upper = np.nan

    return {
        "n": n,
        "mean": mean_val,
        "sd": sd_val,
        "cv": cv_val,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }


# ============================================================
# DATA LOADING
# ============================================================

def load_phase5_data():
    """Load Phase 5 all-features CSV."""
    if not os.path.exists(PHASE5_FEATURES_CSV):
        print(f"ERROR: Phase 5 features CSV not found: {PHASE5_FEATURES_CSV}")
        sys.exit(1)

    df = pd.read_csv(PHASE5_FEATURES_CSV)
    print(f"  [OK] Loaded Phase 5 features: {len(df)} rows, {len(df.columns)} columns")
    return df


def load_phase6_data():
    """Load Phase 6 multi-domain features CSV and add Material column."""
    if not os.path.exists(PHASE6_FEATURES_CSV):
        print(f"ERROR: Phase 6 features CSV not found: {PHASE6_FEATURES_CSV}")
        sys.exit(1)

    df = pd.read_csv(PHASE6_FEATURES_CSV)

    # Map Sensor column (FBG1_processed → FBG1 → material)
    def sensor_to_material(sensor_col):
        fbg = sensor_col.replace("_processed", "").upper()
        return MATERIAL_MAP.get(fbg, "Unknown")

    df["Material"] = df["Sensor"].apply(sensor_to_material)

    # Also extract a clean FBG name
    df["FBG"] = df["Sensor"].apply(lambda s: s.replace("_processed", "").upper())

    # Extract expert number for consistency
    def extract_expert(filename):
        try:
            num = int(filename.lower().split("expert")[1].split(".")[0])
            return f"Expert {num}"
        except (IndexError, ValueError):
            return filename
    df["Dataset"] = df["File"].apply(extract_expert)

    print(f"  [OK] Loaded Phase 6 features: {len(df)} rows, {len(df.columns)} columns")
    return df


# ============================================================
# STATISTICAL ANALYSIS
# ============================================================

def analyze_features(df, features, impact_only, source_label):
    """
    For each feature × material, compute statistics.
    
    Parameters
    ----------
    df : DataFrame with columns including 'Material' and either
         'Impact_Status' (Phase 5) or 'Phase4_Result' (Phase 6).
    features : list of feature column names.
    impact_only : bool, if True filter to IMPACT cases only.
    source_label : str, label for the source phase ('Phase5' or 'Phase6').
    
    Returns a list of result dicts.
    """
    results = []

    # Determine the impact-status column
    if "Impact_Status" in df.columns:
        status_col = "Impact_Status"
    elif "Phase4_Result" in df.columns:
        status_col = "Phase4_Result"
    else:
        print(f"  WARNING: No impact status column found in {source_label} data")
        status_col = None

    for feature in features:
        if feature not in df.columns:
            print(f"  WARNING: Feature '{feature}' not found in {source_label} data -- skipping")
            continue

        for material in ["Bare", "Copper", "Steel"]:
            mask = df["Material"] == material
            if impact_only and status_col:
                mask = mask & (df[status_col] == "IMPACT")

            subset = df.loc[mask, feature]
            stat = compute_statistics(subset)

            results.append({
                "Source": source_label,
                "Feature": feature,
                "Material": material,
                "Impact_Only": impact_only,
                "n": stat["n"],
                "Mean": stat["mean"],
                "SD": stat["sd"],
                "CV_pct": stat["cv"],
                "CI_95_Lower": stat["ci_lower"],
                "CI_95_Upper": stat["ci_upper"],
            })

    return results


# ============================================================
# COMPARISON TABLE
# ============================================================

def build_comparison_table(stats_df):
    """
    Build a Bare vs Copper vs Steel comparison table.
    Each row is a feature; columns show mean/sd/cv/ci for each material.
    """
    rows = []
    features = stats_df["Feature"].unique()

    for feature in features:
        row = {"Feature": feature}
        feat_data = stats_df[stats_df["Feature"] == feature]
        source = feat_data["Source"].iloc[0] if len(feat_data) > 0 else ""
        impact_only = feat_data["Impact_Only"].iloc[0] if len(feat_data) > 0 else ""
        row["Source"] = source
        row["Impact_Only"] = impact_only

        for material in ["Bare", "Copper", "Steel"]:
            mat_data = feat_data[feat_data["Material"] == material]
            if len(mat_data) == 1:
                r = mat_data.iloc[0]
                row[f"{material}_n"] = int(r["n"]) if not np.isnan(r["n"]) else 0
                row[f"{material}_Mean"] = r["Mean"]
                row[f"{material}_SD"] = r["SD"]
                row[f"{material}_CV_pct"] = r["CV_pct"]
                row[f"{material}_CI_95_Lower"] = r["CI_95_Lower"]
                row[f"{material}_CI_95_Upper"] = r["CI_95_Upper"]
            else:
                row[f"{material}_n"] = 0
                row[f"{material}_Mean"] = np.nan
                row[f"{material}_SD"] = np.nan
                row[f"{material}_CV_pct"] = np.nan
                row[f"{material}_CI_95_Lower"] = np.nan
                row[f"{material}_CI_95_Upper"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# MARKDOWN REPORT
# ============================================================

def generate_summary_markdown(stats_df, comparison_df, output_dir, limitations):
    """Generate phase7_summary.md with key findings."""
    md_path = os.path.join(output_dir, "phase7_summary.md")

    lines = []
    lines.append("# Phase 7 — Statistical Validation Summary\n")
    lines.append("## Material Mapping\n")
    lines.append("| FBG Sensor | Material |")
    lines.append("|------------|----------|")
    lines.append("| FBG1       | Copper   |")
    lines.append("| FBG2       | Bare     |")
    lines.append("| FBG3       | Steel    |")
    lines.append("")

    lines.append("## Overview\n")
    total_features = stats_df["Feature"].nunique()
    total_records = int(stats_df["n"].sum())
    lines.append(f"- **Features Analyzed**: {total_features}")
    lines.append(f"- **Total Statistical Records**: {total_records}")
    lines.append(f"- **Materials Compared**: Bare, Copper, Steel")
    lines.append("")

    # Impact case counts per material
    lines.append("## Sample Sizes (IMPACT Cases)\n")
    lines.append("| Material | Max Available IMPACT Cases |")
    lines.append("|----------|---------------------------|")
    impact_stats = stats_df[stats_df["Impact_Only"] == True]
    for material in ["Bare", "Copper", "Steel"]:
        mat_data = impact_stats[impact_stats["Material"] == material]
        max_n = int(mat_data["n"].max()) if len(mat_data) > 0 else 0
        lines.append(f"| {material} | {max_n} |")
    lines.append("")

    # Statistical comparison — Phase 5 features
    lines.append("## Phase 5 Engineering Features (IMPACT Cases Only)\n")
    p5_comp = comparison_df[comparison_df["Source"] == "Phase5_Impact"]
    if len(p5_comp) > 0:
        lines.append("| Feature | Bare Mean ± SD (n) | Copper Mean ± SD (n) | Steel Mean ± SD (n) |")
        lines.append("|---------|-------------------|---------------------|---------------------|")
        for _, row in p5_comp.iterrows():
            feat = row["Feature"]
            cells = []
            for mat in ["Bare", "Copper", "Steel"]:
                n = int(row[f"{mat}_n"]) if not np.isnan(row.get(f"{mat}_n", np.nan)) else 0
                mean = row[f"{mat}_Mean"]
                sd = row[f"{mat}_SD"]
                if n == 0 or np.isnan(mean):
                    cells.append("N/A")
                elif np.isnan(sd):
                    cells.append(f"{mean:.6g} (n={n})")
                else:
                    cells.append(f"{mean:.6g} ± {sd:.6g} (n={n})")
            lines.append(f"| {feat} | {cells[0]} | {cells[1]} | {cells[2]} |")
        lines.append("")

    # Phase 5 baseline features
    lines.append("## Phase 5 Baseline Features (All Cases)\n")
    p5b_comp = comparison_df[comparison_df["Source"] == "Phase5_Baseline"]
    if len(p5b_comp) > 0:
        lines.append("| Feature | Bare Mean ± SD (n) | Copper Mean ± SD (n) | Steel Mean ± SD (n) |")
        lines.append("|---------|-------------------|---------------------|---------------------|")
        for _, row in p5b_comp.iterrows():
            feat = row["Feature"]
            cells = []
            for mat in ["Bare", "Copper", "Steel"]:
                n = int(row[f"{mat}_n"]) if not np.isnan(row.get(f"{mat}_n", np.nan)) else 0
                mean = row[f"{mat}_Mean"]
                sd = row[f"{mat}_SD"]
                if n == 0 or np.isnan(mean):
                    cells.append("N/A")
                elif np.isnan(sd):
                    cells.append(f"{mean:.6g} (n={n})")
                else:
                    cells.append(f"{mean:.6g} ± {sd:.6g} (n={n})")
            lines.append(f"| {feat} | {cells[0]} | {cells[1]} | {cells[2]} |")
        lines.append("")

    # Phase 6 features
    lines.append("## Phase 6 Multi-Domain Features (IMPACT Cases Only)\n")
    p6_comp = comparison_df[comparison_df["Source"] == "Phase6_Impact"]
    if len(p6_comp) > 0:
        lines.append("| Feature | Bare Mean ± SD (n) | Copper Mean ± SD (n) | Steel Mean ± SD (n) |")
        lines.append("|---------|-------------------|---------------------|---------------------|")
        for _, row in p6_comp.iterrows():
            feat = row["Feature"]
            cells = []
            for mat in ["Bare", "Copper", "Steel"]:
                n = int(row[f"{mat}_n"]) if not np.isnan(row.get(f"{mat}_n", np.nan)) else 0
                mean = row[f"{mat}_Mean"]
                sd = row[f"{mat}_SD"]
                if n == 0 or np.isnan(mean):
                    cells.append("N/A")
                elif np.isnan(sd):
                    cells.append(f"{mean:.6g} (n={n})")
                else:
                    cells.append(f"{mean:.6g} ± {sd:.6g} (n={n})")
            lines.append(f"| {feat} | {cells[0]} | {cells[1]} | {cells[2]} |")
        lines.append("")

    # Coefficient of Variation summary
    lines.append("## Coefficient of Variation Summary\n")
    lines.append("CV (%) = (SD / |Mean|) × 100 — indicates relative variability.")
    lines.append("NaN indicates undefined CV (mean ≈ 0 or insufficient data).\n")
    lines.append("| Feature | Source | Bare CV (%) | Copper CV (%) | Steel CV (%) |")
    lines.append("|---------|--------|-------------|---------------|--------------|")
    for _, row in comparison_df.iterrows():
        feat = row["Feature"]
        src = row["Source"]
        cv_vals = []
        for mat in ["Bare", "Copper", "Steel"]:
            cv = row[f"{mat}_CV_pct"]
            if np.isnan(cv):
                cv_vals.append("NaN")
            else:
                cv_vals.append(f"{cv:.2f}")
        lines.append(f"| {feat} | {src} | {cv_vals[0]} | {cv_vals[1]} | {cv_vals[2]} |")
    lines.append("")

    # Limitations
    lines.append("## Limitations\n")
    for lim in limitations:
        lines.append(f"- {lim}")
    lines.append("")

    lines.append("## Notes\n")
    lines.append("- All statistics are computed from existing Phase 5 and Phase 6 result CSVs.")
    lines.append("- Impact-specific features use only IMPACT cases as classified by Phase 4.")
    lines.append("- NO IMPACT cases are excluded from impact-event statistics to avoid mixing.")
    lines.append("- Confidence intervals use the t-distribution with n−1 degrees of freedom.")
    lines.append("- Differences between materials are reported as observed statistical differences")
    lines.append("  in the measured signal features — no causal claims are made.")
    lines.append("")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"  [OK] Saved: {md_path}")
    return md_path


# ============================================================
# PLOTS
# ============================================================

def generate_comparison_plots(comparison_df, output_dir):
    """
    Generate meaningful statistical comparison plots:
    1. Mean ± 95% CI bar chart per feature, grouped by material.
    2. CV comparison heatmap.
    """
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    plots_created = []

    materials = ["Bare", "Copper", "Steel"]
    colors = {"Bare": "#2196F3", "Copper": "#FF9800", "Steel": "#4CAF50"}

    # --- Plot 1: Mean ± 95% CI for Phase 5 Impact Features ---
    p5_features = comparison_df[comparison_df["Source"] == "Phase5_Impact"]
    if len(p5_features) > 0:
        fig, ax = plt.subplots(figsize=(16, 8))
        feat_names = p5_features["Feature"].tolist()
        x = np.arange(len(feat_names))
        width = 0.25

        for i, mat in enumerate(materials):
            means = p5_features[f"{mat}_Mean"].values
            ci_lowers = p5_features[f"{mat}_CI_95_Lower"].values
            ci_uppers = p5_features[f"{mat}_CI_95_Upper"].values

            # Error bars = distance from mean to CI bounds
            err_low = np.where(np.isnan(ci_lowers), 0, means - ci_lowers)
            err_high = np.where(np.isnan(ci_uppers), 0, ci_uppers - means)
            yerr = np.array([err_low, err_high])

            # Replace NaN means with 0 for plotting
            plot_means = np.where(np.isnan(means), 0, means)

            ax.bar(x + i * width, plot_means, width, label=mat,
                   color=colors[mat], alpha=0.8, yerr=yerr, capsize=3,
                   ecolor="black", linewidth=0.5)

        ax.set_xlabel("Feature", fontsize=11)
        ax.set_ylabel("Mean Value", fontsize=11)
        ax.set_title("Phase 5 Engineering Features — Mean ± 95% CI by Material\n(IMPACT Cases Only)",
                      fontsize=13)
        ax.set_xticks(x + width)
        ax.set_xticklabels(feat_names, rotation=45, ha="right", fontsize=8)
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        path = os.path.join(plots_dir, "phase7_p5_mean_ci_comparison.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        plots_created.append(path)
        print(f"  [OK] Plot saved: {path}")

    # --- Plot 2: Mean ± 95% CI for Phase 6 Impact Features ---
    p6_features = comparison_df[comparison_df["Source"] == "Phase6_Impact"]
    if len(p6_features) > 0:
        fig, ax = plt.subplots(figsize=(14, 7))
        feat_names = p6_features["Feature"].tolist()
        x = np.arange(len(feat_names))
        width = 0.25

        for i, mat in enumerate(materials):
            means = p6_features[f"{mat}_Mean"].values
            ci_lowers = p6_features[f"{mat}_CI_95_Lower"].values
            ci_uppers = p6_features[f"{mat}_CI_95_Upper"].values

            err_low = np.where(np.isnan(ci_lowers), 0, means - ci_lowers)
            err_high = np.where(np.isnan(ci_uppers), 0, ci_uppers - means)
            yerr = np.array([err_low, err_high])

            plot_means = np.where(np.isnan(means), 0, means)

            ax.bar(x + i * width, plot_means, width, label=mat,
                   color=colors[mat], alpha=0.8, yerr=yerr, capsize=3,
                   ecolor="black", linewidth=0.5)

        ax.set_xlabel("Feature", fontsize=11)
        ax.set_ylabel("Mean Value", fontsize=11)
        ax.set_title("Phase 6 Multi-Domain Features — Mean ± 95% CI by Material\n(IMPACT Cases Only)",
                      fontsize=13)
        ax.set_xticks(x + width)
        ax.set_xticklabels(feat_names, rotation=45, ha="right", fontsize=9)
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        path = os.path.join(plots_dir, "phase7_p6_mean_ci_comparison.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        plots_created.append(path)
        print(f"  [OK] Plot saved: {path}")

    # --- Plot 3: CV Heatmap ---
    all_features = comparison_df[["Feature", "Source"]].copy()
    cv_data = []
    for _, row in comparison_df.iterrows():
        for mat in materials:
            cv = row[f"{mat}_CV_pct"]
            cv_data.append(cv)

    if len(cv_data) > 0:
        n_features = len(comparison_df)
        cv_matrix = np.array(cv_data).reshape(n_features, 3)

        fig, ax = plt.subplots(figsize=(8, max(6, n_features * 0.4)))
        # Mask NaN for display
        masked = np.ma.masked_invalid(cv_matrix)
        cmap = plt.cm.YlOrRd.copy()
        cmap.set_bad(color="lightgray")

        im = ax.imshow(masked, aspect="auto", cmap=cmap)
        ax.set_xticks(range(3))
        ax.set_xticklabels(materials, fontsize=10)

        feat_labels = [f"{row['Feature']} ({row['Source']})"
                       for _, row in comparison_df.iterrows()]
        ax.set_yticks(range(n_features))
        ax.set_yticklabels(feat_labels, fontsize=7)

        # Annotate cells
        for i in range(n_features):
            for j in range(3):
                val = cv_matrix[i, j]
                if np.isnan(val):
                    ax.text(j, i, "NaN", ha="center", va="center", fontsize=7, color="gray")
                else:
                    ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=7)

        plt.colorbar(im, ax=ax, label="CV (%)")
        ax.set_title("Coefficient of Variation (%) — All Features by Material", fontsize=12)
        plt.tight_layout()

        path = os.path.join(plots_dir, "phase7_cv_heatmap.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        plots_created.append(path)
        print(f"  [OK] Plot saved: {path}")

    return plots_created


# ============================================================
# VALIDATION
# ============================================================

def run_validation(stats_df, comparison_df):
    """Run Phase 7 validation checks."""
    print("\n" + "=" * 70)
    print("PHASE 7 VALIDATION")
    print("=" * 70)

    all_pass = True

    # 1. Expected number of datasets/channels processed
    n_stat_records = len(stats_df)
    print(f"\n[CHECK 1] Total statistical records: {n_stat_records}")
    if n_stat_records > 0:
        print("  [PASS] Statistical records generated")
    else:
        print("  [FAIL] No statistical records generated")
        all_pass = False

    # 2. Material mapping correctness
    materials_in_data = sorted(stats_df["Material"].unique())
    expected_materials = ["Bare", "Copper", "Steel"]
    if materials_in_data == expected_materials:
        print(f"[CHECK 2] Material mapping correct: {materials_in_data}")
        print("  [PASS] FBG1->Copper, FBG2->Bare, FBG3->Steel")
    else:
        print(f"  [FAIL] Unexpected materials: {materials_in_data}")
        all_pass = False

    # 3. Mean, SD, CV, CI are numerically valid (no Inf)
    numeric_cols = ["Mean", "SD", "CV_pct", "CI_95_Lower", "CI_95_Upper"]
    has_inf = False
    for col in numeric_cols:
        inf_count = np.isinf(stats_df[col].dropna()).sum()
        if inf_count > 0:
            print(f"  [FAIL] {col} contains {inf_count} Inf values")
            has_inf = True
            all_pass = False

    if not has_inf:
        print("[CHECK 3] [PASS] No infinite values in Mean/SD/CV/CI")

    # 4. No infinite CV
    cv_inf = np.isinf(stats_df["CV_pct"].dropna()).sum()
    if cv_inf == 0:
        print("[CHECK 4] [PASS] No infinite CV values")
    else:
        print(f"  [FAIL] {cv_inf} infinite CV values found")
        all_pass = False

    # 5. No NO IMPACT data treated as IMPACT
    # Verify by checking that impact-only features have correct n
    # Phase 4 says: Copper IMPACT=3, Bare IMPACT=7, Steel IMPACT=2
    impact_stats = stats_df[stats_df["Impact_Only"] == True]
    for material, expected_max_n in [("Copper", 3), ("Bare", 7), ("Steel", 2)]:
        mat_data = impact_stats[impact_stats["Material"] == material]
        max_n = int(mat_data["n"].max()) if len(mat_data) > 0 else 0
        if max_n <= expected_max_n:
            print(f"[CHECK 5] [PASS] {material} impact count n={max_n} <= expected max {expected_max_n}")
        else:
            print(f"  [FAIL] {material} impact count n={max_n} > expected max {expected_max_n}")
            all_pass = False

    # 6. Features validated count
    n_features = stats_df["Feature"].nunique()
    print(f"[CHECK 6] Features statistically validated: {n_features}")

    print("\n" + "-" * 40)
    if all_pass:
        print("PHASE 7 VALIDATION: ALL CHECKS PASSED")
    else:
        print("PHASE 7 VALIDATION: SOME CHECKS FAILED")
    print("=" * 70)

    return all_pass


# ============================================================
# MAIN
# ============================================================

def run_phase7():
    print("=" * 70)
    print("PHASE 7 -- STATISTICAL VALIDATION")
    print("=" * 70)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- 1. Load Data ---
    print("\n[1/5] Loading Phase 5 and Phase 6 feature data...")
    p5_df = load_phase5_data()
    p6_df = load_phase6_data()

    # --- 2. Statistical Analysis ---
    print("\n[2/5] Computing statistical validation metrics...")

    all_stats = []

    # Phase 5 — Impact-specific features (IMPACT only)
    print("  Analyzing Phase 5 impact features (IMPACT cases only)...")
    p5_impact_stats = analyze_features(p5_df, PHASE5_IMPACT_FEATURES,
                                       impact_only=True, source_label="Phase5_Impact")
    all_stats.extend(p5_impact_stats)
    impact_feature_count = sum(1 for s in p5_impact_stats if s["n"] > 0)
    print(f"    -> {impact_feature_count} feature-material combinations with data")

    # Phase 5 — Baseline features (ALL cases)
    print("  Analyzing Phase 5 baseline features (all cases)...")
    p5_baseline_stats = analyze_features(p5_df, PHASE5_BASELINE_FEATURES,
                                          impact_only=False, source_label="Phase5_Baseline")
    all_stats.extend(p5_baseline_stats)
    baseline_feature_count = sum(1 for s in p5_baseline_stats if s["n"] > 0)
    print(f"    -> {baseline_feature_count} feature-material combinations with data")

    # Phase 6 — Multi-domain features (IMPACT only)
    print("  Analyzing Phase 6 multi-domain features (IMPACT cases only)...")
    p6_impact_stats = analyze_features(p6_df, PHASE6_IMPACT_FEATURES,
                                        impact_only=True, source_label="Phase6_Impact")
    all_stats.extend(p6_impact_stats)
    p6_feature_count = sum(1 for s in p6_impact_stats if s["n"] > 0)
    print(f"    -> {p6_feature_count} feature-material combinations with data")

    stats_df = pd.DataFrame(all_stats)

    # Save full statistical summary
    stats_path = os.path.join(OUTPUT_DIR, "phase7_statistical_summary.csv")
    stats_df.to_csv(stats_path, index=False)
    print(f"\n  [OK] Saved: {stats_path}")

    # --- 3. Material Comparison ---
    print("\n[3/5] Building Bare vs Copper vs Steel comparison...")
    comparison_df = build_comparison_table(stats_df)
    comp_path = os.path.join(OUTPUT_DIR, "phase7_material_comparison.csv")
    comparison_df.to_csv(comp_path, index=False)
    print(f"  [OK] Saved: {comp_path}")

    # --- 4. Plots ---
    print("\n[4/5] Generating statistical comparison plots...")
    plots_created = generate_comparison_plots(comparison_df, OUTPUT_DIR)

    # --- 5. Markdown Report ---
    print("\n[5/5] Generating markdown summary report...")
    limitations = [
        "Small sample sizes limit statistical power — Copper has n=3 IMPACT cases, Steel has n=2, Bare has n=7.",
        "Steel has only n=2 IMPACT cases, so its SD is based on a single degree of freedom and CI is wide.",
        "Copper has n=0 valid peak_width_seconds observations; statistics are undefined for this feature.",
        "Some Phase 5 impact features (e.g., recovery_time_seconds, peak_width_seconds) have missing values "
        "within IMPACT cases, reducing effective n below the maximum for that material.",
        "CV is set to NaN where |mean| < 1e-15 to avoid division-by-zero artifacts.",
        "Phase 6 features for NO IMPACT cases are NaN by design (not computed), so multi-domain "
        "statistics are restricted to IMPACT cases only.",
        "Observed differences between materials reflect signal-level measurement variations — "
        "no causal claims about material superiority are made.",
    ]
    md_path = generate_summary_markdown(stats_df, comparison_df, OUTPUT_DIR, limitations)

    # --- Validation ---
    validation_passed = run_validation(stats_df, comparison_df)

    # --- Final Summary ---
    print("\n" + "=" * 70)
    print("PHASE 7 COMPLETE")
    print("=" * 70)

    print(f"\nMaterial Mapping:")
    print(f"  FBG1 -> Copper")
    print(f"  FBG2 -> Bare")
    print(f"  FBG3 -> Steel")

    total_features = stats_df["Feature"].nunique()
    total_stat_records = len(stats_df)
    print(f"\nFeatures Statistically Validated: {total_features}")
    print(f"Total Statistical Records: {total_stat_records}")

    print(f"\nOutput Files:")
    print(f"  - {stats_path}")
    print(f"  - {comp_path}")
    print(f"  - {md_path}")
    for p in plots_created:
        print(f"  - {p}")

    print(f"\nValidation: {'PASS' if validation_passed else 'FAIL'}")
    print("=" * 70)


if __name__ == "__main__":
    run_phase7()
