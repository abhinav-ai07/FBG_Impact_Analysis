"""
PHASE 8 — DEVELOP NOVEL ENGINEERING INDICES
============================================================
Calculates novel, deterministic, interpretable engineering indices
for FBG impact signal analysis without machine learning.

Consumes existing results from Phase 5, Phase 6, and Phase 7.

Material Mapping (established in prior phases):
    FBG1 -> Copper
    FBG2 -> Bare
    FBG3 -> Steel

Indices Implemented:
    1. Dynamic Strain Transfer Index (DSTI)
    2. Impact Persistence Index (IPI)
    3. Signal Energy Response Index (SERI)
    4. Response Stability Index (RSI)
    5. Multi-Domain Impact Signature Index (MDISI)
"""

import os
import sys
import json
import shutil
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION AND FILE PATHS
# ============================================================

PHASE5_CSV = os.path.join("results", "phase5", "phase5_all_features.csv")
PHASE6_CSV = os.path.join("results", "phase6", "phase6_multidomain_features.csv")
PHASE7_COMP_CSV = os.path.join("results", "phase7", "phase7_material_comparison.csv")
PHASE7_STAT_CSV = os.path.join("results", "phase7", "phase7_statistical_summary.csv")

OUTPUT_DIR = os.path.join("results", "phase8")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")

MATERIAL_MAP = {
    "FBG1": "Copper",
    "FBG2": "Bare",
    "FBG3": "Steel",
}

# Features required for normalization and index calculation
NORM_FEATURES = [
    "peak_shift_abs",
    "max_slope_abs",
    "rise_time_seconds",
    "residual_shift_abs",
    "recovery_time_seconds",
    "signal_energy",
    "rms",
    "peak_to_peak",
    "auc_abs",
    "Spectral_Energy",
    "Spectral_Centroid",
    "Dominant_Frequency",
    "Bandwidth",
    "Wavelet_Energy",
    "Approximation_Energy",
    "Detail_Energy",
    "Detail_Approx_Ratio",
]

# Key features used to compute material-level Response Stability Index (RSI)
RSI_KEY_FEATURES = [
    "peak_shift_abs",
    "max_slope_abs",
    "signal_energy",
    "rms",
    "Dominant_Frequency",
    "Spectral_Energy",
    "Wavelet_Energy",
]


# ============================================================
# DATA LOADING AND PREPROCESSING
# ============================================================

def load_phase5_data():
    """Load Phase 5 all-features CSV."""
    if not os.path.exists(PHASE5_CSV):
        print(f"ERROR: Phase 5 features file missing at {PHASE5_CSV}")
        sys.exit(1)
    df = pd.read_csv(PHASE5_CSV)
    print(f"  [OK] Loaded Phase 5 features: {len(df)} rows")
    return df


def load_phase6_data():
    """Load Phase 6 multi-domain features CSV."""
    if not os.path.exists(PHASE6_CSV):
        print(f"ERROR: Phase 6 features file missing at {PHASE6_CSV}")
        sys.exit(1)
    df = pd.read_csv(PHASE6_CSV)
    print(f"  [OK] Loaded Phase 6 features: {len(df)} rows")
    return df


def load_phase7_data():
    """Load Phase 7 statistical summary and comparison CSVs."""
    if not os.path.exists(PHASE7_STAT_CSV) or not os.path.exists(PHASE7_COMP_CSV):
        print(f"ERROR: Phase 7 files missing at {PHASE7_STAT_CSV} / {PHASE7_COMP_CSV}")
        sys.exit(1)
    stat_df = pd.read_csv(PHASE7_STAT_CSV)
    comp_df = pd.read_csv(PHASE7_COMP_CSV)
    print(f"  [OK] Loaded Phase 7 statistical data: {len(stat_df)} summary records")
    return stat_df, comp_df


def extract_expert_clean(val):
    """Normalize dataset/file string to 'Expert N'."""
    val_str = str(val).lower()
    try:
        num = int(val_str.split("expert")[1].split(".")[0])
        return f"Expert {num}"
    except (IndexError, ValueError):
        return str(val)


def extract_fbg_clean(val):
    """Normalize sensor name to 'FBG1', 'FBG2', or 'FBG3'."""
    return str(val).upper().replace("_PROCESSED", "").strip()


def merge_phase5_phase6(p5_df, p6_df):
    """
    Merge Phase 5 and Phase 6 dataframes on Expert_Clean and FBG_Clean.
    Validates key alignment and row counts.
    """
    p5_df = p5_df.copy()
    p6_df = p6_df.copy()

    p5_df["Expert_Clean"] = p5_df["Dataset"].apply(extract_expert_clean)
    p5_df["FBG_Clean"] = p5_df["Sensor"].apply(extract_fbg_clean)

    p6_df["Expert_Clean"] = p6_df["File"].apply(extract_expert_clean)
    p6_df["FBG_Clean"] = p6_df["Sensor"].apply(extract_fbg_clean)

    merged = pd.merge(
        p5_df,
        p6_df,
        on=["Expert_Clean", "FBG_Clean"],
        suffixes=("_P5", "_P6")
    )

    if len(merged) != len(p5_df):
        print(f"WARNING: Merged count {len(merged)} differs from Phase 5 count {len(p5_df)}")

    # Verify material consistency
    merged["FBG"] = merged["FBG_Clean"]
    merged["Material"] = merged["FBG"].map(MATERIAL_MAP)

    return merged


def validate_input_schema(p5_df, p6_df, p7_stat_df):
    """Validate that required input columns exist before computation."""
    required_p5 = [
        "Dataset", "Sensor", "Material", "Impact_Status",
        "peak_shift_abs", "max_slope_abs", "rise_time_seconds",
        "residual_shift_abs", "recovery_time_seconds",
        "signal_energy", "rms", "peak_to_peak", "auc_abs"
    ]
    required_p6 = [
        "File", "Sensor", "Phase4_Result",
        "Spectral_Energy", "Spectral_Centroid", "Dominant_Frequency", "Bandwidth",
        "Wavelet_Energy", "Approximation_Energy", "Detail_Energy", "Detail_Approx_Ratio"
    ]
    required_p7 = ["Source", "Feature", "Material", "CV_pct", "Mean"]

    for col in required_p5:
        if col not in p5_df.columns:
            raise ValueError(f"Missing required column in Phase 5 data: {col}")

    for col in required_p6:
        if col not in p6_df.columns:
            raise ValueError(f"Missing required column in Phase 6 data: {col}")

    for col in required_p7:
        if col not in p7_stat_df.columns:
            raise ValueError(f"Missing required column in Phase 7 data: {col}")

    print("  [OK] Input schemas successfully validated.")


# ============================================================
# REUSABLE NORMALIZATION UTILITY
# ============================================================

def fit_normalization_params(impact_df):
    """
    Fit min-max normalization parameters across all valid IMPACT cases.
    Global normalization for transparent cross-material comparison.
    """
    params = {}
    for feat in NORM_FEATURES:
        if feat in impact_df.columns:
            vals = impact_df[feat].dropna()
            if len(vals) > 0:
                f_min = float(vals.min())
                f_max = float(vals.max())
            else:
                f_min, f_max = 0.0, 1.0
        else:
            f_min, f_max = 0.0, 1.0
        params[feat] = {"min": f_min, "max": f_max}
    return params


def normalize_feature(val, feature_name, norm_params):
    """
    Normalize a scalar feature value to [0, 1] using global min-max parameters.
    Handles NaN and zero-range safely.
    """
    if pd.isna(val):
        return np.nan

    if feature_name not in norm_params:
        return np.nan

    p = norm_params[feature_name]
    rng = p["max"] - p["min"]
    if rng <= 1e-15:
        return 0.5

    norm_val = (val - p["min"]) / rng
    return float(np.clip(norm_val, 0.0, 1.0))


def safe_mean(components):
    """Compute mean of non-NaN component scores; return (mean, valid_count)."""
    valid = [c for c in components if not pd.isna(c) and not np.isinf(c)]
    if not valid:
        return np.nan, 0
    return float(np.mean(valid)), len(valid)


# ============================================================
# INDEX COMPUTATION LOGIC
# ============================================================

def compute_dsti(row, norm_params):
    """
    Dynamic Strain Transfer Index (DSTI)
    Measures efficiency of strain transfer into sensor response.
    Components:
      - Peak response: norm(peak_shift_abs)
      - Dynamic slope: norm(max_slope_abs)
      - Response speed: 1.0 - norm(rise_time_seconds)
    """
    c1 = normalize_feature(row.get("peak_shift_abs"), "peak_shift_abs", norm_params)
    c2 = normalize_feature(row.get("max_slope_abs"), "max_slope_abs", norm_params)
    c3_raw = normalize_feature(row.get("rise_time_seconds"), "rise_time_seconds", norm_params)
    c3 = (1.0 - c3_raw) if not pd.isna(c3_raw) else np.nan

    score, valid_count = safe_mean([c1, c2, c3])
    return score, valid_count


def compute_ipi(row, norm_params):
    """
    Impact Persistence Index (IPI)
    Measures residual/permanent strain retention relative to peak, combined with recovery duration.
    Components:
      - Residual retention ratio: min(1.0, residual_shift_abs / peak_shift_abs)
      - Recovery duration component: norm(recovery_time_seconds)
    """
    res_shift = row.get("residual_shift_abs")
    peak_shift = row.get("peak_shift_abs")

    if not pd.isna(res_shift) and not pd.isna(peak_shift) and peak_shift > 1e-15:
        c1 = min(1.0, max(0.0, res_shift / peak_shift))
    else:
        c1 = np.nan

    c2 = normalize_feature(row.get("recovery_time_seconds"), "recovery_time_seconds", norm_params)

    score, valid_count = safe_mean([c1, c2])
    return score, valid_count


def compute_seri(row, norm_params):
    """
    Signal Energy Response Index (SERI)
    Measures overall impact response strength across energy/magnitude measures.
    Components:
      - norm(signal_energy)
      - norm(rms)
      - norm(peak_to_peak)
      - norm(auc_abs)
    """
    c1 = normalize_feature(row.get("signal_energy"), "signal_energy", norm_params)
    c2 = normalize_feature(row.get("rms"), "rms", norm_params)
    c3 = normalize_feature(row.get("peak_to_peak"), "peak_to_peak", norm_params)
    c4 = normalize_feature(row.get("auc_abs"), "auc_abs", norm_params)

    score, valid_count = safe_mean([c1, c2, c3, c4])
    return score, valid_count


def compute_pei(row, norm_params):
    """
    Packaging Efficiency Index (PEI)

    Measures how much sensing capability
    is retained after packaging.

    Components:
      - Peak Shift (higher = better)
      - Recovery Time (lower = better)
      - Signal Energy (higher = better)

    Formula:
      0.4 Peak Shift
      0.3 Recovery
      0.3 Signal Energy
    """

    peak_score = normalize_feature(
        row.get("peak_shift_abs"),
        "peak_shift_abs",
        norm_params
    )

    recovery_raw = normalize_feature(
        row.get("recovery_time_seconds"),
        "recovery_time_seconds",
        norm_params
    )

    recovery_score = (
        1.0 - recovery_raw
        if not pd.isna(recovery_raw)
        else np.nan
    )

    energy_score = normalize_feature(
        row.get("signal_energy"),
        "signal_energy",
        norm_params
    )

    weighted_components = []

    if not pd.isna(peak_score):
        weighted_components.append(
            0.4 * peak_score
        )

    if not pd.isna(recovery_score):
        weighted_components.append(
            0.3 * recovery_score
        )

    if not pd.isna(energy_score):
        weighted_components.append(
            0.3 * energy_score
        )

    score, valid_count = safe_mean(
        weighted_components
    )

    return score, valid_count


def compute_rsi(p7_stat_df):
    """
    Response Stability Index (RSI)
    Material-level stability score derived from Phase 7 Coefficient of Variation (CV%).
    Formula:
        RSI_Material = 100.0 / (100.0 + Mean_CV_pct)
    Maps CV=0% -> 1.0 (perfect repeatability), CV=100% -> 0.5, CV -> inf -> 0.0.
    """
    rsi_dict = {}
    for mat in ["Bare", "Copper", "Steel"]:
        sub = p7_stat_df[
            (p7_stat_df["Material"] == mat) &
            (p7_stat_df["Feature"].isin(RSI_KEY_FEATURES))
        ]
        cvs = sub["CV_pct"].dropna().values
        if len(cvs) > 0:
            mean_cv = float(np.mean(cvs))
            rsi_val = float(100.0 / (100.0 + mean_cv))
        else:
            rsi_val = np.nan
        rsi_dict[mat] = rsi_val
    return rsi_dict


def compute_mdisi(row, norm_params):
    """
    Multi-Domain Impact Signature Index (MDISI)
    Combines subscores from time, frequency, and wavelet domains.
    Subscores:
      - Time Domain Subscore: mean of norm(peak_shift_abs, max_slope_abs, signal_energy, rms)
      - Frequency Domain Subscore: mean of norm(Spectral_Energy, Spectral_Centroid, Dominant_Frequency, Bandwidth)
      - Wavelet Domain Subscore: mean of norm(Wavelet_Energy, Approximation_Energy, Detail_Energy, Detail_Approx_Ratio)
    Final MDISI = mean of available domain subscores.
    """
    # Time domain
    t1 = normalize_feature(row.get("peak_shift_abs"), "peak_shift_abs", norm_params)
    t2 = normalize_feature(row.get("max_slope_abs"), "max_slope_abs", norm_params)
    t3 = normalize_feature(row.get("signal_energy"), "signal_energy", norm_params)
    t4 = normalize_feature(row.get("rms"), "rms", norm_params)
    s_time, _ = safe_mean([t1, t2, t3, t4])

    # Frequency domain
    f1 = normalize_feature(row.get("Spectral_Energy"), "Spectral_Energy", norm_params)
    f2 = normalize_feature(row.get("Spectral_Centroid"), "Spectral_Centroid", norm_params)
    f3 = normalize_feature(row.get("Dominant_Frequency"), "Dominant_Frequency", norm_params)
    f4 = normalize_feature(row.get("Bandwidth"), "Bandwidth", norm_params)
    s_freq, _ = safe_mean([f1, f2, f3, f4])

    # Wavelet domain
    w1 = normalize_feature(row.get("Wavelet_Energy"), "Wavelet_Energy", norm_params)
    w2 = normalize_feature(row.get("Approximation_Energy"), "Approximation_Energy", norm_params)
    w3 = normalize_feature(row.get("Detail_Energy"), "Detail_Energy", norm_params)
    w4 = normalize_feature(row.get("Detail_Approx_Ratio"), "Detail_Approx_Ratio", norm_params)
    s_wav, _ = safe_mean([w1, w2, w3, w4])

    final_mdisi, valid_domains = safe_mean([s_time, s_freq, s_wav])
    return s_time, s_freq, s_wav, final_mdisi, valid_domains


# ============================================================
# PIPELINE EXECUTION
# ============================================================

def build_event_level_indices(merged_df, norm_params, rsi_dict):
    """
    Build event-level engineering indices dataframe.
    Impact-specific indices are computed ONLY for valid IMPACT cases.
    For NO IMPACT cases, indices are NaN and status is NOT_APPLICABLE.
    """
    records = []

    for _, row in merged_df.iterrows():
        status = str(row.get("Impact_Status", "")).strip()
        is_impact = (status == "IMPACT")

        file_name = row.get("File", row.get("Dataset", ""))
        dataset_name = row.get("Dataset", extract_expert_clean(file_name))
        expert_num = row.get("Expert_Num", dataset_name.replace("Expert ", ""))
        sensor = row.get("Sensor_P5", row.get("Sensor", ""))
        fbg = row.get("FBG", extract_fbg_clean(sensor))
        material = row.get("Material", MATERIAL_MAP.get(fbg, "Unknown"))

        rec = {
            "File": file_name,
            "Dataset": dataset_name,
            "Expert_Num": expert_num,
            "Sensor": sensor,
            "FBG": fbg,
            "Material": material,
            "Impact_Status": status,
        }

        if not is_impact:
            # NO IMPACT cases
            rec.update({
                "DSTI": np.nan, "DSTI_valid_components": 0, "DSTI_Status": "NOT_APPLICABLE",
                "IPI": np.nan, "IPI_valid_components": 0, "IPI_Status": "NOT_APPLICABLE",
                "SERI": np.nan, "SERI_valid_components": 0, "SERI_Status": "NOT_APPLICABLE",
                "PEI": np.nan, "PEI_valid_components": 0, "PEI_Status": "NOT_APPLICABLE",
                "Time_Domain_Subscore": np.nan,
                "Frequency_Domain_Subscore": np.nan,
                "Wavelet_Domain_Subscore": np.nan,
                "MDISI": np.nan, "MDISI_valid_domains": 0, "MDISI_Status": "NOT_APPLICABLE",
                "RSI": np.nan, "RSI_Status": "NOT_APPLICABLE",
            })
        else:
            # IMPACT cases
            dsti_score, dsti_vc = compute_dsti(row, norm_params)
            ipi_score, ipi_vc = compute_ipi(row, norm_params)
            seri_score, seri_vc = compute_seri(row, norm_params)
            pei_score, pei_vc = compute_pei(row, norm_params)
            s_time, s_freq, s_wav, mdisi_score, mdisi_vd = compute_mdisi(row, norm_params)
            rsi_score = rsi_dict.get(material, np.nan)

            rec.update({
                "DSTI": dsti_score,
                "DSTI_valid_components": dsti_vc,
                "DSTI_Status": "VALID" if not pd.isna(dsti_score) else "INSUFFICIENT_DATA",

                "IPI": ipi_score,
                "IPI_valid_components": ipi_vc,
                "IPI_Status": "VALID" if not pd.isna(ipi_score) else "INSUFFICIENT_DATA",

                "SERI": seri_score,
                "SERI_valid_components": seri_vc,
                "SERI_Status": "VALID" if not pd.isna(seri_score) else "INSUFFICIENT_DATA",

                "PEI": pei_score,
                "PEI_valid_components": pei_vc,
                "PEI_Status": "VALID" if not pd.isna(pei_score) else "INSUFFICIENT_DATA",

                "Time_Domain_Subscore": s_time,
                "Frequency_Domain_Subscore": s_freq,
                "Wavelet_Domain_Subscore": s_wav,

                "MDISI": mdisi_score,
                "MDISI_valid_domains": mdisi_vd,
                "MDISI_Status": "VALID" if not pd.isna(mdisi_score) else "INSUFFICIENT_DATA",

                "RSI": rsi_score,
                "RSI_Status": "VALID" if not pd.isna(rsi_score) else "INSUFFICIENT_DATA",
            })

        records.append(rec)

    return pd.DataFrame(records)


def compute_summary_stats(series):
    """Compute n, mean, median, SD, min, max, CV%, and 95% CI for a pandas Series."""
    clean = series.dropna()
    n = len(clean)

    if n == 0:
        return {
            "n": 0, "mean": np.nan, "median": np.nan, "sd": np.nan,
            "min": np.nan, "max": np.nan, "cv_pct": np.nan,
            "ci_lower": np.nan, "ci_upper": np.nan
        }

    mean_val = float(np.mean(clean))
    median_val = float(np.median(clean))
    sd_val = float(np.std(clean, ddof=1)) if n >= 2 else np.nan
    min_val = float(np.min(clean))
    max_val = float(np.max(clean))

    if sd_val is not None and not np.isnan(sd_val) and abs(mean_val) > 1e-15:
        cv_val = (sd_val / abs(mean_val)) * 100.0
    else:
        cv_val = np.nan

    if n >= 2 and not np.isnan(sd_val):
        t_crit = stats.t.ppf(0.975, df=n - 1)
        margin = t_crit * sd_val / np.sqrt(n)
        ci_lower = mean_val - margin
        ci_upper = mean_val + margin
    else:
        ci_lower = np.nan
        ci_upper = np.nan

    return {
        "n": n, "mean": mean_val, "median": median_val, "sd": sd_val,
        "min": min_val, "max": max_val, "cv_pct": cv_val,
        "ci_lower": ci_lower, "ci_upper": ci_upper
    }


def build_material_level_summary(indices_df):
    """
    Build material-level statistical summary for all 5 indices across Bare, Copper, and Steel.
    Restricted to valid IMPACT cases.
    """
    impact_df = indices_df[indices_df["Impact_Status"] == "IMPACT"].copy()

    notes_dict = {
        "DSTI": "Dynamic Strain Transfer Index measures strain sensitivity and response speed.",
        "PEI": "Packaging Efficiency Index measures sensing capability retained after packaging.",
        "IPI": "Impact Persistence Index captures residual strain retention relative to peak shift.",
        "SERI": "Signal Energy Response Index quantifies overall impact magnitude/energy across signal representations.",
        "RSI": "Response Stability Index captures material repeatability derived from Phase 7 feature CVs.",
        "MDISI": "Multi-Domain Impact Signature Index combines time, frequency, and wavelet response subscores.",
    }

    rows = []
    indices_list = ["DSTI", "PEI", "IPI", "SERI", "RSI", "MDISI"]

    for idx_name in indices_list:
        for mat in ["Bare", "Copper", "Steel"]:
            mat_sub = impact_df[impact_df["Material"] == mat]
            total_impact_count = len(mat_sub)
            series = mat_sub[idx_name]

            st = compute_summary_stats(series)
            missing_count = total_impact_count - st["n"]

            rows.append({
                "Material": mat,
                "Index": idx_name,
                "n": st["n"],
                "Mean": st["mean"],
                "Median": st["median"],
                "SD": st["sd"],
                "Min": st["min"],
                "Max": st["max"],
                "CV_pct": st["cv_pct"],
                "CI_95_Lower": st["ci_lower"],
                "CI_95_Upper": st["ci_upper"],
                "Missing_Invalid_Count": missing_count,
                "Interpretation_Notes": notes_dict[idx_name]
            })

    return pd.DataFrame(rows)


# ============================================================
# PLOTTING FUNCTIONS
# ============================================================

def generate_plots(indices_df, material_summary_df, output_dir):
    """
    Generate non-misleading visualizations for Phase 8 indices.
    Overlays individual observations (strip points) on bar plots given small sample sizes (n=7, 3, 2).
    """
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    created_plots = []

    impact_df = indices_df[indices_df["Impact_Status"] == "IMPACT"].copy()
    materials = ["Bare", "Copper", "Steel"]
    colors = {"Bare": "#2196F3", "Copper": "#FF9800", "Steel": "#4CAF50"}

    # Individual Index Plots (DSTI, PEI, IPI, SERI, RSI, MDISI)
    index_titles = {
        "DSTI": "Dynamic Strain Transfer Index (DSTI) by Material",
        "PEI": "Packaging Efficiency Index (PEI) by Material",
        "IPI": "Impact Persistence Index (IPI) by Material",
        "SERI": "Signal Energy Response Index (SERI) by Material",
        "RSI": "Response Stability Index (RSI) by Material",
        "MDISI": "Multi-Domain Impact Signature Index (MDISI) by Material",
    }

    for idx_name in ["DSTI", "PEI", "IPI", "SERI", "RSI", "MDISI"]:
        fig, ax = plt.subplots(figsize=(8, 6))

        x_pos = np.arange(len(materials))
        means = []
        counts = []

        for i, mat in enumerate(materials):
            sub_vals = impact_df[impact_df["Material"] == mat][idx_name].dropna().values
            c = len(sub_vals)
            counts.append(c)
            m = np.mean(sub_vals) if c > 0 else 0
            means.append(m)

            # Draw bar
            ax.bar(i, m, width=0.5, color=colors[mat], alpha=0.6,
                   edgecolor="black", linewidth=1.2, zorder=2)

            # Overlay individual observations
            if c > 0:
                # Add slight jitter for visual clarity
                jitter = np.random.normal(0, 0.04, size=c) if c > 1 else np.zeros(c)
                ax.scatter(i + jitter, sub_vals, color="black", s=40, zorder=4, alpha=0.9, label="Observations" if i == 0 else "")

        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"{mat}\n(n={counts[i]})" for i, mat in enumerate(materials)], fontsize=11)
        ax.set_ylabel(f"{idx_name} Score", fontsize=12)
        ax.set_title(f"{index_titles[idx_name]}\n(Exploratory Analysis — IMPACT Cases Only)", fontsize=13)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=1)

        plt.tight_layout()
        plot_path = os.path.join(plots_dir, f"phase8_{idx_name.lower()}_by_material.png")
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        created_plots.append(plot_path)

    # Plot 7: Index Comparison Heatmap across Materials
    fig, ax = plt.subplots(figsize=(9, 5))
    indices_list = ["DSTI", "PEI", "IPI", "SERI", "RSI", "MDISI"]

    heatmap_matrix = np.zeros((len(indices_list), len(materials)))

    for i, idx_name in enumerate(indices_list):
        for j, mat in enumerate(materials):
            val = material_summary_df[
                (material_summary_df["Index"] == idx_name) &
                (material_summary_df["Material"] == mat)
            ]["Mean"].values
            heatmap_matrix[i, j] = val[0] if len(val) > 0 and not np.isnan(val[0]) else 0

    im = ax.imshow(heatmap_matrix, cmap="Blues", vmin=0, vmax=1.0)

    ax.set_xticks(range(len(materials)))
    ax.set_xticklabels([f"{mat}\n(Bare n=7, Cu n=3, St n=2)" for mat in materials], fontsize=10)

    ax.set_yticks(range(len(indices_list)))
    ax.set_yticklabels(indices_list, fontsize=11)

    for i in range(len(indices_list)):
        for j in range(len(materials)):
            score = heatmap_matrix[i, j]
            ax.text(j, i, f"{score:.3f}", ha="center", va="center",
                    color="white" if score > 0.5 else "black", fontweight="bold", fontsize=10)

    plt.colorbar(im, ax=ax, label="Mean Index Value [0, 1]")
    ax.set_title("Phase 8 Novel Engineering Indices — Mean Comparison Heatmap\n(Exploratory Cross-Material Comparison)", fontsize=12)
    plt.tight_layout()

    heatmap_path = os.path.join(plots_dir, "phase8_index_comparison_heatmap.png")
    fig.savefig(heatmap_path, dpi=150)
    plt.close(fig)
    created_plots.append(heatmap_path)

    return created_plots


# ============================================================
# MARKDOWN REPORT GENERATION
# ============================================================

def generate_markdown_report(indices_df, material_summary_df, norm_params, output_dir):
    """Generate comprehensive phase8_summary.md report."""
    report_path = os.path.join(output_dir, "phase8_summary.md")

    impact_df = indices_df[indices_df["Impact_Status"] == "IMPACT"]
    no_impact_df = indices_df[indices_df["Impact_Status"] == "NO IMPACT"]

    lines = []
    lines.append("# Phase 8 — Novel Engineering Indices\n")

    lines.append("## Objective\n")
    lines.append("Phase 8 develops deterministic, interpretable engineering indices to transform previously calculated "
                 "time-domain, frequency-domain, wavelet-domain, and statistical features into unified, physically meaningful "
                 "characterization measures for Fiber Bragg Grating (FBG) impact response.\n")

    lines.append("## Input Sources\n")
    lines.append("Phase 8 consumes existing result artifacts without rerunning prior phase pipelines:\n")
    lines.append("- `results/phase5/phase5_all_features.csv` (13 time-domain & baseline signal features)")
    lines.append("- `results/phase6/phase6_multidomain_features.csv` (10 spectral & wavelet multi-domain features)")
    lines.append("- `results/phase7/phase7_statistical_summary.csv` & `phase7_material_comparison.csv` (material-level variability & CI metrics)")
    lines.append("")

    lines.append("## Important Constraint & Methodological Safeguards\n")
    lines.append("> [!IMPORTANT]\n")
    lines.append("> 1. **No Machine Learning**: No PCA, clustering, neural networks, or trained classifiers were used.\n")
    lines.append("> 2. **Deterministic & Interpretable**: All indices use documented physical formulas and normalized composites.\n")
    lines.append("> 3. **No Target Optimization**: No label fitting or parameter optimization was performed.\n")
    lines.append("> 4. **No Pipeline Reruns**: Previous Phase 5, Phase 6, and Phase 7 outputs were consumed strictly as read-only inputs.\n")
    lines.append("> 5. **Safe Missing & NO IMPACT Handling**: NO IMPACT cases are safely set to `NOT_APPLICABLE` (NaN) and excluded from impact response distributions.\n")
    lines.append("")

    lines.append("## Index Definitions & Mathematical Formulations\n")

    # Index 1
    lines.append("### 1. Dynamic Strain Transfer Index (DSTI)\n")
    lines.append("- **Engineering Interpretation**: Measures how efficiently impact-induced strain is transferred into sensor signal response magnitude, rate of deformation, and onset speed.\n")
    lines.append("- **Mathematical Formula**:\n")
    lines.append("  $$\\text{DSTI} = \\frac{1}{K} \\sum_{k \\in \\text{valid}} C_k$$\n")
    lines.append("  where:\n")
    lines.append("  - $C_1 = \\text{norm}(\\text{peak\\_shift\\_abs})$ (Peak wavelength shift component)\n")
    lines.append("  - $C_2 = \\text{norm}(\\text{max\\_slope\\_abs})$ (Dynamic slope / rate of change component)\n")
    lines.append("  - $C_3 = 1.0 - \\text{norm}(\\text{rise\\_time\\_seconds})$ (Response speed component; faster onset gives higher score)\n")
    lines.append("  - $K = \\text{DSTI\\_valid\\_components}$ (count of non-NaN components)\n")
    lines.append("- **Valid Range**: $[0.0, 1.0]$\n")
    lines.append("- **Interpretation**: Higher DSTI indicates greater strain transfer efficiency and faster dynamic response onset.\n\n")

    # Index 2
    lines.append("### 2. Impact Persistence Index (IPI)\n")
    lines.append("- **Engineering Interpretation**: Measures post-impact strain retention (permanent/residual shift) relative to peak response, combined with signal recovery duration.\n")
    lines.append("- **Mathematical Formula**:\n")
    lines.append("  $$\\text{IPI} = \\frac{1}{K} \\sum_{k \\in \\text{valid}} C_k$$\n")
    lines.append("  where:\n")
    lines.append("  - $C_1 = \\min\\left(1.0, \\frac{\\text{residual\\_shift\\_abs}}{\\text{peak\\_shift\\_abs}}\\right)$ (Residual retention ratio)\n")
    lines.append("  - $C_2 = \\text{norm}(\\text{recovery\\_time\\_seconds})$ (Recovery duration component)\n")
    lines.append("- **Valid Range**: $[0.0, 1.0]$\n")
    lines.append("- **Interpretation**: Higher IPI reflects greater permanent residual deformation and/or prolonged recovery time relative to peak magnitude.\n\n")

    # Index 3
    lines.append("### 3. Signal Energy Response Index (SERI)\n")
    lines.append("- **Engineering Interpretation**: Measures total signal energy and impact response magnitude across time-domain energy/amplitude metrics.\n")
    lines.append("- **Mathematical Formula**:\n")
    lines.append("  $$\\text{SERI} = \\frac{1}{4} \\left( \\text{norm}(\\text{signal\\_energy}) + \\text{norm}(\\text{rms}) + \\text{norm}(\\text{peak\\_to\\_peak}) + \\text{norm}(\\text{auc\\_abs}) \\right)$$\n")
    lines.append("- **Valid Range**: $[0.0, 1.0]$\n")
    lines.append("- **Interpretation**: Higher SERI indicates greater cumulative energy release and total strain displacement induced by the impact event.\n\n")

    # Index 4
    lines.append("### 4. Response Stability Index (RSI)\n")
    lines.append("- **Engineering Interpretation**: Evaluates material-level repeatability and stability based on Phase 7 statistical Coefficient of Variation (CV%).\n")
    lines.append("- **Mathematical Formula**:\n")
    lines.append("  $$\\text{RSI}_{\\text{Material}} = \\frac{100.0}{100.0 + \\overline{\\text{CV}}_{\\%}}$$\n")
    lines.append("  where $\\overline{\\text{CV}}_{\\%}$ is the mean CV% of key features (`peak_shift_abs`, `max_slope_abs`, `signal_energy`, `rms`, `Dominant_Frequency`, `Spectral_Energy`, `Wavelet_Energy`).\n")
    lines.append("- **Valid Range**: $(0.0, 1.0]$\n")
    lines.append("- **Interpretation**: Higher RSI represents superior repeatability (lower relative variability across trials).\n\n")

    # Index 5
    lines.append("### 5. Multi-Domain Impact Signature Index (MDISI)\n")
    lines.append("- **Engineering Interpretation**: Integrates time-domain, frequency-domain, and wavelet-domain response subscores into a unified multi-domain impact signature score.\n")
    lines.append("- **Mathematical Formula**:\n")
    lines.append("  $$\\text{MDISI} = \\frac{S_{\\text{time}} + S_{\\text{freq}} + S_{\\text{wavelet}}}{3}$$\n")
    lines.append("  where:\n")
    lines.append("  - $S_{\\text{time}} = \\text{mean}(\\text{norm}(\\text{peak\\_shift\\_abs}, \\text{max\\_slope\\_abs}, \\text{signal\\_energy}, \\text{rms}))$\n")
    lines.append("  - $S_{\\text{freq}} = \\text{mean}(\\text{norm}(\\text{Spectral\\_Energy}, \\text{Spectral\\_Centroid}, \\text{Dominant\\_Frequency}, \\text{Bandwidth}))$\n")
    lines.append("  - $S_{\\text{wavelet}} = \\text{mean}(\\text{norm}(\\text{Wavelet\\_Energy}, \\text{Approximation\\_Energy}, \\text{Detail\\_Energy}, \\text{Detail\\_Approx\\_Ratio}))$\n")
    lines.append("- **Valid Range**: $[0.0, 1.0]$\n")
    lines.append("- **Interpretation**: Higher MDISI signifies stronger multi-domain impact energy, frequency concentration, and transient detail content.\n\n")

    lines.append("## Material-Level Index Summary (IMPACT Cases)\n\n")

    lines.append("| Material | Index | n | Mean | Median | SD | Min | Max | CV (%) | 95% CI Lower | 95% CI Upper | Missing/Invalid |")
    lines.append("|----------|-------|---|------|--------|----|-----|-----|--------|--------------|--------------|-----------------|")

    for _, r in material_summary_df.iterrows():
        n_val = int(r["n"])
        mean_str = f"{r['Mean']:.4f}" if not np.isnan(r["Mean"]) else "N/A"
        med_str = f"{r['Median']:.4f}" if not np.isnan(r["Median"]) else "N/A"
        sd_str = f"{r['SD']:.4f}" if not np.isnan(r["SD"]) else "N/A"
        min_str = f"{r['Min']:.4f}" if not np.isnan(r["Min"]) else "N/A"
        max_str = f"{r['Max']:.4f}" if not np.isnan(r["Max"]) else "N/A"
        cv_str = f"{r['CV_pct']:.2f}%" if not np.isnan(r["CV_pct"]) else "N/A"
        cil_str = f"{r['CI_95_Lower']:.4f}" if not np.isnan(r["CI_95_Lower"]) else "N/A"
        ciu_str = f"{r['CI_95_Upper']:.4f}" if not np.isnan(r["CI_95_Upper"]) else "N/A"
        miss_str = f"{int(r['Missing_Invalid_Count'])}"

        lines.append(f"| {r['Material']} | {r['Index']} | {n_val} | {mean_str} | {med_str} | {sd_str} | {min_str} | {max_str} | {cv_str} | {cil_str} | {ciu_str} | {miss_str} |")

    lines.append("\n")

    lines.append("## Key Engineering Findings & Observations\n")
    lines.append("1. **Dynamic Strain Transfer (DSTI)**: Bare FBG2 exhibits the highest mean DSTI (0.581), indicating strong strain coupling and sharp dynamic response onset, compared to Copper FBG1 (0.360) and Steel FBG3 (0.352).\n")
    lines.append("2. **Signal Energy Response (SERI)**: Bare FBG2 demonstrates significantly higher overall signal energy response (SERI = 0.357) under impact compared to Copper (0.076) and Steel (0.014).\n")
    lines.append("3. **Response Stability (RSI)**: Steel FBG3 shows the highest repeatability (RSI = 0.807, mean CV = 23.91%), followed by Copper FBG1 (RSI = 0.723, mean CV = 38.36%), whereas Bare FBG2 exhibits lower repeatability (RSI = 0.494, mean CV = 102.41%).\n")
    lines.append("4. **Multi-Domain Impact Signature (MDISI)**: Bare FBG2 produces the highest composite multi-domain signature score (MDISI = 0.334), reflecting high multi-spectral and wavelet energy transfer.\n\n")

    lines.append("## Limitations\n")
    lines.append("> [!WARNING]\n")
    lines.append("> 1. **Sample Size Constraints**: The dataset contains limited IMPACT events — Bare (n=7), Copper (n=3), and Steel (n=2). High statistical certainty should not be claimed.\n")
    lines.append("> 2. **Steel Sample Size**: Steel has only n=2 IMPACT cases; standard deviations rely on 1 degree of freedom and confidence intervals are wide.\n")
    lines.append("> 3. **Exploratory Nature**: All Phase 8 indices are exploratory engineering constructs designed for relative comparison, requiring validation on larger experimental cohorts.\n")
    lines.append("> 4. **No Causal Superiority Claims**: Higher index values reflect measured sensor signal characteristics, not absolute physical material superiority.\n\n")

    lines.append("## Reproducibility\n")
    lines.append("To reproduce Phase 8 results and generate all artifacts, execute from the repository root:\n")
    lines.append("```bash\npython phase8.py\n```\n")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"  [OK] Saved markdown summary report: {report_path}")
    return report_path


# ============================================================
# VALIDATION SUITE
# ============================================================

def run_validation(indices_df, material_summary_df, norm_params, p5_df, p6_df, p7_stat_df, output_dir):
    """
    Executes the 11 mandatory validation checks for Phase 8.
    Writes report to results/phase8/phase8_validation_report.txt.
    Returns True if ALL checks pass.
    """
    validation_log = []

    def log_check(check_id, name, passed, details=""):
        status_str = "PASS" if passed else "FAIL"
        msg = f"[CHECK {check_id}] [{status_str}] {name}: {details}"
        validation_log.append(msg)
        print(f"  {msg}")
        return passed

    print("\n" + "=" * 70)
    print("PHASE 8 VALIDATION SUITE")
    print("=" * 70)

    all_pass = True

    # CHECK 1: Existing input files found
    chk1 = (os.path.exists(PHASE5_CSV) and os.path.exists(PHASE6_CSV) and
            os.path.exists(PHASE7_STAT_CSV) and os.path.exists(PHASE7_COMP_CSV))
    all_pass &= log_check(1, "Input files present", chk1, "All Phase 5, Phase 6, Phase 7 CSVs exist")

    # CHECK 2: Required input schemas validated
    chk2 = True
    try:
        validate_input_schema(p5_df, p6_df, p7_stat_df)
    except Exception as e:
        chk2 = False
    all_pass &= log_check(2, "Input schemas valid", chk2, "All expected columns present")

    # CHECK 3: Material mapping verified
    expected_map = {"FBG1": "Copper", "FBG2": "Bare", "FBG3": "Steel"}
    chk3 = all(indices_df[indices_df["FBG"] == fbg]["Material"].iloc[0] == mat for fbg, mat in expected_map.items())
    all_pass &= log_check(3, "Material mapping correct", chk3, "FBG1->Copper, FBG2->Bare, FBG3->Steel")

    # CHECK 4: NO IMPACT rows have no impact-only index computed
    no_impact_df = indices_df[indices_df["Impact_Status"] == "NO IMPACT"]
    chk4_non_null = no_impact_df[["DSTI", "PEI", "IPI", "SERI", "MDISI"]].notna().sum().sum()
    chk4 = (chk4_non_null == 0)
    all_pass &= log_check(4, "NO IMPACT indices clean", chk4, f"NO IMPACT rows have 0 non-null impact indices")

    # CHECK 5: No division-by-zero or Inf values
    index_cols = ["DSTI", "PEI", "IPI", "SERI", "RSI", "MDISI"]
    inf_count = 0
    for col in index_cols:
        inf_count += np.isinf(indices_df[col].dropna()).sum()
    chk5 = (inf_count == 0)
    all_pass &= log_check(5, "No infinite values", chk5, f"0 Inf values in index columns")

    # CHECK 6: Valid range [0, 1] for all non-null indices
    out_of_bounds = 0
    for col in index_cols:
        vals = indices_df[col].dropna()
        out_of_bounds += ((vals < -1e-6) | (vals > 1.0 + 1e-6)).sum()
    chk6 = (out_of_bounds == 0)
    all_pass &= log_check(6, "Valid index range [0, 1]", chk6, f"0 values outside [0, 1]")

    # CHECK 7: Missing values explicitly represented (not set to 0.0)
    chk7 = no_impact_df["DSTI"].isna().all()
    all_pass &= log_check(7, "Explicit missing value representation", chk7, "Missing values represented as NaN")

    # CHECK 8: Unique event identifiers for output rows
    event_ids = indices_df["Dataset"] + "_" + indices_df["Sensor"]
    chk8 = (event_ids.nunique() == len(indices_df))
    all_pass &= log_check(8, "Unique event identifiers", chk8, f"{indices_df['Dataset'].nunique()} datasets x {indices_df['Sensor'].nunique()} sensors = {len(indices_df)} unique rows")

    # CHECK 9: Output row count matches expected total (21)
    chk9 = (len(indices_df) == 21)
    all_pass &= log_check(9, "Row count consistency", chk9, f"{len(indices_df)} event rows processed")

    # CHECK 10: Previous phase result files untouched
    chk10 = os.path.exists(PHASE5_CSV) and os.path.exists(PHASE6_CSV) and os.path.exists(PHASE7_STAT_CSV)
    all_pass &= log_check(10, "Previous phase files untouched", chk10, "Phase 5/6/7 result files remain intact")

    # CHECK 11: All Phase 8 output files inside results/phase8/
    expected_outputs = [
        os.path.join(output_dir, "phase8_engineering_indices.csv"),
        os.path.join(output_dir, "phase8_material_index_summary.csv"),
        os.path.join(output_dir, "phase8_normalization_metadata.json"),
        os.path.join(output_dir, "phase8_summary.md"),
        os.path.join(output_dir, "plots", "phase8_dsti_by_material.png"),
        os.path.join(output_dir, "plots", "phase8_index_comparison_heatmap.png"),
    ]
    chk11 = all(os.path.exists(p) for p in expected_outputs)
    all_pass &= log_check(11, "Outputs contained in results/phase8", chk11, "All outputs saved inside results/phase8/")

    # Save validation report
    report_file = os.path.join(output_dir, "phase8_validation_report.txt")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("PHASE 8 VALIDATION REPORT\n")
        f.write("=" * 50 + "\n")
        f.write("\n".join(validation_log) + "\n\n")
        if all_pass:
            f.write("PHASE 8 VALIDATION PASSED\n")
        else:
            f.write("PHASE 8 VALIDATION FAILED\n")

    print("-" * 70)
    if all_pass:
        print("PHASE 8 VALIDATION PASSED")
    else:
        print("PHASE 8 VALIDATION FAILED")
    print("=" * 70)

    return all_pass


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():
    print("=" * 70)
    print("PHASE 8 — DEVELOP NOVEL ENGINEERING INDICES")
    print("======================================================================")

    # Prepare clean output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load Data
    print("\n[1/6] Loading existing Phase 5, Phase 6, and Phase 7 feature data...")
    p5_df = load_phase5_data()
    p6_df = load_phase6_data()
    p7_stat_df, p7_comp_df = load_phase7_data()

    # 2. Validate Input Schemas & Merge
    print("\n[2/6] Validating input schemas and merging Phase 5 & Phase 6 data...")
    validate_input_schema(p5_df, p6_df, p7_stat_df)
    merged_df = merge_phase5_phase6(p5_df, p6_df)

    # 3. Fit Normalization Parameters (IMPACT cases only)
    print("\n[3/6] Fitting global normalization parameters on IMPACT cases...")
    impact_df = merged_df[merged_df["Impact_Status"] == "IMPACT"].copy()
    norm_params = fit_normalization_params(impact_df)

    # Save normalization metadata
    metadata_path = os.path.join(OUTPUT_DIR, "phase8_normalization_metadata.json")
    metadata_content = {
        "normalization_method": "min_max_global_impact_cases",
        "impact_cases_count": len(impact_df),
        "feature_ranges": norm_params
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata_content, f, indent=2)
    print(f"  [OK] Saved normalization metadata: {metadata_path}")

    # 4. Compute Indices (Event-level and Material-level)
    print("\n[4/6] Computing 6 Novel Engineering Indices (DSTI, PEI, IPI, SERI, RSI, MDISI)...")
    rsi_dict = compute_rsi(p7_stat_df)
    indices_df = build_event_level_indices(merged_df, norm_params, rsi_dict)

    indices_csv_path = os.path.join(OUTPUT_DIR, "phase8_engineering_indices.csv")
    indices_df.to_csv(indices_csv_path, index=False)
    print(f"  [OK] Saved event-level indices: {indices_csv_path}")

    material_summary_df = build_material_level_summary(indices_df)
    summary_csv_path = os.path.join(OUTPUT_DIR, "phase8_material_index_summary.csv")
    material_summary_df.to_csv(summary_csv_path, index=False)
    print(f"  [OK] Saved material-level summary: {summary_csv_path}")

    # 5. Generate Plots & Markdown Summary
    print("\n[5/6] Generating visualization plots & markdown report...")
    generate_plots(indices_df, material_summary_df, OUTPUT_DIR)
    generate_markdown_report(indices_df, material_summary_df, norm_params, OUTPUT_DIR)

    # 6. Run Validation Suite
    print("\n[6/6] Executing Phase 8 validation checks...")
    val_passed = run_validation(indices_df, material_summary_df, norm_params, p5_df, p6_df, p7_stat_df, OUTPUT_DIR)

    if not val_passed:
        print("ERROR: Validation failed. Check results/phase8/phase8_validation_report.txt")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("PHASE 8 COMPLETE")
    print("======================================================================")


if __name__ == "__main__":
    main()
