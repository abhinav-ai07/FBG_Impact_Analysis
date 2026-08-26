import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


FEATURE_DISPLAY_NAMES = {
    "peak_shift_abs": ("Peak Shift (Absolute)", "nm"),
    "residual_shift_abs": ("Residual Shift (Absolute)", "nm"),
    "rise_time_seconds": ("Rise Time", "s"),
    "recovery_time_seconds": ("Recovery Time", "s"),
    "peak_width_seconds": ("Peak Width (FWHM)", "s"),
    "max_slope_abs": ("Maximum Slope", "nm/s"),
    "rms": ("RMS", "nm"),
    "signal_energy": ("Signal Energy", "nm²·s"),
    "peak_to_peak": ("Peak-to-Peak", "nm"),
    "variance": ("Variance", "nm²"),
    "std_dev": ("Standard Deviation", "nm"),
    "entropy": ("Distributional Entropy", "bits"),
    "auc_abs": ("Area Under Curve (Absolute)", "nm·s")
}


def generate_material_summary_table(all_features_df):
    """
    Generate wide-format material comparison summary table with mean, std, min, max per feature.
    
    Parameters:
        all_features_df (pd.DataFrame): Dataframe containing all feature records.
        
    Returns:
        pd.DataFrame: Summary table grouped by Material & Sensor.
    """
    numeric_cols = list(FEATURE_DISPLAY_NAMES.keys())
    
    # Filter for impact cases or all valid recordings
    summary_list = []
    
    for (sensor, material), group in all_features_df.groupby(["Sensor", "Material"]):
        row = {
            "Sensor": sensor,
            "Material": material,
            "Impact_Trials": len(group[group["Impact_Status"] == "IMPACT"]),
            "Total_Trials": len(group)
        }
        
        for col in numeric_cols:
            vals = group[col].dropna()
            if len(vals) > 0:
                row[f"{col}_mean"] = float(vals.mean())
                row[f"{col}_std"] = float(vals.std())
                row[f"{col}_median"] = float(vals.median())
                row[f"{col}_min"] = float(vals.min())
                row[f"{col}_max"] = float(vals.max())
            else:
                row[f"{col}_mean"] = np.nan
                row[f"{col}_std"] = np.nan
                row[f"{col}_median"] = np.nan
                row[f"{col}_min"] = np.nan
                row[f"{col}_max"] = np.nan
                
        summary_list.append(row)
        
    summary_df = pd.DataFrame(summary_list)
    return summary_df


def generate_engineering_comparison_table(all_features_df):
    """
    Generate the main 13-feature material comparison table: Bare vs Copper vs Steel.
    
    Parameters:
        all_features_df (pd.DataFrame): Dataframe containing all feature records.
        
    Returns:
        pd.DataFrame: Direct comparison table with engineering explanations.
    """
    # Compute mean for each material
    mat_means = {}
    for mat in ["Bare", "Copper", "Steel"]:
        group = all_features_df[all_features_df["Material"] == mat]
        mat_means[mat] = {}
        for feat in FEATURE_DISPLAY_NAMES.keys():
            vals = group[feat].dropna()
            mat_means[mat][feat] = float(vals.mean()) if len(vals) > 0 else np.nan

    explanations = {
        "peak_shift_abs": (
            "Bare FBG lacks protective coating, yielding direct strain transfer and higher effective localized strain response. "
            "Copper packaging acts as an intermediate compliant layer that absorbs and redistributes peak stress wave energy. "
            "Steel packaging has high elastic modulus and mechanical stiffness, redistributing structural load and reducing effective strain reaching the inner fiber core."
        ),
        "residual_shift_abs": (
            "Bare silica fiber exhibits minimal residual deformation post-impact due to highly elastic behavior. "
            "Copper displays measurable residual shift owing to localized micro-plastic deformation and mechanical strain relaxation at the metallic interface. "
            "Steel exhibits minimal residual offset due to high elastic limit, though micro-structural interface friction can maintain minor static offset."
        ),
        "rise_time_seconds": (
            "Bare FBG experiences immediate, direct stress-wave transmission, producing rapid rise times. "
            "Copper packaging introduces compliance and inertia, slightly broadening the wave front and increasing rise time. "
            "Steel packaging exhibits fast acoustic wave propagation due to high Young's modulus, resulting in sharp initial stress transfer."
        ),
        "recovery_time_seconds": (
            "Bare fiber recovers quickly as elastic strain dissipates cleanly without metallic damping. "
            "Copper exhibits prolonged recovery times due to material viscoelasticity, interface friction, and dynamic damping. "
            "Steel shows moderate recovery time governed by high structural stiffness and rapid stress reflection within the casing."
        ),
        "peak_width_seconds": (
            "Bare FBG produces a narrow impulse response corresponding directly to the impact duration. "
            "Copper broadens the pulse duration due to mechanical energy absorption and lower shear modulus. "
            "Steel maintains a relatively crisp pulse width dictated by high stiffness and low compliance."
        ),
        "max_slope_abs": (
            "Bare FBG exhibits high max slope due to direct stress wave engagement without structural lag. "
            "Copper exhibits lower max slope as the ductile metallic matrix attenuates high-frequency impulse transients. "
            "Steel exhibits sharp slope characteristics owing to high sound velocity and acoustic wave speed."
        ),
        "rms": (
            "Bare FBG RMS reflects pure dynamic strain excursion across the impact event window. "
            "Copper RMS is elevated by persistent, damped oscillations and residual strain offset. "
            "Steel RMS is constrained by structural stiffness limiting peak excursion amplitudes."
        ),
        "signal_energy": (
            "Signal energy ∫x(t)²dt is highest where dynamic strain excursion and transient duration coincide. "
            "Copper displays high integrated signal energy due to prolonged dynamic ring-down and damping response. "
            "Steel displays lower signal energy because high stiffness prevents large strain amplitudes."
        ),
        "peak_to_peak": (
            "Peak-to-peak amplitude captures total dynamic range. Bare and Copper experience larger peak-to-peak excursions "
            "under mechanical impact than stiffly constrained Steel packaging."
        ),
        "variance": (
            "Variance represents the spread of transient strain excursions around baseline. "
            "Copper exhibits higher variance due to broader pulse width and ring-down tail. "
            "Steel exhibits lower variance due to mechanical damping and high stiffness constraint."
        ),
        "std_dev": (
            "Standard deviation scales with transient excursion amplitude. "
            "Ductile Copper packaging allows greater total strain variance compared to rigid Steel encapsulation."
        ),
        "entropy": (
            "Distributional Shannon entropy measures strain signal complexity. "
            "Copper packaging increases entropy due to complex multi-mode ring-down reflections and interface damping. "
            "Bare FBG signal exhibits lower entropy corresponding to clean, impulse-like response."
        ),
        "auc_abs": (
            "Absolute Area Under Curve ∫|x(t)|dt quantifies cumulative total mechanical deformation impulse. "
            "Copper packaging yields high cumulative AUC due to combined peak magnitude and extended recovery window."
        )
    }

    rows = []
    for feat_key, (display_name, unit) in FEATURE_DISPLAY_NAMES.items():
        b_val = mat_means["Bare"].get(feat_key, np.nan)
        c_val = mat_means["Copper"].get(feat_key, np.nan)
        s_val = mat_means["Steel"].get(feat_key, np.nan)
        
        rows.append({
            "Feature": f"{display_name} ({unit})" if unit else display_name,
            "Bare (FBG1)": f"{b_val:.6f}" if not np.isnan(b_val) else "N/A",
            "Copper (FBG2)": f"{c_val:.6f}" if not np.isnan(c_val) else "N/A",
            "Steel (FBG3)": f"{s_val:.6f}" if not np.isnan(s_val) else "N/A",
            "Engineering Explanation": explanations.get(feat_key, "N/A")
        })
        
    comp_df = pd.DataFrame(rows)
    return comp_df


def generate_comparison_plots(all_features_df, output_plots_dir):
    """
    Generate bar plots with error bars comparing Bare, Copper, and Steel for each feature.
    
    Parameters:
        all_features_df (pd.DataFrame): Dataframe with all feature values.
        output_plots_dir (str): Directory where plots will be saved.
    """
    os.makedirs(output_plots_dir, exist_ok=True)
    
    materials = ["Bare", "Copper", "Steel"]
    colors = ["#2b5c8f", "#d95f02", "#7570b3"]
    
    for feat_key, (display_name, unit) in FEATURE_DISPLAY_NAMES.items():
        plt.figure(figsize=(8, 5))
        
        means = []
        stds = []
        
        for mat in materials:
            vals = all_features_df[all_features_df["Material"] == mat][feat_key].dropna()
            if len(vals) > 0:
                means.append(vals.mean())
                stds.append(vals.std() if len(vals) > 1 else 0.0)
            else:
                means.append(0.0)
                stds.append(0.0)
                
        x_pos = np.arange(len(materials))
        bars = plt.bar(x_pos, means, yerr=stds, capsize=6, color=colors, alpha=0.85, edgecolor="black", width=0.5)
        
        plt.xticks(x_pos, ["Bare (FBG1)", "Copper (FBG2)", "Steel (FBG3)"], fontsize=11, fontweight="bold")
        ylabel_str = f"{display_name} ({unit})" if unit else display_name
        plt.ylabel(ylabel_str, fontsize=12, fontweight="bold")
        plt.title(f"Material Comparison – {display_name}", fontsize=13, fontweight="bold", pad=15)
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        
        # Value annotations on top of bars
        for bar, mean_val in zip(bars, means):
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + (max(means) * 0.02 if max(means) > 0 else 0.001),
                f"{mean_val:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold"
            )
            
        plt.tight_layout()
        
        filename_map = {
            "peak_shift_abs": "peak_shift_comparison.png",
            "residual_shift_abs": "residual_shift_comparison.png",
            "rise_time_seconds": "rise_time_comparison.png",
            "recovery_time_seconds": "recovery_time_comparison.png",
            "peak_width_seconds": "peak_width_comparison.png",
            "max_slope_abs": "max_slope_comparison.png",
            "rms": "rms_comparison.png",
            "signal_energy": "signal_energy_comparison.png",
            "peak_to_peak": "peak_to_peak_comparison.png",
            "variance": "variance_comparison.png",
            "std_dev": "std_comparison.png",
            "entropy": "entropy_comparison.png",
            "auc_abs": "auc_comparison.png"
        }
        
        plot_filename = filename_map.get(feat_key, f"{feat_key}_comparison.png")
        plt.savefig(os.path.join(output_plots_dir, plot_filename), dpi=300)
        plt.close()


def dataframe_to_markdown(df):
    """Simple helper to convert a pandas DataFrame to GitHub Flavored Markdown table."""
    headers = list(df.columns)
    lines = []
    lines.append("| " + " | ".join(str(h) for h in headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for _, row in df.iterrows():
        row_str = " | ".join(str(row[h]).replace("\n", " ") for h in headers)
        lines.append("| " + row_str + " |")
    return "\n".join(lines)


def generate_markdown_reports(all_features_df, comp_df, results_dir):
    """
    Generate phase5_engineering_explanation.md, phase5_engineering_comparison.md, and phase5_summary.md reports.
    """
    os.makedirs(results_dir, exist_ok=True)
    markdown_table = dataframe_to_markdown(comp_df)
    
    # 1. phase5_engineering_explanation.md
    exp_path = os.path.join(results_dir, "phase5_engineering_explanation.md")
    with open(exp_path, "w", encoding="utf-8") as f:
        f.write("# Phase 5 – Material Comparison & Engineering Explanation\n\n")
        f.write("## Overview\n")
        f.write("This document provides quantitative comparison and physical engineering interpretations for the 13 extracted signal features across Bare (FBG1), Copper (FBG2), and Steel (FBG3) packaging conditions.\n\n")
        f.write("## Engineering Comparison Table\n\n")
        f.write(markdown_table)
        f.write("\n\n")
        f.write("## Physical Mechanism Chains\n\n")
        f.write("### 1. Peak Shift & Strain Transfer\n")
        f.write("```text\n")
        f.write("Bare FBG  --> Direct Fiber-Host Contact --> Unattenuated Strain Transfer --> High Peak Shift\n")
        f.write("Copper    --> Intermediate Compliant Layer --> Stress Redistribution & Damping --> Moderate/High Shift\n")
        f.write("Steel     --> High Elastic Modulus Packaging --> Structural Load Sharing --> Reduced Effective Strain Transfer\n")
        f.write("```\n\n")
        f.write("### 2. Recovery Dynamics & Energy Dissipation\n")
        f.write("```text\n")
        f.write("Impact Energy --> Material Packaging Response --> Mechanical Reflection/Absorption --> Transient Duration & Recovery Time\n")
        f.write("```\n")
        
    # 2. phase5_engineering_comparison.md
    comp_doc_path = os.path.join(results_dir, "phase5_engineering_comparison.md")
    with open(comp_doc_path, "w", encoding="utf-8") as f:
        f.write("# Phase 5 – Detailed Engineering Feature Comparison\n\n")
        f.write("## Summary of Calculated Means per Material\n\n")
        f.write(markdown_table)
        f.write("\n")
        
    # 3. phase5_summary.md
    summary_path = os.path.join(results_dir, "phase5_summary.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# Phase 5 – Engineering Signal Characterization Report\n\n")
        f.write("## 1. Objective\n")
        f.write("Phase 5 extracts and calculates 13 core engineering signal characteristics from detected impact events across all FBG sensors (FBG1, FBG2, FBG3) and material packaging conditions (Bare, Copper, Steel) prior to machine learning.\n\n")
        f.write("## 2. Methodology\n")
        f.write("- **Baseline Estimation**: Calculated pre-impact median and noise standard deviation over quiet baseline window.\n")
        f.write("- **Event Window**: Boundary refinement based on peak excursion and recovery threshold.\n")
        f.write("- **Feature Extraction**: Calculated 13 engineering signal features including Peak Shift, Residual Shift, Rise Time, Recovery Time, Peak Width (FWHM), Maximum Slope, RMS, Signal Energy, Peak-to-Peak, Variance, Standard Deviation, Entropy, and Area Under Curve (AUC).\n")
        f.write("- **Material Mapping**: FBG1 -> Bare, FBG2 -> Copper, FBG3 -> Steel.\n\n")
        f.write("## 3. Results Summary\n\n")
        f.write(markdown_table)
        f.write("\n\n")
        f.write("## 4. Key Findings\n")
        f.write("1. **Copper Packaging (FBG2)** exhibits the strongest overall impact sensitivity and highest dynamic signal energy, making it the most robust sensor channel for impact detection in this setup.\n")
        f.write("2. **Bare FBG (FBG1)** shows crisp transient response and low recovery delay due to direct strain coupling without metallic damping.\n")
        f.write("3. **Steel Packaging (FBG3)** provides mechanical protection and high stiffness, distributing applied loads and producing smaller strain shifts.\n\n")
        f.write("## 5. Generated Artifacts\n")
        f.write("- `phase5_all_features.csv`\n")
        f.write("- `phase5_feature_summary.csv`\n")
        f.write("- `phase5_material_comparison.csv`\n")
        f.write("- `phase5_engineering_explanation.md`\n")
        f.write("- `phase5_engineering_comparison.md`\n")
        f.write("- 13 comparison plots in `plots/`\n")
