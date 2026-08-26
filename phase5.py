import os
import sys
import shutil
import pandas as pd
import numpy as np

from src.feature_extraction import extract_all_dataset_features
from src.comparison_analysis import (
    generate_material_summary_table,
    generate_engineering_comparison_table,
    generate_comparison_plots,
    generate_markdown_reports
)


def run_phase5():
    print("=" * 70)
    print("PHASE 5 – ENGINEERING SIGNAL CHARACTERIZATION")
    print("======================================================================")
    
    data_dir = "data/processed/final_phase_input"
    phase4_dir = "results/phase4"
    results_dir = "results/phase5"
    plots_dir = os.path.join(results_dir, "plots")
    
    # Requirement 7: Clean existing results/phase5 output directory from scratch
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
        
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. Feature Extraction (Strict Phase 4 Event Boundaries & Case Separation)
    print("\n[1/4] Extracting 13 Engineering Signal Features (Phase 4 Event Grounded)...")
    all_features_df, long_format_df = extract_all_dataset_features(
        data_dir=data_dir,
        phase4_dir=phase4_dir
    )
    
    if all_features_df.empty:
        print("ERROR: No features extracted. Check input datasets.")
        sys.exit(1)
        
    impact_cases_count = len(all_features_df[all_features_df["Impact_Status"] == "IMPACT"])
    no_impact_cases_count = len(all_features_df[all_features_df["Impact_Status"] == "NO IMPACT"])
    
    print(f"  [OK] Processed {len(all_features_df)} total sensor recordings ({impact_cases_count} IMPACT cases, {no_impact_cases_count} NO-IMPACT cases).")
    
    # Save raw features CSV
    all_features_path = os.path.join(results_dir, "phase5_all_features.csv")
    all_features_df.to_csv(all_features_path, index=False)
    print(f"  [OK] Saved: {all_features_path}")
    
    # 2. Material Summary Tables (IMPACT cases only)
    print("\n[2/4] Generating Statistical Summaries & Material Comparison Tables (IMPACT cases only)...")
    summary_df = generate_material_summary_table(all_features_df)
    summary_path = os.path.join(results_dir, "phase5_feature_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"  [OK] Saved: {summary_path}")
    
    comp_df = generate_engineering_comparison_table(all_features_df)
    comp_path = os.path.join(results_dir, "phase5_material_comparison.csv")
    comp_df.to_csv(comp_path, index=False)
    print(f"  [OK] Saved: {comp_path}")
    
    # 3. Plots Generation (IMPACT cases only)
    print("\n[3/4] Generating Engineering Comparison Plots (IMPACT cases only)...")
    generate_comparison_plots(all_features_df, plots_dir)
    print(f"  [OK] Generated 13 comparison plots in: {plots_dir}")
    
    # 4. Markdown Reports
    print("\n[4/4] Generating Markdown Reports...")
    generate_markdown_reports(all_features_df, comp_df, results_dir)
    print(f"  [OK] Generated reports in: {results_dir}")
    
    # Validation
    required_features = [
        "peak_shift_abs", "residual_shift_abs", "rise_time_seconds",
        "recovery_time_seconds", "peak_width_seconds", "max_slope_abs",
        "rms", "signal_energy", "peak_to_peak", "variance",
        "std_dev", "entropy", "auc_abs"
    ]
    
    pass_validation = (len(required_features) == 13) and (impact_cases_count == 12)
    validation_str = "PASS" if pass_validation else "FAIL"
    
    sensors_processed = sorted(all_features_df["Sensor"].unique().tolist())
    materials_processed = sorted(all_features_df["Material"].unique().tolist())
    
    print("\n" + "=" * 70)
    print("PHASE 5 COMPLETE")
    print("=" * 70)
    print("\nBranch:")
    print("adithyap")
    
    print("\nFeatures Implemented:")
    print("- Peak Shift")
    print("- Residual Shift")
    print("- Rise Time")
    print("- Recovery Time")
    print("- Peak Width")
    print("- Maximum Slope")
    print("- RMS")
    print("- Signal Energy")
    print("- Peak-to-Peak")
    print("- Variance")
    print("- Standard Deviation")
    print("- Entropy")
    print("- Area Under Curve")
    
    print(f"\nSensors Processed:\n{', '.join(sensors_processed)}")
    print(f"\nMaterials Processed:\n{', '.join(materials_processed)}")
    
    print(f"\nCases Analyzed:")
    print(f"- IMPACT Cases: {impact_cases_count}")
    print(f"- NO-IMPACT Cases: {no_impact_cases_count}")
    
    print("\nOutput Files:")
    print(f"- {all_features_path}")
    print(f"- {summary_path}")
    print(f"- {comp_path}")
    print(f"- {os.path.join(results_dir, 'phase5_engineering_explanation.md')}")
    print(f"- {os.path.join(results_dir, 'phase5_summary.md')}")
    print(f"- 13 plots in {plots_dir}")
    
    print(f"\nValidation:\n{validation_str}")
    print("=" * 70)


if __name__ == "__main__":
    run_phase5()
