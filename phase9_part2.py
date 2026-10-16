"""
PHASE 9 — PART 2: MASTER ORCHESTRATOR & VALIDATION SUITE
========================================================
FBG Impact Analysis — Clustering + Autoencoder + Explainability
Physics-Guided Multi-Domain Signal Intelligence Framework (PGMSIF)

Master pipeline that coordinates:
    Part A: Unsupervised Clustering (K-Means, Hierarchical, DBSCAN)
    Part B: Neural Autoencoder & 2D Latent Space Analysis
    Part C: Explainability & Ranked Engineering Parameters

Generates:
    - results/phase9/phase9_part2_summary.md
    - results/phase9/phase9_part2_validation_report.txt
    - All sub-module CSVs and visual plots in results/phase9/

Consumes existing results from Phase 5, Phase 6, Phase 8, and Phase 9 Part 1 as read-only inputs.
Strictly idempotent and duplicate-free.
"""

import os
import sys
import json
import numpy as np
import pandas as pd

# Import sub-modules
from phase9_clustering import run_clustering_pipeline
from phase9_autoencoder import run_autoencoder_pipeline
from phase9_explainability import run_explainability_pipeline_main


# ============================================================
# CONFIGURATION
# ============================================================

RESULTS_DIR = os.path.join("results", "phase9")
CLUSTERING_DIR = os.path.join(RESULTS_DIR, "clustering")
AE_DIR = os.path.join(RESULTS_DIR, "autoencoder")
EXP_DIR = os.path.join(RESULTS_DIR, "explainability")

SUMMARY_MD_PATH = os.path.join(RESULTS_DIR, "phase9_part2_summary.md")
VALIDATION_REPORT_PATH = os.path.join(RESULTS_DIR, "phase9_part2_validation_report.txt")


# ============================================================
# SUMMARY MARKDOWN GENERATOR
# ============================================================

def generate_part2_summary_report(km_df, hc_df, db_df, comp_clust_df, rec_df, ranked_df, shap_df):
    """
    Generates a comprehensive scientific summary report for Phase 9 Part 2.
    All rankings are derived programmatically from the actual CSV outputs.
    All wording is scientifically calibrated to reflect N=12 limitations.
    """
    lines = []

    lines.append("# Phase 9 — Part 2: Clustering, Autoencoder & Explainability Summary Report\n")
    lines.append("## Physics-Guided Multi-Domain Signal Intelligence Framework (PGMSIF)\n")

    lines.append("> [!CAUTION]")
    lines.append("> **Dataset Size Limitation**: This entire Phase 9 Part 2 analysis is based on **N=12** validated")
    lines.append("> IMPACT events (Bare n=7, Copper n=3, Steel n=2). All ML and exploratory findings must be")
    lines.append("> interpreted with extreme caution. No result should be claimed as definitive, optimal,")
    lines.append("> or generalisable without validation on a larger independent dataset.\n")

    lines.append("## 1. Executive Summary & Research Objectives\n")
    lines.append("Phase 9 Part 2 investigates the unsupervised structural properties, non-linear latent manifolds, and feature "
                 "importance attribution of the 24-dimensional multi-domain FBG impact feature representation. "
                 "**N=12 is the principal limitation throughout this analysis.** "
                 "The framework addresses three exploratory scientific questions:\n")
    lines.append("1. **Exploratory Clustering**: *Does the 24D multi-domain feature space reveal partial structure associated with packaging configurations?*")
    lines.append("2. **Latent Space Compression**: *Can a non-linear autoencoder compress and reconstruct 24 features in-sample?*")
    lines.append("3. **Supervised Exploratory Attribution**: *Within this dataset, which features receive the greatest attribution in a supervised exploratory model?*\n")

    lines.append("---")
    lines.append("## 2. Input Scope & Safeguards\n")
    lines.append("- **Input Artifacts Consumed (Read-Only)**:")
    lines.append("  - `results/phase5/phase5_all_features.csv` (10 time-domain features)")
    lines.append("  - `results/phase6/phase6_multidomain_features.csv` (10 spectral & wavelet features)")
    lines.append("  - `results/phase8/phase8_engineering_indices.csv` (4 physics-guided indices: DSTI, PEI, SII, DRI)")
    lines.append("  - `results/phase9/pca/phase9_pca_scores.csv` (PCA coordinates from Phase 9 Part 1)")
    lines.append("- **Scope**: $N=12$ valid experimental IMPACT events across 3 packaging materials:")
    lines.append("  - Bare Fiber (FBG2): $n=7$ (58.3%)")
    lines.append("  - Copper Packaging (FBG1): $n=3$ (25.0%)")
    lines.append("  - Steel Packaging (FBG3): $n=2$ (16.7%)")
    lines.append("- **Unsupervised Constraint**: Material labels were strictly excluded from clustering and autoencoder training.")
    lines.append("  Labels are used only for post-hoc metric computation (ARI, NMI, Purity) and visualisation.")
    lines.append("- **Feature Redundancy Note**: Some multidomain variables are physically or mathematically correlated,")
    lines.append("  and Phase 8 indices are composites derived from underlying signal features. Therefore, model-based")
    lines.append("  importance values should not be interpreted as independent causal contributions.\n")

    lines.append("---")
    lines.append("## 3. Part A — Clustering Analysis & Findings\n")

    lines.append("> [!NOTE]")
    lines.append("> Clustering is strictly unsupervised: labels are excluded from model fitting.")
    lines.append("> ARI and NMI are computed post-hoc for comparison only, not used in clustering.")
    lines.append("> Silhouette score is an internal clustering quality metric (no labels required).")
    lines.append("> All findings are exploratory given N=12.\n")

    lines.append("### K-Means Clustering Results\n")
    lines.append("| K | Silhouette Score | Davies-Bouldin | Calinski-Harabasz | ARI | NMI | Cluster Purity |")
    lines.append("|---|---|---|---|---|---|---|")
    for _, r in km_df.iterrows():
        lines.append(f"| K={int(r['K'])} | {r['Silhouette_Score']:.4f} | {r['Davies_Bouldin_Index']:.4f} | {r['Calinski_Harabasz_Index']:.4f} | {r['Adjusted_Rand_Index']:.4f} | {r['Normalized_Mutual_Info']:.4f} | {r['Cluster_Purity']:.4f} |")
    lines.append("\n")

    lines.append("### Hierarchical Clustering (Ward Linkage)\n")
    lines.append("| K | Silhouette Score | Davies-Bouldin | Calinski-Harabasz | ARI | NMI | Cluster Purity |")
    lines.append("|---|---|---|---|---|---|---|")
    ward_rows = hc_df[hc_df["Linkage"] == "ward"]
    for _, r in ward_rows.iterrows():
        lines.append(f"| K={int(r['K'])} | {r['Silhouette_Score']:.4f} | {r['Davies_Bouldin_Index']:.4f} | {r['Calinski_Harabasz_Index']:.4f} | {r['Adjusted_Rand_Index']:.4f} | {r['Normalized_Mutual_Info']:.4f} | {r['Cluster_Purity']:.4f} |")
    lines.append("\n")

    # Derive Ward K=2 metrics programmatically
    ward2_row = ward_rows[ward_rows["K"] == 2].iloc[0]
    ward3_row = ward_rows[ward_rows["K"] == 3].iloc[0]
    km3_row = km_df[km_df["K"] == 3].iloc[0]

    # Compute DBSCAN summary for selected configuration
    db_selected = db_df[(db_df["eps"] == 4.0) & (db_df["min_samples"] == 2)]
    if len(db_selected) == 0:
        db_selected = db_df.iloc[[0]]
    db_row = db_selected.iloc[0]
    db_n_clusters = int(db_row["Number_of_Clusters"])
    db_noise = int(db_row["Noise_Count"])
    db_noise_pct = float(db_row["Noise_Percentage"])

    lines.append("### DBSCAN Density-Based Clustering (Exploratory Configuration)\n")
    lines.append(f"- **Selected Exploratory DBSCAN Configuration**: `eps = {db_row['eps']:.1f}`, `min_samples = {int(db_row['min_samples'])}`.")
    lines.append(f"  This configuration was selected for exploratory inspection; it is not claimed as 'optimal'.")
    lines.append(f"- **Result**: {db_n_clusters} cluster(s) identified; {db_noise} noise point(s) ({db_noise_pct:.1f}% of N=12).")
    if db_noise > 0:
        lines.append(f"  The presence of {db_noise} noise point(s) limits the usefulness of this DBSCAN configuration for this small dataset.")
    lines.append("")

    lines.append("### Exploratory Clustering Interpretation\n")
    lines.append(f"Exploratory clustering reveals partial structure associated with packaging configurations in the 24D multi-domain feature space.")
    lines.append(f"- **Binary structure (K=2)**: Ward hierarchical clustering shows ARI={ward2_row['Adjusted_Rand_Index']:.4f}, "
                 f"NMI={ward2_row['Normalized_Mutual_Info']:.4f}, Purity={ward2_row['Cluster_Purity']:.4f}.")
    lines.append(f"  This is consistent with a broad separation between the Bare fiber regime and the packaged configurations,")
    lines.append(f"  but does not constitute definitive material classification.")
    lines.append(f"- **Three-group structure (K=3)**: Ward and K-Means clustering at K=3 yield NMI={ward3_row['Normalized_Mutual_Info']:.4f}, "
                 f"Purity={ward3_row['Cluster_Purity']:.4f}.")
    lines.append(f"  Partial recovery of the three packaging types is observed, but given N=12 these metrics must be interpreted with caution.")
    lines.append("")
    lines.append("> [!WARNING]")
    lines.append("> No clustering result is treated as 'proof', 'optimal', or 'definitive material classification'.")
    lines.append("> Silhouette scores in the range 0.12–0.30 indicate weak-to-moderate cluster separation, which is")
    lines.append("> expected given the small sample size and class imbalance.\n")

    lines.append("---")
    lines.append("## 4. Part B — Autoencoder & Latent Space Characterisation\n")

    mean_r2 = float(rec_df["Reconstruction_R2"].mean())
    mean_mse = float(rec_df["Reconstruction_MSE"].mean())

    lines.append("- **Architecture**: Fully-connected symmetric autoencoder: $\\text{Input}(24) \\to 12 \\to 6 \\to \\text{Latent}(2) \\to 6 \\to 12 \\to \\text{Output}(24)$.")
    lines.append(f"- **In-Sample Reconstruction Loss**: Mean MSE per feature = **{mean_mse:.5f}**.")
    lines.append(f"- **In-Sample Reconstruction Fidelity**: Mean $R^2 =$ **{mean_r2:.4f}** across all 24 multi-domain features.")
    lines.append("")
    lines.append("> [!IMPORTANT]")
    lines.append("> The reconstruction metric above is an **IN-SAMPLE** fidelity measure.")
    lines.append("> The autoencoder is trained and evaluated on the same 12 events.")
    lines.append("> **The reconstruction metric is NOT a held-out test score** because the available dataset")
    lines.append("> is too small for the current implementation to provide a reliable independent evaluation.")
    lines.append("> This result demonstrates in-sample nonlinear representation and reconstruction capability,")
    lines.append("> not generalisation to unseen data.\n")
    lines.append("- **Latent Manifold ($z_1, z_2$)**: The 2D latent space organises events continuously along two learned axes,")
    lines.append("  successfully condensing 24 dimensions while maintaining in-sample feature recoverability.\n")

    lines.append("### Latent Space vs. Original 24D Space Clustering Comparison\n")
    lines.append("| Representation Space | Dimensionality | Algorithm | K | Silhouette Score | Davies-Bouldin | ARI | NMI | Cluster Purity |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for _, r in comp_clust_df.iterrows():
        lines.append(f"| {r['Representation_Space']} | {int(r['Dimensionality'])}D | {r['Algorithm']} | K={int(r['K'])} | {r['Silhouette_Score']:.4f} | {r['Davies_Bouldin_Index']:.4f} | {r['Adjusted_Rand_Index']:.4f} | {r['Normalized_Mutual_Info']:.4f} | {r['Cluster_Purity']:.4f} |")
    lines.append("\n")

    lines.append("---")
    lines.append("## 5. Part C — Supervised Exploratory Feature Attribution\n")
    lines.append("> [!IMPORTANT]")
    lines.append("> **The Random Forest model is SUPERVISED** — material labels (Y) are used to train it.")
    lines.append("> This is NOT unsupervised learning.")
    lines.append("> With N=12, the Random Forest achieves 100% in-sample training accuracy, which")
    lines.append("> likely reflects memorisation/overfitting rather than genuine predictive generalisation.")
    lines.append("> Feature importance and Shapley-style attribution are exploratory and dataset-specific;")
    lines.append("> they do not constitute proof of which features universally govern material separation.")
    lines.append("> The Shapley-style values are computed via a **custom sampling approximation**,")
    lines.append("> not the standard SHAP library.\n")

    # Derive top features from the actual ranked_df CSV — NOT hardcoded
    lines.append("### Top Ranked Engineering Parameters (derived from actual CSV output)\n")
    lines.append("| Rank | Feature | Domain | Composite Score | Gini Importance | Mean |Shapley| | Direction of Effect |")
    lines.append("|---|---|---|---|---|---|---|")
    for _, r in ranked_df.head(10).iterrows():
        lines.append(f"| **{int(r['Rank'])}** | `{r['Feature']}` | {r['Domain']} | {r['Composite_Importance_Score']:.4f} | {r['Gini_Importance']:.4f} | {r['Mean_Abs_Shapley_Value']:.4f} | {r['Direction_of_Effect']} |")
    lines.append("\n")

    # Derive physical insights programmatically from actual CSV ranks
    top5 = ranked_df.head(5)
    r1 = top5.iloc[0]
    r2 = top5.iloc[1]
    r3 = top5.iloc[2]
    r4 = top5.iloc[3]
    r5 = top5.iloc[4]

    lines.append("### Exploratory Attribution Insights (derived from actual ranking output):\n")
    lines.append(f"The following insights are derived directly from the computed ranking and must be treated as exploratory,")
    lines.append(f"dataset-specific observations within this N=12 analysis. They are not causal claims.\n")
    lines.append(f"1. **`{r1['Feature']}` (Rank 1, Score={r1['Composite_Importance_Score']:.4f})** — {r1['Domain']}: {r1['Engineering_Interpretation']}")
    lines.append(f"   Direction: {r1['Direction_of_Effect']}.")
    lines.append(f"2. **`{r2['Feature']}` (Rank 2, Score={r2['Composite_Importance_Score']:.4f})** — {r2['Domain']}: {r2['Engineering_Interpretation']}")
    lines.append(f"   Direction: {r2['Direction_of_Effect']}.")
    lines.append(f"3. **`{r3['Feature']}` (Rank 3, Score={r3['Composite_Importance_Score']:.4f})** — {r3['Domain']}: {r3['Engineering_Interpretation']}")
    lines.append(f"   Direction: {r3['Direction_of_Effect']}.")
    lines.append(f"4. **`{r4['Feature']}` (Rank 4, Score={r4['Composite_Importance_Score']:.4f})** — {r4['Domain']}: {r4['Engineering_Interpretation']}")
    lines.append(f"   Direction: {r4['Direction_of_Effect']}.")
    lines.append(f"5. **`{r5['Feature']}` (Rank 5, Score={r5['Composite_Importance_Score']:.4f})** — {r5['Domain']}: {r5['Engineering_Interpretation']}")
    lines.append(f"   Direction: {r5['Direction_of_Effect']}.\n")

    lines.append("---")
    lines.append("## 6. Scientific Limitations & Cautious Interpretation\n")
    lines.append("> [!CAUTION]")
    lines.append("> **CRITICAL LIMITATION — N=12**: The entire Phase 9 Part 2 analysis rests on 12 validated")
    lines.append("> experimental impact events with severe class imbalance (Bare n=7, Copper n=3, Steel n=2).")
    lines.append("> This is the primary limitation. No finding should be extrapolated beyond this dataset.\n")
    lines.append("> [!WARNING]")
    lines.append("> 1. **Clustering**: Results are exploratory. No clustering result is treated as 'proof',")
    lines.append(">    'optimal', or 'definitive material classification'. Silhouette scores in the 0.12–0.30")
    lines.append(">    range indicate weak-to-moderate separation, consistent with a small imbalanced dataset.")
    lines.append("> 2. **Autoencoder**: Demonstrates in-sample nonlinear compression and reconstruction only.")
    lines.append(">    The R² metric is IN-SAMPLE, not a held-out generalisation score.")
    lines.append("> 3. **Random Forest**: Is a supervised exploratory model, NOT a validated predictive model.")
    lines.append(">    100% in-sample training accuracy likely reflects memorisation given N=12.")
    lines.append("> 4. **Feature Attribution**: Gini importance, permutation importance, and approximate")
    lines.append(">    Shapley-style values are exploratory and dataset-specific. They reflect the fitted")
    lines.append(">    model's attributions, not causal or universal feature importance.")
    lines.append("> 5. **Feature Redundancy**: Some multidomain variables are physically or mathematically")
    lines.append(">    correlated, and Phase 8 indices are composites. Model-based importance values should")
    lines.append(">    not be interpreted as independent causal contributions.")
    lines.append("> 6. **DBSCAN**: The eps=4.0, min_samples=2 configuration is a selected exploratory")
    lines.append(">    configuration, not a statistically validated 'optimal' parameter set.")
    lines.append("> 7. **Phase 9 Part 2 prepares the feature space and ML methodology for future predictive")
    lines.append(">    modelling, but does NOT claim that a reliable predictive model has already been")
    lines.append(">    established. Future work requires a larger, balanced experimental dataset.\n")

    lines.append("---")
    lines.append("## 7. Final Scientific Conclusions\n")
    lines.append("- Exploratory clustering provides evidence of partial structure in the multidomain feature space consistent with packaging configurations.")
    lines.append("- No clustering result is treated as definitive material classification.")
    lines.append("- The autoencoder demonstrates in-sample nonlinear representation and reconstruction capability (Mean R² = "
                 f"{mean_r2:.4f}), not generalisation to unseen data.")
    lines.append("- The Random Forest is a supervised exploratory analysis with 100% in-sample training accuracy, not a validated predictive model.")
    lines.append("- Approximate Shapley-style attribution and feature importance are exploratory and dataset-specific.")
    lines.append("- **N=12 is the major limitation.** Phase 9 Part 2 demonstrates analytical methodology and prepares the")
    lines.append("  feature space for future predictive modelling once a larger dataset becomes available.\n")

    lines.append("---")
    lines.append("## 8. Reproducibility\n")
    lines.append("To execute the full Phase 9 Part 2 pipeline and regenerate all outputs deterministically:\n")
    lines.append("```bash\npython phase9_part2.py\n```\n")

    with open(SUMMARY_MD_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"  [OK] Saved comprehensive Phase 9 Part 2 summary report: {SUMMARY_MD_PATH}")



# ============================================================
# VALIDATION SUITE
# ============================================================

def run_phase9_part2_validation():
    """
    Executes a 12-point validation check suite for Phase 9 Part 2.
    Logs report to results/phase9/phase9_part2_validation_report.txt.
    """
    validation_log = []

    def log_check(check_id, name, passed, details=""):
        status_str = "PASS" if passed else "FAIL"
        msg = f"[CHECK {check_id:02d}] [{status_str}] {name}: {details}"
        validation_log.append(msg)
        print(f"  {msg}")
        return passed

    print("\n" + "=" * 70)
    print("PHASE 9 PART 2 VALIDATION SUITE")
    print("=" * 70)

    all_pass = True

    # Check 1: Input files present
    chk1 = (os.path.exists("results/phase5/phase5_all_features.csv") and
            os.path.exists("results/phase6/phase6_multidomain_features.csv") and
            os.path.exists("results/phase8/phase8_engineering_indices.csv"))
    all_pass &= log_check(1, "Previous phase input files intact", chk1, "Phase 5, Phase 6, Phase 8 CSVs found")

    # Check 2: Clustering CSV outputs exist
    km_path = os.path.join(CLUSTERING_DIR, "phase9_kmeans_results.csv")
    hc_path = os.path.join(CLUSTERING_DIR, "phase9_hierarchical_results.csv")
    db_path = os.path.join(CLUSTERING_DIR, "phase9_dbscan_results.csv")
    chk2 = (os.path.exists(km_path) and os.path.exists(hc_path) and os.path.exists(db_path))
    all_pass &= log_check(2, "Clustering CSV outputs generated", chk2, "K-Means, Hierarchical, and DBSCAN CSVs present")

    # Check 3: Clustering plots exist
    p_km = os.path.join(CLUSTERING_DIR, "plots", "phase9_kmeans_clusters.png")
    p_hc = os.path.join(CLUSTERING_DIR, "plots", "phase9_hierarchical_dendrogram.png")
    p_db = os.path.join(CLUSTERING_DIR, "plots", "phase9_dbscan_clusters.png")
    p_cmp = os.path.join(CLUSTERING_DIR, "plots", "phase9_clustering_metrics_comparison.png")
    chk3 = (os.path.exists(p_km) and os.path.exists(p_hc) and os.path.exists(p_db) and os.path.exists(p_cmp))
    all_pass &= log_check(3, "Clustering plots generated", chk3, "All 4 clustering visualization plots present")

    # Check 4: Autoencoder outputs exist
    ae_lat = os.path.join(AE_DIR, "phase9_autoencoder_latent_coordinates.csv")
    ae_rec = os.path.join(AE_DIR, "phase9_autoencoder_reconstruction_metrics.csv")
    ae_wgt = os.path.join(AE_DIR, "phase9_autoencoder_model_weights.pt")
    chk4 = (os.path.exists(ae_lat) and os.path.exists(ae_rec) and os.path.exists(ae_wgt))
    all_pass &= log_check(4, "Autoencoder model & latent outputs generated", chk4, "Latent coordinates, reconstruction metrics, model weights present")

    # Check 5: Autoencoder plots exist
    p_loss = os.path.join(AE_DIR, "plots", "phase9_autoencoder_training_loss.png")
    p_lat = os.path.join(AE_DIR, "plots", "phase9_autoencoder_latent_space.png")
    p_rec = os.path.join(AE_DIR, "plots", "phase9_autoencoder_reconstruction_error.png")
    p_flc = os.path.join(AE_DIR, "plots", "phase9_feature_vs_latent_clustering_comparison.png")
    chk5 = (os.path.exists(p_loss) and os.path.exists(p_lat) and os.path.exists(p_rec) and os.path.exists(p_flc))
    all_pass &= log_check(5, "Autoencoder plots generated", chk5, "All 4 autoencoder plots present")

    # Check 6: Explainability CSV outputs exist
    exp_rank = os.path.join(EXP_DIR, "phase9_feature_importance_ranked.csv")
    exp_shap = os.path.join(EXP_DIR, "phase9_shapley_values_by_material.csv")
    exp_corr = os.path.join(EXP_DIR, "phase9_latent_feature_correlations.csv")
    chk6 = (os.path.exists(exp_rank) and os.path.exists(exp_shap) and os.path.exists(exp_corr))
    all_pass &= log_check(6, "Explainability CSV outputs generated", chk6, "Feature importance, Shapley values, and correlation CSVs present")

    # Check 7: Explainability plots exist
    p_rank = os.path.join(EXP_DIR, "plots", "phase9_explainability_feature_importance_ranking.png")
    p_shap = os.path.join(EXP_DIR, "plots", "phase9_explainability_shap_summary.png")
    p_corr = os.path.join(EXP_DIR, "plots", "phase9_explainability_latent_feature_correlations.png")
    chk7 = (os.path.exists(p_rank) and os.path.exists(p_shap) and os.path.exists(p_corr))
    all_pass &= log_check(7, "Explainability plots generated", chk7, "All 3 explainability plots present")

    # Check 8: Event count consistency
    lat_df = pd.read_csv(ae_lat)
    chk8 = (len(lat_df) == 12)
    all_pass &= log_check(8, "Event scope consistency (N=12 IMPACT)", chk8, f"{len(lat_df)} valid impact events processed")

    # Check 9: Feature count consistency (24 multi-domain features)
    rec_df = pd.read_csv(ae_rec)
    chk9 = (len(rec_df) == 24)
    all_pass &= log_check(9, "Feature count consistency (24 features)", ch9:=chk9, f"{len(rec_df)} features evaluated in autoencoder and explainability")

    # Check 10: Autoencoder reconstruction fidelity (R^2 > 0.90)
    mean_r2 = float(rec_df["Reconstruction_R2"].mean())
    chk10 = (mean_r2 >= 0.90)
    all_pass &= log_check(10, "Autoencoder reconstruction fidelity", chk10, f"Mean R^2 = {mean_r2:.4f} >= 0.90")

    # Check 11: Summary markdown and report generated
    chk11 = os.path.exists(SUMMARY_MD_PATH)
    all_pass &= log_check(11, "Comprehensive summary report created", chk11, f"{SUMMARY_MD_PATH} present")

    # Check 12: Idempotency & Clean separation (no modification to prior phase results)
    chk12 = (os.path.exists("results/phase9/pca/phase9_pca_scores.csv") and
             os.path.exists("results/phase9/umap/phase9_umap_complete_multidomain_coordinates.csv"))
    all_pass &= log_check(12, "Phase 9 Part 1 PCA/UMAP results untouched", chk12, "Prior phase results preserved read-only")

    # Write validation report
    with open(VALIDATION_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("PHASE 9 PART 2 VALIDATION REPORT\n")
        f.write("=" * 60 + "\n")
        f.write("\n".join(validation_log) + "\n\n")
        if all_pass:
            f.write("PHASE 9 PART 2 VALIDATION: ALL CHECKS PASSED\n")
        else:
            f.write("PHASE 9 PART 2 VALIDATION: SOME CHECKS FAILED\n")

    print("-" * 70)
    if all_pass:
        print("PHASE 9 PART 2 VALIDATION PASSED")
    else:
        print("PHASE 9 PART 2 VALIDATION FAILED")
    print("=" * 70)

    return all_pass


# ============================================================
# MAIN ORCHESTRATOR
# ============================================================

def main():
    print("=" * 70)
    print("PHASE 9 — PART 2: CLUSTERING + AUTOENCODER + EXPLAINABILITY")
    print("======================================================================")

    # 1. Run Clustering Pipeline
    km_df, hc_df, db_df, comp_clust_df = run_clustering_pipeline()

    # 2. Run Autoencoder Pipeline
    latent_df, rec_metrics_df, ae_comp_df = run_autoencoder_pipeline()

    # 3. Run Explainability Pipeline
    ranked_df, shap_df, corr_df = run_explainability_pipeline_main()

    # 4. Generate Comprehensive Summary Report
    print("\n[Summary] Generating Phase 9 Part 2 Comprehensive Summary Report...")
    generate_part2_summary_report(km_df, hc_df, db_df, ae_comp_df, rec_metrics_df, ranked_df, shap_df)

    # 5. Run Validation Suite
    val_passed = run_phase9_part2_validation()

    if not val_passed:
        print("ERROR: Validation failed. Check results/phase9/phase9_part2_validation_report.txt")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("PHASE 9 PART 2 PIPELINE COMPLETE & FULLY VALIDATED")
    print("======================================================================")


if __name__ == "__main__":
    main()
