"""
PHASE 9 — PART 2C: EXPLAINABILITY & ENGINEERING PARAMETER RANKING
================================================================
Physics-Guided Multi-Domain Signal Intelligence Framework (PGMSIF)

Identifies and ranks the key engineering parameters and multi-domain features
that govern the observed FBG sensor response, packaging differentiation,
and latent space representation.

IMPORTANT SCIENTIFIC NOTES:
    - Random Forest is used as a SUPERVISED EXPLORATORY model (X = features, Y = material labels).
      It is NOT an unsupervised method. With N=12, 100% in-sample training accuracy indicates
      likely memorization/overfitting rather than genuine predictive generalisation.
    - Feature importance and attribution are EXPLORATORY and DATASET-SPECIFIC.
      They must not be interpreted as causal or universally generalisable.
    - The Shapley-style attribution uses a CUSTOM SAMPLING APPROXIMATION
      (NOT the standard SHAP library). Results are reported as
      "approximate Shapley-style attribution" to avoid misleading terminology.
    - All results must be interpreted cautiously given N=12.

Explainability Methodologies:
    1. Random Forest Gini / Impurity-based Feature Importance (MDI) — supervised, exploratory
    2. Permutation Feature Importance (50 repeats) — computed on training set, dataset-specific
    3. Approximate Shapley-Style Attribution (custom sampling, not the SHAP library)
    4. Autoencoder Latent Space Correlation & Gradient Sensitivity

Core Question Addressed (Exploratory):
    "Within this 12-event dataset, which engineering parameters receive the greatest
     attribution in a supervised exploratory model separating packaging configurations?"

Outputs Generated:
    - results/phase9/explainability/phase9_feature_importance_ranked.csv
    - results/phase9/explainability/phase9_shapley_values_by_material.csv
    - results/phase9/explainability/phase9_latent_feature_correlations.csv
    - results/phase9/explainability/plots/phase9_explainability_feature_importance_ranking.png
    - results/phase9/explainability/plots/phase9_explainability_shap_summary.png
    - results/phase9/explainability/plots/phase9_explainability_latent_feature_correlations.png
"""

import os
import sys
import re
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION AND FILE PATHS
# ============================================================

PHASE5_CSV = os.path.join("results", "phase5", "phase5_all_features.csv")
PHASE6_CSV = os.path.join("results", "phase6", "phase6_multidomain_features.csv")
PHASE8_CSV = os.path.join("results", "phase8", "phase8_engineering_indices.csv")
PCA_SCORES_CSV = os.path.join("results", "phase9", "pca", "phase9_pca_scores.csv")
AE_LATENT_CSV = os.path.join("results", "phase9", "autoencoder", "phase9_autoencoder_latent_coordinates.csv")

OUTPUT_DIR = os.path.join("results", "phase9", "explainability")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")

RANDOM_STATE = 42

# Domain classification for 24 multi-domain features
FEATURE_DOMAINS = {
    "peak_shift_abs": "Phase 5 Time-Domain",
    "rise_time_seconds": "Phase 5 Time-Domain",
    "signal_energy": "Phase 5 Time-Domain",
    "rms": "Phase 5 Time-Domain",
    "peak_to_peak": "Phase 5 Time-Domain",
    "std_dev": "Phase 5 Time-Domain",
    "entropy": "Phase 5 Time-Domain",
    "max_slope_abs": "Phase 5 Time-Domain",
    "auc_abs": "Phase 5 Time-Domain",
    "noise_std_nm": "Phase 5 Time-Domain",
    "Dominant_Frequency": "Phase 6 Spectral FFT",
    "Spectral_Energy": "Phase 6 Spectral FFT",
    "Spectral_Entropy": "Phase 6 Spectral FFT",
    "Spectral_Centroid": "Phase 6 Spectral FFT",
    "Bandwidth": "Phase 6 Spectral FFT",
    "Approximation_Energy": "Phase 6 Wavelet",
    "Detail_Energy": "Phase 6 Wavelet",
    "Wavelet_Energy": "Phase 6 Wavelet",
    "Wavelet_Entropy": "Phase 6 Wavelet",
    "Detail_Approx_Ratio": "Phase 6 Wavelet",
    "DSTI": "Phase 8 Physics-Guided",
    "PEI": "Phase 8 Physics-Guided",
    "SII": "Phase 8 Physics-Guided",
    "DRI": "Phase 8 Physics-Guided",
}

DOMAIN_COLORS = {
    "Phase 5 Time-Domain": "#1f77b4",
    "Phase 6 Spectral FFT": "#9467bd",
    "Phase 6 Wavelet": "#2ca02c",
    "Phase 8 Physics-Guided": "#ff7f0e",
}

# Physical interpretation glossary for all 24 features
FEATURE_INTERPRETATIONS = {
    "peak_shift_abs": "Peak wavelength excursion directly proportional to maximum mechanical strain transfer.",
    "rms": "Root mean square amplitude reflecting effective dynamic deformation energy over the impact event.",
    "signal_energy": "Cumulative energy of the transient strain burst; elevated in direct, unbuffered impacts.",
    "DSTI": "Dynamic Strain Transfer Index capturing strain transfer magnitude and dynamic response speed.",
    "max_slope_abs": "Peak strain rate of change; measures structural velocity of impact onset.",
    "PEI": "Packaging Efficiency Index quantifying dynamic attenuation and mechanical damping capacity.",
    "Spectral_Centroid": "Center frequency of the vibration power spectrum; higher in stiffened packaging (Steel).",
    "Dominant_Frequency": "Primary structural resonance excited by the kinetic impact.",
    "Wavelet_Energy": "Total multi-scale discrete wavelet transform energy across decomposition levels.",
    "Detail_Energy": "High-frequency transient wavelet sub-band energy indicative of sharp shock waves.",
    "Approximation_Energy": "Low-frequency baseline wavelet energy corresponding to bulk flexural deformation.",
    "Detail_Approx_Ratio": "Ratio of high-frequency shock content to bulk deformation; elevated in rigid packaging.",
    "Spectral_Energy": "Total power spectral density magnitude across all Fourier frequencies.",
    "Spectral_Entropy": "Flatness and complexity of the spectral distribution; high entropy implies broadband shock.",
    "Wavelet_Entropy": "Distribution complexity across wavelet sub-bands.",
    "DRI": "Dynamic Response Index evaluating onset velocity, rise time, and recovery speed.",
    "SII": "Signal Integrity Index reflecting optical signal-to-noise quality and baseline stability.",
    "peak_to_peak": "Total span between maximum positive and negative strain excursions.",
    "std_dev": "Standard deviation of dynamic strain oscillations.",
    "entropy": "Information entropy of the raw optical strain time series.",
    "auc_abs": "Absolute area under the curve; measure of total integrated impulse.",
    "rise_time_seconds": "Duration from impact onset to peak deformation; shorter in direct coupling.",
    "Bandwidth": "Frequency spread of the optical sensor response.",
    "noise_std_nm": "Pre-impact optical baseline noise level reflecting intrinsic interrogation stability.",
}


# ============================================================
# 1. DATA LOADING
# ============================================================

def load_and_prepare_data():
    """
    Loads Phase 5, Phase 6, Phase 8, PCA, and Autoencoder outputs.
    Extracts the unified 24 multi-domain features for the 12 IMPACT events.
    """
    p5 = pd.read_csv(PHASE5_CSV)
    p6 = pd.read_csv(PHASE6_CSV)
    p8 = pd.read_csv(PHASE8_CSV)

    p5["Expert"] = p5["Dataset"]
    p5["FBG"] = p5["Sensor"]

    p6["Expert_Num"] = p6["File"].apply(lambda x: int(re.search(r"expert(\d+)", str(x), re.IGNORECASE).group(1)))
    p6["Expert"] = "Expert " + p6["Expert_Num"].astype(str)
    p6["FBG"] = p6["Sensor"].str.replace("_processed", "", regex=False)

    p5_imp = p5[p5["Impact_Status"] == "IMPACT"].copy()
    p6_imp = p6[p6["Phase4_Result"] == "IMPACT"].copy()
    p8_imp = p8[p8["Impact_Status"] == "IMPACT"].copy()

    merged = p5_imp.merge(p6_imp, on=["Expert", "FBG"], suffixes=("_p5", "_p6"))
    merged = merged.merge(p8_imp, on=["Expert", "FBG"], suffixes=("", "_p8"))

    all_features = [
        "peak_shift_abs", "rise_time_seconds", "signal_energy", "rms",
        "peak_to_peak", "std_dev", "entropy", "max_slope_abs", "auc_abs", "noise_std_nm",
        "Dominant_Frequency", "Spectral_Energy", "Spectral_Entropy", "Spectral_Centroid", "Bandwidth",
        "Approximation_Energy", "Detail_Energy", "Wavelet_Energy", "Wavelet_Entropy", "Detail_Approx_Ratio",
        "DSTI", "PEI", "SII", "DRI"
    ]

    pca_df = pd.read_csv(PCA_SCORES_CSV) if os.path.exists(PCA_SCORES_CSV) else None
    ae_latent_df = pd.read_csv(AE_LATENT_CSV) if os.path.exists(AE_LATENT_CSV) else None

    return merged, all_features, pca_df, ae_latent_df


# ============================================================
# 2. SHAPLEY VALUE ATTRIBUTION ALGORITHM
# ============================================================

def compute_sampled_shapley_values(rf_model, X_scaled, labels_encoded, feature_names, n_samples=100):
    """
    Computes approximate Shapley-style attribution values for multi-class classification.

    NOTE: This is a CUSTOM SAMPLING APPROXIMATION — it is NOT the standard SHAP library
    (e.g., TreeExplainer). The algorithm samples random permutations and computes marginal
    contributions of each feature to the model's probability output. Results should be
    reported as 'approximate Shapley-style attribution' rather than 'SHAP values'.

    Because the model is trained on only 12 events (N=12), the resulting attribution
    values are exploratory and dataset-specific. They do not constitute causal proof
    of any feature's universal importance.
    """
    np.random.seed(RANDOM_STATE)
    n_instances, n_features = X_scaled.shape
    n_classes = len(np.unique(labels_encoded))

    shap_values = np.zeros((n_classes, n_features))

    for j in range(n_features):
        X_with = np.zeros((n_samples, n_features))
        X_without = np.zeros((n_samples, n_features))

        for s in range(n_samples):
            idx = np.random.randint(0, n_instances)
            x_inst = X_scaled[idx]

            perm = np.random.permutation(n_features)
            j_idx = np.where(perm == j)[0][0]
            subset = perm[:j_idx]

            ref_idx = np.random.randint(0, n_instances)
            x_ref = X_scaled[ref_idx].copy()

            x_wo = x_ref.copy()
            x_wo[subset] = x_inst[subset]

            x_w = x_wo.copy()
            x_w[j] = x_inst[j]

            X_without[s] = x_wo
            X_with[s] = x_w

        # Batched prediction for all samples and classes
        prob_with = rf_model.predict_proba(X_with) # (n_samples, n_classes)
        prob_without = rf_model.predict_proba(X_without) # (n_samples, n_classes)

        diff = prob_with - prob_without # (n_samples, n_classes)
        for c in range(n_classes):
            shap_values[c, j] = float(np.mean(diff[:, c]))

    return shap_values



# ============================================================
# 3. EXPLAINABILITY PIPELINE COMPUTATION
# ============================================================

def run_explainability_pipeline(X_scaled, labels_true, all_features, pca_df, ae_latent_df):
    """
    Executes multi-method exploratory feature attribution suite.

    IMPORTANT: Random Forest is SUPERVISED — labels are used to fit the model.
    This is an exploratory supervised model, NOT unsupervised learning.
    With N=12, training accuracy is expected to be 100% (in-sample), which
    likely reflects memorisation/overfitting rather than genuine generalisation.
    All feature importance results are exploratory and dataset-specific.
    """
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels_true)
    class_names = le.classes_ # ["Bare", "Copper", "Steel"]

    # 1. Random Forest Gini Importance & Permutation Importance (SUPERVISED EXPLORATORY)
    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    rf.fit(X_scaled, labels_encoded)

    # Compute and report in-sample training accuracy
    from sklearn.metrics import accuracy_score
    train_preds = rf.predict(X_scaled)
    train_accuracy = float(accuracy_score(labels_encoded, train_preds))
    print(f"  [RF] IN-SAMPLE training accuracy: {train_accuracy*100:.1f}% (N=12; likely reflects memorisation, not generalisation)")

    # Permutation importance computed on the training set — dataset-specific
    perm = permutation_importance(rf, X_scaled, labels_encoded, n_repeats=50, random_state=RANDOM_STATE)

    # 2. Approximate Shapley-Style Attribution (custom sampling — NOT the SHAP library)
    shap_matrix = compute_sampled_shapley_values(rf, X_scaled, labels_encoded, all_features, n_samples=100)
    mean_abs_shap = np.mean(np.abs(shap_matrix), axis=0)

    # 3. Latent Space & PCA Correlations
    z1_corr, z2_corr = [], []
    pc1_corr, pc2_corr = [], []

    z1 = ae_latent_df["z1"].values if ae_latent_df is not None else np.zeros(len(X_scaled))
    z2 = ae_latent_df["z2"].values if ae_latent_df is not None else np.zeros(len(X_scaled))
    pc1 = pca_df["PC1"].values if pca_df is not None else np.zeros(len(X_scaled))
    pc2 = pca_df["PC2"].values if pca_df is not None else np.zeros(len(X_scaled))

    for i in range(len(all_features)):
        feat_vals = X_scaled[:, i]
        z1_corr.append(float(np.corrcoef(feat_vals, z1)[0, 1]) if np.std(z1) > 0 else 0.0)
        z2_corr.append(float(np.corrcoef(feat_vals, z2)[0, 1]) if np.std(z2) > 0 else 0.0)
        pc1_corr.append(float(np.corrcoef(feat_vals, pc1)[0, 1]) if np.std(pc1) > 0 else 0.0)
        pc2_corr.append(float(np.corrcoef(feat_vals, pc2)[0, 1]) if np.std(pc2) > 0 else 0.0)

    # 4. Determine Direction of Effect (Packaging Discrimination)
    # Compare mean standardized value of Bare vs Packaged (Copper + Steel)
    bare_mask = (labels_true == "Bare")
    pkg_mask = ~bare_mask
    effect_directions = []
    for i in range(len(all_features)):
        bare_mean = float(np.mean(X_scaled[bare_mask, i]))
        pkg_mean = float(np.mean(X_scaled[pkg_mask, i]))
        if bare_mean > pkg_mean + 0.3:
            effect_directions.append("Elevated in Bare (Direct Strain Transfer)")
        elif pkg_mean > bare_mean + 0.3:
            effect_directions.append("Elevated in Packaged (Buffering & Stiffness)")
        else:
            effect_directions.append("Balanced across configurations")

    # 5. Composite Ranking Score
    # Composite score combining Gini Importance (0.35), Permutation Importance (0.35),
    # and Mean Absolute Approximate Shapley-Style Attribution (0.30).
    # All components are in-sample and exploratory (N=12).
    gini_norm = rf.feature_importances_ / (np.max(rf.feature_importances_) + 1e-12)
    perm_norm = perm.importances_mean / (np.max(perm.importances_mean) + 1e-12) if np.max(perm.importances_mean) > 0 else gini_norm
    shap_norm = mean_abs_shap / (np.max(mean_abs_shap) + 1e-12)

    composite_importance = 0.35 * gini_norm + 0.35 * perm_norm + 0.30 * shap_norm

    ranked_records = []
    for i, feat in enumerate(all_features):
        ranked_records.append({
            "Feature": feat,
            "Domain": FEATURE_DOMAINS.get(feat, "Unknown"),
            "Composite_Importance_Score": float(composite_importance[i]),
            "Gini_Importance": float(rf.feature_importances_[i]),
            "Permutation_Importance_Mean": float(perm.importances_mean[i]),
            "Permutation_Importance_Std": float(perm.importances_std[i]),
            "Mean_Abs_Shapley_Value": float(mean_abs_shap[i]),
            "Direction_of_Effect": effect_directions[i],
            "Engineering_Interpretation": FEATURE_INTERPRETATIONS.get(feat, "N/A"),
        })

    ranked_df = pd.DataFrame(ranked_records).sort_values(by="Composite_Importance_Score", ascending=False)
    ranked_df["Rank"] = range(1, len(ranked_df) + 1)
    ranked_df = ranked_df[["Rank", "Feature", "Domain", "Composite_Importance_Score",
                           "Gini_Importance", "Permutation_Importance_Mean", "Mean_Abs_Shapley_Value",
                           "Direction_of_Effect", "Engineering_Interpretation"]]

    # Shapley values table by material
    shap_records = []
    for i, feat in enumerate(all_features):
        row = {"Feature": feat, "Domain": FEATURE_DOMAINS.get(feat, "Unknown")}
        for c_idx, c_name in enumerate(class_names):
            row[f"Shapley_{c_name}"] = float(shap_matrix[c_idx, i])
        row["Mean_Absolute_Shapley"] = float(mean_abs_shap[i])
        shap_records.append(row)
    shap_df = pd.DataFrame(shap_records).sort_values(by="Mean_Absolute_Shapley", ascending=False)

    # Latent correlations table
    corr_records = []
    for i, feat in enumerate(all_features):
        corr_records.append({
            "Feature": feat,
            "Domain": FEATURE_DOMAINS.get(feat, "Unknown"),
            "Corr_Latent_Z1": z1_corr[i],
            "Corr_Latent_Z2": z2_corr[i],
            "Corr_PCA_PC1": pc1_corr[i],
            "Corr_PCA_PC2": pc2_corr[i],
        })
    corr_df = pd.DataFrame(corr_records)

    return ranked_df, shap_df, corr_df


# ============================================================
# 4. VISUALIZATION FUNCTIONS
# ============================================================

def generate_explainability_plots(ranked_df, shap_df, corr_df, output_dir):
    """Generates analytical explainability plots."""
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    created_plots = []

    # ---------------------------------------------------------
    # Plot 1: Composite Feature Importance Ranking
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 7.5))
    df_sorted = ranked_df.sort_values(by="Composite_Importance_Score", ascending=True)

    colors = [DOMAIN_COLORS[d] for d in df_sorted["Domain"]]
    bars = ax.barh(df_sorted["Feature"], df_sorted["Composite_Importance_Score"], color=colors, edgecolor="black", height=0.65)

    # Legend for feature domains
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor="black") for color in DOMAIN_COLORS.values()]
    ax.legend(handles, DOMAIN_COLORS.keys(), loc="lower right", fontsize=9.5, title="Feature Domain", title_fontsize=10)

    ax.set_title("Phase 9 — Ranked Engineering Parameters by Composite Importance Score\n(Integrating RF Gini, Permutation Importance, and Approx. Shapley-Style Attribution — In-Sample, Exploratory)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Composite Importance Score [0, 1]", fontsize=10.5)
    ax.grid(True, linestyle="--", alpha=0.35, axis="x")

    plt.tight_layout()
    p1 = os.path.join(plots_dir, "phase9_explainability_feature_importance_ranking.png")
    fig.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    created_plots.append(p1)

    # ---------------------------------------------------------
    # Plot 2: Approximate Shapley-Style Attribution by Material Class
    # (Custom sampling approximation — NOT the standard SHAP library)
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(13, 6.5))
    top_shap = shap_df.head(12).sort_values(by="Mean_Absolute_Shapley", ascending=True)

    y = np.arange(len(top_shap))
    height = 0.25

    ax.barh(y + height, np.abs(top_shap["Shapley_Bare"]), height, label="Bare (FBG2)", color="#1f77b4", edgecolor="black")
    ax.barh(y, np.abs(top_shap["Shapley_Copper"]), height, label="Copper (FBG1)", color="#d62728", edgecolor="black")
    ax.barh(y - height, np.abs(top_shap["Shapley_Steel"]), height, label="Steel (FBG3)", color="#2ca02c", edgecolor="black")

    ax.set_yticks(y)
    ax.set_yticklabels(top_shap["Feature"], fontsize=9.5)
    ax.set_title("Phase 9 — Top 12 Parameters: Approximate Shapley-Style Attribution by Packaging Configuration\n(Custom Sampling Approximation, NOT SHAP Library — In-Sample, Exploratory)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Mean Absolute Approximate Shapley-Style Attribution |φ|", fontsize=10.5)
    ax.grid(True, linestyle="--", alpha=0.35, axis="x")
    ax.legend(loc="lower right", fontsize=10)

    plt.tight_layout()
    p2 = os.path.join(plots_dir, "phase9_explainability_shap_summary.png")
    fig.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    created_plots.append(p2)

    # ---------------------------------------------------------
    # Plot 3: Latent Space & PCA Feature Correlations Heatmap
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 8.5))
    corr_matrix = corr_df[["Corr_Latent_Z1", "Corr_Latent_Z2", "Corr_PCA_PC1", "Corr_PCA_PC2"]].values

    im = ax.imshow(corr_matrix, cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(4))
    ax.set_xticklabels(["Latent z1", "Latent z2", "PCA PC1", "PCA PC2"], fontsize=11, fontweight="bold")

    ax.set_yticks(range(len(corr_df)))
    ax.set_yticklabels(corr_df["Feature"], fontsize=9)

    # Annotate values
    for i in range(len(corr_df)):
        for j in range(4):
            val = corr_matrix[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color="white" if abs(val) > 0.6 else "black", fontsize=8)

    plt.colorbar(im, ax=ax, label="Pearson Correlation Coefficient")
    ax.set_title("Phase 9 — Feature Alignment with Autoencoder Latent Space (z1, z2) & PCA (PC1, PC2)", fontsize=12, fontweight="bold")
    plt.tight_layout()

    p3 = os.path.join(plots_dir, "phase9_explainability_latent_feature_correlations.png")
    fig.savefig(p3, dpi=150, bbox_inches="tight")
    plt.close(fig)
    created_plots.append(p3)

    return created_plots


# ============================================================
# 5. MAIN EXECUTION
# ============================================================

def run_explainability_pipeline_main():
    """Master pipeline for Phase 9 Part 2C: Explainability."""
    print("=" * 70)
    print("PHASE 9 — PART 2C: EXPLAINABILITY & PARAMETER RANKING")
    print("======================================================================")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # 1. Load Data
    print("\n[1/4] Loading multi-domain feature representation and latent coordinates...")
    merged, all_features, pca_df, ae_latent_df = load_and_prepare_data()
    X_raw = merged[all_features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    labels_true = merged["Material"].values

    # 2. Run Explainability Algorithms
    print("\n[2/4] Running multi-method feature importance & Shapley attribution suite...")
    ranked_df, shap_df, corr_df = run_explainability_pipeline(X_scaled, labels_true, all_features, pca_df, ae_latent_df)

    # 3. Save CSV Artifacts
    print("\n[3/4] Saving explainability CSV results...")
    ranked_path = os.path.join(OUTPUT_DIR, "phase9_feature_importance_ranked.csv")
    ranked_df.to_csv(ranked_path, index=False)
    print(f"  [OK] Saved ranked feature importance: {ranked_path}")

    shap_path = os.path.join(OUTPUT_DIR, "phase9_shapley_values_by_material.csv")
    shap_df.to_csv(shap_path, index=False)
    print(f"  [OK] Saved Shapley values by material: {shap_path}")

    corr_path = os.path.join(OUTPUT_DIR, "phase9_latent_feature_correlations.csv")
    corr_df.to_csv(corr_path, index=False)
    print(f"  [OK] Saved latent/PCA feature correlations: {corr_path}")

    # Display Top 5 Important Engineering Parameters
    print("\nTop 5 Most Influential Engineering Parameters:")
    for _, r in ranked_df.head(5).iterrows():
        print(f"  Rank {int(r['Rank'])}: {r['Feature']} ({r['Domain']}) — Score: {r['Composite_Importance_Score']:.4f} | {r['Direction_of_Effect']}")

    # 4. Generate Visualizations
    print("\n[4/4] Generating explainability plots...")
    plots_created = generate_explainability_plots(ranked_df, shap_df, corr_df, OUTPUT_DIR)
    for p in plots_created:
        print(f"  [OK] Plot generated: {p}")

    print("\n" + "=" * 70)
    print("PHASE 9 PART 2C: EXPLAINABILITY COMPLETE")
    print("======================================================================")
    return ranked_df, shap_df, corr_df


if __name__ == "__main__":
    run_explainability_pipeline_main()
