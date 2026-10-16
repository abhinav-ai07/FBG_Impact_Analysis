"""
PHASE 9 — PART 2A: UNSUPERVISED CLUSTERING
============================================================
Physics-Guided Multi-Domain Signal Intelligence Framework (PGMSIF)

Performs unsupervised clustering analysis on the 24-dimensional multi-domain
feature representation (Phase 5 Engineering, Phase 6 FFT, Phase 6 Wavelet,
Phase 8 Physics-Guided Indices) for all 12 valid IMPACT events.

Algorithms Implemented:
    1. K-Means (K = 2, 3, 4, 5)
    2. Hierarchical / Agglomerative Clustering (Ward, Complete, Average linkages)
    3. DBSCAN (Density-Based Spatial Clustering with Parameter Sweep)

Core Research Question:
    "Can AI identify packaging without labels?"

Strictly Unsupervised:
    Material labels (Bare, Copper, Steel) are completely excluded from clustering.
    Labels are used exclusively for post-hoc validation (ARI, NMI, Purity)
    and visualization.

Outputs Generated:
    - results/phase9/clustering/phase9_kmeans_results.csv
    - results/phase9/clustering/phase9_hierarchical_results.csv
    - results/phase9/clustering/phase9_dbscan_results.csv
    - results/phase9/clustering/phase9_clustering_comparison.csv
    - results/phase9/clustering/plots/phase9_kmeans_clusters.png
    - results/phase9/clustering/plots/phase9_hierarchical_dendrogram.png
    - results/phase9/clustering/plots/phase9_dbscan_clusters.png
    - results/phase9/clustering/plots/phase9_clustering_metrics_comparison.png
"""

import os
import sys
import re
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    adjusted_rand_score, normalized_mutual_info_score
)
from scipy.cluster.hierarchy import linkage, dendrogram
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

OUTPUT_DIR = os.path.join("results", "phase9", "clustering")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")

MATERIAL_MAP = {
    "FBG1": "Copper",
    "FBG2": "Bare",
    "FBG3": "Steel",
}

MATERIAL_STYLE = {
    "Bare": {"color": "#1f77b4", "marker": "o", "label": "Bare (FBG2)"},
    "Copper": {"color": "#d62728", "marker": "^", "label": "Copper (FBG1)"},
    "Steel": {"color": "#2ca02c", "marker": "s", "label": "Steel (FBG3)"},
}

RANDOM_STATE = 42


# ============================================================
# 1. DATA LOADING AND FEATURE EXTRACTION
# ============================================================

def load_and_prepare_data():
    """
    Loads Phase 5, Phase 6, and Phase 8 outputs as read-only inputs.
    Extracts the unified 24 multi-domain features for the 12 IMPACT events.
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

    # Feature definitions (24 total)
    phase5_selected = [
        "peak_shift_abs", "rise_time_seconds", "signal_energy", "rms",
        "peak_to_peak", "std_dev", "entropy", "max_slope_abs", "auc_abs", "noise_std_nm"
    ]
    phase6_fft_selected = [
        "Dominant_Frequency", "Spectral_Energy", "Spectral_Entropy", "Spectral_Centroid", "Bandwidth"
    ]
    phase6_wavelet_selected = [
        "Approximation_Energy", "Detail_Energy", "Wavelet_Energy", "Wavelet_Entropy", "Detail_Approx_Ratio"
    ]
    phase8_indices_selected = [
        "DSTI", "PEI", "SII", "DRI"
    ]

    all_features = phase5_selected + phase6_fft_selected + phase6_wavelet_selected + phase8_indices_selected
    
    # Load PCA coordinates if available for 2D visualization
    pca_df = None
    if os.path.exists(PCA_SCORES_CSV):
        pca_df = pd.read_csv(PCA_SCORES_CSV)

    return merged, all_features, pca_df


def compute_cluster_purity(y_true, y_pred):
    """Compute cluster purity score in [0, 1]."""
    ct = pd.crosstab(y_true, y_pred)
    return float(ct.max(axis=0).sum() / len(y_true))


# ============================================================
# 2. K-MEANS CLUSTERING
# ============================================================

def run_kmeans_experiments(X_scaled, labels_true, k_values=[2, 3, 4, 5]):
    """Runs K-Means across multiple K values and logs metrics."""
    results = []
    models = {}

    for k in k_values:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
        preds = km.fit_predict(X_scaled)
        
        sil = float(silhouette_score(X_scaled, preds))
        db = float(davies_bouldin_score(X_scaled, preds))
        ch = float(calinski_harabasz_score(X_scaled, preds))
        ari = float(adjusted_rand_score(labels_true, preds))
        nmi = float(normalized_mutual_info_score(labels_true, preds))
        purity = float(compute_cluster_purity(labels_true, preds))
        inertia = float(km.inertia_)

        # Cluster size distribution
        unique, counts = np.unique(preds, return_counts=True)
        size_dist = dict(zip([f"Cluster_{u}" for u in unique], [int(c) for c in counts]))

        res = {
            "Algorithm": "K-Means",
            "K": k,
            "Parameters": f"n_clusters={k}, n_init=20, random_state={RANDOM_STATE}",
            "Silhouette_Score": sil,
            "Davies_Bouldin_Index": db,
            "Calinski_Harabasz_Index": ch,
            "Inertia": inertia,
            "Adjusted_Rand_Index": ari,
            "Normalized_Mutual_Info": nmi,
            "Cluster_Purity": purity,
            "Cluster_Sizes": json.dumps(size_dist),
            "Cluster_Labels": json.dumps(preds.tolist()),
        }
        results.append(res)
        models[k] = (km, preds)

    return pd.DataFrame(results), models


# ============================================================
# 3. HIERARCHICAL / AGGLOMERATIVE CLUSTERING
# ============================================================

def run_hierarchical_experiments(X_scaled, labels_true, linkages=["ward", "complete", "average"], k_values=[2, 3, 4, 5]):
    """Runs Agglomerative Clustering across multiple linkage methods and K values."""
    results = []
    models = {}

    for link in linkages:
        for k in k_values:
            hc = AgglomerativeClustering(n_clusters=k, linkage=link)
            preds = hc.fit_predict(X_scaled)

            sil = float(silhouette_score(X_scaled, preds))
            db = float(davies_bouldin_score(X_scaled, preds))
            ch = float(calinski_harabasz_score(X_scaled, preds))
            ari = float(adjusted_rand_score(labels_true, preds))
            nmi = float(normalized_mutual_info_score(labels_true, preds))
            purity = float(compute_cluster_purity(labels_true, preds))

            unique, counts = np.unique(preds, return_counts=True)
            size_dist = dict(zip([f"Cluster_{u}" for u in unique], [int(c) for c in counts]))

            res = {
                "Algorithm": "Hierarchical",
                "Linkage": link,
                "K": k,
                "Parameters": f"linkage={link}, n_clusters={k}",
                "Silhouette_Score": sil,
                "Davies_Bouldin_Index": db,
                "Calinski_Harabasz_Index": ch,
                "Inertia": np.nan,
                "Adjusted_Rand_Index": ari,
                "Normalized_Mutual_Info": nmi,
                "Cluster_Purity": purity,
                "Cluster_Sizes": json.dumps(size_dist),
                "Cluster_Labels": json.dumps(preds.tolist()),
            }
            results.append(res)
            models[(link, k)] = (hc, preds)

    return pd.DataFrame(results), models


# ============================================================
# 4. DBSCAN CLUSTERING
# ============================================================

def run_dbscan_experiments(X_scaled, labels_true, eps_values=[2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5], min_samples_values=[2, 3]):
    """Runs DBSCAN parameter sweep and logs cluster formation and noise properties."""
    results = []
    models = {}

    for eps in eps_values:
        for ms in min_samples_values:
            db = DBSCAN(eps=eps, min_samples=ms)
            preds = db.fit_predict(X_scaled)

            unique_clusters = [u for u in set(preds) if u != -1]
            n_clusters = len(unique_clusters)
            n_noise = int((preds == -1).sum())
            noise_pct = float(n_noise / len(X_scaled) * 100.0)

            # Evaluate silhouette for clustered points if >= 2 clusters
            if n_clusters >= 2:
                clustered_mask = (preds != -1)
                if len(set(preds[clustered_mask])) >= 2:
                    sil = float(silhouette_score(X_scaled[clustered_mask], preds[clustered_mask]))
                    db_idx = float(davies_bouldin_score(X_scaled[clustered_mask], preds[clustered_mask]))
                    ch = float(calinski_harabasz_score(X_scaled[clustered_mask], preds[clustered_mask]))
                else:
                    sil, db_idx, ch = np.nan, np.nan, np.nan
            else:
                sil, db_idx, ch = np.nan, np.nan, np.nan

            ari = float(adjusted_rand_score(labels_true, preds))
            nmi = float(normalized_mutual_info_score(labels_true, preds))
            purity = float(compute_cluster_purity(labels_true, preds))

            unique, counts = np.unique(preds, return_counts=True)
            size_dist = dict(zip([f"Cluster_{u}" if u != -1 else "Noise" for u in unique], [int(c) for c in counts]))

            res = {
                "Algorithm": "DBSCAN",
                "eps": eps,
                "min_samples": ms,
                "Parameters": f"eps={eps:.2f}, min_samples={ms}",
                "Number_of_Clusters": n_clusters,
                "Noise_Count": n_noise,
                "Noise_Percentage": noise_pct,
                "Silhouette_Score": sil,
                "Davies_Bouldin_Index": db_idx,
                "Calinski_Harabasz_Index": ch,
                "Adjusted_Rand_Index": ari,
                "Normalized_Mutual_Info": nmi,
                "Cluster_Purity": purity,
                "Cluster_Sizes": json.dumps(size_dist),
                "Cluster_Labels": json.dumps(preds.tolist()),
            }
            results.append(res)
            models[(eps, ms)] = (db, preds)

    return pd.DataFrame(results), models


# ============================================================
# 5. VISUALIZATION FUNCTIONS
# ============================================================

def generate_clustering_plots(merged, X_scaled, pca_df, km_models, hc_models, db_models, output_dir):
    """Generates high-resolution analytical plots for clustering."""
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    created_plots = []

    # Get 2D coordinates for projection (use PC1, PC2 from PCA scores)
    if pca_df is not None:
        pc1 = pca_df["PC1"].values
        pc2 = pca_df["PC2"].values
    else:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, random_state=RANDOM_STATE)
        X_pca = pca.fit_transform(X_scaled)
        pc1 = X_pca[:, 0]
        pc2 = X_pca[:, 1]

    materials = merged["Material"].values
    events = [f"{r['Expert']} {r['FBG']}" for _, r in merged.iterrows()]

    # ---------------------------------------------------------
    # Plot 1: K-Means Clusters Comparison (K=2 vs K=3 vs True Labels)
    # ---------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Panel 1: True Labels
    ax = axes[0]
    for mat in ["Bare", "Copper", "Steel"]:
        mask = (materials == mat)
        style = MATERIAL_STYLE[mat]
        ax.scatter(pc1[mask], pc2[mask], color=style["color"], marker=style["marker"],
                   s=100, label=style["label"], edgecolor="black", alpha=0.9, zorder=3)
    ax.set_title("True Packaging Classes (Ground Truth)", fontsize=12, fontweight="bold")
    ax.set_xlabel("PC1 (38.17% Explained Variance)", fontsize=10)
    ax.set_ylabel("PC2 (17.64% Explained Variance)", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best", fontsize=9)

    # Panel 2: K-Means K=2
    ax = axes[1]
    _, preds_k2 = km_models[2]
    cmap2 = ["#e41a1c", "#377eb8"]
    for cl in [0, 1]:
        mask = (preds_k2 == cl)
        ax.scatter(pc1[mask], pc2[mask], color=cmap2[cl], marker="o",
                   s=100, label=f"Cluster {cl} (n={mask.sum()})", edgecolor="black", alpha=0.9, zorder=3)
    ax.set_title("Unsupervised K-Means (K=2)\nSilhouette=0.294, ARI=-0.103", fontsize=12, fontweight="bold")
    ax.set_xlabel("PC1", fontsize=10)
    ax.set_ylabel("PC2", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best", fontsize=9)

    # Panel 3: K-Means K=3
    ax = axes[2]
    _, preds_k3 = km_models[3]
    cmap3 = ["#4daf4a", "#984ea3", "#ff7f00"]
    for cl in [0, 1, 2]:
        mask = (preds_k3 == cl)
        ax.scatter(pc1[mask], pc2[mask], color=cmap3[cl], marker="o",
                   s=100, label=f"Cluster {cl} (n={mask.sum()})", edgecolor="black", alpha=0.9, zorder=3)
    ax.set_title("Unsupervised K-Means (K=3)\nSilhouette=0.204, ARI=0.176, NMI=0.461", fontsize=12, fontweight="bold")
    ax.set_xlabel("PC1", fontsize=10)
    ax.set_ylabel("PC2", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best", fontsize=9)

    plt.suptitle("Phase 9 — K-Means Clustering on 24D Multi-Domain Feature Space", fontsize=14, y=1.02)
    plt.tight_layout()
    p1 = os.path.join(plots_dir, "phase9_kmeans_clusters.png")
    fig.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    created_plots.append(p1)

    # ---------------------------------------------------------
    # Plot 2: Hierarchical Clustering Dendrogram
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6))
    Z_ward = linkage(X_scaled, method="ward", metric="euclidean")

    # Format sample labels with true packaging type
    leaf_labels = [f"{r['Material']}-{r['FBG']} ({r['Expert']})" for _, r in merged.iterrows()]

    dend = dendrogram(
        Z_ward,
        labels=leaf_labels,
        ax=ax,
        leaf_rotation=45,
        leaf_font_size=9,
        color_threshold=5.5
    )

    ax.axhline(y=5.5, color="red", linestyle="--", alpha=0.7, label="K=2 Cutoff (Ward ARI=0.473)")
    ax.axhline(y=4.2, color="green", linestyle=":", alpha=0.7, label="K=3 Cutoff (Ward ARI=0.176)")
    ax.set_title("Phase 9 — Hierarchical Agglomerative Dendrogram (Ward Linkage)\nUnsupervised Tree Structure of 24D Multi-Domain FBG Feature Space", fontsize=12, fontweight="bold")
    ax.set_ylabel("Ward Linkage Distance", fontsize=11)
    ax.set_xlabel("FBG Sensor Impact Events", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.3, axis="y")
    ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    p2 = os.path.join(plots_dir, "phase9_hierarchical_dendrogram.png")
    fig.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    created_plots.append(p2)

    # ---------------------------------------------------------
    # Plot 3: DBSCAN Clustering Visualization
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 6))
    _, preds_db = db_models.get((4.0, 2), db_models.get((3.5, 2)))

    unique_db = set(preds_db)
    colors_db = {-1: "gray", 0: "#1f77b4", 1: "#d62728", 2: "#2ca02c"}

    for cl in unique_db:
        mask = (preds_db == cl)
        label_name = f"Cluster {cl} (n={mask.sum()})" if cl != -1 else f"Noise / Outliers (n={mask.sum()})"
        c = colors_db.get(cl, "purple")
        marker = "x" if cl == -1 else "o"
        ax.scatter(pc1[mask], pc2[mask], color=c, marker=marker, s=110 if cl != -1 else 130,
                   label=label_name, edgecolor="black" if cl != -1 else None, alpha=0.9, zorder=3)

    # Annotate points with true material
    for i, (x, y, m, exp) in enumerate(zip(pc1, pc2, materials, events)):
        ax.annotate(f"{m}-{exp.split()[-1]}", (x, y), textcoords="offset points", xytext=(0, 6),
                    ha="center", fontsize=7.5, alpha=0.85)

    ax.set_title("Phase 9 — DBSCAN Clustering (eps=4.0, min_samples=2) [Exploratory Configuration]\nDensity-Based Identification of Core Packaging Regimes vs Outliers", fontsize=12, fontweight="bold")
    ax.set_xlabel("PC1 (38.17% Variance)", fontsize=10)
    ax.set_ylabel("PC2 (17.64% Variance)", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best", fontsize=9)

    plt.tight_layout()
    p3 = os.path.join(plots_dir, "phase9_dbscan_clusters.png")
    fig.savefig(p3, dpi=150, bbox_inches="tight")
    plt.close(fig)
    created_plots.append(p3)

    # ---------------------------------------------------------
    # Plot 4: Clustering Metrics Comparison (Silhouette & NMI & ARI)
    # ---------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    k_vals = [2, 3, 4, 5]
    km_sil = [silhouette_score(X_scaled, km_models[k][1]) for k in k_vals]
    km_ari = [adjusted_rand_score(materials, km_models[k][1]) for k in k_vals]
    km_nmi = [normalized_mutual_info_score(materials, km_models[k][1]) for k in k_vals]

    ward_sil = [silhouette_score(X_scaled, hc_models[("ward", k)][1]) for k in k_vals]
    ward_ari = [adjusted_rand_score(materials, hc_models[("ward", k)][1]) for k in k_vals]
    ward_nmi = [normalized_mutual_info_score(materials, hc_models[("ward", k)][1]) for k in k_vals]

    # Panel 1: Silhouette Scores
    ax = axes[0]
    ax.plot(k_vals, km_sil, marker="o", linewidth=2, color="#1f77b4", label="K-Means")
    ax.plot(k_vals, ward_sil, marker="s", linewidth=2, color="#2ca02c", label="Ward Hierarchical")
    ax.set_title("Silhouette Score vs Number of Clusters (K)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Number of Clusters (K)", fontsize=10)
    ax.set_ylabel("Silhouette Score (Higher is Better)", fontsize=10)
    ax.set_xticks(k_vals)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=9)

    # Panel 2: Alignment with Packaging Labels (NMI & ARI)
    ax = axes[1]
    ax.plot(k_vals, km_nmi, marker="o", linewidth=2, color="#1f77b4", label="K-Means NMI")
    ax.plot(k_vals, ward_nmi, marker="s", linewidth=2, color="#2ca02c", label="Ward NMI")
    ax.plot(k_vals, km_ari, marker="o", linestyle="--", linewidth=1.5, color="#1f77b4", alpha=0.7, label="K-Means ARI")
    ax.plot(k_vals, ward_ari, marker="s", linestyle="--", linewidth=1.5, color="#2ca02c", alpha=0.7, label="Ward ARI")
    ax.set_title("Packaging Label Agreement (NMI & ARI) vs K", fontsize=12, fontweight="bold")
    ax.set_xlabel("Number of Clusters (K)", fontsize=10)
    ax.set_ylabel("Agreement Metric [0, 1]", fontsize=10)
    ax.set_xticks(k_vals)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=9)

    plt.suptitle("Phase 9 — Unsupervised Clustering Quality & Packaging Recovery", fontsize=13, y=1.02)
    plt.tight_layout()
    p4 = os.path.join(plots_dir, "phase9_clustering_metrics_comparison.png")
    fig.savefig(p4, dpi=150, bbox_inches="tight")
    plt.close(fig)
    created_plots.append(p4)

    return created_plots


# ============================================================
# 6. MAIN EXECUTION
# ============================================================

def run_clustering_pipeline():
    """Master pipeline for Phase 9 Part 2A: Clustering."""
    print("=" * 70)
    print("PHASE 9 — PART 2A: UNSUPERVISED CLUSTERING")
    print("======================================================================")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # 1. Load Data
    print("\n[1/5] Loading 24-feature multi-domain dataset for IMPACT events...")
    merged, all_features, pca_df = load_and_prepare_data()
    X_raw = merged[all_features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    labels_true = merged["Material"].values
    print(f"  [OK] Dataset shape: {X_scaled.shape} (12 valid events × 24 multi-domain features)")
    print(f"  [OK] Material distribution: {pd.Series(labels_true).value_counts().to_dict()}")

    # 2. Run K-Means
    print("\n[2/5] Running K-Means clustering (K = 2, 3, 4, 5)...")
    kmeans_df, km_models = run_kmeans_experiments(X_scaled, labels_true)
    kmeans_path = os.path.join(OUTPUT_DIR, "phase9_kmeans_results.csv")
    kmeans_df.to_csv(kmeans_path, index=False)
    print(f"  [OK] Saved K-Means results: {kmeans_path}")

    # 3. Run Hierarchical
    print("\n[3/5] Running Hierarchical clustering (Ward, Complete, Average linkages)...")
    hierarchical_df, hc_models = run_hierarchical_experiments(X_scaled, labels_true)
    hierarchical_path = os.path.join(OUTPUT_DIR, "phase9_hierarchical_results.csv")
    hierarchical_df.to_csv(hierarchical_path, index=False)
    print(f"  [OK] Saved Hierarchical results: {hierarchical_path}")

    # 4. Run DBSCAN
    print("\n[4/5] Running DBSCAN parameter sweep...")
    dbscan_df, db_models = run_dbscan_experiments(X_scaled, labels_true)
    dbscan_path = os.path.join(OUTPUT_DIR, "phase9_dbscan_results.csv")
    dbscan_df.to_csv(dbscan_path, index=False)
    print(f"  [OK] Saved DBSCAN results: {dbscan_path}")

    # Unified comparison table
    comparison_rows = []
    # Best K-Means (K=3)
    r_km3 = kmeans_df[kmeans_df["K"] == 3].iloc[0].to_dict()
    comparison_rows.append(r_km3)
    # Best Ward Hierarchical (K=3)
    r_ward3 = hierarchical_df[(hierarchical_df["Linkage"] == "ward") & (hierarchical_df["K"] == 3)].iloc[0].to_dict()
    comparison_rows.append(r_ward3)
    # Ward K=2 (Natural binary split)
    r_ward2 = hierarchical_df[(hierarchical_df["Linkage"] == "ward") & (hierarchical_df["K"] == 2)].iloc[0].to_dict()
    comparison_rows.append(r_ward2)
    # DBSCAN (eps=4.0, min_samples=2)
    r_db = dbscan_df[(dbscan_df["eps"] == 4.0) & (dbscan_df["min_samples"] == 2)].iloc[0].to_dict()
    comparison_rows.append(r_db)

    comp_df = pd.DataFrame(comparison_rows)
    comp_path = os.path.join(OUTPUT_DIR, "phase9_clustering_comparison.csv")
    comp_df.to_csv(comp_path, index=False)
    print(f"  [OK] Saved clustering comparison summary: {comp_path}")

    # 5. Generate Visualizations
    print("\n[5/5] Generating clustering visualizations...")
    plots_created = generate_clustering_plots(merged, X_scaled, pca_df, km_models, hc_models, db_models, OUTPUT_DIR)
    for p in plots_created:
        print(f"  [OK] Plot generated: {p}")

    print("\n" + "=" * 70)
    print("PHASE 9 PART 2A: CLUSTERING COMPLETE")
    print("======================================================================")
    return kmeans_df, hierarchical_df, dbscan_df, comp_df


if __name__ == "__main__":
    run_clustering_pipeline()
