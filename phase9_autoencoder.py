"""
PHASE 9 — PART 2B: MULTI-DOMAIN AUTOENCODER & LATENT SPACE ANALYSIS
===================================================================
Physics-Guided Multi-Domain Signal Intelligence Framework (PGMSIF)

Implements a non-linear neural autoencoder to compress the 24-dimensional
multi-domain feature representation into a compact, continuous 2D latent space.

Components:
    1. Neural Autoencoder Architecture:
       Input (24) -> Hidden1 (12) -> Hidden2 (6) -> Latent (2) -> Hidden2' (6) -> Hidden1' (12) -> Output (24)
    2. Model Training & Reconstruction Quality Evaluation:
       Loss tracking, total MSE, and per-feature reconstruction R^2 scores.
    3. Latent Space Structural Characterization:
       Coordinate extraction, latent distribution analysis, packaging separation inspection.
    4. Comparative Latent vs. Original Feature Clustering:
       Evaluates K-Means and Hierarchical clustering on Original 24D, Latent 2D, and PCA 2D spaces.

Consumes existing results from Phase 5, Phase 6, and Phase 8 as read-only inputs.

Outputs Generated:
    - results/phase9/autoencoder/phase9_autoencoder_latent_coordinates.csv
    - results/phase9/autoencoder/phase9_autoencoder_reconstruction_metrics.csv
    - results/phase9/autoencoder/phase9_autoencoder_training_history.csv
    - results/phase9/autoencoder/phase9_autoencoder_model_weights.pt
    - results/phase9/autoencoder/plots/phase9_autoencoder_training_loss.png
    - results/phase9/autoencoder/plots/phase9_autoencoder_latent_space.png
    - results/phase9/autoencoder/plots/phase9_autoencoder_reconstruction_error.png
    - results/phase9/autoencoder/plots/phase9_feature_vs_latent_clustering_comparison.png
"""

import os
import sys
import re
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    adjusted_rand_score, normalized_mutual_info_score
)
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

OUTPUT_DIR = os.path.join("results", "phase9", "autoencoder")
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
EPOCHS = 250
LEARNING_RATE = 0.015
WEIGHT_DECAY = 1e-5


# ============================================================
# 1. AUTOENCODER MODEL ARCHITECTURE
# ============================================================

class MultiDomainAutoencoder(nn.Module):
    """
    Non-linear symmetric autoencoder for multi-domain FBG feature compression.
    Compresses 24 standardized multi-domain features to a 2D latent representation.
    """
    def __init__(self, input_dim=24, latent_dim=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 12),
            nn.Tanh(),
            nn.Linear(12, 6),
            nn.Tanh(),
            nn.Linear(6, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 6),
            nn.Tanh(),
            nn.Linear(6, 12),
            nn.Tanh(),
            nn.Linear(12, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_rec = self.decoder(z)
        return x_rec, z


# ============================================================
# 2. DATA PREPARATION
# ============================================================

def load_and_prepare_data():
    """
    Loads Phase 5, Phase 6, and Phase 8 outputs as read-only inputs.
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
    
    pca_df = None
    if os.path.exists(PCA_SCORES_CSV):
        pca_df = pd.read_csv(PCA_SCORES_CSV)

    return merged, all_features, pca_df


# ============================================================
# 3. AUTOENCODER TRAINING & EVALUATION
# ============================================================

def train_autoencoder(X_scaled, epochs=EPOCHS, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY):
    """
    Trains the multi-domain autoencoder with fixed random seed and logs history.
    """
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    input_dim = X_scaled.shape[1]
    model = MultiDomainAutoencoder(input_dim=input_dim, latent_dim=2)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    history = []
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        x_rec, z = model(X_tensor)
        loss = criterion(x_rec, X_tensor)
        loss.backward()
        optimizer.step()

        history.append({
            "Epoch": epoch + 1,
            "Reconstruction_Loss_MSE": float(loss.item()),
        })

    model.eval()
    with torch.no_grad():
        x_rec, z_latent = model(X_tensor)
        final_loss = float(criterion(x_rec, X_tensor).item())
        z_np = z_latent.numpy()
        x_rec_np = x_rec.numpy()

    history_df = pd.DataFrame(history)
    return model, history_df, z_np, x_rec_np, final_loss


def evaluate_reconstruction(X_scaled, x_rec_np, feature_names):
    """Computes per-feature MSE and R^2 reconstruction metrics."""
    feat_mse = np.mean((X_scaled - x_rec_np) ** 2, axis=0)
    feat_var = np.var(X_scaled, axis=0) # = 1.0 because standardized
    feat_r2 = 1.0 - (feat_mse / (feat_var + 1e-12))

    records = []
    for i, feat in enumerate(feature_names):
        records.append({
            "Feature": feat,
            "Reconstruction_MSE": float(feat_mse[i]),
            "Reconstruction_R2": float(feat_r2[i]),
            "Original_Mean": float(np.mean(X_scaled[:, i])),
            "Original_Std": float(np.std(X_scaled[:, i])),
            "Reconstructed_Mean": float(np.mean(x_rec_np[:, i])),
            "Reconstructed_Std": float(np.std(x_rec_np[:, i])),
        })

    return pd.DataFrame(records)


def compute_cluster_purity(y_true, y_pred):
    """Compute cluster purity score in [0, 1]."""
    ct = pd.crosstab(y_true, y_pred)
    return float(ct.max(axis=0).sum() / len(y_true))


# ============================================================
# 4. LATENT VS ORIGINAL CLUSTERING COMPARISON
# ============================================================

def compare_latent_clustering(X_scaled, z_np, pca_df, labels_true):
    """
    Compares clustering performance in:
    1. Original 24D Feature Space
    2. Autoencoder 2D Latent Space
    3. PCA 2D Subspace (PC1, PC2)
    """
    spaces = {
        "Original_24D": X_scaled,
        "Autoencoder_2D_Latent": z_np,
    }
    if pca_df is not None:
        spaces["PCA_2D"] = pca_df[["PC1", "PC2"]].values

    records = []

    for space_name, data in spaces.items():
        # K-Means K=2 and K=3
        for k in [2, 3]:
            km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
            preds_km = km.fit_predict(data)

            sil = float(silhouette_score(data, preds_km))
            db = float(davies_bouldin_score(data, preds_km))
            ch = float(calinski_harabasz_score(data, preds_km))
            ari = float(adjusted_rand_score(labels_true, preds_km))
            nmi = float(normalized_mutual_info_score(labels_true, preds_km))
            
            # Purity
            purity = compute_cluster_purity(labels_true, preds_km)

            records.append({
                "Representation_Space": space_name,
                "Dimensionality": data.shape[1],
                "Algorithm": "K-Means",
                "K": k,
                "Silhouette_Score": sil,
                "Davies_Bouldin_Index": db,
                "Calinski_Harabasz_Index": ch,
                "Adjusted_Rand_Index": ari,
                "Normalized_Mutual_Info": nmi,
                "Cluster_Purity": purity,
            })

        # Ward Hierarchical K=2 and K=3
        for k in [2, 3]:
            hc = AgglomerativeClustering(n_clusters=k, linkage="ward")
            preds_hc = hc.fit_predict(data)

            sil = float(silhouette_score(data, preds_hc))
            db = float(davies_bouldin_score(data, preds_hc))
            ch = float(calinski_harabasz_score(data, preds_hc))
            ari = float(adjusted_rand_score(labels_true, preds_hc))
            nmi = float(normalized_mutual_info_score(labels_true, preds_hc))
            
            purity = compute_cluster_purity(labels_true, preds_hc)

            records.append({
                "Representation_Space": space_name,
                "Dimensionality": data.shape[1],
                "Algorithm": "Ward_Hierarchical",
                "K": k,
                "Silhouette_Score": sil,
                "Davies_Bouldin_Index": db,
                "Calinski_Harabasz_Index": ch,
                "Adjusted_Rand_Index": ari,
                "Normalized_Mutual_Info": nmi,
                "Cluster_Purity": purity,
            })

    return pd.DataFrame(records)


# ============================================================
# 5. VISUALIZATION FUNCTIONS
# ============================================================

def generate_autoencoder_plots(history_df, z_np, rec_metrics_df, comp_clustering_df, merged, output_dir):
    """Generates analytical plots for Autoencoder and Latent space."""
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    created_plots = []

    materials = merged["Material"].values
    events = [f"{r['Expert']} {r['FBG']}" for _, r in merged.iterrows()]

    # ---------------------------------------------------------
    # Plot 1: Training Loss Curve
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history_df["Epoch"], history_df["Reconstruction_Loss_MSE"], color="#1f77b4", linewidth=2)
    ax.set_title("Phase 9 — Autoencoder Training Loss (MSE vs Epoch)\nArchitecture: 24 -> 12 -> 6 -> 2 -> 6 -> 12 -> 24", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Reconstruction Loss (MSE)", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    final_loss = history_df["Reconstruction_Loss_MSE"].iloc[-1]
    ax.annotate(f"Final MSE = {final_loss:.5f}",
                xy=(len(history_df), final_loss),
                xytext=(len(history_df) - 70, final_loss + 0.08),
                arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=6),
                fontsize=9, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.5))

    plt.tight_layout()
    p1 = os.path.join(plots_dir, "phase9_autoencoder_training_loss.png")
    fig.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    created_plots.append(p1)

    # ---------------------------------------------------------
    # Plot 2: 2D Latent Space Visualization
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 6.5))
    z1 = z_np[:, 0]
    z2 = z_np[:, 1]

    for mat in ["Bare", "Copper", "Steel"]:
        mask = (materials == mat)
        style = MATERIAL_STYLE[mat]
        ax.scatter(z1[mask], z2[mask], color=style["color"], marker=style["marker"],
                   s=120, label=f"{style['label']} (n={mask.sum()})", edgecolor="black", linewidth=1.2, alpha=0.9, zorder=3)

    for i, (x, y, m, exp) in enumerate(zip(z1, z2, materials, events)):
        ax.annotate(f"{m} ({exp.split()[-1]})", (x, y), textcoords="offset points", xytext=(0, 7),
                    ha="center", fontsize=8, fontweight="bold", alpha=0.85)

    ax.set_title("Phase 9 — Autoencoder 2D Latent Space (z1 vs z2)\nUnsupervised Non-Linear Compression of 24D Multi-Domain Features", fontsize=12, fontweight="bold")
    ax.set_xlabel("Latent Dimension z1", fontsize=11)
    ax.set_ylabel("Latent Dimension z2", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best", fontsize=9.5)

    plt.tight_layout()
    p2 = os.path.join(plots_dir, "phase9_autoencoder_latent_space.png")
    fig.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    created_plots.append(p2)

    # ---------------------------------------------------------
    # Plot 3: Feature Reconstruction Fidelity (R^2 Bar Chart)
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6))
    rec_sorted = rec_metrics_df.sort_values(by="Reconstruction_R2", ascending=True)

    # Assign colors by domain
    domain_colors = []
    for f in rec_sorted["Feature"]:
        if f in ["DSTI", "PEI", "SII", "DRI"]:
            domain_colors.append("#ff7f0e") # Phase 8 orange
        elif "Wavelet" in f or "Detail" in f or "Approximation" in f:
            domain_colors.append("#2ca02c") # Phase 6 wavelet green
        elif "Spectral" in f or "Frequency" in f or "Bandwidth" in f:
            domain_colors.append("#9467bd") # Phase 6 FFT purple
        else:
            domain_colors.append("#1f77b4") # Phase 5 time-domain blue

    bars = ax.barh(rec_sorted["Feature"], rec_sorted["Reconstruction_R2"], color=domain_colors, edgecolor="black", height=0.65)
    ax.axvline(x=1.0, color="gray", linestyle="--", alpha=0.7)
    ax.axvline(x=np.mean(rec_metrics_df["Reconstruction_R2"]), color="red", linestyle=":", linewidth=1.5,
               label=f"Mean R^2 = {np.mean(rec_metrics_df['Reconstruction_R2']):.3f}")

    ax.set_title("Phase 9 — Autoencoder Reconstruction Fidelity per Feature (R^2 Score)\nEvaluating Information Preservation Through 2D Latent Bottleneck", fontsize=12, fontweight="bold")
    ax.set_xlabel("Reconstruction R^2 Score (1.0 = Perfect Reconstruction)", fontsize=10)
    ax.set_xlim(0.85, 1.02)
    ax.grid(True, linestyle="--", alpha=0.3, axis="x")
    ax.legend(loc="lower left", fontsize=9)

    plt.tight_layout()
    p3 = os.path.join(plots_dir, "phase9_autoencoder_reconstruction_error.png")
    fig.savefig(p3, dpi=150, bbox_inches="tight")
    plt.close(fig)
    created_plots.append(p3)

    # ---------------------------------------------------------
    # Plot 4: Clustering Comparison (Original 24D vs Latent 2D vs PCA 2D)
    # ---------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Silhouette Scores
    ax = axes[0]
    spaces = comp_clustering_df["Representation_Space"].unique()
    x = np.arange(len(spaces))
    width = 0.35

    km_k3 = comp_clustering_df[(comp_clustering_df["Algorithm"] == "K-Means") & (comp_clustering_df["K"] == 3)]
    ward_k3 = comp_clustering_df[(comp_clustering_df["Algorithm"] == "Ward_Hierarchical") & (comp_clustering_df["K"] == 3)]

    ax.bar(x - width/2, km_k3["Silhouette_Score"], width, label="K-Means (K=3)", color="#1f77b4", edgecolor="black")
    ax.bar(x + width/2, ward_k3["Silhouette_Score"], width, label="Ward (K=3)", color="#2ca02c", edgecolor="black")

    ax.set_title("Cluster Separation (Silhouette Score) by Feature Space", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", " ") for s in spaces], fontsize=9)
    ax.set_ylabel("Silhouette Score", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.3, axis="y")
    ax.legend(fontsize=9)

    # Panel 2: Agreement with Packaging (NMI & ARI)
    ax = axes[1]
    ax.bar(x - width/2, km_k3["Normalized_Mutual_Info"], width, label="K-Means NMI", color="#ff7f0e", edgecolor="black")
    ax.bar(x + width/2, ward_k3["Normalized_Mutual_Info"], width, label="Ward NMI", color="#9467bd", edgecolor="black")

    ax.set_title("Packaging Alignment (Normalized Mutual Information) by Space", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", " ") for s in spaces], fontsize=9)
    ax.set_ylabel("Normalized Mutual Information [0, 1]", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.3, axis="y")
    ax.legend(fontsize=9)

    plt.suptitle("Phase 9 — Impact of Latent Space Compression on Clustering Quality", fontsize=13, y=1.02)
    plt.tight_layout()
    p4 = os.path.join(plots_dir, "phase9_feature_vs_latent_clustering_comparison.png")
    fig.savefig(p4, dpi=150, bbox_inches="tight")
    plt.close(fig)
    created_plots.append(p4)

    return created_plots


# ============================================================
# 6. MAIN EXECUTION
# ============================================================

def run_autoencoder_pipeline():
    """Master pipeline for Phase 9 Part 2B: Autoencoder & Latent Space."""
    print("=" * 70)
    print("PHASE 9 — PART 2B: AUTOENCODER & LATENT SPACE ANALYSIS")
    print("======================================================================")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # 1. Load Data
    print("\n[1/5] Loading multi-domain feature representation...")
    merged, all_features, pca_df = load_and_prepare_data()
    X_raw = merged[all_features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    labels_true = merged["Material"].values

    # 2. Train Autoencoder
    print(f"\n[2/5] Training PyTorch Autoencoder (24 -> 12 -> 6 -> 2 -> 6 -> 12 -> 24) for {EPOCHS} epochs...")
    model, history_df, z_np, x_rec_np, final_loss = train_autoencoder(X_scaled)
    print(f"  [OK] Training completed. Final Reconstruction MSE: {final_loss:.6f}")

    # Save training history
    history_path = os.path.join(OUTPUT_DIR, "phase9_autoencoder_training_history.csv")
    history_df.to_csv(history_path, index=False)
    print(f"  [OK] Saved training history: {history_path}")

    # Save model weights
    weights_path = os.path.join(OUTPUT_DIR, "phase9_autoencoder_model_weights.pt")
    torch.save(model.state_dict(), weights_path)
    print(f"  [OK] Saved model weights: {weights_path}")

    # 3. Save Latent Coordinates
    print("\n[3/5] Extracting 2D latent space coordinates...")
    latent_df = merged[["Expert", "FBG", "Material", "Impact_Status"]].copy()
    latent_df["z1"] = z_np[:, 0]
    latent_df["z2"] = z_np[:, 1]
    latent_path = os.path.join(OUTPUT_DIR, "phase9_autoencoder_latent_coordinates.csv")
    latent_df.to_csv(latent_path, index=False)
    print(f"  [OK] Saved latent coordinates: {latent_path}")

    # 4. Reconstruction Metrics
    print("\n[4/5] Evaluating feature reconstruction fidelity...")
    rec_metrics_df = evaluate_reconstruction(X_scaled, x_rec_np, all_features)
    rec_path = os.path.join(OUTPUT_DIR, "phase9_autoencoder_reconstruction_metrics.csv")
    rec_metrics_df.to_csv(rec_path, index=False)
    mean_r2 = float(rec_metrics_df["Reconstruction_R2"].mean())
    print(f"  [OK] IN-SAMPLE Mean Feature R^2 across 24 features: {mean_r2:.4f}")
    print(f"  [NOTE] This is an IN-SAMPLE reconstruction metric. The dataset contains only 12 events,")
    print(f"         which is too small to support a reliable independent held-out evaluation.")
    print(f"  [OK] Saved reconstruction metrics: {rec_path}")

    # 5. Latent Space Clustering Comparison
    print("\n[5/5] Comparing clustering in 24D Original vs 2D Latent vs 2D PCA spaces...")
    comp_clustering_df = compare_latent_clustering(X_scaled, z_np, pca_df, labels_true)
    comp_clust_path = os.path.join(OUTPUT_DIR, "phase9_autoencoder_clustering_comparison.csv")
    comp_clustering_df.to_csv(comp_clust_path, index=False)
    print(f"  [OK] Saved clustering comparison: {comp_clust_path}")

    # Visualizations
    print("\nGenerating Autoencoder visualizations...")
    plots_created = generate_autoencoder_plots(history_df, z_np, rec_metrics_df, comp_clustering_df, merged, OUTPUT_DIR)
    for p in plots_created:
        print(f"  [OK] Plot generated: {p}")

    print("\n" + "=" * 70)
    print("PHASE 9 PART 2B: AUTOENCODER COMPLETE")
    print("======================================================================")
    return latent_df, rec_metrics_df, comp_clustering_df


if __name__ == "__main__":
    run_autoencoder_pipeline()
