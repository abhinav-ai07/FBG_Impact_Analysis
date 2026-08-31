import os
import pandas as pd
import umap
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import silhouette_score

INPUT_FILE = "results/phase8/phase8_engineering_indices.csv"

# =====================================================
# LOAD DATA
# =====================================================
df = pd.read_csv(INPUT_FILE)

df = df[df["Index_Status"] == "VALID"].copy()

os.makedirs("results/phase9", exist_ok=True)

# =====================================================
# FEATURE SETS
# =====================================================
feature_sets = {
    "All Features": {
        "features": [
            "Peak_Shift",
            "Residual_Shift",
            "Rise_Time",
            "Recovery_Time",
            "Signal_Energy",
            "DSTI",
            "PEI",
            "SII",
            "DRI"
        ],
        "plot": "umap_all_features.png"
    },

    "PGMSIF Only": {
        "features": [
            "DSTI",
            "PEI",
            "SII",
            "DRI"
        ],
        "plot": "umap_pgmsif_only.png"
    },

    "Traditional Only": {
        "features": [
            "Peak_Shift",
            "Residual_Shift",
            "Rise_Time",
            "Recovery_Time",
            "Signal_Energy"
        ],
        "plot": "umap_traditional_only.png"
    }
}

results = []

# =====================================================
# CREATE INDIVIDUAL PLOTS
# =====================================================
for name, config in feature_sets.items():

    X = df[config["features"]].copy()

    X = X.fillna(X.median())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    reducer = umap.UMAP(
        n_neighbors=5,
        min_dist=0.1,
        n_components=2,
        random_state=42
    )

    embedding = reducer.fit_transform(X_scaled)

    encoder = LabelEncoder()
    labels = encoder.fit_transform(df["Material"])

    score = silhouette_score(embedding, labels)

    results.append((name, score))

    temp_df = df.copy()
    temp_df["UMAP1"] = embedding[:, 0]
    temp_df["UMAP2"] = embedding[:, 1]

    # Save coordinates
    csv_name = (
        name.lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

    temp_df.to_csv(
        f"results/phase9/{csv_name}_coordinates.csv",
        index=False
    )

    # Individual plot
    plt.figure(figsize=(8, 6))

    for material in temp_df["Material"].unique():

        subset = temp_df[temp_df["Material"] == material]

        plt.scatter(
            subset["UMAP1"],
            subset["UMAP2"],
            s=120,
            label=material
        )

    plt.title(
        f"{name}\nSilhouette Score = {score:.4f}"
    )

    plt.xlabel("UMAP Component 1")
    plt.ylabel("UMAP Component 2")
    plt.legend()
    plt.grid(True)

    plt.savefig(
        f"results/phase9/{config['plot']}",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

# =====================================================
# COMPARISON FIGURE
# =====================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for ax, (name, config) in zip(axes, feature_sets.items()):

    X = df[config["features"]].copy()

    X = X.fillna(X.median())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    reducer = umap.UMAP(
        n_neighbors=5,
        min_dist=0.1,
        n_components=2,
        random_state=42
    )

    embedding = reducer.fit_transform(X_scaled)

    encoder = LabelEncoder()
    labels = encoder.fit_transform(df["Material"])

    score = silhouette_score(embedding, labels)

    temp_df = df.copy()
    temp_df["UMAP1"] = embedding[:, 0]
    temp_df["UMAP2"] = embedding[:, 1]

    for material in temp_df["Material"].unique():

        subset = temp_df[temp_df["Material"] == material]

        ax.scatter(
            subset["UMAP1"],
            subset["UMAP2"],
            s=100,
            label=material
        )

    ax.set_title(
        f"{name}\nScore = {score:.4f}"
    )

    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.grid(True)

handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=3
)

plt.tight_layout()

plt.savefig(
    "results/phase9/umap_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# =====================================================
# SUMMARY
# =====================================================
print("\nUMAP COMPARISON\n")
print("-" * 40)

for name, score in results:
    print(f"{name:20s} : {score:.4f}")

print("\nBest Result:")

best = max(results, key=lambda x: x[1])

print(
    f"{best[0]} "
    f"(Silhouette Score = {best[1]:.4f})"
)