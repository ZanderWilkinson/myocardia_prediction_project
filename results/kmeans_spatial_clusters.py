"""Cluster patients by spatial geometry and save a two-dimensional diagnostic plot.

Run from the project root with .venv/bin/python results/kmeans_spatial_clusters.py.
The projection uses full XYZ point sequences from all ten frames; metadata and
anatomical labels are not clustering features.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "training_data"
OUT = ROOT / "results"
metadata = pd.read_csv(DATA / "mixed_demographics.csv")

# Files follow row index + MI label, and rows are sorted by patient index.
features = np.empty((len(metadata), 10 * 18000 * 3), dtype=np.float32)
patient_ids = np.arange(len(metadata))
for i, row in metadata.iterrows():
    path = DATA / "mixed_samples" / f"{i}{row.MI}.npy"
    cloud = np.load(path, allow_pickle=False)
    if cloud.shape != (10, 18000, 4):
        raise ValueError(f"Unexpected shape in {path}: {cloud.shape}")
    if not np.isfinite(cloud[..., :3]).all():
        raise ValueError(f"Non-finite spatial coordinates in {path}")
    features[i] = cloud[..., :3].reshape(-1)
    if (i + 1) % 180 == 0:
        print(f"Loaded {i + 1}/{len(metadata)} patients", flush=True)

# Scale each coordinate-time feature, then cluster in the first 50 PCA
# components. This preserves major spatial variation while making k-means
# practical for the 540,000-dimensional patient vectors.
scaler = StandardScaler()
scaled = scaler.fit_transform(features)
pca = PCA(n_components=50, svd_solver="randomized", random_state=42)
scores = pca.fit_transform(scaled)
model = KMeans(n_clusters=2, n_init=20, random_state=42)
labels = model.fit_predict(scores)
distances = np.linalg.norm(scores - model.cluster_centers_[labels], axis=1)
projection = scores[:, :2]

fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), constrained_layout=True)
colors = ["#2878B5", "#E07A2D"]
for cluster in range(2):
    member = labels == cluster
    axes[0].scatter(projection[member, 0], projection[member, 1], s=24,
                    alpha=0.75, color=colors[cluster], label=f"Cluster {cluster} (n={member.sum()})")
axes[0].set_title("K-means on 50 spatial PCs, shown on first two PCs")
axes[0].set_xlabel(f"PC 1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
axes[0].set_ylabel(f"PC 2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
axes[0].legend(frameon=False)

# Mark the most distant patients from their assigned cluster centroid.
outlier_n = min(12, len(patient_ids))
outlier_ix = np.argsort(distances)[-outlier_n:]
axes[0].scatter(projection[outlier_ix, 0], projection[outlier_ix, 1],
                s=85, facecolors="none", edgecolors="#222222", linewidths=1.2,
                label=f"Top {outlier_n} centroid distances")
for i in outlier_ix:
    axes[0].annotate(str(i), projection[i], xytext=(3, 3),
                     textcoords="offset points", fontsize=7)
axes[0].legend(frameon=False, fontsize=8)

for cluster in range(2):
    member_dist = np.sort(distances[labels == cluster])[::-1]
    axes[1].plot(np.arange(1, len(member_dist) + 1), member_dist,
                 color=colors[cluster], linewidth=2, label=f"Cluster {cluster}")
axes[1].set_title("Distance to assigned centroid")
axes[1].set_xlabel("Patients in cluster, sorted most to least distant")
axes[1].set_ylabel("Euclidean distance (standardized feature space)")
axes[1].legend(frameon=False)

fig.suptitle("Patient spatial data: K-means (k=2)", fontsize=15)
OUT.mkdir(exist_ok=True)
plot_path = OUT / "kmeans_spatial_k2.png"
fig.savefig(plot_path, dpi=180)
plt.close(fig)

report = pd.DataFrame({
    "patient_index": patient_ids,
    "diagnosis": metadata.MI,
    "cluster": labels,
    "centroid_distance": distances,
    "pc1": projection[:, 0],
    "pc2": projection[:, 1],
})
report.sort_values("centroid_distance", ascending=False).to_csv(
    OUT / "kmeans_spatial_k2_patients.csv", index=False)

print("Cluster counts:", dict(zip(*np.unique(labels, return_counts=True))), flush=True)
print("Diagnosis by cluster:\n", pd.crosstab(labels, metadata.MI).to_string(), flush=True)
print(f"PCA variance in 50 components: {pca.explained_variance_ratio_.sum():.1%}", flush=True)
print("Most distant patients:", flush=True)
print(report.iloc[outlier_ix].sort_values("centroid_distance", ascending=False)
      [["patient_index", "diagnosis", "cluster", "centroid_distance"]].to_string(index=False), flush=True)
print(f"Saved {plot_path}", flush=True)
