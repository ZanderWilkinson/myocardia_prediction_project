"""Exploratory data audit and patient-level CV; does not use held-out test patients.

Run from the project root with .venv/bin/python results/dimensionality_reduction_audit.py.
The classification comparison uses 200 fixed point indices per anatomical label,
all 10 frames, and a fixed logistic-regression C=1. It is a screening experiment,
not an unbiased estimate of the winning model's final test performance.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
BASE = ROOT / "data/training_data"
df = pd.read_csv(BASE / "mixed_demographics.csv")
y_all = (df.MI == "pMI").to_numpy(dtype=int)
dev, holdout = train_test_split(
    np.arange(len(df)), test_size=0.2, random_state=42, stratify=df.MI
)
rng = np.random.default_rng(42)


def load(i):
    return np.load(BASE / "mixed_samples" / f"{i}{df.MI.iloc[i]}.npy", allow_pickle=False)


ref = load(dev[0])
labels = ref[0, :, 3]
selected = np.sort(np.concatenate([
    rng.choice(np.flatnonzero(labels == label), 200, replace=False)
    for label in np.unique(labels)
]))
X = np.empty((len(dev), 10 * len(selected) * 3), dtype=np.float64)
summaries = []
audit_clouds = []
all_shapes_match = True
all_labels_match = True
all_finite = True
for row, i in enumerate(dev):
    cloud = load(i)
    all_shapes_match &= cloud.shape == (10, 18000, 4)
    all_labels_match &= np.array_equal(cloud[:, :, 3], np.broadcast_to(labels, (10, 18000)))
    all_finite &= np.isfinite(cloud).all()
    X[row] = cloud[:, selected, :3].reshape(-1)
    features = []
    for label in np.unique(labels):
        xyz = cloud[:, labels == label, :3]
        center = xyz.mean(axis=1)
        centered = xyz - center[:, None, :]
        cov = np.einsum("tni,tnj->tij", centered, centered) / xyz.shape[1]
        eig = np.linalg.eigvalsh(cov)
        features.extend([center.ravel(), np.ptp(xyz, axis=1).ravel(), eig.ravel()])
    summaries.append(np.concatenate(features))
    if row < 24:
        audit_clouds.append(cloud)
    if (row + 1) % 180 == 0:
        print(f"Loaded {row + 1}/{len(dev)} development patients", flush=True)
S = np.asarray(summaries)
meta_columns = [c for c in df if c != "MI" and not c.lower().startswith("unnamed")]
M = df.iloc[dev][meta_columns].to_numpy(dtype=float)
y = y_all[dev]
report = {
    "population": {"total_training": len(df), "development": len(dev),
                   "untouched_internal_test": len(holdout),
                   "class_counts": df.MI.value_counts().to_dict()},
    "audit": {"patients_checked": len(dev), "shape": [10, 18000, 4],
              "all_shapes_match": bool(all_shapes_match),
              "all_labels_match_across_patients_and_frames": bool(all_labels_match),
              "all_finite": bool(all_finite),
              "label_counts": {str(int(l)): int((labels == l).sum()) for l in np.unique(labels)},
              "label_block_order": labels[np.r_[0, np.flatnonzero(np.diff(labels)) + 1]].tolist()},
    "benchmark": {"points_per_label": 200, "coordinate_features": X.shape[1],
                  "summary_features": S.shape[1], "metadata_features": meta_columns,
                  "folds": 5, "logistic_C": 1.0,
                  "scope": "exploratory fixed-setting CV on development patients; all learned transforms fit within each fold"},
}

for kind in ["between_patients", "between_frames_0_and_3"]:
    same, shuffled, correlations, overlaps = [], [], [], []
    for cloud in audit_clouds[1:]:
        for label in np.unique(labels):
            mask = labels == label
            a = audit_clouds[0][0, mask, :3] if kind == "between_patients" else cloud[0, mask, :3]
            b = cloud[0, mask, :3] if kind == "between_patients" else cloud[3, mask, :3]
            a, b = a - a.mean(0), b - b.mean(0)
            a, b = a / np.sqrt(np.mean(a * a)), b / np.sqrt(np.mean(b * b))
            same.append(float(np.sqrt(np.mean((a - b) ** 2))))
            shuffled.append(float(np.sqrt(np.mean((a - b[rng.permutation(len(b))]) ** 2))))
            correlations.append(float(np.corrcoef(a.ravel(), b.ravel())[0, 1]))
            ix = np.arange(0, len(a), 40)
            na = cKDTree(a).query(a[ix], k=11)[1][:, 1:]
            nb = cKDTree(b).query(b[ix], k=11)[1][:, 1:]
            overlaps.append(float(np.mean([len(set(u) & set(v)) / 10 for u, v in zip(na, nb)])))
    report["audit"][kind] = {
        "comparisons": len(same), "patients_sampled": len(audit_clouds),
        "median_same_index_normalized_rms": float(np.median(same)),
        "median_shuffled_normalized_rms": float(np.median(shuffled)),
        "median_coordinate_correlation": float(np.median(correlations)),
        "median_10_neighbor_overlap": float(np.median(overlaps)),
        "chance_10_neighbor_overlap": 10 / 5999,
        "minimum_coordinate_correlation": min(correlations),
    }
print(json.dumps(report["audit"], indent=2), flush=True)

scores = {}
spectra = {}


def evaluate(name, train_features, val_features, train, val):
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000))
    model.fit(train_features, y[train])
    probability = model.predict_proba(val_features)[:, 1]
    scores.setdefault(name, []).append({
        "auc": float(roc_auc_score(y[val], probability)),
        "balanced_accuracy": float(balanced_accuracy_score(y[val], probability >= 0.5)),
    })


folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train, val) in enumerate(folds.split(X, y), start=1):
    evaluate("demographics_only", M[train], M[val], train, val)
    evaluate("shape_summaries_only", S[train], S[val], train, val)
    evaluate("shape_summaries_plus_demographics", np.c_[S[train], M[train]], np.c_[S[val], M[val]], train, val)
    for scaled in [False, True]:
        name = "coordinate_standardized_pca" if scaled else "coordinate_centered_pca"
        pre = StandardScaler(with_std=scaled)
        a, b = pre.fit_transform(X[train]), pre.transform(X[val])
        pca = PCA(n_components=100, svd_solver="randomized", random_state=42)
        u = pca.fit_transform(a)
        v = pca.transform(b)
        ev = np.cumsum(pca.explained_variance_ratio_)
        spectra.setdefault(name, []).append({str(k): float(ev[k - 1]) for k in [2, 5, 10, 20, 50, 100]})
        for k in [2, 5, 10, 20, 50, 100]:
            evaluate(f"{name}_{k}", u[:, :k], v[:, :k], train, val)
            evaluate(f"{name}_{k}_plus_demographics", np.c_[u[:, :k], M[train]], np.c_[v[:, :k], M[val]], train, val)
    # PLS is supervised: fit only on this fold's training patients, then classify its scores.
    pls = PLSRegression(n_components=10, scale=False, max_iter=1000)
    pls.fit(X[train], y[train])
    u, v = pls.transform(X[train]), pls.transform(X[val])
    for k in [2, 5, 10]:
        evaluate(f"pls_{k}", u[:, :k], v[:, :k], train, val)
        evaluate(f"pls_{k}_plus_demographics", np.c_[u[:, :k], M[train]], np.c_[v[:, :k], M[val]], train, val)
    print(f"Completed fold {fold}/5", flush=True)

report["variance_explained"] = spectra
report["results"] = {
    name: {"mean_auc": float(np.mean([s["auc"] for s in values])),
           "sd_auc": float(np.std([s["auc"] for s in values], ddof=1)),
           "mean_balanced_accuracy": float(np.mean([s["balanced_accuracy"] for s in values])),
           "folds": values}
    for name, values in scores.items()
}
(OUT / "dimensionality_reduction_audit.json").write_text(json.dumps(report, indent=2))
table = pd.DataFrame(report["results"]).T.drop(columns="folds").sort_values("mean_auc", ascending=False)
table.to_csv(OUT / "dimensionality_reduction_comparison.csv")
print(table.to_string(), flush=True)
print("Variance explained:", json.dumps(spectra), flush=True)
