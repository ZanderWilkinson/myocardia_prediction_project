"""Fit 540-component PCA once, then tune XGBoost without using test outcomes.

Run from the project root with:
    VECLIB_MAXIMUM_THREADS=4 OMP_NUM_THREADS=4 .venv/bin/python -u results/tune_xgboost.py

Writes tuning metrics and cached PCA scores under results/xgboost_tuning/.
The cache lets notebooks reuse the expensive features after a kernel restart.
"""

import json
import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.model_selection import ParameterSampler, train_test_split
from xgboost import XGBClassifier

from mcpp.pca import pca_datasubset
from mcpp.xgboost import PreparedData, encode_labels, evaluate_xgboost


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/training_data"
OUT = ROOT / "results/xgboost_tuning"
OUT.mkdir(parents=True, exist_ok=True)
COMPONENTS = 540
BASELINE = {
    "n_estimators": 1000,
    "learning_rate": 0.03,
    "max_depth": 3,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 10.0,
    "reg_alpha": 0.0,
    "gamma": 0.0,
}
TUNABLE_KEYS = tuple(k for k in BASELINE if k != "n_estimators")
EARLY_STOPPING = 60


class LazyPatient:
    """Memory-map a sample on access to avoid keeping all point clouds in RAM."""

    def __init__(self, path):
        self.path = path

    @property
    def data(self):
        return np.load(self.path, mmap_mode="r", allow_pickle=False)


class PatientSubset:
    def __init__(self, metadata, indices, paths):
        self.indices = np.asarray(indices)
        self.data = metadata.iloc[self.indices]
        self.patients = [LazyPatient(paths[i]) for i in self.indices]


class PatientData:
    pass


def build_patient_splits():
    metadata = pd.read_csv(DATA / "mixed_demographics.csv")
    paths = [DATA / "mixed_samples" / f"{i}{label}.npy"
             for i, label in enumerate(metadata.MI)]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    indices = np.arange(len(metadata))
    train_val, test = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=metadata.MI,
    )
    train, val = train_test_split(
        train_val, test_size=0.25, random_state=42,
        stratify=metadata.MI.iloc[train_val],
    )
    dataset = PatientData()
    dataset.train = PatientSubset(metadata, train, paths)
    dataset.val = PatientSubset(metadata, val, paths)
    dataset.test = PatientSubset(metadata, test, paths)
    dataset.metadata = metadata
    dataset.paths = paths
    return dataset


def get_pca_features(dataset):
    print("Fitting PCA on 540 training patients with 540 XYZ components...", flush=True)
    started = time.monotonic()
    # Exactly the notebook's patient splits and 540-component, geometry-only PCA.
    X_train, pca_model = pca_datasubset(
        dataset.train, n_components=COMPONENTS, motion_only=True,
    )
    X_val, _ = pca_datasubset(dataset.val, model=pca_model)
    X_test, _ = pca_datasubset(dataset.test, model=pca_model)
    print(f"PCA fit and three transforms completed in {(time.monotonic()-started)/60:.1f} minutes.", flush=True)

    columns = [c for c in dataset.train.data.columns
               if c != "MI" and not c.lower().startswith("unnamed")]
    subsets = (dataset.train, dataset.val, dataset.test)
    matrices = [np.column_stack((scores, subset.data.loc[:, columns].to_numpy(dtype=float)))
                for scores, subset in zip((X_train, X_val, X_test), subsets)]
    result = PreparedData(
        *matrices,
        encode_labels(dataset.train), encode_labels(dataset.val), encode_labels(dataset.test),
        pca_model, [f"PC{i+1}" for i in range(COMPONENTS)] + columns,
    )

    # Store transformed scores, labels, and the PCA variance spectrum. This is
    # small; the dense 540 x 540,000 PCA component matrix is deliberately omitted.
    np.savez_compressed(
        OUT / "pca_scores_540.npz",
        X_train=result.X_train, X_val=result.X_val, X_test=result.X_test,
        y_train=result.y_train, y_val=result.y_val, y_test=result.y_test,
        explained_variance_ratio=pca_model[-1].explained_variance_ratio_,
    )
    source = hashlib.sha256()
    csv_path = DATA / "mixed_demographics.csv"
    source.update(csv_path.read_bytes())
    for path in dataset.paths:
        stat = path.stat()
        source.update(f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}\n".encode())
    manifest = {
        "n_components": COMPONENTS,
        "motion_only": True,
        "include_demographics_after_pca": True,
        "split_seed": 42,
        "split_sizes": {"train": len(result.y_train), "validation": len(result.y_val),
                        "test": len(result.y_test)},
        "train_indices": dataset.train.indices.tolist(),
        "validation_indices": dataset.val.indices.tolist(),
        "test_indices": dataset.test.indices.tolist(),
        "source_signature_sha256": source.hexdigest(),
        "label_counts": {k: int(v) for k, v in dataset.metadata.MI.value_counts().items()},
        "explained_variance_ratio_sum": float(pca_model[-1].explained_variance_ratio_.sum()),
        "feature_names": result.feature_names,
        "xgboost_version": __import__("xgboost").__version__,
        "score_cache": "pca_scores_540.npz",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return result


def make_model(params, seed=42):
    settings = dict(
        objective="binary:logistic", eval_metric="auc", tree_method="hist",
        n_estimators=2500, early_stopping_rounds=EARLY_STOPPING,
        random_state=seed, n_jobs=4, verbosity=0,
    )
    settings.update(params)
    settings["random_state"] = seed
    settings["early_stopping_rounds"] = EARLY_STOPPING
    return XGBClassifier(**settings)


def score_one(prepared, params, phase, trial, seed=42):
    model = make_model(params, seed=seed)
    model.fit(prepared.X_train, prepared.y_train,
              eval_set=[(prepared.X_val, prepared.y_val)], verbose=False)
    metrics = evaluate_xgboost(model, prepared.X_val, prepared.y_val)
    row = {"phase": phase, "trial": trial, "seed": seed,
           "n_estimators": int(model.n_estimators),
           **params, "best_iteration": int(model.best_iteration), **metrics}
    return model, row


def write_scores(rows):
    table = pd.DataFrame(rows).sort_values(
        ["roc_auc", "balanced_accuracy", "accuracy"], ascending=False,
    )
    table.to_csv(OUT / "trials.csv", index=False)
    seed_rows = table[table.phase == "seed_check"]
    if not seed_rows.empty:
        grouped = seed_rows.groupby(list(TUNABLE_KEYS), dropna=False).agg(
            mean_auc=("roc_auc", "mean"), sd_auc=("roc_auc", "std"),
            min_auc=("roc_auc", "min"), max_auc=("roc_auc", "max"),
        ).reset_index().sort_values("mean_auc", ascending=False)
        selected = grouped.iloc[0]
        robust_settings = {key: (int(selected[key]) if key == "max_depth" else float(selected[key]))
                           for key in TUNABLE_KEYS}
        matching = table[(table.seed == 42) & table.phase.isin(["random_search", "local_refinement"])]
        for key in TUNABLE_KEYS:
            target = robust_settings[key]
            if key == "max_depth":
                matching = matching[matching[key].astype(int) == target]
            else:
                matching = matching[np.isclose(matching[key].astype(float), target, rtol=1e-7, atol=1e-8)]
        metric_row = matching.sort_values("roc_auc", ascending=False).iloc[0] if not matching.empty else None
        selected_settings = {
            **robust_settings, "n_estimators": 1000, "early_stopping_rounds": EARLY_STOPPING,
            "random_state": 42, "validation_roc_auc_mean_3_seeds": float(selected.mean_auc),
            "validation_roc_auc_sd_3_seeds": float(selected.sd_auc),
            "validation_roc_auc_range_3_seeds": [float(selected.min_auc), float(selected.max_auc)],
        }
        if metric_row is not None:
            selected_settings.update({
                "validation_roc_auc_seed_42": float(metric_row.roc_auc),
                "validation_accuracy_seed_42": float(metric_row.accuracy),
                "validation_balanced_accuracy_seed_42": float(metric_row.balanced_accuracy),
                "validation_f1_pMI_seed_42": float(metric_row.f1_pMI),
                "best_iteration_seed_42": int(metric_row.best_iteration),
            })
        selection_metric = "mean validation ROC-AUC across random seeds 11, 22, and 33 among seed-checked settings"
    else:
        best = table.iloc[0]
        selected_settings = {k: best[k].item() if isinstance(best[k], np.generic) else best[k]
                             for k in ["n_estimators", *TUNABLE_KEYS, "best_iteration", "roc_auc",
                                       "accuracy", "balanced_accuracy", "f1_pMI"]}
        selection_metric = "validation ROC-AUC"
    (OUT / "best_parameters.json").write_text(json.dumps({
        "selection_metric": selection_metric,
        "selected": selected_settings,
        "validation_mean_auc_of_top_five": float(table.head(5).roc_auc.mean()),
        "validation_auc_range_of_top_five": [float(table.head(5).roc_auc.min()),
                                              float(table.head(5).roc_auc.max())],
        "trials": len(table),
        "test_set_scored": False,
    }, indent=2))
    return table


def main():
    dataset = build_patient_splits()
    cache_path = OUT / "pca_scores_540.npz"
    manifest_path = OUT / "manifest.json"
    expected_train = dataset.train.indices.tolist()
    expected_val = dataset.val.indices.tolist()
    expected_test = dataset.test.indices.tolist()
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    current_signature = hashlib.sha256()
    csv_path = DATA / "mixed_demographics.csv"
    current_signature.update(csv_path.read_bytes())
    for path in dataset.paths:
        stat = path.stat()
        current_signature.update(f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}\n".encode())
    cache_is_current = (
        cache_path.is_file()
        and manifest.get("n_components") == COMPONENTS
        and manifest.get("motion_only") is True
        and manifest.get("include_demographics_after_pca") is True
        and manifest.get("train_indices") == expected_train
        and manifest.get("validation_indices") == expected_val
        and manifest.get("test_indices") == expected_test
        and manifest.get("source_signature_sha256") == current_signature.hexdigest()
    )
    if cache_is_current:
        print("Loading matching cached 540-component PCA features; no PCA refit.", flush=True)
        with np.load(cache_path, allow_pickle=False) as saved:
            columns = [c for c in dataset.train.data.columns
                       if c != "MI" and not c.lower().startswith("unnamed")]
            features = [saved[k].copy() for k in ("X_train", "X_val", "X_test")]
            labels = [saved[k].copy() for k in ("y_train", "y_val", "y_test")]
            variance = saved["explained_variance_ratio"].copy()
        class CachedPCA:
            def __getitem__(self, key):
                if key == -1:
                    return type("PCASummary", (), {"explained_variance_ratio_": variance})()
                raise IndexError(key)
        prepared = PreparedData(
            *features, *labels, CachedPCA(),
            [f"PC{i+1}" for i in range(COMPONENTS)] + columns,
        )
    else:
        prepared = get_pca_features(dataset)
    trials_file = OUT / "trials.csv"
    rows = pd.read_csv(trials_file).to_dict(orient="records") if trials_file.is_file() else []
    for row in rows:
        if pd.isna(row.get("n_estimators")):
            row["n_estimators"] = 2500 if row["phase"] != "notebook_baseline" else BASELINE["n_estimators"]
    baseline_rows = [r for r in rows if r["phase"] == "notebook_baseline"]
    if baseline_rows:
        print("Reusing the saved notebook-parameter baseline.", flush=True)
    else:
        _, baseline = score_one(prepared, BASELINE, "notebook_baseline", 0)
        rows.append(baseline)
        print("Notebook-parameter baseline:", json.dumps(baseline), flush=True)

    # Broad randomized search; use only the existing validation subset.
    space = {
        "learning_rate": loguniform(0.01, 0.15),
        "max_depth": randint(1, 7),
        "min_child_weight": loguniform(1, 30),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.3, 0.7),
        "reg_lambda": loguniform(0.1, 100),
        "reg_alpha": loguniform(1e-4, 10),
        "gamma": loguniform(1e-4, 3),
    }
    candidates = list(ParameterSampler(space, n_iter=120, random_state=704))
    for idx, params in enumerate(candidates, start=1):
        if any(r["phase"] == "random_search" and int(r["trial"]) == idx for r in rows):
            continue
        _, row = score_one(prepared, params, "random_search", idx)
        rows.append(row)
        if idx % 10 == 0:
            top = max(rows, key=lambda r: r["roc_auc"])
            print(f"Random search {idx}/120; best validation AUC={top['roc_auc']:.4f}; "
                  f"depth={top['max_depth']}, best_iteration={top['best_iteration']}", flush=True)

    table = write_scores(rows)
    # Refine depth, shrinkage, sampling, and regularization around the leading
    # candidates. Round continuous values to keep each candidate identifiable.
    centers = table.head(4).to_dict(orient="records")
    refinements = {}
    depth_deltas = (-1, 0, 1)
    lr_scales = (0.7, 1.0, 1.4)
    reg_scales = (0.5, 1.0, 2.0)
    for center in centers:
        for dd in depth_deltas:
            for lr_scale, reg_scale in zip(lr_scales, reg_scales):
                params = {
                    "learning_rate": round(float(center["learning_rate"]) * lr_scale, 5),
                    "max_depth": int(np.clip(int(center["max_depth"]) + dd, 1, 8)),
                    "min_child_weight": round(float(center["min_child_weight"]), 4),
                    "subsample": round(float(center["subsample"]), 4),
                    "colsample_bytree": round(float(center["colsample_bytree"]), 4),
                    "reg_lambda": round(float(center["reg_lambda"]) * reg_scale, 5),
                    "reg_alpha": round(float(center["reg_alpha"]), 5),
                    "gamma": round(float(center["gamma"]), 5),
                }
                refinements[tuple(params.items())] = params
    for idx, params in enumerate(refinements.values(), start=1):
        _, row = score_one(prepared, params, "local_refinement", idx)
        rows.append(row)
        if idx % 10 == 0:
            top = max(rows, key=lambda r: r["roc_auc"])
            print(f"Refinement {idx}/{len(refinements)}; best validation AUC={top['roc_auc']:.4f}", flush=True)

    # Repeat the strongest distinct settings with several random seeds to
    # check whether their ranking depends strongly on row/column subsampling.
    table = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
    robust_candidates = []
    for _, row in table.iterrows():
        params = {key: row[key] for key in TUNABLE_KEYS}
        signature = tuple(round(float(params[k]), 8) if k != "max_depth"
                          else int(params[k]) for k in TUNABLE_KEYS)
        if signature not in [item[0] for item in robust_candidates]:
            robust_candidates.append((signature, params))
        if len(robust_candidates) == 4:
            break
    for idx, (_, params) in enumerate(robust_candidates, start=1):
        for seed in (11, 22, 33):
            _, row = score_one(prepared, params, "seed_check", idx, seed=seed)
            rows.append(row)

    table = write_scores(rows)
    print("\nTOP 12 VALIDATION TRIALS", flush=True)
    print(table.head(12)[["phase", "trial", "seed", "learning_rate", "max_depth",
                           "min_child_weight", "subsample", "colsample_bytree",
                           "reg_lambda", "reg_alpha", "gamma", "best_iteration",
                           "roc_auc", "accuracy", "balanced_accuracy", "f1_pMI"]].to_string(index=False),
          flush=True)
    print("\nSelected settings saved to", OUT / "best_parameters.json", flush=True)
    print("Validation scores saved to", OUT / "trials.csv", flush=True)
    print("PCA feature cache saved to", OUT / "pca_scores_540.npz", flush=True)


if __name__ == "__main__":
    main()
