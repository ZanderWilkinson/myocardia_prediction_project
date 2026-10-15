"""Prepare PCA features once, then fit independent XGBoost experiments."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier

from .pca import pca_datasubset


@dataclass
class PreparedData:
    """Reusable feature matrices; no raw point clouds are retained here."""

    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    pca_model: object
    feature_names: list[str]


def encode_labels(subset):
    """Encode healthy=0 and pMI=1, rejecting missing or unexpected labels."""
    labels = subset.data["MI"]
    if not labels.isin(["healthy", "pMI"]).all():
        raise ValueError("MI must contain only 'healthy' and 'pMI', with no missing labels.")
    return (labels == "pMI").to_numpy(dtype=np.int32)


def prepare_xgboost_data(dataset, n_components=100, *, motion_only=True,
                         include_demographics=True):
    """Fit PCA on train only and transform val/test once, ready for repeated fits.

    By default PCA sees XYZ only; demographics are appended to its scores.
    If motion_only=False, demographics already enter PCA (legacy behavior),
    so include_demographics does not append a duplicate copy. To exclude
    demographics entirely, use motion_only=True, include_demographics=False.
    """
    subsets = (dataset.train, dataset.val, dataset.test)
    labels = [encode_labels(subset) for subset in subsets]
    X_train, pca_model = pca_datasubset(
        dataset.train, n_components=n_components, motion_only=motion_only,
    )
    X_val, _ = pca_datasubset(dataset.val, model=pca_model)
    X_test, _ = pca_datasubset(dataset.test, model=pca_model)
    matrices = [X_train, X_val, X_test]
    names = [f"PC{i + 1}" for i in range(X_train.shape[1])]

    if motion_only and include_demographics:
        columns = [c for c in dataset.train.data.columns
                   if c != "MI" and not c.lower().startswith("unnamed")]
        matrices = [np.column_stack((scores, subset.data.loc[:, columns].to_numpy(dtype=float)))
                    for scores, subset in zip(matrices, subsets)]
        names.extend(columns)

    return PreparedData(*matrices, *labels, pca_model, names)


def _source_signature(dataset):
    """Fingerprint source metadata and point arrays to prevent stale cache reuse."""
    metadata_path = Path(dataset.directory) / "mixed_demographics.csv"
    digest = hashlib.sha256(metadata_path.read_bytes())
    for patient in dataset.patients:
        path = Path(patient.sample_file)
        stat = path.stat()
        digest.update(f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}\n".encode())
    return digest.hexdigest()


def save_xgboost_feature_cache(cache_dir, dataset, prepared, *, n_components,
                              motion_only=True, include_demographics=True):
    """Persist only PCA scores and labels so tuning survives a kernel restart."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_name = f"pca_scores_{n_components}.npz"
    cache_path = cache_dir / cache_name
    temporary_path = cache_dir / f"{cache_name[:-4]}.tmp.npz"
    with temporary_path.open("wb") as stream:
        np.savez_compressed(
            stream, X_train=prepared.X_train, X_val=prepared.X_val,
            X_test=prepared.X_test, y_train=prepared.y_train,
            y_val=prepared.y_val, y_test=prepared.y_test,
            explained_variance_ratio=prepared.pca_model[-1].explained_variance_ratio_,
        )
    temporary_path.replace(cache_path)
    manifest = {
        "n_components": n_components,
        "motion_only": bool(motion_only),
        "include_demographics_after_pca": bool(include_demographics and motion_only),
        "source_signature_sha256": _source_signature(dataset),
        "train_indices": dataset.train.indices.tolist(),
        "validation_indices": dataset.val.indices.tolist(),
        "test_indices": dataset.test.indices.tolist(),
        "feature_names": prepared.feature_names,
        "explained_variance_ratio_sum": float(
            prepared.pca_model[-1].explained_variance_ratio_.sum()
        ),
        "score_cache": cache_path.name,
    }
    temporary_manifest = cache_dir / "manifest.tmp.json"
    temporary_manifest.write_text(json.dumps(manifest, indent=2))
    temporary_manifest.replace(cache_dir / "manifest.json")
    return manifest


def load_xgboost_feature_cache(cache_dir, dataset, *, n_components,
                               motion_only=True, include_demographics=True):
    """Load cached features only when PCA settings, splits, and source match."""
    cache_dir = Path(cache_dir)
    cache_path = cache_dir / f"pca_scores_{n_components}.npz"
    manifest_path = cache_dir / "manifest.json"
    if not cache_path.is_file() or not manifest_path.is_file():
        return None, None
    manifest = json.loads(manifest_path.read_text())
    expected = {
        "n_components": n_components,
        "motion_only": bool(motion_only),
        "include_demographics_after_pca": bool(include_demographics and motion_only),
        "source_signature_sha256": _source_signature(dataset),
        "train_indices": dataset.train.indices.tolist(),
        "validation_indices": dataset.val.indices.tolist(),
        "test_indices": dataset.test.indices.tolist(),
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        return None, None
    with np.load(cache_path, allow_pickle=False) as saved:
        feature_keys = ("X_train", "X_val", "X_test")
        label_keys = ("y_train", "y_val", "y_test")
        matrices = [saved[key].copy() for key in feature_keys]
        labels = [saved[key].copy() for key in label_keys]
        variance = saved["explained_variance_ratio"].copy()
    feature_names = manifest.get("feature_names", [])
    if any(matrix.shape[1] != len(feature_names) for matrix in matrices):
        return None, None
    if not all(np.array_equal(y, encode_labels(subset)) for y, subset in zip(
        labels, (dataset.train, dataset.val, dataset.test),
    )):
        return None, None

    # The classifier needs transformed features, not the large PCA component
    # matrix. Preserve the notebook's variance-summary interface.
    variance_summary = type("PCASummary", (), {"explained_variance_ratio_": variance})()
    model_summary = type("CachedPCA", (), {"__getitem__": lambda self, key:
                                variance_summary if key == -1 else IndexError(key)})()
    return PreparedData(*matrices, *labels, model_summary, feature_names), manifest


def train_xgboost(X_train, y_train, X_val, y_val, **hyperparameters):
    """Fit a fresh classifier on prepared features; never fit or transform PCA.

    Override any XGBClassifier setting with keyword arguments. Validation is
    used only when early stopping is enabled. With early_stopping_rounds=None,
    the model fits on training data alone; the test set must not be passed here.
    """
    parameters = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "n_estimators": 1000,
        "learning_rate": 0.03,
        "max_depth": 3,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 10,
        "early_stopping_rounds": 50,
        "random_state": 42,
        "n_jobs": 4,
    }
    parameters.update(hyperparameters)
    model = XGBClassifier(**parameters)
    if parameters.get("early_stopping_rounds") is None:
        model.fit(X_train, y_train, verbose=False)
    else:
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def calibrate_xgboost_threshold(X, y, *, objective="macro_f1", n_splits=5,
                               calibration_fraction=0.15, random_state=42,
                               **hyperparameters):
    """Calibrate a cutoff on training-only splits for the requested metric.

    For each outer fold, a stratified inner holdout calibrates the cutoff while
    the model is trained on the remaining fold-training patients. The returned
    value is the median fold cutoff and does not use validation/test outcomes.
    ``objective`` may be ``"macro_f1"`` or ``"balanced_accuracy"``. Pass the
    same XGBoost hyperparameters that will be used for the final model.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    if X.ndim != 2 or len(X) != len(y):
        raise ValueError("X must be a 2D matrix with one row per label.")
    if np.unique(y).size != 2:
        raise ValueError("Threshold calibration requires exactly two classes.")
    if objective not in {"macro_f1", "balanced_accuracy"}:
        raise ValueError("objective must be 'macro_f1' or 'balanced_accuracy'.")
    parameters = {
        "objective": "binary:logistic", "eval_metric": "auc", "tree_method": "hist",
        "n_estimators": 1000, "learning_rate": 0.03, "max_depth": 3,
        "min_child_weight": 5, "subsample": 0.8, "colsample_bytree": 0.8,
        "reg_lambda": 10, "random_state": random_state, "n_jobs": 4,
        "verbosity": 0,
    }
    parameters.update(hyperparameters)
    # No validation-based early stopping: use the requested fixed tree count.
    parameters.pop("early_stopping_rounds", None)
    thresholds = []
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for fold_id, (outer_train, _) in enumerate(splitter.split(X, y)):
        fit_idx, calibration_idx = train_test_split(
            outer_train, test_size=calibration_fraction,
            random_state=random_state + fold_id, stratify=y[outer_train],
        )
        fold_parameters = dict(parameters)
        fold_parameters["random_state"] = random_state
        model = XGBClassifier(**fold_parameters)
        model.fit(X[fit_idx], y[fit_idx], verbose=False)
        probabilities = model.predict_proba(X[calibration_idx])[:, 1]
        candidates = np.unique(np.r_[0.0, probabilities, 1.0])
        predictions = [probabilities >= cutoff for cutoff in candidates]
        if objective == "macro_f1":
            scores = [f1_score(y[calibration_idx], pred, average="macro", zero_division=0)
                      for pred in predictions]
        else:
            scores = [balanced_accuracy_score(y[calibration_idx], pred)
                      for pred in predictions]
        thresholds.append(float(candidates[int(np.argmax(scores))]))
    return float(np.median(thresholds))


def evaluate_xgboost(model, X, y, *, threshold=0.5):
    """Report patient-level discrimination and classification metrics."""
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1.")
    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "accuracy": float(accuracy_score(y, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
        "f1_pMI": float(f1_score(y, predictions, zero_division=0)),
    }
