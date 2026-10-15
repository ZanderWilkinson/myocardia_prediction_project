"""Compare 100/200/300/540 PCA components with matched XGBoost CV searches.

Uses the existing 540-component feature cache and fits PCA once each for
100/200/300 on the same training patients. Hyperparameters are evaluated on
the same five patient-level folds. Each outer fold gets a private stopping and
threshold-calibration subset, so its held-out fold selects neither.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import ParameterSampler, StratifiedKFold, train_test_split
from xgboost import XGBClassifier

from mcpp.pca import pca_datasubset
from tune_xgboost import build_patient_splits
from tune_xgboost_cv import N_FOLDS, SEED, MAX_ESTIMATORS, OUT, make_model

DIMENSIONS = (10, 25, 50, 100, 200, 300, 540)
N_CANDIDATES = 300
SEARCH_SEED = 20261016


def sample_candidates():
    """Broad randomized coverage; same 300 configurations for every PCA size."""
    space = {
        "learning_rate": loguniform(0.001, 0.5),
        "max_depth": randint(1, 17),
        "min_child_weight": loguniform(0.01, 300),
        "subsample": uniform(0.3, 0.7),
        "colsample_bytree": uniform(0.1, 0.9),
        "reg_lambda": loguniform(1e-6, 1e4),
        "reg_alpha": [0.0, *np.geomspace(1e-8, 1e3, 120)],
        "gamma": [0.0, *np.geomspace(1e-5, 100, 120)],
        "max_bin": [32, 64, 128, 256, 512, 1024],
        "grow_policy": ["depthwise", "lossguide"],
        "max_leaves": randint(8, 257),
        "max_delta_step": [0, 1, 5, 10],
    }
    return list(ParameterSampler(space, n_iter=N_CANDIDATES, random_state=SEARCH_SEED))


def load_or_build_features(n_components):
    path = OUT / f"pca_scores_{n_components}_cv.npz"
    if path.is_file():
        with np.load(path, allow_pickle=False) as z:
            return tuple(z[k].copy() for k in ("X_train", "y_train", "X_val", "y_val"))
    dataset = build_patient_splits()
    if n_components == 540:
        with np.load(OUT / "pca_scores_540.npz", allow_pickle=False) as z:
            X_train, y_train, X_val, y_val = (z[k].copy() for k in ("X_train", "y_train", "X_val", "y_val"))
    else:
        print(f"Fitting {n_components}-component PCA once on training patients...", flush=True)
        X_train, pca_model = pca_datasubset(dataset.train, n_components=n_components, motion_only=True)
        X_val, _ = pca_datasubset(dataset.val, model=pca_model)
        columns = [c for c in dataset.train.data.columns
                   if c != "MI" and not c.lower().startswith("unnamed")]
        X_train = np.column_stack((X_train, dataset.train.data.loc[:, columns].to_numpy(dtype=float)))
        X_val = np.column_stack((X_val, dataset.val.data.loc[:, columns].to_numpy(dtype=float)))
        y_train = (dataset.train.data.MI == "pMI").to_numpy(dtype=np.int32)
        y_val = (dataset.val.data.MI == "pMI").to_numpy(dtype=np.int32)
        np.savez_compressed(path, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)
    return X_train, y_train, X_val, y_val


def fbeta(y, pred, beta=2.0):
    precision = precision_score(y, pred, zero_division=0)
    recall = recall_score(y, pred, zero_division=0)
    b2 = beta * beta
    return float((1 + b2) * precision * recall / (b2 * precision + recall)) if (precision + recall) else 0.0


def evaluate_candidate(params, X, y, folds):
    fold_rows, iterations, cutoffs = [], [], []
    for fold_id, (outer_train, outer_eval) in enumerate(folds):
        fit_idx, stop_idx = train_test_split(
            outer_train, test_size=0.15, random_state=SEED + fold_id,
            stratify=y[outer_train],
        )
        model = make_model(params)
        model.fit(X[fit_idx], y[fit_idx], eval_set=[(X[stop_idx], y[stop_idx])], verbose=False)
        stop_prob = model.predict_proba(X[stop_idx])[:, 1]
        # Calibrate the operating point only on the inner stopping subset.
        candidates = np.unique(np.r_[0.0, stop_prob, 1.0])
        scores = [f1_score(y[stop_idx], stop_prob >= cutoff, average="macro",
                           zero_division=0) for cutoff in candidates]
        cutoff = float(candidates[int(np.argmax(scores))])
        eval_prob = model.predict_proba(X[outer_eval])[:, 1]
        pred = eval_prob >= cutoff
        fold_rows.append({
            "auc": float(roc_auc_score(y[outer_eval], eval_prob)),
            "macro_f1": float(f1_score(y[outer_eval], pred, average="macro", zero_division=0)),
            "f2": fbeta(y[outer_eval], pred),
            "recall": float(recall_score(y[outer_eval], pred, zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y[outer_eval], pred)),
            "threshold": cutoff,
        })
        iterations.append(int(model.best_iteration) + 1)
        cutoffs.append(cutoff)
    return fold_rows, iterations, cutoffs


def main():
    parameter_names = ["colsample_bytree", "gamma", "learning_rate", "max_bin", "max_depth",
                       "min_child_weight", "reg_alpha", "reg_lambda", "subsample",
                       "grow_policy", "max_leaves", "max_delta_step"]
    candidates = sample_candidates()
    all_results = {}
    for n_components in DIMENSIONS:
        started = time.monotonic()
        X_train, y_train, X_val, y_val = load_or_build_features(n_components)
        folds = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(X_train, y_train))
        rows = []
        trial_file = OUT / f"macro_f1_cv_trials_{n_components}pc.csv"
        done = set()
        # A parameter-space version in the filename prevents stale narrow sweeps
        # from being mistaken for this expanded search.
        trial_file = OUT / f"macro_f1_broad_v2_trials_{n_components}pc.csv"
        for idx, params in enumerate(candidates, 1):
            if idx in done:
                continue
            fold_rows, iterations, cutoffs = evaluate_candidate(params, X_train, y_train, folds)
            rows.append({
                "candidate": idx, **params,
                "cv_auc_mean": float(np.mean([r["auc"] for r in fold_rows])),
                "cv_auc_sd": float(np.std([r["auc"] for r in fold_rows], ddof=1)),
                "cv_macro_f1_mean": float(np.mean([r["macro_f1"] for r in fold_rows])),
                "cv_f2_mean": float(np.mean([r["f2"] for r in fold_rows])),
                "cv_recall_mean": float(np.mean([r["recall"] for r in fold_rows])),
                "cv_balanced_accuracy_mean": float(np.mean([r["balanced_accuracy"] for r in fold_rows])),
                "fold_auc": json.dumps([r["auc"] for r in fold_rows]),
                "fold_f2": json.dumps([r["f2"] for r in fold_rows]),
                "fold_macro_f1": json.dumps([r["macro_f1"] for r in fold_rows]),
                "fold_recall": json.dumps([r["recall"] for r in fold_rows]),
                "fold_thresholds": json.dumps(cutoffs),
                "best_iterations": json.dumps(iterations),
            })
            if idx % 10 == 0 or idx == N_CANDIDATES:
                pd.DataFrame(rows).sort_values("cv_macro_f1_mean", ascending=False).to_csv(trial_file, index=False)
                lead = max(rows, key=lambda r: r["cv_macro_f1_mean"])
                print(f"{n_components} PCs: {idx}/{N_CANDIDATES}; best CV macro-F1="
                      f"{lead['cv_macro_f1_mean']:.3f}, recall={lead['cv_recall_mean']:.3f}, "
                      f"AUC={lead['cv_auc_mean']:.3f}; "
                      f"{(time.monotonic()-started)/60:.1f} min", flush=True)
        table = pd.DataFrame(rows).sort_values("cv_macro_f1_mean", ascending=False)
        table.to_csv(trial_file, index=False)
        winner = table.iloc[0]
        params = {key: (int(winner[key]) if key in ("max_bin", "max_depth", "max_leaves", "max_delta_step")
                        else (str(winner[key]) if key == "grow_policy" else float(winner[key])))
                  for key in parameter_names}
        n_estimators = max(1, int(np.median(json.loads(winner.best_iterations))))
        threshold = float(np.median(json.loads(winner.fold_thresholds)))
        model = XGBClassifier(
            **params, objective="binary:logistic", eval_metric="auc", tree_method="hist",
            n_estimators=n_estimators, random_state=SEED, n_jobs=4, verbosity=0,
        )
        model.fit(X_train, y_train, verbose=False)
        val_prob = model.predict_proba(X_val)[:, 1]
        metrics = {}
        for name, cutoff in (("cv_threshold", threshold), ("default_0_5", 0.5)):
            pred = val_prob >= cutoff
            metrics[name] = {
                "threshold": cutoff,
                "roc_auc": float(roc_auc_score(y_val, val_prob)),
                "accuracy": float(accuracy_score(y_val, pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y_val, pred)),
                "precision_pMI": float(precision_score(y_val, pred, zero_division=0)),
                "recall_pMI": float(recall_score(y_val, pred, zero_division=0)),
                "f1_pMI": float(f1_score(y_val, pred, zero_division=0)),
                "f2_pMI": fbeta(y_val, pred),
                "pMI_detected": int(np.sum((y_val == 1) & pred)),
                "pMI_total": int(np.sum(y_val == 1)),
            }
        all_results[str(n_components)] = {
            "selected_by": "highest mean outer-fold macro-F1; cutoff calibrated on inner stopping split to maximize macro-F1",
            "cv": {key: float(winner[key]) for key in ("cv_auc_mean", "cv_auc_sd", "cv_macro_f1_mean",
                                                        "cv_f2_mean", "cv_recall_mean", "cv_balanced_accuracy_mean")},
            "parameters": {**params, "n_estimators": n_estimators,
                            "classification_threshold": threshold, "random_state": SEED},
            "validation": metrics,
            "trials_file": trial_file.name,
        }
        print(f"{n_components} PCs validation: {json.dumps(metrics)}", flush=True)
    result = {
        "components_tested": list(DIMENSIONS),
        "candidate_count_per_dimension": N_CANDIDATES,
        "search_seed": SEARCH_SEED,
        "search_space": {"learning_rate": [0.001, 0.5], "max_depth": [1, 16],
                         "min_child_weight": [0.01, 300], "subsample": [0.3, 1.0],
                         "colsample_bytree": [0.1, 1.0], "reg_lambda": [1e-6, 1e4],
                         "reg_alpha": [0, 1e3], "gamma": [0, 100],
                         "max_bin": [32, 64, 128, 256, 512, 1024],
                         "grow_policy": ["depthwise", "lossguide"], "max_leaves": [8, 256],
                         "max_delta_step": [0, 1, 5, 10]},
        "patient_level_cv_folds": N_FOLDS,
        "validation_used_to_compare_dimensions": True,
        "test_set_scored": False,
        "pca_scope": "each PCA fitted on the 540 training patients only; validation transformed with that PCA",
        "threshold_objective": "Each fold's threshold maximizes macro-F1 on that fold's inner calibration subset; final cutoff is the median fold threshold.",
        "note": "Validation metrics compare candidate dimensions and are selection metrics, not a final unbiased estimate. PCA was fit once on the full training subset before inner CV, so CV PCA transforms are unsupervised-transductive.",
        "results": all_results,
    }
    (OUT / "pca_dimension_comparison_macro_f1.json").write_text(json.dumps(result, indent=2))
    print("\nFINAL COMPONENT COMPARISON", json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
