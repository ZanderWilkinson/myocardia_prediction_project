"""Broad train-only XGBoost search using the saved 540-PC feature cache.

The five CV folds select hyperparameters inside the 540-patient training set.
Each fold has a separate early-stopping subset. The 180-patient validation set
is touched only once, after selecting the CV winner. PCA is never refitted.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.metrics import (roc_auc_score, accuracy_score, balanced_accuracy_score,
                             f1_score, roc_curve)
from sklearn.model_selection import ParameterSampler, StratifiedKFold, train_test_split
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/xgboost_tuning"
CACHE = OUT / "pca_scores_540.npz"
N_CANDIDATES = 240
N_FOLDS = 5
SEED = 20261015
MAX_ESTIMATORS = 5000


def sample_candidates():
    space = {
        "learning_rate": loguniform(0.005, 0.3),
        "max_depth": randint(1, 13),
        "min_child_weight": loguniform(0.1, 100),
        "subsample": uniform(0.5, 0.5),
        "colsample_bytree": uniform(0.15, 0.85),
        "reg_lambda": loguniform(1e-4, 1e3),
        "reg_alpha": [0.0, *np.geomspace(1e-5, 100, 80)],
        "gamma": [0.0, *np.geomspace(1e-4, 30, 80)],
        "max_bin": [64, 128, 256, 512],
    }
    return list(ParameterSampler(space, n_iter=N_CANDIDATES, random_state=SEED))


def make_model(params):
    return XGBClassifier(
        **params, objective="binary:logistic", eval_metric="auc", tree_method="hist",
        n_estimators=MAX_ESTIMATORS, early_stopping_rounds=100,
        random_state=SEED, n_jobs=4, verbosity=0,
    )


def score_candidate(params, X, y, folds):
    aucs, iterations = [], []
    for fold_id, (outer_train, outer_eval) in enumerate(folds):
        fit_idx, stop_idx = train_test_split(
            outer_train, test_size=0.15, random_state=SEED + fold_id,
            stratify=y[outer_train],
        )
        model = make_model(params)
        model.fit(X[fit_idx], y[fit_idx], eval_set=[(X[stop_idx], y[stop_idx])], verbose=False)
        prob = model.predict_proba(X[outer_eval])[:, 1]
        aucs.append(roc_auc_score(y[outer_eval], prob))
        iterations.append(int(model.best_iteration) + 1)
    return aucs, iterations


def cross_validated_probabilities(params, X, y, folds):
    """Get out-of-fold probabilities and choose a training-only cutoff."""
    probabilities = np.full(len(y), np.nan, dtype=float)
    for fold_id, (outer_train, outer_eval) in enumerate(folds):
        fit_idx, stop_idx = train_test_split(
            outer_train, test_size=0.15, random_state=SEED + fold_id,
            stratify=y[outer_train],
        )
        model = make_model(params)
        model.fit(X[fit_idx], y[fit_idx], eval_set=[(X[stop_idx], y[stop_idx])], verbose=False)
        probabilities[outer_eval] = model.predict_proba(X[outer_eval])[:, 1]
    fpr, tpr, thresholds = roc_curve(y, probabilities)
    threshold = float(thresholds[int(np.argmax(tpr - fpr))])
    return probabilities, threshold


def main():
    if not CACHE.is_file():
        raise FileNotFoundError(f"Expected existing PCA cache: {CACHE}")
    with np.load(CACHE, allow_pickle=False) as z:
        X_train, y_train = z["X_train"].copy(), z["y_train"].copy()
        X_val, y_val = z["X_val"].copy(), z["y_val"].copy()
    print(f"Using cached features: train {X_train.shape}, validation {X_val.shape}; PCA not refit.", flush=True)
    folds = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(X_train, y_train))
    candidates = sample_candidates()
    rows = []
    started = time.monotonic()
    for idx, params in enumerate(candidates, 1):
        aucs, iterations = score_candidate(params, X_train, y_train, folds)
        rows.append({"candidate": idx, **params,
                     "cv_auc_mean": float(np.mean(aucs)), "cv_auc_sd": float(np.std(aucs, ddof=1)),
                     "cv_auc_min": float(np.min(aucs)), "cv_auc_max": float(np.max(aucs)),
                     "best_iterations": json.dumps(iterations),
                     **{f"fold_{j+1}_auc": value for j, value in enumerate(aucs)}})
        if idx % 10 == 0:
            pd.DataFrame(rows).sort_values("cv_auc_mean", ascending=False).to_csv(OUT / "cv_trials.csv", index=False)
            leader = max(rows, key=lambda row: row["cv_auc_mean"])
            elapsed = time.monotonic() - started
            print(f"{idx}/{N_CANDIDATES}; best 5-fold AUC={leader['cv_auc_mean']:.4f} "
                  f"(SD {leader['cv_auc_sd']:.4f}); elapsed {elapsed/60:.1f} min", flush=True)
    table = pd.DataFrame(rows).sort_values("cv_auc_mean", ascending=False)
    table.to_csv(OUT / "cv_trials.csv", index=False)

    # Prefer the leading configuration's fold-median iteration count, then
    # train on all 540 development patients and evaluate validation once.
    winner = table.iloc[0]
    params = {key: winner[key].item() if isinstance(winner[key], np.generic) else winner[key]
              for key in candidates[0]}
    iterations = np.asarray(json.loads(winner.best_iterations), dtype=int)
    final_n_estimators = max(1, int(np.median(iterations)))
    oof_probability, cv_threshold = cross_validated_probabilities(params, X_train, y_train, folds)
    model = XGBClassifier(
        **params, objective="binary:logistic", eval_metric="auc", tree_method="hist",
        n_estimators=final_n_estimators, random_state=SEED, n_jobs=4, verbosity=0,
    )
    model.fit(X_train, y_train, verbose=False)
    probability = model.predict_proba(X_val)[:, 1]
    pred = probability >= cv_threshold
    validation = {
        "roc_auc": float(roc_auc_score(y_val, probability)),
        "accuracy": float(accuracy_score(y_val, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_val, pred)),
        "f1_pMI": float(f1_score(y_val, pred, zero_division=0)),
        "n_pMI_missed_at_cv_threshold": int(np.sum((y_val == 1) & ~pred)),
        "n_pMI": int(np.sum(y_val == 1)),
    }
    result = {
        "selection": "highest mean ROC-AUC across 5 stratified CV folds in the 540-patient training set",
        "candidates": N_CANDIDATES, "folds": N_FOLDS,
        "validation_used_once_after_selection": True,
        "test_set_scored": False,
        "pca_refit": False,
        "pca_components": 540,
        "classification_threshold": cv_threshold,
        "threshold_selection": "maximizes Youden J on train-only out-of-fold probabilities",
        "train_oof_auc": float(roc_auc_score(y_train, oof_probability)),
        "train_oof_balanced_accuracy_at_threshold": float(
            balanced_accuracy_score(y_train, oof_probability >= cv_threshold)),
        "features_per_patient": int(X_train.shape[1]),
        "selected_parameters": {**params, "n_estimators": final_n_estimators,
                                "random_state": SEED, "tree_method": "hist"},
        "selected_cv": {"mean_auc": float(winner.cv_auc_mean), "sd_auc": float(winner.cv_auc_sd),
                        "fold_auc": [float(winner[f"fold_{i+1}_auc"]) for i in range(N_FOLDS)],
                        "fold_best_iterations": iterations.tolist()},
        "validation_metrics_single_final_check": validation,
        "caveat": "PCA scores were computed once on all training patients before CV; PCA was unsupervised, but fold feature transforms were therefore transductive.",
    }
    (OUT / "cv_best_parameters.json").write_text(json.dumps(result, indent=2))
    print("\nTOP 10 CV CONFIGURATIONS", flush=True)
    print(table.head(10)[["cv_auc_mean", "cv_auc_sd", "max_depth", "learning_rate",
                          "min_child_weight", "subsample", "colsample_bytree",
                          "reg_lambda", "reg_alpha", "gamma", "max_bin"]].to_string(index=False), flush=True)
    print("\nFINAL HELD-BACK VALIDATION CHECK", json.dumps(validation, indent=2), flush=True)
    print("Selected parameters:", json.dumps(result["selected_parameters"], indent=2), flush=True)


if __name__ == "__main__":
    main()
