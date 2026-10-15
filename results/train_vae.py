"""Run patient-level VAE CV, with the existing 20% test split left untouched."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import re
import time

from filelock import Timeout
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from mcpp.vae import VAEConfig
from mcpp.vae_data import DEMOGRAPHIC_COLUMNS, PreprocessConfig, load_cohort
from mcpp.vae_training import TrainConfig, cross_validate_vae, mi_classification_report, select_device
from mcpp.vae_experiments import atomic_json_save, experiment_lock


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/training_data")
    parser.add_argument("--experiment", default="default",
                        help="Experiment name; matching names continue training (default: default)")
    parser.add_argument("--output-dir", type=Path,
                        help="Experiment directory; default: results/vae_training/<experiment>/")
    parser.add_argument("--epochs", type=int, default=1, help="Additional epochs per fold for this invocation (default: 1)")
    parser.add_argument("--checkpoint-every", type=int, default=100,
                        help="Replace rolling training checkpoints every N epochs and at run end (default: 100)")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--latent-dim", type=int, default=20)
    parser.add_argument("--points", type=int, default=384)
    parser.add_argument("--limit-patients", type=int, help="Stratified development subset for a quick run")
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, cuda or cuda:N (e.g. cuda:1)")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True,
                        help="CUDA mixed precision and gradient scaling (default: enabled); --no-amp uses FP32")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", args.experiment):
        parser.error("Experiment names must use 1-80 letters, digits, dots, underscores or hyphens, starting with a letter/digit.")
    if args.epochs < 1 or args.checkpoint_every < 1:
        parser.error("epochs and checkpoint-every must be positive.")
    # Validate GPU access before reading and preprocessing the cohort.
    try:
        device = select_device(args.device)
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    if args.output_dir is None:
        args.output_dir = ROOT / "results/vae_training" / args.experiment
    try:
        with experiment_lock(args.output_dir) as run_lock:
            run_experiment(args, device, run_lock)
    except Timeout:
        parser.error("This experiment is already being trained by another process.")
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))


def run_experiment(args, device, run_lock):
    meta = pd.read_csv(args.data_dir / "mixed_demographics.csv")
    if not meta.MI.isin(["healthy", "pMI"]).all():
        raise ValueError("Unexpected or missing MI label.")
    # Match DataSet's held-out test membership, regardless of training seed.
    development, held_out = train_test_split(
        np.arange(len(meta)), test_size=.2, stratify=meta.MI, random_state=42,
    )
    if args.limit_patients is not None:
        if not 0 < args.limit_patients <= len(development):
            parser.error("limit-patients must be positive and no larger than the development set.")
        if args.limit_patients < len(development):
            development, _ = train_test_split(
                development, train_size=args.limit_patients, stratify=meta.MI.iloc[development],
                random_state=args.seed,
            )
    preprocess = PreprocessConfig(n_points=args.points)
    model = VAEConfig(n_points=args.points, latent_dim=args.latent_dim)
    training = TrainConfig(epochs=args.epochs, batch_size=args.batch_size,
                           device=str(device), seed=args.seed, amp=args.amp,
                           checkpoint_every=args.checkpoint_every)
    started = time.monotonic()
    print(f"Preparing {len(development)} development patients; {len(held_out)} test patients excluded.", flush=True)
    print(f"Experiment {args.experiment!r}: adding {args.epochs} epoch(s) per fold "
          f"(an unfinished invocation resumes its saved targets).\n"
          f"Results: {args.output_dir}", flush=True)
    xyz, regions, demographics, labels, audit = load_cohort(args.data_dir, development, preprocess)
    development = np.asarray([a["patient_index"] for a in audit if not a["excluded"]], dtype=int)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    retained_audit = [a for a in audit if not a["excluded"]]
    print(f"Repaired {sum(len(a['flagged_frames']) for a in retained_audit)} frames in "
          f"{sum(bool(a['flagged_frames']) for a in retained_audit)} retained patients; "
          f"{len(audit) - len(retained_audit)} cycles excluded.", flush=True)
    result = cross_validate_vae(
        xyz, regions, demographics, labels, n_splits=args.folds, model_config=model,
        train_config=training, preprocess_config=preprocess, output_dir=args.output_dir,
        patient_ids=development,
        experiment_name=args.experiment, run_lock=run_lock,
    )
    atomic_json_save(audit, args.output_dir / "frame_quality.json")
    # Latent axes from independently fitted folds are not aligned for joint analysis.
    probabilities = result.pop("oof_probabilities")
    np.savez_compressed(args.output_dir / "oof_predictions.npz", patient_ids=development,
                        labels=labels, probabilities=probabilities,
                        latent_means=result.pop("oof_latent_means"))
    result.update({"model_config": asdict(model), "train_config": asdict(training),
                   "preprocess_config": asdict(preprocess),
                   "demographic_columns": list(DEMOGRAPHIC_COLUMNS),
                   "test_patient_ids_excluded": held_out.tolist(), "test_set_scored": False,
                   "quality_excluded_patient_ids": [a["patient_index"] for a in audit if a["excluded"]],
                   "classification_threshold": .5,
                   "latent_axes_aligned_across_folds": False,
                   "total_seconds": time.monotonic() - started})
    atomic_json_save(result, args.output_dir / "cv_results.json")
    report = mi_classification_report(labels, probabilities)
    report.update(evaluation="stratified patient-level out-of-fold MI predictions",
                  n_folds=args.folds, epochs_per_fold=max(f["completed_epochs"] for f in result["folds"]),
                  epochs_added_per_invocation=args.epochs, experiment_name=args.experiment,
                  completed_epochs_per_fold=[f["completed_epochs"] for f in result["folds"]],
                  test_set_scored=False,
                  n_quality_excluded=len(result["quality_excluded_patient_ids"]),
                  n_test_held_out=len(held_out),
                  quality_excluded_patient_ids=result["quality_excluded_patient_ids"])
    atomic_json_save(report, args.output_dir / "mi_classification_report.json")
    patient_folds = {patient: fold["fold"] for fold in result["folds"] for patient in fold["score_patient_ids"]}
    pd.DataFrame({"patient_index": development,
                  "fold": [patient_folds[int(patient)] for patient in development],
                  "true_MI": np.where(labels, "pMI", "healthy"),
                  "predicted_MI": np.where(probabilities >= .5, "pMI", "healthy"),
                  "probability_pMI": probabilities}).to_csv(args.output_dir / "mi_predictions.csv", index=False)
    metrics = report["metrics"]
    print(f"\nOverall MI classification: {len(labels)} out-of-fold patients, {args.folds} folds\n"
          f"F1 (pMI): {metrics['f1_pMI']:.4f}\n"
          f"Accuracy: {metrics['accuracy']:.4f} ({100 * metrics['accuracy']:.2f}%)\n"
          f"Macro F1: {metrics['macro_f1']:.4f}\n"
          f"Balanced accuracy: {metrics['balanced_accuracy']:.4f}\n"
          f"ROC-AUC: {metrics['roc_auc']:.4f}\n"
          f"Report: {args.output_dir / 'mi_classification_report.json'}\n"
          f"Patient predictions: {args.output_dir / 'mi_predictions.csv'}\n"
          f"Final test set scored: no", flush=True)


if __name__ == "__main__":
    main()
