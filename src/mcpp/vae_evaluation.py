"""Evaluate validation-selected weights on held-out or external final patients."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch

from .vae_data import DEMOGRAPHIC_COLUMNS, PreprocessConfig, load_cohort
from .vae_experiments import atomic_json_save, experiment_lock
from .vae_training import load_checkpoint, mi_classification_report, predict_prepared


def _experiment_identity(directory, saved):
    identity = json.loads((directory / "experiment.json").read_text())["identity"]
    context = saved.get("metadata", {}).get("context", {})
    if (saved.get("metadata", {}).get("selection_metric") != "validation_macro_f1"
            or context.get("inputs") != identity["input_signature"]
            or saved["model_config"] != identity["model_config"]
            or saved["preprocess_config"] != identity["preprocess_config"]
            or context.get("extra", {}).get("experiment_name") != identity["experiment_name"]):
        raise ValueError("Expected a val_best checkpoint belonging to this experiment directory.")
    if (saved["demographic_columns"] != list(DEMOGRAPHIC_COLUMNS)
            or saved["class_mapping"] != {"healthy": 0, "pMI": 1}):
        raise ValueError("Checkpoint demographic or MI class schema is incompatible.")
    return identity


def _test_membership(directory, metadata, identity):
    if not metadata.MI.isin(["healthy", "pMI"]).all():
        raise ValueError("Unexpected or missing MI label in the cohort.")
    # Training fixes this split at seed 42, independently of its model seed.
    _, test = train_test_split(np.arange(len(metadata)), test_size=.2,
                               stratify=metadata.MI, random_state=42)
    development = np.asarray(identity["patient_ids"])
    if (development.ndim != 1 or not len(development)
            or not np.issubdtype(development.dtype, np.integer)
            or len(np.unique(development)) != len(development)
            or development.min() < 0 or development.max() >= len(metadata)
            or np.intersect1d(development, test).size):
        raise ValueError("Saved development patients overlap the test split or do not match this cohort.")
    # Completed CLI runs also record exact test membership. Check it when present;
    # a still-running experiment can be evaluated before its final report exists.
    result_path = directory / ("holdout_results.json" if identity.get("evaluation_mode") == "validation_holdout"
                               else "cv_results.json")
    source = "original stratified 20% split, random_state=42"
    if result_path.exists():
        recorded = json.loads(result_path.read_text())["test_patient_ids_excluded"]
        if not np.array_equal(np.sort(recorded), np.sort(test)):
            raise ValueError("Recorded test membership differs from this cohort's original held-out split.")
        test, source = np.asarray(recorded, dtype=int), str(result_path.resolve())
    return test, source


def evaluate_test_checkpoint(checkpoint, data_dir, *, output_dir=None, device="auto",
                             batch_size=8, cpu_threads=4, final=False):
    """Score only held-out patients, using saved transforms and cutoff 0.5.

    final=True evaluates every row of test.csv against test_samples/N.npy.
    No fitting, threshold selection or checkpoint updates occur. The same fixed
    frame-quality policy applies; excluded test cycles and coverage are reported.
    """
    if batch_size < 1 or cpu_threads < 1:
        raise ValueError("batch_size and cpu_threads must be positive.")
    checkpoint, data_dir = Path(checkpoint).resolve(), Path(data_dir)
    torch.set_num_threads(cpu_threads)
    model, scaler, saved = load_checkpoint(checkpoint, device=device)
    identity = _experiment_identity(checkpoint.parent, saved)
    metadata = pd.read_csv(data_dir / ("test.csv" if final else "mixed_demographics.csv"))
    loader_options = {}
    if final:
        test = np.arange(len(metadata))
        split_source = str((data_dir / "test.csv").resolve())
        loader_options = dict(metadata_filename="test.csv", samples_subdir="test_samples", labelled_filenames=False)
        # Original patient IDs are 900..999, while sample filenames use 0..99.
        patient_ids = metadata["Unnamed: 0"].to_numpy() if "Unnamed: 0" in metadata else test
        if (not len(patient_ids) or not np.issubdtype(patient_ids.dtype, np.integer)
                or len(np.unique(patient_ids)) != len(patient_ids)):
            raise ValueError("Final test patient IDs must be unique integers.")
        if "Unnamed: 0" in metadata and np.intersect1d(patient_ids, identity["patient_ids"]).size:
            raise ValueError("Final test patient IDs overlap saved development patients.")
    else:
        test, split_source = _test_membership(checkpoint.parent, metadata, identity)
        patient_ids = np.arange(len(metadata))
    output_dir = (Path(output_dir) if output_dir is not None
                  else checkpoint.parent / ("final_evaluation" if final else "test_evaluation") / checkpoint.stem)
    if output_dir.resolve() == checkpoint.parent:
        raise ValueError("Use a separate evaluation output directory to preserve training reports.")
    with experiment_lock(output_dir):
        xyz, regions, demographics, labels, audit = load_cohort(
            data_dir, test, PreprocessConfig(**saved["preprocess_config"]), **loader_options)
        if len(np.unique(labels)) != 2:
            raise ValueError("Both MI classes must remain after test quality filtering to report ROC-AUC.")
        probability, _ = predict_prepared(model, scaler, xyz, regions, demographics, batch_size=batch_size)
        for record in audit:
            record["sample_index"] = record["patient_index"]
            record["patient_index"] = int(patient_ids[record["sample_index"]])
        patients = [a["patient_index"] for a in audit if not a["excluded"]]
        samples = [a["sample_index"] for a in audit if not a["excluded"]]
        excluded = [a["patient_index"] for a in audit if a["excluded"]]
        report = mi_classification_report(labels, probability)
        report.update(evaluation="final_test_samples" if final else "held_out_test", test_set_scored=True,
                      final_test_set_scored=final,
                      experiment_name=identity["experiment_name"], checkpoint=str(checkpoint),
                      checkpoint_epoch=saved["metadata"]["epoch"],
                      selection_metric="validation_macro_f1",
                      selected_validation_macro_f1=saved["metadata"]["validation_macro_f1"],
                      device=str(next(model.parameters()).device),
                      model_config=saved["model_config"], preprocess_config=saved["preprocess_config"],
                      split_source=split_source, n_test_patients=len(test),
                      n_quality_excluded=len(excluded), test_patient_ids=patient_ids[test].tolist(),
                      scored_patient_ids=patients, quality_excluded_patient_ids=excluded)
        pd.DataFrame({"patient_index": patients, "sample_index": samples,
                      "true_MI": np.where(labels, "pMI", "healthy"),
                      "predicted_MI": np.where(probability >= .5, "pMI", "healthy"),
                      "probability_pMI": probability}).to_csv(output_dir / "mi_predictions.csv", index=False)
        atomic_json_save(audit, output_dir / "frame_quality.json")
        atomic_json_save(report, output_dir / "mi_classification_report.json")
    return report, output_dir
