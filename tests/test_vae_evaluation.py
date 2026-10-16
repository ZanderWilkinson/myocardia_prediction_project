"""Frozen-checkpoint evaluation, split isolation and external dataset regression checks."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from mcpp.vae import VAEConfig
from mcpp.vae_data import DEMOGRAPHIC_COLUMNS, PreprocessConfig, prepare_arrays
from mcpp.vae_evaluation import evaluate_test_checkpoint
from mcpp.vae_training import FoldScaler, TrainConfig, train_holdout_vae


def cycle():
    theta = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    template = np.column_stack((20 * np.cos(theta), 15 * np.sin(theta), 5 * np.cos(2 * theta)))
    template = np.concatenate([template + [region * 15, 0, 0] for region in range(3)])
    data = np.zeros((10, 600, 4))
    data[..., :3] = template[None] * (1 + .15 * np.cos(np.arange(10) * 2 * np.pi / 10))[:, None, None]
    data[..., 3] = np.repeat(np.arange(3), 200)
    return data


class TestEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name).resolve()
        self.data = self.directory / "training_data"
        self.experiment = self.directory / "experiment"
        (self.data / "mixed_samples").mkdir(parents=True)
        demographic = np.random.default_rng(42).normal(size=(40, 7)).astype(np.float32)
        self.labels = np.arange(40) % 2
        self.metadata = pd.DataFrame(demographic, columns=DEMOGRAPHIC_COLUMNS)
        self.metadata["sex"] = self.labels.astype(bool)
        self.metadata["MI"] = np.where(self.labels, "pMI", "healthy")
        self.metadata.to_csv(self.data / "mixed_demographics.csv", index=False)
        self.development, self.test = train_test_split(
            np.arange(40), test_size=.2, stratify=self.labels, random_state=42)
        self.preprocess = PreprocessConfig(n_points=12)
        xyz, regions, _ = prepare_arrays([cycle()] * len(self.development), self.preprocess)
        demographic[:, -1] = self.labels
        train_holdout_vae(
            xyz, regions, demographic[self.development], self.labels[self.development],
            model_config=VAEConfig(n_points=12, width=16, heads=2, spatial_tokens=3,
                                   feedforward=32, dropout=0., encoder_depth=1),
            train_config=TrainConfig(epochs=1, batch_size=8, device="cpu", cpu_threads=2),
            preprocess_config=self.preprocess, patient_ids=self.development,
            output_dir=self.experiment, experiment_name="test-evaluation", verbose=False)
        self.checkpoint = self.experiment / "val_best.pt"
        # There are deliberately no development point files on disk: evaluation
        # must never try to open them or refit their scaler.
        for patient in self.test:
            np.save(self.data / "mixed_samples" / f"{patient}{self.metadata.MI.iloc[patient]}.npy", cycle())

    def test_held_out_scores_match_predictions_without_fitting_or_changing_training(self):
        before = {p.name: p.read_bytes() for p in self.experiment.iterdir() if p.is_file()}
        with patch.object(FoldScaler, "fit", side_effect=AssertionError("must not fit on test")):
            report, output = evaluate_test_checkpoint(self.checkpoint, self.data, device="cpu", cpu_threads=2)
        after = {name: (self.experiment / name).read_bytes() for name in before}
        self.assertEqual(before, after)
        self.assertEqual(report["evaluation"], "held_out_test")
        self.assertEqual(set(report["scored_patient_ids"]), set(self.test))
        self.assertFalse(set(report["scored_patient_ids"]) & set(self.development))
        self.assertEqual(report["n_patients"], 8)
        self.assertFalse(report["final_test_set_scored"])
        predictions = pd.read_csv(output / "mi_predictions.csv")
        self.assertAlmostEqual(report["metrics"]["accuracy"],
                               accuracy_score(predictions.true_MI, predictions.predicted_MI))
        self.assertAlmostEqual(report["metrics"]["f1_pMI"],
                               f1_score(predictions.true_MI, predictions.predicted_MI, pos_label="pMI", zero_division=0))
        self.assertAlmostEqual(report["metrics"]["macro_f1"],
                               f1_score(predictions.true_MI, predictions.predicted_MI, average="macro", zero_division=0))
        self.assertEqual(json.loads((output / "mi_classification_report.json").read_text()), report)

    def test_quality_exclusions_keep_patient_alignment_and_coverage(self):
        bad = int(self.test[0])
        broken = cycle()
        broken[:9, :, :3] += np.random.default_rng(7).normal(0, 14, (9, 600, 3))
        np.save(self.data / "mixed_samples" / f"{bad}{self.metadata.MI.iloc[bad]}.npy", broken)
        report, output = evaluate_test_checkpoint(self.checkpoint, self.data, device="cpu", cpu_threads=2)
        self.assertEqual((report["n_test_patients"], report["n_patients"], report["n_quality_excluded"]), (8, 7, 1))
        self.assertEqual(report["quality_excluded_patient_ids"], [bad])
        predictions = pd.read_csv(output / "mi_predictions.csv")
        self.assertEqual(set(predictions.patient_index), set(self.test) - {bad})
        np.testing.assert_array_equal(predictions.true_MI, self.metadata.MI.iloc[predictions.patient_index])
        audit = json.loads((output / "frame_quality.json").read_text())
        self.assertEqual(len(audit), 8)
        self.assertTrue(next(a for a in audit if a["patient_index"] == bad)["excluded"])

    def test_completed_run_uses_recorded_membership_and_rejects_wrong_cohort(self):
        result_path = self.experiment / "holdout_results.json"
        result_path.write_text(json.dumps({"test_patient_ids_excluded": self.test[::-1].tolist()}))
        report, _ = evaluate_test_checkpoint(self.checkpoint, self.data, device="cpu", cpu_threads=2)
        self.assertEqual(report["test_patient_ids"], self.test[::-1].tolist())
        self.assertEqual(report["split_source"], str(result_path))
        result_path.write_text(json.dumps({"test_patient_ids_excluded": self.development[:8].tolist()}))
        with self.assertRaisesRegex(ValueError, "Recorded test membership differs"):
            evaluate_test_checkpoint(self.checkpoint, self.data, device="cpu", cpu_threads=2)

    def test_wrong_checkpoint_and_development_overlap_are_rejected(self):
        path = self.experiment / "experiment.json"
        manifest = json.loads(path.read_text())
        original_signature = manifest["identity"]["input_signature"]
        manifest["identity"]["input_signature"] = "wrong experiment"
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "checkpoint belonging to this experiment"):
            evaluate_test_checkpoint(self.checkpoint, self.data, device="cpu", cpu_threads=2)
        manifest["identity"]["input_signature"] = original_signature
        manifest["identity"]["patient_ids"] = self.test.tolist()
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "overlap the test split"):
            evaluate_test_checkpoint(self.checkpoint, self.data, device="cpu", cpu_threads=2)

    def _external_dataset(self):
        data = self.directory / "test_samples"
        (data / "test_samples").mkdir(parents=True)
        metadata = pd.concat([self.metadata] * 3, ignore_index=True).iloc[:100].copy()
        metadata.insert(0, "Unnamed: 0", np.arange(900, 1000))
        metadata.to_csv(data / "test.csv", index=False)
        for sample in range(100):
            np.save(data / "test_samples" / f"{sample}.npy", cycle())
        return data, metadata

    def test_final_evaluates_all_100_unlabelled_filenames_and_preserves_patient_ids(self):
        data, metadata = self._external_dataset()
        with patch.object(FoldScaler, "fit", side_effect=AssertionError("must not fit on final test")):
            report, output = evaluate_test_checkpoint(self.checkpoint, data, device="cpu", cpu_threads=2, final=True)
        self.assertEqual(report["evaluation"], "final_test_samples")
        self.assertTrue(report["final_test_set_scored"])
        self.assertEqual((report["n_test_patients"], report["n_patients"]), (100, 100))
        self.assertEqual(report["scored_patient_ids"], list(range(900, 1000)))
        self.assertEqual(output, self.experiment / "final_evaluation" / "val_best")
        predictions = pd.read_csv(output / "mi_predictions.csv")
        self.assertEqual(predictions.sample_index.tolist(), list(range(100)))
        np.testing.assert_array_equal(predictions.true_MI, metadata.MI)
        self.assertFalse((self.experiment / "test_evaluation").exists())

    def test_command_supports_held_out_and_final_and_saves_separate_reports(self):
        data, _ = self._external_dataset()
        script = Path(__file__).resolve().parents[1] / "results/evaluate_vae.py"
        command = [sys.executable, str(script), "--checkpoint", str(self.checkpoint),
                   "--device", "cpu", "--cpu-threads", "2"]
        env = {**os.environ, "LOKY_MAX_CPU_COUNT": "2", "OMP_NUM_THREADS": "2"}
        for extra, headline, directory in (
            (["--data-dir", str(self.data)], "Held-out test", "test_evaluation"),
            (["--final", "--data-dir", str(data)], "Full test_samples", "final_evaluation"),
        ):
            result = subprocess.run(command + extra, capture_output=True, text=True, timeout=60, env=env)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn(headline + " MI classification:", result.stdout)
            for label in ("F1 (pMI):", "Accuracy:", "Macro F1:", "ROC-AUC:"):
                self.assertIn(label, result.stdout)
            self.assertTrue((self.experiment / directory / "val_best" / "mi_classification_report.json").exists())

    def test_training_reports_cannot_be_overwritten(self):
        with self.assertRaisesRegex(ValueError, "preserve training reports"):
            evaluate_test_checkpoint(self.checkpoint, self.data, output_dir=self.experiment, device="cpu", cpu_threads=2)


if __name__ == "__main__":
    unittest.main()
