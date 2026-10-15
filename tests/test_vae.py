"""VAE geometry, leakage, gradients and checkpoint regression checks."""

from pathlib import Path
from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import torch

from mcpp.vae import CardiacVAE, VAEConfig, vae_loss
from mcpp.vae_data import (DEMOGRAPHIC_COLUMNS, PreprocessConfig, UnrepairableCycleError,
                           load_cohort, prepare_arrays, prepare_patient, repair_frames)
from mcpp.vae_training import (FoldScaler, TrainConfig, cross_validate_vae, fit_vae,
                                load_checkpoint, predict_patients, predict_prepared, save_checkpoint,
                                mi_classification_report, select_device)


def cycle(points=24, seed=7):
    rng = np.random.default_rng(seed)
    template = rng.normal(size=(points, 3)) * 20
    phase = np.arange(10) * 2 * np.pi / 10
    data = np.zeros((10, points, 4))
    data[..., :3] = template[None] * (1 + .15 * np.cos(phase))[:, None, None]
    data[..., 3] = np.repeat(np.arange(3), points // 3)
    return data


def surface_cycle():
    """Ordered surface-like points, with the project's anatomical region order."""
    theta = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    template = np.column_stack((20 * np.cos(theta), 15 * np.sin(theta), 5 * np.cos(2 * theta)))
    template = np.concatenate([template + [region * 15, 0, 0] for region in range(3)])
    data = np.zeros((10, 600, 4))
    data[..., :3] = template[None] * (1 + .15 * np.cos(np.arange(10) * 2 * np.pi / 10))[:, None, None]
    data[..., 3] = np.repeat(np.arange(3), 200)
    return data


class FrameQualityTests(unittest.TestCase):
    def test_normal_cycle_unchanged_and_source_not_mutated(self):
        data = cycle()
        saved = data.copy()
        repaired, audit = repair_frames(data)
        np.testing.assert_allclose(repaired, data, rtol=1e-6)
        np.testing.assert_array_equal(data, saved)
        self.assertEqual(audit["flagged_frames"], [])

    def test_expansion_and_spike_replaced_with_neighbour_mean(self):
        for local_spike in (False, True):
            data = cycle()
            if local_spike:
                data[4, 0, :3] += 10000
            else:
                data[4, :, :3] *= 50
            before = data.copy()
            repaired, audit = repair_frames(data)
            self.assertEqual(audit["flagged_frames"], [4])
            np.testing.assert_allclose(repaired[4, :, :3], (data[3, :, :3] + data[5, :, :3]) / 2, rtol=1e-6)
            np.testing.assert_array_equal(repaired[..., 3], data[..., 3])
            np.testing.assert_array_equal(data, before)

    def test_translation_nonfinite_runs_and_endpoints(self):
        data = cycle()
        data[3, :, :3] += 10000
        data[4, 0, 0] = np.nan
        data[0, :, :3] *= 50
        repaired, audit = repair_frames(data)
        self.assertEqual(audit["flagged_frames"], [0, 3, 4])
        np.testing.assert_allclose(repaired[0, :, :3], data[1, :, :3], rtol=1e-6)
        np.testing.assert_allclose(repaired[3, :, :3], 2 / 3 * data[2, :, :3] + 1 / 3 * data[5, :, :3], rtol=1e-6)
        np.testing.assert_allclose(repaired[4, :, :3], 1 / 3 * data[2, :, :3] + 2 / 3 * data[5, :, :3], rtol=1e-6)
        self.assertTrue(np.isfinite(repaired).all())

    def test_cyclic_boundary_uses_last_and_second_frame(self):
        data = cycle()
        data[0, :, :3] *= 100
        repaired, audit = repair_frames(data, cyclic=True)
        self.assertEqual(audit["flagged_frames"], [0])
        np.testing.assert_allclose(repaired[0, :, :3], .5 * (data[9, :, :3] + data[1, :, :3]), rtol=1e-6)

    def test_temporal_scatter_with_nearly_normal_global_size(self):
        for frame in (0, 1, 4, 8, 9):
            data = cycle(points=600)
            data[frame, :, :3] += np.random.default_rng(91).normal(0, 14, (600, 3))
            repaired, audit = repair_frames(data)
            self.assertEqual(audit["flagged_frames"], [frame])
            self.assertEqual(audit["temporal_spike_frames"], [frame])
            if frame in (0, 9):
                neighbour = 1 if frame == 0 else 8
                np.testing.assert_allclose(repaired[frame, :, :3], data[neighbour, :, :3], rtol=1e-6)
            else:
                np.testing.assert_allclose(repaired[frame, :, :3],
                                           .5 * (data[frame-1, :, :3] + data[frame+1, :, :3]), rtol=1e-6)

    def test_spatial_scatter_repairs_consecutive_frames(self):
        data = surface_cycle()
        data[2:5, :, :3] += np.random.default_rng(4).normal(0, 14, (3, 600, 3))
        original = data.copy()
        repaired, audit = repair_frames(data)
        self.assertEqual(audit["flagged_frames"], [2, 3, 4])
        self.assertEqual(audit["spatial_scatter_frames"], [2, 3, 4])
        self.assertIn("KMeans", audit["spatial_detection_method"])
        self.assertGreaterEqual(audit["spatial_cluster_ratio"], 3.)
        self.assertEqual([i for i, label in enumerate(audit["spatial_cluster_labels"]) if label == 1], [2, 3, 4])
        for frame in (2, 3, 4):
            weight = (frame - 1) / 4
            np.testing.assert_allclose(repaired[frame, :, :3],
                                       (1 - weight) * data[1, :, :3] + weight * data[5, :, :3], atol=4e-6)
        np.testing.assert_array_equal(data, original)
        np.testing.assert_array_equal(repaired[..., 3], data[..., 3])

    def test_spatial_coherence_preserves_a_smooth_contracting_surface(self):
        data = surface_cycle()
        data[..., 0] *= (1 + .04 * np.sin(np.arange(10)))[:, None]
        repaired, audit = repair_frames(data)
        self.assertEqual(audit["flagged_frames"], [])
        self.assertEqual(len(audit["spatial_cluster_centres"]), 2)
        self.assertLess(audit["spatial_cluster_ratio"], 3.)
        np.testing.assert_allclose(repaired, data, atol=4e-6)

    def test_majority_corruption_retains_frame_audit_for_review(self):
        data = surface_cycle()
        affected = [i for i in range(10) if i != 3]
        data[affected, :, :3] += np.random.default_rng(4).normal(0, 14, (9, 600, 3))
        with self.assertRaises(UnrepairableCycleError) as error:
            repair_frames(data)
        self.assertEqual(error.exception.audit["flagged_frames"], affected)
        self.assertTrue(error.exception.audit["excluded"])
        self.assertEqual(error.exception.audit["replacements"], [])

    def test_cohort_exclusion_preserves_patient_labels_and_demographics(self):
        arrays = [surface_cycle() for _ in range(3)]
        arrays[1][:9, :, :3] += np.random.default_rng(4).normal(0, 14, (9, 600, 3))
        labels = ["pMI", "healthy", "healthy"]
        metadata = pd.DataFrame({name: [10., 20., 30.] for name in DEMOGRAPHIC_COLUMNS})
        metadata["sex"] = [False, True, False]
        metadata["MI"] = labels
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            (root / "mixed_samples").mkdir()
            metadata.to_csv(root / "mixed_demographics.csv", index=False)
            for index, (array, label) in enumerate(zip(arrays, labels)):
                np.save(root / "mixed_samples" / f"{index}{label}.npy", array)
            xyz, regions, demographics, mi, audit = load_cohort(
                root, [2, 1, 0], PreprocessConfig(n_points=12))
            with self.assertRaises(UnrepairableCycleError):
                load_cohort(root, [2, 1, 0], PreprocessConfig(n_points=12), exclude_unrepairable=False)
        self.assertEqual(xyz.shape, (2, 10, 12, 3))
        self.assertEqual(regions.shape, (2, 12))
        np.testing.assert_array_equal(mi, [0, 1])
        np.testing.assert_array_equal(demographics[:, 0], [30, 10])
        self.assertEqual([a["patient_index"] for a in audit if not a["excluded"]], [2, 0])
        self.assertEqual([a["patient_index"] for a in audit if a["excluded"]], [1])

    def test_ambiguous_and_unaligned_data_rejected(self):
        data = cycle()
        data[:6, :, :3] = np.nan
        with self.assertRaisesRegex(ValueError, "half"):
            repair_frames(data)
        data = cycle()
        data[2, 0, 3] = 2
        with self.assertRaisesRegex(ValueError, "correspondence"):
            repair_frames(data)

    def test_sampling_is_balanced_deterministic_and_keeps_motion(self):
        config = PreprocessConfig(n_points=12)
        xyz, regions, audit = prepare_patient(cycle(), config)
        again, _, _ = prepare_patient(cycle(), config)
        np.testing.assert_array_equal(again, xyz)
        np.testing.assert_array_equal(np.bincount(regions), [4, 4, 4])
        chosen = np.asarray(audit["selected_indices"])
        np.testing.assert_allclose(xyz[2] - xyz[1], cycle()[2, chosen, :3] - cycle()[1, chosen, :3], atol=1e-5)
        self.assertGreater(np.std(xyz, axis=0).mean(), 0)


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        self.config = VAEConfig(n_points=24, width=32, heads=4, spatial_tokens=4,
                                feedforward=64, dropout=0., encoder_depth=1)
        self.model = CardiacVAE(self.config)
        xyz, regions, _ = prepare_arrays([cycle(), cycle(seed=8)], PreprocessConfig(n_points=24))
        self.xyz = torch.from_numpy(xyz / 20)
        self.regions = torch.from_numpy(regions)
        self.demographics = torch.randn(2, 7)

    def test_default_parameter_budget(self):
        model = CardiacVAE()
        self.assertTrue(9_000_000 <= model.parameter_count() <= 11_000_000, model.parameter_count())

    def test_demographics_only_affect_head_and_eval_is_deterministic(self):
        self.model.eval()
        a = self.model(self.xyz, self.regions, self.demographics)
        b = self.model(self.xyz, self.regions, self.demographics + 10)
        c = self.model(self.xyz, self.regions, self.demographics)
        self.assertEqual(a["mu"].shape, (2, 20))
        self.assertEqual(a["reconstruction"].shape, self.xyz.shape)
        for key in ("mu", "logvar", "reconstruction"):
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
            torch.testing.assert_close(a[key], c[key], rtol=0, atol=0)
        self.assertFalse(torch.allclose(a["logits"], b["logits"]))

    def test_reparameterization_loss_and_gradients(self):
        self.model.train()
        a = self.model(self.xyz, self.regions, self.demographics)
        b = self.model(self.xyz, self.regions, self.demographics)
        self.assertFalse(torch.allclose(a["reconstruction"], b["reconstruction"]))
        loss = vae_loss(a, self.xyz, torch.tensor([0, 1]))
        self.assertTrue(all(torch.isfinite(v) for v in loss.values()))
        loss["loss"].backward()
        for part in (self.model.mu, self.model.logvar, self.model.classifier,
                     self.model.decoder, self.model.encoder, self.model.pool_attention):
            gradients = [p.grad for p in part.parameters()]
            self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients))
            self.assertGreater(sum(float(g.abs().sum()) for g in gradients), 0)

    def test_kl_zero_for_unit_gaussian(self):
        output = {"reconstruction": self.xyz, "mu": torch.zeros(2, 20),
                  "logvar": torch.zeros(2, 20), "logits": torch.zeros(2)}
        losses = vae_loss(output, self.xyz, torch.tensor([0, 1]))
        self.assertEqual(losses["reconstruction"].item(), 0)
        self.assertEqual(losses["kl"].item(), 0)

    def test_half_precision_outputs_have_finite_float32_loss_and_gradients(self):
        output = {"reconstruction": self.xyz.half().requires_grad_(),
                  "mu": torch.full((2, 20), 300., dtype=torch.float16, requires_grad=True),
                  "logvar": torch.full((2, 20), 8., dtype=torch.float16, requires_grad=True),
                  "logits": torch.tensor([500., -500.], dtype=torch.float16, requires_grad=True)}
        losses = vae_loss(output, self.xyz, torch.tensor([0, 1]))
        self.assertEqual(losses["loss"].dtype, torch.float32)
        self.assertTrue(all(torch.isfinite(value) for value in losses.values()))
        expected_kl = -.5 * (1 + 8 - 300 ** 2 - np.exp(8))
        self.assertAlmostEqual(float(losses["kl"].detach()), expected_kl, delta=.01)
        losses["loss"].backward()
        for value in output.values():
            self.assertIsNotNone(value.grad)
            self.assertTrue(torch.isfinite(value.grad).all())


class DeviceSelectionTests(unittest.TestCase):
    def test_auto_prefers_cuda_and_accepts_an_explicit_gpu(self):
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.current_device", return_value=0), \
             patch("torch.cuda.device_count", return_value=2):
            self.assertEqual(str(select_device()), "cuda:0")
            self.assertEqual(str(select_device("cuda:1")), "cuda:1")
            with self.assertRaisesRegex(ValueError, "does not exist"):
                select_device("cuda:2")

    def test_unavailable_cuda_has_an_actionable_error(self):
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CUDA-enabled PyTorch"):
                select_device("cuda")
            with patch("torch.backends.mps.is_available", return_value=True):
                self.assertEqual(str(select_device()), "mps")
            with patch("torch.backends.mps.is_available", return_value=False):
                self.assertEqual(str(select_device()), "cpu")
                with self.assertRaisesRegex(RuntimeError, "Apple GPU"):
                    select_device("mps")

    def test_explicit_cpu_and_unsupported_devices(self):
        self.assertEqual(str(select_device("cpu")), "cpu")
        with self.assertRaisesRegex(ValueError, "Supported training devices"):
            select_device("meta")
        with self.assertRaisesRegex(ValueError, "Use auto"):
            select_device("invalid")


class MIReportTests(unittest.TestCase):
    def test_report_scores_patient_mi_with_explicit_positive_class(self):
        report = mi_classification_report([0, 0, 0, 0, 1, 1], [.1, .9, .2, .8, .7, .3])
        self.assertEqual(report["n_patients"], 6)
        self.assertEqual(report["metrics"]["accuracy"], .5)
        self.assertEqual(report["metrics"]["f1_pMI"], .4)
        self.assertEqual(report["confusion_matrix"], [[2, 2], [1, 1]])
        self.assertEqual(report["per_class"]["healthy"]["support"], 4)
        self.assertEqual(report["per_class"]["pMI"]["support"], 2)
        self.assertEqual(json.loads(json.dumps(report)), report)

    def test_invalid_probabilities_cannot_be_reported_as_results(self):
        for probability in ([.1, np.nan], [.1, 1.2], [.1]):
            with self.assertRaisesRegex(ValueError, "one binary MI label"):
                mi_classification_report([0, 1], probability)


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.raw = [cycle(seed=i) for i in range(40)]
        self.preprocess = PreprocessConfig(n_points=12)
        self.xyz, self.regions, _ = prepare_arrays(self.raw, self.preprocess)
        self.demographics = np.random.default_rng(42).normal(size=(40, 7)).astype(np.float32)
        self.labels = np.arange(40, dtype=np.int64) % 2
        self.model_config = VAEConfig(n_points=12, width=16, heads=2, spatial_tokens=3,
                                      feedforward=32, dropout=0., encoder_depth=1)
        self.training = TrainConfig(epochs=1, batch_size=8, device="cpu", cpu_threads=2)

    def test_training_scaler_ignores_validation_and_roundtrip_predicts_raw(self):
        train, validation = np.arange(30), np.arange(30, 40)
        self.demographics[validation] += 100
        model, scaler, report = fit_vae(
            self.xyz, self.regions, self.demographics, self.labels, train, validation,
            model_config=self.model_config, train_config=self.training, verbose=False,
        )
        np.testing.assert_allclose(scaler.demographic_mean, self.demographics[train].mean(0))
        self.assertEqual(report["best_epoch"], 1)
        self.assertFalse(report["amp_enabled"])  # amp=True is CUDA-only.
        expected, expected_latent = predict_prepared(model, scaler, self.xyz[:2], self.regions[:2], self.demographics[:2])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            save_checkpoint(path, model, scaler, preprocess_config=self.preprocess)
            _, loaded_scale, saved = load_checkpoint(path, device="cpu")
            actual, latent, audit = predict_patients(path, self.raw[:2], self.demographics[:2], device="cpu")
        np.testing.assert_allclose(actual, expected)
        np.testing.assert_allclose(latent, expected_latent)
        self.assertEqual(len(saved["demographic_columns"]), 7)
        self.assertEqual(loaded_scale.xyz_scale, scaler.xyz_scale)
        self.assertEqual(len(audit), 2)

    def test_cross_validation_uses_disjoint_patients_and_fresh_scalers(self):
        with patch.object(FoldScaler, "fit", wraps=FoldScaler.fit) as fit:
            result = cross_validate_vae(
                self.xyz, self.regions, self.demographics, self.labels, n_splits=2,
                model_config=self.model_config, train_config=self.training,
                preprocess_config=self.preprocess, patient_ids=np.arange(40) + 100, verbose=False,
            )
        self.assertEqual(fit.call_count, 2)
        scored = []
        for fold, call in zip(result["folds"], fit.call_args_list):
            a, b, c = (set(fold[key]) for key in
                       ("fit_patient_ids", "early_stop_patient_ids", "score_patient_ids"))
            self.assertFalse(a & b or a & c or b & c)
            self.assertEqual(a | b | c, set(range(100, 140)))
            np.testing.assert_allclose(call.args[1], self.demographics[np.asarray(fold["fit_patient_ids"]) - 100])
            scored.extend(c)
        self.assertEqual(sorted(scored), list(range(100, 140)))
        self.assertTrue(np.isfinite(result["oof_probabilities"]).all())
        self.assertEqual(result["oof_latent_means"].shape, (40, 20))

    def test_epoch_metrics_pool_patient_predictions_and_match_validation(self):
        batches = []
        def capture_loss(output, xyz, labels, **kwargs):
            batches.append((labels.detach().cpu().numpy(), (output["logits"].detach() >= 0).cpu().numpy()))
            return vae_loss(output, xyz, labels, **kwargs)
        stdout = io.StringIO()
        with patch("mcpp.vae_training.vae_loss", side_effect=capture_loss), redirect_stdout(stdout):
            model, scaler, report = fit_vae(
                self.xyz, self.regions, self.demographics, self.labels, np.arange(30), np.arange(30, 40),
                model_config=self.model_config, train_config=replace(self.training, epochs=2),
            )
        self.assertEqual(len(report["history"]), 2)
        for epoch, record in enumerate(report["history"]):
            labels = np.concatenate([b[0] for b in batches[epoch * 4:(epoch + 1) * 4]])
            prediction = np.concatenate([b[1] for b in batches[epoch * 4:(epoch + 1) * 4]])
            self.assertEqual(len(labels), 30)  # Includes the final partial minibatch.
            self.assertAlmostEqual(record["train_accuracy"], accuracy_score(labels, prediction))
            self.assertAlmostEqual(record["train_f1_pMI"], f1_score(labels, prediction, zero_division=0))
            self.assertAlmostEqual(record["train_macro_f1"], f1_score(labels, prediction, average="macro", zero_division=0))
            for split in ("train", "validation"):
                for metric in ("accuracy", "f1_pMI", "macro_f1"):
                    self.assertTrue(0 <= record[f"{split}_{metric}"] <= 1)
        probability, _ = predict_prepared(
            model, scaler, self.xyz[30:], self.regions[30:], self.demographics[30:],
        )
        best = report["history"][report["best_epoch"] - 1]
        self.assertAlmostEqual(best["validation_accuracy"], accuracy_score(self.labels[30:], probability >= .5))
        self.assertAlmostEqual(best["validation_f1_pMI"], f1_score(self.labels[30:], probability >= .5, zero_division=0))
        self.assertAlmostEqual(best["validation_macro_f1"], f1_score(self.labels[30:], probability >= .5, average="macro", zero_division=0))
        self.assertEqual(stdout.getvalue().count("train acc="), 2)
        self.assertEqual(stdout.getvalue().count("val acc="), 2)
        self.assertEqual(stdout.getvalue().count("F1(pMI)="), 4)
        self.assertEqual(json.loads(json.dumps(report))["history"], report["history"])

    def test_overlapping_patients_rejected(self):
        with self.assertRaisesRegex(ValueError, "overlap"):
            fit_vae(self.xyz, self.regions, self.demographics, self.labels, np.arange(30), np.arange(20, 40),
                    model_config=self.model_config, train_config=self.training, verbose=False)

    def test_command_defaults_to_one_epoch_and_writes_overall_mi_scores(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            data_dir, output = directory / "cohort", directory / "experiment"
            (data_dir / "mixed_samples").mkdir(parents=True)
            metadata = pd.DataFrame(self.demographics, columns=DEMOGRAPHIC_COLUMNS)
            metadata["sex"] = self.labels.astype(bool)
            metadata["MI"] = np.where(self.labels, "pMI", "healthy")
            metadata.to_csv(data_dir / "mixed_demographics.csv", index=False)
            for patient, label in enumerate(metadata.MI):
                np.save(data_dir / "mixed_samples" / f"{patient}{label}.npy", cycle(points=48, seed=patient))
            completed = subprocess.run(
                [sys.executable, str(root / "results/train_vae.py"), "--data-dir", str(data_dir),
                 "--output-dir", str(output), "--folds", "2", "--points", "48", "--device", "cpu"],
                capture_output=True, text=True, timeout=60,
                env={**os.environ, "LOKY_MAX_CPU_COUNT": "2", "OMP_NUM_THREADS": "2"},
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            cv = json.loads((output / "cv_results.json").read_text())
            report = json.loads((output / "mi_classification_report.json").read_text())
            predictions = pd.read_csv(output / "mi_predictions.csv")
        self.assertEqual(cv["train_config"]["epochs"], 1)
        self.assertTrue(all(len(f["history"]) == 1 for f in cv["folds"]))
        self.assertEqual(report["n_patients"], 32)
        self.assertEqual(report["epochs_per_fold"], 1)
        self.assertFalse(report["test_set_scored"])
        self.assertTrue(set(predictions.patient_index).isdisjoint(cv["test_patient_ids_excluded"]))
        np.testing.assert_array_equal(predictions.true_MI, metadata.MI.iloc[predictions.patient_index])
        expected_accuracy = accuracy_score(predictions.true_MI, predictions.predicted_MI)
        expected_f1 = f1_score(predictions.true_MI, predictions.predicted_MI, pos_label="pMI", zero_division=0)
        self.assertAlmostEqual(report["metrics"]["accuracy"], expected_accuracy)
        self.assertAlmostEqual(report["metrics"]["f1_pMI"], expected_f1)
        self.assertIn("Overall MI classification:", completed.stdout)
        self.assertIn("F1 (pMI):", completed.stdout)
        self.assertIn("Accuracy:", completed.stdout)

    @unittest.skipUnless(torch.cuda.is_available(), "Requires an NVIDIA CUDA GPU")
    def test_cuda_training_with_amp_and_float32_and_portable_checkpoint(self):
        for amp in (True, False):
            with self.subTest(amp=amp):
                model, scaler, report = fit_vae(
                    self.xyz, self.regions, self.demographics, self.labels, np.arange(30), np.arange(30, 40),
                    model_config=self.model_config,
                    train_config=replace(self.training, device="cuda:0", amp=amp), verbose=False,
                )
                self.assertEqual(next(model.parameters()).device.type, "cuda")
                self.assertEqual(report["amp_enabled"], amp)
                for record in report["history"]:
                    self.assertTrue(all(np.isfinite(value) for value in record.values()))
                probability, latent = predict_prepared(
                    model, scaler, self.xyz[:2], self.regions[:2], self.demographics[:2])
                self.assertTrue(np.isfinite(probability).all())
                self.assertTrue(np.isfinite(latent).all())
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "cuda.pt"
                    save_checkpoint(path, model, scaler, preprocess_config=self.preprocess, metadata=report)
                    cpu_model, cpu_scaler, saved = load_checkpoint(path, device="cpu")
                    cpu_probability, cpu_latent = predict_prepared(
                        cpu_model, cpu_scaler, self.xyz[:2], self.regions[:2], self.demographics[:2])
                self.assertEqual(saved["metadata"]["amp_enabled"], amp)
                np.testing.assert_allclose(cpu_probability, probability, atol=1e-4, rtol=1e-4)
                np.testing.assert_allclose(cpu_latent, latent, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
