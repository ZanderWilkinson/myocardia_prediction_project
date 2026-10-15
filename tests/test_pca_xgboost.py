"""Patient-feature and reuse checks, runnable with unittest discovery."""

import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from mcpp.pca import pca_datasubset
from mcpp.xgboost import prepare_xgboost_data, train_xgboost, evaluate_xgboost


def make_subset(n, seed=42):
    rng = np.random.default_rng(seed)
    labels = np.arange(n) % 2
    clouds = rng.normal(size=(n, 3, 6, 4))
    clouds[..., :3] += labels[:, None, None, None] * 2
    clouds[..., 3] = np.repeat([0, 1, 2], 2)
    return SimpleNamespace(
        patients=[SimpleNamespace(data=cloud) for cloud in clouds],
        data=pd.DataFrame({
            "Unnamed: 0": np.arange(n),
            "MI": np.where(labels, "pMI", "healthy"),
            "age": rng.normal(60, 5, n),
            "sex": rng.integers(0, 2, n).astype(bool),
        }),
    )


class PCAFeatureTests(unittest.TestCase):
    def test_default_keeps_legacy_combined_features(self):
        subset = make_subset(12)
        scores, model = pca_datasubset(subset, n_components=3)
        self.assertEqual(scores.shape, (12, 3))
        self.assertFalse(model.motion_only_)
        self.assertEqual(model.n_features_in_, 3 * 6 * 4 + 2)
        np.testing.assert_allclose(model[0].mean_[:2], subset.data[["age", "sex"]].mean())

    def test_motion_only_ignores_metadata_and_anatomical_labels_but_keeps_position(self):
        subset = make_subset(12)
        original = subset.patients[0].data.copy()
        scores, model = pca_datasubset(subset, n_components=3, motion_only=True)
        altered = copy.deepcopy(subset)
        altered.data["age"] += 500
        altered.data["sex"] = ~altered.data["sex"]
        for patient in altered.patients:
            patient.data[..., 3] += 123
        transformed, reused = pca_datasubset(altered, model=model)
        self.assertIs(reused, model)
        self.assertTrue(model.motion_only_)
        self.assertEqual(model.n_features_in_, 3 * 6 * 3)
        np.testing.assert_allclose(transformed, scores, atol=1e-5)
        np.testing.assert_array_equal(subset.patients[0].data, original)

        # Constant offsets across all frames would vanish under displacement;
        # this flag must retain absolute heart coordinates instead.
        for patient in altered.patients:
            patient.data[..., :3] += 10
        translated, _ = pca_datasubset(altered, model=model)
        self.assertGreater(np.linalg.norm(translated - transformed), 1)

    def test_transform_preserves_training_fit_and_rejects_mode_change(self):
        _, model = pca_datasubset(make_subset(12), n_components=3, motion_only=True)
        mean = model[0].mean_.copy()
        components = model[-1].components_.copy()
        validation = make_subset(6, seed=19)
        with patch.object(model, "fit_transform", side_effect=AssertionError("PCA refitted")):
            scores, _ = pca_datasubset(validation, model=model)
        self.assertEqual(scores.shape, (6, 3))
        np.testing.assert_array_equal(model[0].mean_, mean)
        np.testing.assert_array_equal(model[-1].components_, components)
        with self.assertRaisesRegex(ValueError, "match"):
            pca_datasubset(validation, model=model, motion_only=False)


class XGBoostReuseTests(unittest.TestCase):
    def setUp(self):
        self.dataset = SimpleNamespace(
            train=make_subset(40), val=make_subset(12, seed=3), test=make_subset(12, seed=4),
        )

    def test_preparation_and_multiple_classifier_fits_reuse_pca(self):
        # Validation columns may arrive in another order; training schema wins.
        self.dataset.val.data = self.dataset.val.data.iloc[:, ::-1]
        with patch("mcpp.xgboost.pca_datasubset", wraps=pca_datasubset) as pca_calls:
            prepared = prepare_xgboost_data(self.dataset, n_components=3)
            self.assertEqual(pca_calls.call_count, 3)
            self.assertIsNone(pca_calls.call_args_list[0].kwargs.get("model"))
            self.assertIs(pca_calls.call_args_list[1].kwargs["model"], prepared.pca_model)
            self.assertIs(pca_calls.call_args_list[2].kwargs["model"], prepared.pca_model)
            self.assertEqual(prepared.X_train.shape, (40, 5))
            self.assertEqual(prepared.feature_names, ["PC1", "PC2", "PC3", "age", "sex"])
            np.testing.assert_allclose(
                prepared.X_val[:, -2:], self.dataset.val.data[["age", "sex"]].to_numpy(dtype=float),
            )
            np.testing.assert_array_equal(prepared.y_train, np.arange(40) % 2)
            original = prepared.X_train.copy()
            for depth in [1, 2]:
                classifier = train_xgboost(
                    prepared.X_train, prepared.y_train, prepared.X_val, prepared.y_val,
                    n_estimators=12, max_depth=depth, min_child_weight=1,
                    early_stopping_rounds=3, n_jobs=1,
                )
                self.assertEqual(classifier.get_params()["max_depth"], depth)
                metrics = evaluate_xgboost(classifier, prepared.X_val, prepared.y_val)
                self.assertTrue(all(0 <= value <= 1 for value in metrics.values()))
            self.assertEqual(pca_calls.call_count, 3)
            np.testing.assert_array_equal(prepared.X_train, original)

    def test_heart_only_classifier_and_invalid_labels(self):
        prepared = prepare_xgboost_data(self.dataset, n_components=3, include_demographics=False)
        self.assertEqual(prepared.X_train.shape[1], 3)
        self.dataset.val.data.loc[0, "MI"] = "unknown"
        with patch("mcpp.xgboost.pca_datasubset") as pca_calls:
            with self.assertRaisesRegex(ValueError, "MI"):
                prepare_xgboost_data(self.dataset, n_components=3)
            pca_calls.assert_not_called()


if __name__ == "__main__":
    unittest.main()
