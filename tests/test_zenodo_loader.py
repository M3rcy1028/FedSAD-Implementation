"""Protect train-only preprocessing and the published ULAK split."""

from experiment.backends import activate
activate()
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from taae import get_datasets_zenodo


class ZenodoLoaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        pd.DataFrame({
            " feature": [0., 2., 4., -1., np.inf, 1e9],
            " Fwd Header Length": [0., 1., 2., 3., 4., 5.],
            " Fwd Header Length.1": [0., 1., 2., 3., 4., 5.],
            "constant": [0., 0., 0., 0., 0., 99.],
            " Label": ["BENIGN"] * 5 + ["Attack"],
        }).to_csv(self.root / "Train_ULAK.csv", index=False)
        pd.DataFrame({
            " feature": np.arange(24, dtype=float) + 100.,
            " Fwd Header Length": np.arange(24, dtype=float),
            " Fwd Header Length.1": np.arange(24, dtype=float),
            "constant": np.ones(24),
            " Label": ["BENIGN"] * 16 + ["Attack A"] * 4 + ["Attack B"] * 4,
        }).to_csv(self.root / "Test_ULAK.csv", index=False)

    def load(self, **kwargs):
        return get_datasets_zenodo(self.root, max_train_samples=None,
                                   return_metadata=True, **kwargs)

    def test_train_only_imputation_scaling_and_native_split(self):
        train, test, labels, metadata = self.load(balance_test=False)
        self.assertEqual(train.shape, (5, 3))
        self.assertEqual(test.shape, (24, 3))
        np.testing.assert_array_equal(np.sort(train[:, 0]), [0., .5, .5, .5, 1.])
        self.assertAlmostEqual(metadata["imputer"].statistics_[0], 2.)
        self.assertEqual(metadata["scaler"].data_max_[0], 4.)
        self.assertTrue(np.all(test[:, 0] > 1))
        self.assertTrue(np.all(test[:, 2] == 1))
        self.assertTrue(np.isfinite(train).all() and np.isfinite(test).all())
        self.assertEqual(set(metadata["train_source_rows"]), set(range(5)))
        self.assertEqual(set(metadata["test_source_rows"]), set(range(24)))
        self.assertEqual(metadata["report"]["dropped_redundant_columns"], ["Fwd Header Length.1"])
        np.testing.assert_array_equal(np.bincount(labels), [16, 8])

    def test_balancing_retains_every_attack_and_is_reproducible(self):
        first = self.load(balance_test=True)
        second = self.load(balance_test=True)
        np.testing.assert_array_equal(np.bincount(first[2]), [8, 8])
        for left, right in zip(first[:3], second[:3]):
            np.testing.assert_array_equal(left, right)
        np.testing.assert_array_equal(first[3]["test_source_rows"], second[3]["test_source_rows"])
        counts = first[3]["report"]["test_class_counts"]
        self.assertEqual(counts["Attack A"], 4)
        self.assertEqual(counts["Attack B"], 4)
        self.assertEqual(len(np.unique(first[3]["test_source_rows"])), 16)

    def test_missing_annotation_and_schema_mismatch_fail(self):
        path = self.root / "Test_ULAK.csv"
        frame = pd.read_csv(path)
        frame.iloc[0, -1] = "No Label"
        frame.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "Unannotated"):
            self.load()
        frame.iloc[0, -1] = "BENIGN"
        frame[frame.columns[::-1]].to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "feature names and order"):
            self.load()


if __name__ == "__main__":
    unittest.main()
