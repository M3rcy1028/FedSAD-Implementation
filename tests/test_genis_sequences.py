"""Keep benchmark holdouts identical without leaking labeled training rows."""

from experiment.backends import activate
activate()
import json
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd

import test_genis as fixtures
from utils.genis import prepare_genis
from utils.genis_sequences import prepare_sequence_data, capped_attack_sample
from utils.genis_outputs import publish_rnep_outputs


class GeNISSequenceTests(unittest.TestCase):
    setUp = fixtures.GeNISPreprocessingTests.setUp

    def prepare(self):
        base = self.root / "processed" / "base"
        prepare_genis(self.root, base, layout="flows")
        return base, prepare_sequence_data(base, max_normal_samples=16)

    def test_shared_holdout_disjoint_training_and_train_only_scaling(self):
        base, prepared = self.prepare()
        with np.load(base / "source_rows.npz") as z:
            original = dict(z)
        with np.load(prepared / "source_rows.npz") as z:
            source = dict(z)
        with np.load(prepared / "prepared_data.npz") as z:
            arrays = dict(z)
        for part in ["val", "test"]:
            np.testing.assert_array_equal(source[f"{part}_ids"], original[f"{part}_ids"])
            np.testing.assert_array_equal(source[f"{part}_labels"], original[f"{part}_labels"])
            np.testing.assert_array_equal(np.bincount(arrays[f"y_{part}"]), [32, 32])
        for left, right in [("train", "val"), ("train", "test"), ("val", "test")]:
            self.assertFalse(set(source[f"{left}_groups"]) & set(source[f"{right}_groups"]))
            self.assertFalse(set(source[f"{left}_ids"]) & set(source[f"{right}_ids"]))
        np.testing.assert_array_equal(np.bincount(arrays["y_train"]), [16, 16])
        self.assertGreaterEqual(arrays["X_train"].min(), -1e-6)
        self.assertLessEqual(arrays["X_train"].max(), 1 + 1e-6)
        report = json.loads((prepared / "preprocessing_report.json").read_text())
        self.assertEqual(arrays["X_train"].shape[1:], (10, report["features"]))
        flat = arrays["X_train"].reshape(len(arrays["y_train"]), -1)
        np.testing.assert_array_equal(flat[:, report["flat_input_dim"]:], 0)
        raw = {}
        for path in (self.root / "raw" / "2-flows" / "flows-10-sec").glob("*.csv"):
            for index, row in pd.read_csv(path).iterrows():
                raw[f"{path.name}:{index}"] = row
        pipeline = joblib.load(prepared / "preprocessing.joblib")
        selected = pd.DataFrame([raw[key] for key in source["train_ids"]])[pipeline["numeric_features"]]
        np.testing.assert_array_equal(pipeline["scaler"].data_min_, selected.min().to_numpy())
        np.testing.assert_array_equal(pipeline["scaler"].data_max_, selected.max().to_numpy())
        repeated = prepare_sequence_data(base, max_normal_samples=16)
        self.assertEqual(prepared, repeated)

    def test_portless_groups_match_rnep(self):
        path = self.root / "raw" / "2-flows" / "flows-10-sec" / "attack-a.csv"
        frame = pd.read_csv(path)
        frame["FlowID"], frame["Proto"] = "single-capture", "icmp"
        frame.to_csv(path, index=False)
        base, prepared = self.prepare()
        with np.load(base / "source_rows.npz") as a, np.load(prepared / "source_rows.npz") as b:
            for part in ["val", "test"]:
                np.testing.assert_array_equal(a[f"{part}_groups"], b[f"{part}_groups"])

    def test_capped_training_attacks_keep_rare_classes(self):
        frame = pd.DataFrame({"SubCategoryLabel": ["large"] * 1000 + ["rare"] * 3})
        sampled = capped_attack_sample(frame, 20, 123)
        self.assertEqual(sampled.SubCategoryLabel.value_counts().to_dict(), {"large": 17, "rare": 3})
        self.assertFalse(sampled.index.duplicated().any())


class GeNISOutputTests(unittest.TestCase):
    def test_publish_keeps_selected_threshold_predictions_and_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepared, run, published = [root / name for name in ["prepared", "run", "rnep"]]
            for path in [prepared, run]:
                path.mkdir()
            np.savez(prepared / "source_rows.npz", test_labels=["user", "user", "dos", "dos"],
                     test_ids=["f:1", "f:2", "f:3", "f:4"])
            np.savez(run / "scores.npz", y_test=[0, 0, 1, 1], test_errors=[.1, .5, .3, .9])
            (run / "model_config.json").write_text(json.dumps({"prepared_data": str(prepared), "taae_version": "current"}))
            (run / "selected_threshold.json").write_text(json.dumps({"percentile": 80, "threshold": .5,
                "test_metrics": {"confusion_matrix": [[2, 0], [1, 1]]}}))
            weights = b"unchanged model weights"
            (run / "fedsad_weights.h5").write_bytes(weights)
            publish_rnep_outputs(run, published)
            self.assertEqual((published / "rnep_frame_genis_weights.h5").read_bytes(), weights)
            predictions = pd.read_csv(published / "GeNIS_predictions.csv")
            np.testing.assert_array_equal(predictions.y_pred, [0, 0, 0, 1])
            summary = pd.read_csv(published / "GeNIS_summary.csv").iloc[0]
            self.assertEqual(summary.Accuracy, .75)
            self.assertEqual(summary.Pred_Normal, 3)
            self.assertEqual(summary.Pred_Anomaly, 1)
            self.assertTrue((published / "GeNIS_cm.png").is_file())
            self.assertTrue((published / "GeNIS_roc.png").is_file())
            self.assertTrue((published / "rnep_server.txt").is_file())


if __name__ == "__main__":
    unittest.main()
