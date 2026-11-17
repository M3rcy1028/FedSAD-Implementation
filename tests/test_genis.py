"""Protect the matched GeNIS partitions and train-only transformations."""

from experiment.backends import activate
activate()
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from utils.genis import load_genis, balance


class GeNISPreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        metadata = self.root / "raw" / "0-info"
        metadata.mkdir(parents=True)
        pd.DataFrame({"FeatureName": ["amount", "PCRatio", "all_missing", "Rank"],
                      "FeatureType": ["Continuous", "Continuous", "Continuous", "Integer"]}).to_csv(
                          metadata / "genis-features.csv", index=False)
        folder = self.root / "raw" / "2-flows" / "flows-10-sec"
        folder.mkdir(parents=True)
        for label, binary, groups in [("benign-admin", 0, 80), ("benign-user", 0, 80),
                                      ("attack-a", 1, 80), ("attack-b", 1, 80)]:
            records = []
            for group in range(groups):
                for window in range(2):
                    records.append({"FlowID": f"{label}-{group}", "StartTime": group * 10 + window,
                        "LastTime": group * 10 + window + 1, "BinaryLabel": binary,
                        "CategoryLabel": "attack" if binary else "benign", "SubCategoryLabel": label,
                        "amount": 1e9 if binary else float(group + window),
                        "PCRatio": -.5 if binary else (group % 5 - 2) / 2,
                        "all_missing": 999 if binary else np.nan, "Rank": group,
                        "Proto": "tcp", "Flgs": "e", "Dir": "->", "Cause": "Start", "TcpOpt": "M",
                        "State": "attack-only" if binary else "normal"})
            pd.DataFrame(records).to_csv(folder / f"{label}.csv", index=False)

    def test_disjoint_connections_balancing_and_reproducibility(self):
        first = load_genis(self.root, layout="flows")
        second = load_genis(self.root, layout="flows")
        for name, array in first[0].items():
            np.testing.assert_array_equal(array, second[0][name])
        groups = [set(first[1][f"{name}_groups"]) for name in ["train", "val", "test"]]
        for i, j in [(0, 1), (0, 2), (1, 2)]:
            self.assertFalse(groups[i] & groups[j])
        for name in ["val", "test"]:
            counts = np.bincount(first[0][f"y_{name}"])
            self.assertEqual(counts[0], counts[1])
            self.assertEqual(set(first[1][f"{name}_labels"][first[0][f"y_{name}"] == 1]),
                             {"attack-a", "attack-b"})
        self.assertTrue(all(label.startswith("benign") for label in first[1]["train_labels"]))

    def test_signed_measurements_and_train_only_fitting(self):
        arrays, sources, pipeline, report = load_genis(self.root, layout="flows")
        self.assertEqual(pipeline["numeric_features"], ["amount", "PCRatio"])
        self.assertLess(pipeline["scaler"].data_min_[1], 0)
        self.assertLess(pipeline["scaler"].data_max_[0], 100)
        self.assertIn("all_missing", report["all_missing_training_features_removed"])
        self.assertNotIn("Rank", report["expanded_features"])
        categories = dict(zip(pipeline["categorical_features"], pipeline["encoder"].categories_))
        self.assertNotIn("attack-only", categories["State"])
        self.assertTrue((arrays["X_test"][arrays["y_test"] == 1, 0] > 1).all())
        self.assertTrue(all(np.isfinite(value).all() for value in arrays.values()))

    def test_portless_attack_uses_disjoint_time_blocks(self):
        path = self.root / "raw" / "2-flows" / "flows-10-sec" / "attack-a.csv"
        frame = pd.read_csv(path)
        frame["FlowID"] = "single-icmp-capture"
        frame["Proto"] = "icmp"
        frame.to_csv(path, index=False)
        _, sources, _, report = load_genis(self.root, layout="flows")
        groups = [set(sources[f"{name}_groups"][sources[f"{name}_labels"] == "attack-a"])
                  for name in ["val", "test"]]
        self.assertTrue(groups[0] and groups[1])
        self.assertFalse(groups[0] & groups[1])
        self.assertTrue(all("|window=" in group for group in groups[0] | groups[1]))

    def test_scenario_clock_strings_and_reused_capture_records(self):
        source = self.root / "raw" / "2-flows" / "flows-10-sec"
        frame = pd.concat([pd.read_csv(path) for path in sorted(source.glob("*.csv"))], ignore_index=True)
        mask = frame.SubCategoryLabel == "attack-a"
        frame.loc[mask, "Proto"] = "icmp"
        frame.loc[mask, "FlowID"] = "single-icmp-capture"
        for column in ["StartTime", "LastTime"]:
            frame[column] = frame[column].map(lambda n: f"{int(n)//3600:02d}:{int(n)//60%60:02d}:{int(n)%60:02d}")
        destination = self.root / "raw" / "3-scenarios" / "scenarios-10-sec"
        destination.mkdir(parents=True)
        frame.to_csv(destination / "scenario-1.csv", index=False)
        frame.to_csv(destination / "scenario-2.csv", index=False)
        _, sources, _, report = load_genis(self.root)
        self.assertEqual(report["original_rows"], 1280)
        self.assertEqual(report["duplicate_records_reused_across_scenarios_removed"], 640)
        self.assertFalse(set(sources["val_groups"]) & set(sources["test_groups"]))

    def test_class_cap_retains_rare_attacks(self):
        frame = pd.DataFrame({"BinaryLabel": [0] * 100 + [1] * 1010,
                              "SubCategoryLabel": ["normal"] * 100 + ["large"] * 1000 + ["rare"] * 10})
        result = balance(frame, seed=123)
        self.assertEqual(result.SubCategoryLabel.value_counts().to_dict(),
                         {"normal": 100, "large": 90, "rare": 10})


if __name__ == "__main__":
    unittest.main()
