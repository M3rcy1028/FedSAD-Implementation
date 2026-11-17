"""Protect main's window tools and the retained CICIoT integration after relocation."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from experiment.backends import load


class MainMergeCompatibilityTests(unittest.TestCase):
    def test_gsad_accepts_both_branches_dataset_options(self):
        arguments = load("utils.arguments")
        for name in ["cic", "unsw", "ciciot", "genis", "genis_frame"]:
            with self.subTest(dataset=name), patch.object(sys, "argv", ["gsad.py", "--eval_dataset", name]):
                self.assertEqual(arguments.get_gsad_args().eval_dataset, name)

    def test_time_sorted_graph_rows_keep_their_matching_taae_rows(self):
        datasets = load("utils.get_datasets")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            taae, gsad = root / "taae", root / "gsad"
            taae.mkdir(); gsad.mkdir()
            normal = pd.DataFrame({"f": [10., 20., 30., 40.]})
            graph = pd.DataFrame({
                "Timestamp": [f"01/01/2025 00:0{i}:00" for i in [3, 2, 1, 0]],
                "Dst Port": normal.f,
                "split": ["train", "test", "train", "test"],
            })
            normal.to_csv(taae / "normal.csv", index=False)
            graph.to_csv(gsad / "normal.csv", index=False)
            for name, value, minute in [("a.csv", 101., 2), ("b.csv", 201., 0)]:
                pd.DataFrame({"f": [value]}).to_csv(taae / name, index=False)
                pd.DataFrame({"Timestamp": [f"01/01/2025 00:0{minute}:00"],
                              "Dst Port": [value]}).to_csv(gsad / name, index=False)
            graph_normal, graph_attack, normal_test, attack_test, scaled_train = datasets.fedsad_data_preprocessing(
                taae / "normal.csv", [taae / "b.csv", taae / "a.csv"],
                gsad / "normal.csv", [gsad / "a.csv", gsad / "b.csv"], MinMaxScaler())
            self.assertEqual(graph_normal.row_id.tolist(), [3, 1])
            self.assertEqual(normal_test.index.tolist(), [1, 3])
            for graph_rows, flow_rows in [(graph_normal, normal_test), (graph_attack, attack_test)]:
                np.testing.assert_array_equal(flow_rows.loc[graph_rows.row_id, "f"].to_numpy(),
                                              graph_rows["Dst Port"].to_numpy())
            np.testing.assert_array_equal(scaled_train[:, 0], [0., 1.])
            with self.assertRaises(ValueError):
                datasets._pair_files([taae / "a.csv"], [gsad / "b.csv"])

    def test_window_metrics_use_anomaly_as_positive_and_include_window_wait(self):
        metrics = load("utils.window_metrics")
        records = []
        for minute, (label, predicted, rows, latency) in enumerate([
                ("Normal", 0, 2, 2.), ("Normal", 1, 3, 3.),
                ("Attack", 0, 1, 4.), ("Attack", 1, 4, 5.)]):
            start = pd.Timestamp("2025-01-01") + pd.Timedelta(minutes=minute)
            records.append({"window_time": start, "first_ts": start + pd.Timedelta(seconds=10),
                            "label": label, "pred": predicted, "n_rows": rows, "infer_ms": latency})
        result = metrics.compute_window_metrics(records, "1min")
        self.assertEqual(result["confusion"], {"tn": 1, "fp": 1, "fn": 1, "tp": 1})
        self.assertEqual(result["precision"], .5)
        self.assertEqual(result["recall"], .5)
        self.assertEqual(result["f1"], .5)
        self.assertAlmostEqual(result["inference_ms_per_flow"], 1.4)
        self.assertAlmostEqual(result["window_detection_delay_s"]["mean"], 50.005)


if __name__ == "__main__":
    unittest.main()
