"""Protect Gotham's connection holdout, normal-only fit and device clients."""

from experiment.backends import activate
activate()
import json
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd

from utils.genis import file_sha256
from utils.gotham import GROUP_COLUMNS, NUMERIC, CATEGORICAL, connection_ids, prepare_gotham


class GothamTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        raw = self.root / "raw" / "processed"
        raw.mkdir(parents=True)
        members = []
        for device in range(6):
            rows = []
            for label, day, groups in [("Benign", 14, 40), ("Attack-A", 18, 20), ("Attack-B", 19, 20), ("Unknown", 20, 2)]:
                for group in range(groups):
                    for reverse in range(2):
                        row = {column: np.nan for column in set(GROUP_COLUMNS + NUMERIC + CATEGORICAL)}
                        row.update({"frame.time": f"Jan {day}, 2025 18:40:22.123456000 GMT",
                            "frame.len": str(group + 50 if label == "Benign" else 1000000),
                            "frame.protocols": "eth:ip:tcp" if label == "Benign" else "eth:ip:tcp:attack",
                            "ip.src": f"192.168.0.{device+1}", "ip.dst": "10.0.0.1", "ip.proto": "6",
                            "eth.src": "aa:bb:cc:dd:ee:01", "eth.dst": "aa:bb:cc:dd:ee:02",
                            "tcp.srcport": str(20000 + group), "tcp.dstport": "80", "label": label,
                            "tcp.flags": "0x0018", "ip.flags": "0x02", "ip.ttl": "64", "ip.tos": "0",
                            "tcp.window_size_value": "65535", "tcp.window_size_scalefactor": "128",
                            "tcp.pdu.size": "32", "ip.checksum": "0xFFFF", "tcp.checksum": "0xEEEE",
                            "tcp.options": "0101080a_deadbeef"})
                        if reverse:
                            row["ip.src"], row["ip.dst"] = row["ip.dst"], row["ip.src"]
                            row["tcp.srcport"], row["tcp.dstport"] = row["tcp.dstport"], row["tcp.srcport"]
                        rows.append(row)
            path = raw / f"device-{device}.csv"
            pd.DataFrame(rows).to_csv(path, index=False)
            members.append({"name": "processed/" + path.name, "sha256": file_sha256(path)})
        (self.root / "download_manifest.json").write_text(json.dumps({"members": members}))

    def test_bidirectional_connections_share_group(self):
        frame = pd.read_csv(self.root / "raw/processed/device-0.csv", dtype=str)
        groups = connection_ids(frame)
        np.testing.assert_array_equal(groups.iloc[::2], groups.iloc[1::2])

    def test_disjoint_balanced_normal_only_and_train_only_scaling(self):
        output = self.root / "prepared"
        report = prepare_gotham(self.root, output, max_train_samples=120,
                                eval_normal_samples=24, client_nums=3, chunksize=37, workers=1)
        with np.load(output / "source_rows.npz") as z:
            rows = dict(z)
        with np.load(output / "prepared_data.npz") as z:
            arrays = dict(z)
        self.assertEqual(set(rows["train_labels"]), {"Benign"})
        self.assertEqual(len(arrays["X_train"]), 120)
        for part in ["val", "test"]:
            np.testing.assert_array_equal(np.bincount(arrays[f"y_{part}"]), [24, 24])
            self.assertNotIn("Unknown", rows[f"{part}_labels"])
            self.assertEqual(set(rows[f"{part}_labels"]), {"Benign", "Attack-A", "Attack-B"})
        for left, right in [("train", "val"), ("train", "test"), ("val", "test")]:
            self.assertFalse(set(rows[f"{left}_groups"]) & set(rows[f"{right}_groups"]))
            self.assertFalse(set(rows[f"{left}_ids"]) & set(rows[f"{right}_ids"]))
        self.assertEqual(report["unknown_rows_excluded"], {"Unknown": 24})
        self.assertGreater(arrays["X_val"][arrays["y_val"] == 1, 0].min(), 1)
        self.assertTrue(np.isfinite(arrays["X_train"]).all())
        pipeline = joblib.load(output / "preprocessing.joblib")
        self.assertLess(pipeline["scaler"].data_max_[0], 100)
        self.assertNotIn("eth:ip:tcp:attack", pipeline["encoder"].categories_[0])
        self.assertEqual(pipeline["scaler"].data_min_[pipeline["numeric_features"].index("tcp.flags")], 24)
        self.assertEqual(set(arrays["train_client_ids"]), {0, 1, 2})
        assignments = pd.DataFrame({"device": rows["train_devices"], "client": arrays["train_client_ids"]})
        self.assertEqual(assignments.groupby("device").client.nunique().max(), 1)
        again = self.root / "repeated"
        prepare_gotham(self.root, again, max_train_samples=120,
                       eval_normal_samples=24, client_nums=3, chunksize=53, workers=2)
        with np.load(again / "prepared_data.npz") as repeated:
            for key, original in arrays.items():
                np.testing.assert_array_equal(original, repeated[key])


if __name__ == "__main__":
    unittest.main()
