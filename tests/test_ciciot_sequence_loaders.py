"""Check split integrity and scaling on small, identifiable flow fixtures."""

from experiment.backends import activate
activate()
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from pathlib import Path
import sys
import subprocess
import tempfile
import unittest

import numpy as np
import pandas as pd

from AE_LSTM import get_datasets_ciciot_semi
from CNN_LSTM import get_datasets_ciciot_supervised
from scripts.test_aelstm import evaluate_ciciot_aelstm
from scripts.test_cnnlstm import evaluate_ciciot_cnnlstm


class CICIoTSequenceLoaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / "ciciot_rnep"
        self.data.mkdir()
        self.columns = [f"feature_{i}" for i in range(78)]
        pd.DataFrame({"id": [1, 2, 3], "class": ["attack_1", "attack_2", "attack_3"]}).to_csv(
            self.root / "ciciot_attack_map.csv", index=False
        )
        self.write_flows("CICIoT_ae_normal.csv", 40, 0, 0)
        for class_id, count in [(1, 10), (2, 14), (3, 18)]:
            self.write_flows(f"CICIoT_anomaly_ae_{class_id}.csv", count, 1000 * class_id, class_id)
        self.loaders = [get_datasets_ciciot_semi, get_datasets_ciciot_supervised]

    def write_flows(self, name, count, first_id, class_id):
        values = np.zeros((count, 78))
        values[:, 0] = np.arange(first_id, first_id + count)
        values[:, 1] = class_id
        values[:, 3] = -5
        values[:, 4] = np.inf
        pd.DataFrame(values, columns=self.columns).to_csv(self.data / name, index=False)

    def test_balanced_stratified_disjoint_split_and_padding(self):
        for load in self.loaders:
            with self.subTest(loader=load.__name__):
                train, y_train, test, y_test = load(base_dir=str(self.data))
                self.assertEqual(train.shape, (41, 10, 8))
                self.assertEqual(test.shape, (41, 10, 8))
                self.assertEqual(np.bincount(y_train).tolist(), [20, 21])
                self.assertEqual(np.bincount(y_test).tolist(), [20, 21])
                for x, y in [(train, y_train), (test, y_test)]:
                    ids, counts = np.unique(x[y == 1, 0, 1], return_counts=True)
                    np.testing.assert_array_equal(ids, [1, 2, 3])
                    np.testing.assert_array_equal(counts, [5, 7, 9])
                    self.assertEqual(x.dtype, np.float32)
                    self.assertTrue(np.isfinite(x).all())
                    np.testing.assert_array_equal(x.reshape(len(x), -1)[:, -2:], 0)
                    np.testing.assert_array_equal(x[:, 0, 3:5], 0)
                    # Attack IDs exceed the normal-train scaling range.
                    self.assertTrue((x[y == 1, 0, 0] > 1).all())
                train_rows = set(map(tuple, train.reshape(len(train), -1)))
                test_rows = set(map(tuple, test.reshape(len(test), -1)))
                self.assertFalse(train_rows & test_rows)

    def test_both_models_share_reproducible_splits(self):
        first = self.loaders[0](base_dir=str(self.data), random_seed=123)
        second = self.loaders[1](base_dir=str(self.data), random_seed=123)
        repeated = self.loaders[0](base_dir=str(self.data), random_seed=123)
        for a, b, c in zip(first, second, repeated):
            np.testing.assert_array_equal(a, b)
            np.testing.assert_array_equal(a, c)

    def test_evaluation_metadata_tracks_held_out_rows(self):
        metadata_by_model = []
        for load in self.loaders:
            plain = load(base_dir=str(self.data))
            *data, metadata = load(base_dir=str(self.data), return_metadata=True)
            for before, after in zip(plain, data):
                np.testing.assert_array_equal(before, after)
            self.assertFalse(set(metadata['train_indices']) & set(metadata['test_indices']))
            np.testing.assert_array_equal(metadata['test_attack_ids'] > 0, data[3] == 1)
            metadata_by_model.append(metadata)
        for key in metadata_by_model[0]:
            np.testing.assert_array_equal(metadata_by_model[0][key], metadata_by_model[1][key])

    def test_ciciot_evaluation_uses_training_calibration_and_held_out_test_only(self):
        calls = []

        class Model:
            def predict(self, x, **kwargs):
                calls.append(len(x))
                return {"decoded": x.copy(),
                        "pred": np.where(x[:, 0, 1] > 0, 0.9, 0.1).reshape(-1, 1)}

        output = self.root / "eval"
        summary, metadata = evaluate_ciciot_aelstm(Model(), self.data, output, batch_size=8)
        self.assertEqual(calls, [20, 41])
        self.assertEqual(metadata['normal_test_rows'], 20)
        self.assertEqual(metadata['attack_test_rows'], 21)
        self.assertEqual(metadata['reconstruction_threshold'], 0)
        self.assertEqual(summary.iloc[0]['Acc_Cls'], 1)
        self.assertEqual(summary['Anomaly'].tolist(), [21, 5, 7, 9])
        self.assertEqual(len(pd.read_csv(output / 'CICIoT2023_predictions.csv')), 41)
        self.assertTrue((output / 'CICIoT2023_cls_cm.png').is_file())
        self.assertTrue((output / 'CICIoT2023_recon_roc.png').is_file())

    def test_cnn_evaluation_predicts_only_held_out_rows(self):
        calls = []

        class Model:
            def predict(self, x, **kwargs):
                calls.append(len(x))
                return np.where(x[:, 0, 1] > 0, 0.9, 0.1).reshape(-1, 1)

        output = self.root / 'cnn_eval'
        summary, metadata = evaluate_ciciot_cnnlstm(Model(), self.data, output, batch_size=8)
        self.assertEqual(calls, [41])
        self.assertEqual(metadata['normal_test_rows'], 20)
        self.assertEqual(metadata['attack_test_rows'], 21)
        self.assertEqual(summary.iloc[0]['Accuracy'], 1)
        self.assertEqual(summary['Anomaly'].tolist(), [21, 5, 7, 9])
        self.assertTrue((output / 'CICIoT2023_cm.png').is_file())
        self.assertTrue((output / 'CICIoT2023_roc.png').is_file())

    def test_missing_class_and_schema_mismatch_fail(self):
        path = self.data / "CICIoT_anomaly_ae_3.csv"
        original = path.read_bytes()
        path.unlink()
        for load in self.loaders:
            with self.assertRaises(FileNotFoundError):
                load(base_dir=str(self.data))
        path.write_bytes(original)
        frame = pd.read_csv(path)
        frame[frame.columns[::-1]].to_csv(path, index=False)
        for load in self.loaders:
            with self.assertRaisesRegex(ValueError, "columns/order"):
                load(base_dir=str(self.data))


class WorkerImportTests(unittest.TestCase):
    def test_evaluation_script_runs_from_scripts_directory(self):
        repo = Path(__file__).resolve().parents[1] / "experiment/_legacy"
        for name in ['test_aelstm.py', 'test_cnnlstm.py']:
            with self.subTest(script=name):
                result = subprocess.run(
                    [sys.executable, str(repo / 'scripts' / name), '--help'],
                    capture_output=True, text=True, cwd=repo / 'scripts',
                    env={**os.environ, 'PYTHONPATH': ''}, timeout=90,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn('--eval_dataset', result.stdout)

    def test_model_imports_do_not_parse_ray_worker_arguments(self):
        script = """
import sys
sys.argv = ['default_worker.py', '--node-ip-address=127.0.0.1',
            '--node-manager-port=45375', '--object-store-name=/tmp/test-store']
import models.model_cnnlstm
import models.model_aelstm
import CNN_LSTM
import AE_LSTM
import scripts.test_aelstm
import scripts.test_cnnlstm
print('worker imports OK')
"""
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True,
            cwd=Path(__file__).resolve().parents[1] / "experiment/_legacy", timeout=90,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("worker imports OK", result.stdout)


if __name__ == "__main__":
    unittest.main()
