"""Protect the public entry point, paper artifacts and numeric split protocol."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle

from experiment.artifacts import ROOT, writable_output
from experiment.cli import main
from experiment.datasets.tabular import prepare
from experiment.evaluation.metrics import report, choose_threshold


class UnifiedInterfaceTests(unittest.TestCase):
    def resolve(self, *args):
        with contextlib.redirect_stdout(io.StringIO()) as stream:
            main([*args, "--dry-run"])
        return json.loads(stream.getvalue())

    def test_defaults_aliases_and_explicit_json_override(self):
        config = self.resolve()
        self.assertEqual(config['epochs'], 30)
        self.assertEqual(config['clients'], [15])
        self.assertEqual(config['rounds'], 10)
        self.assertEqual(config['variant'], 'previous')
        self.assertEqual(self.resolve('--model','mnids','--dataset','unsw')['model'],'mvids')
        self.assertEqual(self.resolve('--model','cnnlstm','--dataset','genis')['model'],'cnnlstm')
        self.assertEqual(self.resolve('--data-dir','relative-data')['data_dir'],
                         str((Path.cwd()/'relative-data').resolve()))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'experiment.json'
            path.write_text(json.dumps({'model':'taae','dataset':'kdd99','epochs':70,'clients':5}))
            config=self.resolve('--config',str(path),'--epochs','30')
            self.assertEqual(config['epochs'],30)
            self.assertEqual(config['clients'],[5])

    def test_paper_paths_including_symlinks_are_not_writable_outputs(self):
        with self.assertRaises(ValueError):writable_output(ROOT/'results/new_run')
        with tempfile.TemporaryDirectory() as tmp:
            link=Path(tmp)/'archive'
            link.symlink_to(ROOT/'results',target_is_directory=True)
            with self.assertRaises(ValueError):writable_output(link/'new_run')

    def test_help_does_not_import_machine_learning_frameworks(self):
        script="from experiment.cli import main; import sys; main(['--list']); assert 'tensorflow' not in sys.modules; assert 'torch' not in sys.modules"
        result=subprocess.run([sys.executable,'-B','-c',script],cwd=ROOT,capture_output=True,text=True,timeout=20)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_fedsad_options_and_dataset_thresholds(self):
        cic = self.resolve('--model','fedsad','--dataset','cic')
        self.assertEqual((cic['time_window'], cic['threshold_std'], cic['clients']), ('1min', .25, [1]))
        self.assertEqual(self.resolve('--model','fedsad','--dataset','unsw')['threshold_std'], 5.0)
        genis = self.resolve('--model','fedsad','--dataset','genis','--time-window','2min','--threshold-std','2')
        self.assertEqual((genis['time_window'], genis['threshold_std']), ('2min', 2.0))
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(['--model','fedsad','--dataset','kdd99','--dry-run'])

    def test_unsupported_model_dataset_fails_before_creating_output(self):
        with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
            main(['--model','mvids','--dataset','genis','--dry-run'])

    def test_kdd_uses_existing_half_normal_split_and_train_only_minmax(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            normal=pd.DataFrame({'a':np.arange(80.),'b':np.arange(80.)%3})
            attack=pd.DataFrame({'a':np.arange(20.)+1000,'b':np.ones(20)})
            normal.to_csv(root/'KDD99_normal.csv',index=False)
            attack.to_csv(root/'KDD99_anomaly.csv',index=False)
            config=self.resolve('--data-dir',str(root))
            config['clients']=15
            bundle,pipeline,details=prepare(config)
            ordered=shuffle(normal,random_state=123)
            scaler=MinMaxScaler().fit(ordered.iloc[:40])
            np.testing.assert_array_equal(bundle['X_train'],scaler.transform(ordered.iloc[:40]))
            self.assertEqual(bundle['X_test'].shape,(60,2))
            self.assertEqual(np.bincount(bundle['y_test']).tolist(),[40,20])
            self.assertFalse(set(bundle['train_ids'])&set(bundle['test_ids']))
            self.assertTrue((bundle['X_test'][bundle['y_test']==1,0]>1).all())
            self.assertEqual(details['normal_train_ratio'],.5)

    def test_metric_positive_class_and_exact_threshold_boundary(self):
        metrics=report(np.array([0,0,1,1]),np.array([.1,.5,.5,.9]),.5)
        self.assertEqual(metrics['confusion_matrix'],[[2,0],[1,1]])
        inclusive=report(np.array([0,0,1,1]),np.array([.1,.5,.5,.9]),.5,inclusive=True)
        self.assertEqual(inclusive['confusion_matrix'],[[1,1],[0,2]])
        config={'model':'taae','percentile':82,'threshold_policy':'validation','independent_validation':False}
        with self.assertRaises(ValueError):choose_threshold(config,np.array([.1,.2]),np.array([.1,.3]),np.array([0,1]))

    def test_legacy_insdn_split_and_unsw_numeric_cleanup_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            normal = pd.DataFrame({'a': -np.arange(1., 81.), 'b': ['0x000b'] * 80})
            attack = pd.DataFrame({'a': [-1000.] * 20, 'b': ['bad'] * 20})
            normal.to_csv(root / 'InSDN_normal.csv', index=False)
            attack.to_csv(root / 'InSDN_anomaly.csv', index=False)
            for model, expected in [('aelstm', 40), ('cnnlstm', 64)]:
                config = self.resolve('--model', model, '--dataset', 'insdn', '--data-dir', str(root))
                bundle, pipeline, details = prepare(config)
                self.assertEqual(details['normal_training_rows'], expected)
                self.assertLess(pipeline['scaler'].data_min_[0], 0)
                self.assertTrue((bundle['X_test'][bundle['y_test'] == 1, 0, 0] < 0).all())
            normal = pd.concat([normal, pd.DataFrame({f'c{i}': np.arange(80.) for i in range(41)})], axis=1)
            attack = pd.concat([attack, pd.DataFrame({f'c{i}': np.arange(20.) for i in range(41)})], axis=1)
            normal.to_csv(root / 'UNSW_NB15_normal.csv', index=False)
            attack.to_csv(root / 'UNSW_NB15_anomaly.csv', index=False)
            config = self.resolve('--model', 'cnnlstm', '--dataset', 'unsw', '--data-dir', str(root))
            bundle, pipeline, details = prepare(config)
            self.assertEqual(bundle['X_train'].shape[1:], (6, 7))
            self.assertTrue(details['source_feature_truncation_preserved'])
            self.assertEqual(details['model_feature_order'], normal.columns[:42].tolist())


if __name__=='__main__':
    unittest.main()
