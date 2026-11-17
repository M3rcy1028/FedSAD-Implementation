"""Load the preserved implementations without parsing the public CLI twice."""
from contextlib import contextmanager
import importlib
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "experiment/_legacy"


def activate():
    for path in [LEGACY / "models", LEGACY]:
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def load(name, batch_size=16, percentile=82, dropout=.1):
    activate()
    previous = sys.argv
    try:
        sys.argv = ["backend", "--batch_size", str(batch_size), "--percentile", str(percentile),
                    "--dropout_rate", str(dropout)]
        return importlib.import_module(name)
    finally:
        sys.argv = previous


def worker_setup():
    activate()
    for key, value in {"TF_NUM_INTRAOP_THREADS": "2", "TF_NUM_INTEROP_THREADS": "1",
                       "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "1"}.items():
        os.environ[key] = value


@contextmanager
def working_directory(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)
