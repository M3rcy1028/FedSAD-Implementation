"""Percentile-threshold helpers for one fixed RNEP checkpoint.

Used by the GeNIS experiment and threshold selection. The maximum on the test
set is an exploratory result, not an independently validated estimate.
"""

import csv
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("MPLBACKEND", "Agg")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np


def threshold_metrics(train_errors, test_errors, labels, percentiles):
    """Use sorted scores to count predictions for each threshold, including ties."""
    train_errors = np.asarray(train_errors)
    test_errors = np.asarray(test_errors)
    labels = np.asarray(labels)
    percentiles = np.asarray(percentiles, dtype=float)
    if train_errors.ndim != 1 or not len(train_errors):
        raise ValueError("Training errors must be a nonempty vector")
    if test_errors.ndim != 1 or labels.shape != test_errors.shape:
        raise ValueError("Test scores and labels must be aligned vectors")
    if not all(np.isfinite(x).all() for x in (train_errors, test_errors, percentiles)):
        raise ValueError("Errors and percentiles must be finite")
    if percentiles.ndim != 1 or not len(percentiles) or np.any((percentiles < 0) | (percentiles > 100)):
        raise ValueError("Percentiles must be a nonempty vector in [0, 100]")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Both Normal=0 and Anomaly=1 test samples are required")

    normal = np.sort(test_errors[labels == 0])
    attack = np.sort(test_errors[labels == 1])
    thresholds = np.percentile(train_errors, percentiles)
    # Prediction is Anomaly only when error > threshold; ties predict Normal.
    tn = np.searchsorted(normal, thresholds, side="right")
    fn = np.searchsorted(attack, thresholds, side="right")
    fp, tp = len(normal) - tn, len(attack) - fn

    def divide(numerator, denominator):
        return np.divide(numerator, denominator, out=np.zeros(len(percentiles)), where=denominator != 0)

    normal_precision = divide(tn, tn + fn)
    normal_recall = tn / len(normal)
    normal_f1 = divide(2 * tn, 2 * tn + fp + fn)
    attack_precision = divide(tp, tp + fp)
    attack_recall = tp / len(attack)
    attack_f1 = divide(2 * tp, 2 * tp + fp + fn)
    columns = {
        "percentile": percentiles,
        "threshold": thresholds,
        "accuracy": (tn + tp) / len(labels),
        "macro_f1": (normal_f1 + attack_f1) / 2,
        "balanced_accuracy": (normal_recall + attack_recall) / 2,
        "normal_precision": normal_precision,
        "normal_recall": normal_recall,
        "normal_f1": normal_f1,
        "attack_precision": attack_precision,
        "attack_recall": attack_recall,
        "attack_f1": attack_f1,
        "false_positive_rate": fp / len(normal),
        "false_negative_rate": fn / len(attack),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }
    return [{key: value[i].item() for key, value in columns.items()} for i in range(len(percentiles))]


def best_row(rows, metric):
    # Resolve equal metrics consistently without looking at another checkpoint.
    return max(rows, key=lambda r: (r[metric], r["macro_f1"], r["accuracy"], -r["percentile"]))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reconstruction_errors(model, data, batch_size):
    predictions = model.predict(data, batch_size=batch_size, verbose=2)
    errors = np.mean(np.square(data - predictions), axis=1)
    if not np.isfinite(errors).all():
        raise ValueError("Model returned nonfinite reconstruction errors")
    return errors


def save_plot(rows, output):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    p = [r["percentile"] for r in rows]
    for key, label in [("accuracy", "Accuracy"), ("macro_f1", "Macro F1"), ("balanced_accuracy", "Balanced accuracy")]:
        axes[0].plot(p, [r[key] for r in rows], label=label)
    for key, label in [("normal_precision", "Normal precision"), ("normal_recall", "Normal recall"), ("attack_recall", "Attack recall")]:
        axes[1].plot(p, [r[key] for r in rows], label=label)
    for axis in axes:
        axis.set(xlabel="Percentile of normal training errors", ylabel="Score", ylim=(0, 1.02))
        axis.axvline(80, color="grey", linestyle="--", alpha=0.6)
        axis.grid(alpha=0.25)
        axis.legend()
    fig.suptitle("RNEP threshold search (exploratory test-set comparison)")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
