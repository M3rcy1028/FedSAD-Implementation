"""Compare train-error percentile thresholds on one fixed RNEP checkpoint.

The maximum on the test set is an exploratory result, not an independently
validated estimate. This script does not train or modify the checkpoint.
"""

import argparse
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
from utils.taae_checkpoint import resolve_latent_dim


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--latent_dim", type=int, default=None,
                        help="Use checkpoint architecture metadata, defaulting to legacy latent dimension 64")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--base_dir", type=Path, default=Path("/data/SDP_Dataset/Unified_model/ciciot_rnep"))
    parser.add_argument("--train_ratio", type=float, default=0.5)
    parser.add_argument("--step", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--selection_metric", choices=["macro_f1", "accuracy", "normal_precision"], default="macro_f1")
    parser.add_argument("--min_normal_recall", type=float, default=0.0,
                        help="Minimum Normal recall required for the selected threshold (0 to 1)")
    parser.add_argument("--reuse_scores", action="store_true", help="Reuse cached scores only if input hashes and split match")
    args = parser.parse_args()
    if not 0 < args.train_ratio < 1 or not 0 < args.step <= 100 or args.batch_size < 1:
        parser.error("Invalid train ratio, percentile step, or batch size")
    if not 0 <= args.min_normal_recall <= 1:
        parser.error("Minimum Normal recall must be between 0 and 1")
    if not args.weights.is_file():
        parser.error(f"Missing checkpoint: {args.weights}")
    latent_dim = resolve_latent_dim(args.weights, args.latent_dim)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_paths = [args.weights.resolve(), args.base_dir / "CICIoT_ae_normal.csv", args.base_dir / "CICIoT_ae_anomaly.csv"]
    fingerprint = {"files": {str(p): sha256(p) for p in input_paths}, "normal_train_ratio": args.train_ratio, "data_split_seed": 123}
    # Keep existing latent-64 score caches compatible with their original schema.
    if latent_dim != 64:
        fingerprint["latent_dim"] = latent_dim
    cache_path = args.output_dir / "scores.npz"
    fingerprint_path = args.output_dir / "score_inputs.json"
    if args.reuse_scores and cache_path.is_file() and fingerprint_path.is_file():
        if json.loads(fingerprint_path.read_text()) != fingerprint:
            raise ValueError("Cached scores do not match the checkpoint, data, and split")
        with np.load(cache_path) as cached:
            train_errors, test_errors, labels = (cached[k] for k in ["train_errors", "test_errors", "y_test"])
        print("Reusing matching cached scores", flush=True)
    else:
        import tensorflow as tf
        from taae import TransformerAAE
        from utils.get_datasets import get_datasets_ciciot

        X_train, X_test, labels = get_datasets_ciciot(str(args.base_dir), train_ratio=args.train_ratio)
        model = TransformerAAE(X_train.shape[1], latent_dim=latent_dim)
        model(tf.zeros((1, X_train.shape[1])), training=False, prior_labels=tf.zeros((1, 1)))
        model.load_weights(str(args.weights))
        print(f"Scoring normal train={len(X_train)}, held-out test={len(X_test)}", flush=True)
        train_errors = reconstruction_errors(model, X_train, args.batch_size)
        test_errors = reconstruction_errors(model, X_test, args.batch_size)
        np.savez_compressed(cache_path, train_errors=train_errors, test_errors=test_errors, y_test=labels)
        fingerprint_path.write_text(json.dumps(fingerprint, indent=2) + "\n")

    percentiles = np.unique(np.r_[np.arange(0, 100, args.step), 80.0, 82.0, 100.0])
    rows = threshold_metrics(train_errors, test_errors, labels, percentiles)
    with (args.output_dir / "percentile_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    from sklearn.metrics import classification_report, roc_auc_score

    eligible = [row for row in rows if row["normal_recall"] >= args.min_normal_recall]
    if not eligible:
        raise ValueError("No candidate satisfies the minimum Normal recall")
    best = best_row(eligible, args.selection_metric)
    selected_predictions = (test_errors > best["threshold"]).astype(int)
    summary = {
        "selection_data": "held-out test set (exploratory; not independent validation)",
        "selection_metric": args.selection_metric,
        "latent_dim": latent_dim,
        "minimum_normal_recall": args.min_normal_recall,
        "eligible_candidates": len(eligible),
        "percentile_step": args.step,
        "candidates": len(rows),
        "fingerprint": fingerprint,
        "train_rows": len(train_errors), "test_rows": len(test_errors),
        "normal_test_rows": int(np.sum(labels == 0)), "attack_test_rows": int(np.sum(labels == 1)),
        "roc_auc": float(roc_auc_score(labels, test_errors)),
        "selected": best,
        "best_macro_f1": best_row(rows, "macro_f1"),
        "best_accuracy": best_row(rows, "accuracy"),
        "best_normal_precision": best_row(eligible, "normal_precision"),
        "unconstrained_best_normal_precision": best_row(rows, "normal_precision"),
        "reference_80": next(r for r in rows if r["percentile"] == 80),
        "reference_82": next(r for r in rows if r["percentile"] == 82),
        "selected_classification_report": classification_report(labels, selected_predictions, target_names=["Normal", "Anomaly"], output_dict=True, zero_division=0),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    report = classification_report(labels, selected_predictions, target_names=["Normal", "Anomaly"], digits=6, zero_division=0)
    report = (f"Exploratory test-set selection by {args.selection_metric}\n"
              f"Minimum Normal recall: {args.min_normal_recall:.6f}\n"
              f"Percentile: {best['percentile']:.3f}\n"
              f"MSE threshold: {best['threshold']:.12g}\n\n") + report
    (args.output_dir / "best_report.txt").write_text(report + "\n")
    save_plot(rows, args.output_dir / "percentile_tradeoffs.png")
    print(report, flush=True)
    print("Compared thresholds on one fixed checkpoint:", flush=True)
    for name in ["reference_80", "reference_82", "best_macro_f1", "best_accuracy", "best_normal_precision"]:
        r = summary[name]
        print(f"{name}: percentile={r['percentile']:.2f}, threshold={r['threshold']:.9g}, accuracy={r['accuracy']:.6f}, macro_f1={r['macro_f1']:.6f}, normal_precision={r['normal_precision']:.6f}, normal_recall={r['normal_recall']:.6f}, attack_recall={r['attack_recall']:.6f}", flush=True)
    print(f"Saved results: {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
