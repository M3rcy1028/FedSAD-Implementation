"""Compare saved predictions on exactly the AE/CNN held-out source flow IDs."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.utils import shuffle

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.sweep_rnep_percentiles import best_row, threshold_metrics


def prediction_metrics(name, labels, scores, predictions, supervision, percentile=None):
    report = classification_report(labels, predictions, target_names=["Normal", "Anomaly"], output_dict=True, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
    return {
        "model": name, "supervision": supervision, "percentile": percentile,
        "accuracy": report["accuracy"], "macro_f1": report["macro avg"]["f1-score"],
        "normal_precision": report["Normal"]["precision"], "normal_recall": report["Normal"]["recall"],
        "attack_precision": report["Anomaly"]["precision"], "attack_recall": report["Anomaly"]["recall"],
        "roc_auc": roc_auc_score(labels, scores), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rnep_sweep_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--audit_dir", type=Path, required=True)
    parser.add_argument("--aelstm_dir", type=Path, default=ROOT / "results/CICIoT2023/aelstm/eval")
    parser.add_argument("--cnnlstm_dir", type=Path, default=ROOT / "results/CICIoT2023/cnnlstm/eval")
    args = parser.parse_args()
    ae = pd.read_csv(args.aelstm_dir / "CICIoT2023_predictions.csv")
    cnn = pd.read_csv(args.cnnlstm_dir / "CICIoT2023_predictions.csv")
    for key in ["flow_index", "attack_id", "y_true"]:
        np.testing.assert_array_equal(ae[key].to_numpy(), cnn[key].to_numpy())
    ids = ae["flow_index"].to_numpy(dtype=np.int64)
    labels = ae["y_true"].to_numpy(dtype=np.int64)
    if len(np.unique(ids)) != len(ids):
        raise ValueError("Baseline predictions contain repeated source IDs")
    ae_meta = json.loads((args.aelstm_dir / "CICIoT2023_metrics.json").read_text())
    cnn_meta = json.loads((args.cnnlstm_dir / "CICIoT2023_metrics.json").read_text())
    for metadata in [ae_meta, cnn_meta]:
        if (metadata["random_seed"], metadata["normal_train_ratio"], metadata["attack_train_ratio"]) != (123, 0.5, 0.5):
            raise ValueError("Source-ID reconstruction requires the documented seed123 and 50:50 splits")

    summary = json.loads((args.rnep_sweep_dir / "summary.json").read_text())
    audit = json.loads((args.audit_dir / "audit.json").read_text())
    if not audit["merged_attack_exact_order_equal"]:
        raise ValueError("Merged attack CSV order was not verified against the per-class files")
    manifest = pd.read_csv(args.audit_dir / "sha256_manifest.csv").set_index("file")
    for path, digest in summary["fingerprint"]["files"].items():
        if path.endswith(".csv"):
            relative = "ciciot_rnep/" + Path(path).name
            if manifest.loc[relative, "unified_sha256"] != digest:
                raise ValueError("Current scores do not match the audited source rows")

    with np.load(args.rnep_sweep_dir / "scores.npz") as scores:
        train_errors = scores["train_errors"]
        full_errors = scores["test_errors"]
        full_labels = scores["y_test"]
    if summary["fingerprint"]["normal_train_ratio"] != 0.5:
        raise ValueError("This comparison expects the normal 50:50 RNEP split")
    n_normal = 2 * len(train_errors)
    n_attack = int(np.sum(full_labels == 1))
    normal_test_ids = shuffle(np.arange(n_normal), random_state=123)[len(train_errors):]
    attack_ids = n_normal + shuffle(np.arange(n_attack), random_state=123)
    rnep_ids = shuffle(np.r_[normal_test_ids, attack_ids], random_state=123)
    np.testing.assert_array_equal((rnep_ids >= n_normal).astype(int), full_labels)
    if not np.isin(ids, rnep_ids).all():
        raise ValueError("Some baseline test rows are outside the RNEP held-out set")
    lookup = np.full(n_normal + n_attack, np.nan)
    lookup[rnep_ids] = full_errors
    common_errors = lookup[ids]
    if not np.isfinite(common_errors).all():
        raise ValueError("Missing common-test RNEP scores")
    np.testing.assert_array_equal((ids >= n_normal).astype(int), labels)

    rows = threshold_metrics(train_errors, common_errors, labels, np.arange(1001) / 10)
    selected = best_row(rows, "macro_f1")
    chosen = [("RNEP p80", next(r for r in rows if r["percentile"] == 80)),
              ("RNEP p82", next(r for r in rows if r["percentile"] == 82)),
              ("RNEP p70.4", next(r for r in rows if r["percentile"] == 70.4)),
              ("RNEP best Macro F1 (exploratory)", selected)]
    precision_choices = []
    for floor in [0.5, 0.7, 0.8, 0.9]:
        b = best_row([r for r in rows if r["normal_recall"] >= floor], "normal_precision")
        precision_choices.append({"minimum_normal_recall": floor, **b})
        chosen.append((f"RNEP max Normal precision, recall >= {floor:g}", b))
    results = [prediction_metrics(name, labels, common_errors, (common_errors > row["threshold"]).astype(int), "normal-only", row["percentile"]) for name, row in chosen]
    results.extend([
        prediction_metrics("AE-LSTM classifier", labels, ae["classifier_probability"], ae["pred_cls"], "normal + 100000 attack training rows"),
        prediction_metrics("AE-LSTM reconstruction p85", labels, ae["reconstruction_error"], ae["pred_recon"], "normal + 100000 attack training rows", 85),
        prediction_metrics("CNN-LSTM classifier", labels, cnn["probability"], cnn["y_pred"], "normal + 100000 attack training rows"),
    ])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(args.output_dir / "comparison.csv", index=False)
    pd.DataFrame(rows).to_csv(args.output_dir / "rnep_percentile_metrics.csv", index=False)
    np.savez_compressed(args.output_dir / "rnep_common_test_scores.npz", train_errors=train_errors, test_errors=common_errors, y_test=labels, source_ids=ids)
    record = {
        "rows": len(ids), "normal_rows": int(np.sum(labels == 0)), "attack_rows": int(np.sum(labels == 1)),
        "source_ids_sha256": hashlib.sha256(ids.tobytes()).hexdigest(),
        "identical_ae_cnn_test_ids_and_labels": True, "rnep_rows_verified_held_out": True,
        "source_order_audit": str(args.audit_dir.resolve()), "source_hashes_match_audit": True,
        "selection_note": "RNEP maxima are exploratory selections on this common test set. Supervision differs across models.",
        "rnep_precision_tradeoffs": precision_choices, "comparison": results,
    }
    (args.output_dir / "comparison.json").write_text(json.dumps(record, indent=2) + "\n")
    columns = ["model", "percentile", "accuracy", "macro_f1", "normal_precision", "normal_recall", "attack_recall", "roc_auc"]
    print(pd.DataFrame(results)[columns].to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"Verified identical test source rows: {len(ids)}; saved: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
