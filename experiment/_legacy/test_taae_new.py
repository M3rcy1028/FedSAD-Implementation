"""Evaluate the current CICIoT2023 RNEP checkpoint using its training loader.

Normal training rows stay at 80%, matching the current checkpoint. By default,
evaluation contains equal numbers of held-out normal and class-stratified attack
rows. The training loader owns cleanup, log1p, MinMax fitting, and shuffling.
"""

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

PROJECT_ROOT = Path(__file__).resolve().parent
TRAIN_RATIO = 0.5
SPLIT_SEED = 123


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_metadata(base_dir, train_count, labels, attack_map):
    """Reproduce loader row order without changing its feature transformations."""
    normal_count = train_count + int(np.sum(labels == 0))
    normal_order = shuffle(np.arange(normal_count), random_state=SPLIT_SEED)
    merged = pd.read_csv(base_dir / "CICIoT_ae_anomaly.csv", low_memory=False)
    attack_ids = []
    offset = 0
    for entry in attack_map.itertuples(index=False):
        frame = pd.read_csv(base_dir / f"CICIoT_anomaly_ae_{entry.id}.csv")
        if (frame.columns.tolist() != merged.columns.tolist()
                or not np.array_equal(frame.to_numpy(),
                                      merged.iloc[offset:offset + len(frame)].to_numpy())):
            raise ValueError(f"Merged/per-class attack rows differ for attack {entry.id}")
        attack_ids.extend([int(entry.id)] * len(frame))
        offset += len(frame)
    if offset != len(merged) or offset != int(np.sum(labels == 1)):
        raise ValueError("Attack map does not cover the merged attack dataset")
    attack_order = shuffle(np.arange(offset), random_state=SPLIT_SEED)
    source_ids = np.r_[normal_order[train_count:], normal_count + attack_order]
    class_ids = np.r_[np.zeros(normal_count - train_count, dtype=int),
                      np.asarray(attack_ids)[attack_order]]
    source_ids, class_ids = shuffle(source_ids, class_ids, random_state=SPLIT_SEED)
    if not np.array_equal((class_ids != 0).astype(int), labels):
        raise ValueError("Reconstructed source order does not match training-loader labels")
    if np.intersect1d(normal_order[:train_count], source_ids).size:
        raise ValueError("Training normal rows overlap evaluation rows")
    return source_ids, class_ids, merged.columns.tolist()


def evaluation_indices(labels, attack_ids, balance):
    if balance == "all":
        return np.arange(len(labels))
    normal = np.flatnonzero(labels == 0)
    attack = np.flatnonzero(labels == 1)
    if len(attack) < len(normal):
        raise ValueError("Not enough attack rows for equal normal/attack evaluation")
    if len(attack) > len(normal):
        attack, _ = train_test_split(
            attack, train_size=len(normal), random_state=SPLIT_SEED,
            stratify=attack_ids[attack],
        )
    # Keep the original loader's shuffled order within the selected subset.
    return np.sort(np.r_[normal, attack])


def metric_row(name, attack_id, labels, errors, threshold):
    predictions = (errors > threshold).astype(int)
    report = classification_report(
        labels, predictions, labels=[0, 1], target_names=["Normal", "Anomaly"],
        output_dict=True, zero_division=0,
    )
    return {
        "Attack_ID": attack_id, "Attack": name, "Samples": len(labels),
        "Normal": int(np.sum(labels == 0)), "Anomaly": int(np.sum(labels == 1)),
        "Accuracy": report["accuracy"], "Precision": report["Anomaly"]["precision"],
        "Recall": report["Anomaly"]["recall"], "F1": report["Anomaly"]["f1-score"],
        "Normal_Precision": report["Normal"]["precision"],
        "Normal_Recall": report["Normal"]["recall"], "Macro_F1": report["macro avg"]["f1-score"],
        "ROC_AUC": float(roc_auc_score(labels, errors)),
        "Pred_Normal": int(np.sum(predictions == 0)),
        "Pred_Anomaly": int(np.sum(predictions == 1)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval_dataset", choices=["CICIoT2023", "ciciot"], default="CICIoT2023")
    parser.add_argument("--weights", type=Path, default=PROJECT_ROOT / "results/CICIoT2023/taae/fedsad_weights.h5")
    parser.add_argument("--base_dir", type=Path, default=Path("/data/SDP_Dataset/Unified_model/ciciot_rnep"))
    parser.add_argument("--result_path", type=Path, default=PROJECT_ROOT / "results/CICIoT2023/taae/eval_new")
    parser.add_argument("--percentile", type=float, default=82,
                        help="Normal training-error percentile (82 matches the training default)")
    parser.add_argument("--test_balance", choices=["balanced", "all"], default="balanced",
                        help="balanced: equal normal/attack test counts; all: all held-out rows")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--latent_dim", type=int, default=None)
    args = parser.parse_args()
    if not 0 <= args.percentile <= 100 or args.batch_size < 1:
        parser.error("percentile must be in [0, 100] and batch_size must be positive")
    weights = args.weights if args.weights.is_absolute() else PROJECT_ROOT / args.weights
    base_dir = args.base_dir if args.base_dir.is_absolute() else PROJECT_ROOT / args.base_dir
    output = args.result_path if args.result_path.is_absolute() else PROJECT_ROOT / args.result_path
    if not weights.is_file():
        parser.error(f"Weight file not found: {weights}")
    if output.resolve() == (PROJECT_ROOT / "results/CICIoT2023/taae/eval").resolve():
        parser.error("Use a separate output directory to preserve the previous evaluation")

    import tensorflow as tf
    import matplotlib.pyplot as plt
    from taae import TransformerAAE
    from utils.get_datasets import get_datasets_ciciot
    from utils.taae_checkpoint import resolve_latent_dim
    from models.test_taae import _plot_and_print_cm, plot_error_distributions

    weight_hash = sha256(weights)
    data_hashes = {name: sha256(base_dir / name) for name in
                   ["CICIoT_ae_normal.csv", "CICIoT_ae_anomaly.csv"]}
    X_train, X_test, labels = get_datasets_ciciot(str(base_dir), train_ratio=TRAIN_RATIO)
    attack_map = pd.read_csv(base_dir.parent / "ciciot_attack_map.csv").sort_values("id")
    source_ids, attack_ids, features = source_metadata(base_dir, len(X_train), labels, attack_map)
    selected = evaluation_indices(labels, attack_ids, args.test_balance)
    X_test, labels = X_test[selected], labels[selected].astype(int)
    source_ids, attack_ids = source_ids[selected], attack_ids[selected]
    print(f"Evaluation: Normal={np.sum(labels == 0):,}, Attack={np.sum(labels == 1):,}; "
          f"normal training ratio={TRAIN_RATIO}, seed={SPLIT_SEED}", flush=True)
    print("Preprocessing: shared RNEP loader (log1p + MinMax fitted on normal-train only)", flush=True)

    latent_dim = resolve_latent_dim(weights, args.latent_dim)
    model = TransformerAAE(input_dim=X_train.shape[1], latent_dim=latent_dim)
    model(tf.zeros((1, X_train.shape[1])), prior_labels=tf.zeros((1, 1)))
    model.load_weights(str(weights))
    train_predictions = model.predict(X_train, batch_size=args.batch_size, verbose=2)
    train_errors = np.mean(np.square(X_train - train_predictions), axis=1)
    del train_predictions, X_train
    predictions = model.predict(X_test, batch_size=args.batch_size, verbose=2)
    errors = np.mean(np.square(X_test - predictions), axis=1)
    if not np.isfinite(train_errors).all() or not np.isfinite(errors).all():
        raise ValueError("Model returned nonfinite reconstruction errors")
    threshold = float(np.percentile(train_errors, args.percentile))
    predicted_labels = (errors > threshold).astype(int)
    if sha256(weights) != weight_hash:
        raise RuntimeError("Checkpoint changed during evaluation; rerun with a stable checkpoint")

    output.mkdir(parents=True, exist_ok=True)
    stem = "CICIoT2023"
    rows = [metric_row("All", 0, labels, errors, threshold)]
    error_by_attack = {}
    for entry in attack_map.itertuples(index=False):
        attack_mask = attack_ids == entry.id
        if not attack_mask.any():
            continue
        mask = (labels == 0) | attack_mask
        rows.append(metric_row(entry[1], int(entry.id), labels[mask], errors[mask], threshold))
        error_by_attack[int(entry.id)] = (entry[1], errors[attack_mask])
    pd.DataFrame(rows).to_csv(output / f"{stem}_summary.csv", index=False)
    pd.DataFrame({"flow_index": source_ids, "attack_id": attack_ids, "y_true": labels,
                  "reconstruction_error": errors, "y_pred": predicted_labels}).to_csv(
        output / f"{stem}_predictions.csv", index=False)
    pd.DataFrame({"y_true": labels, "anomaly_score": errors}).to_csv(
        output / f"{stem}_roc_data.csv", index=False)
    np.savez_compressed(output / f"{stem}_scores.npz", train_errors=train_errors,
                        test_errors=errors, y_test=labels, source_ids=source_ids,
                        attack_ids=attack_ids)
    report = classification_report(labels, predicted_labels, labels=[0, 1],
                                   target_names=["Normal", "Anomaly"], digits=6,
                                   zero_division=0)
    text = f"Percentile: {args.percentile}\nThreshold: {threshold:.16g}\n\n{report}"
    (output / f"{stem}_report.txt").write_text(text)
    metadata = {
        "dataset": stem, "weights": str(weights.resolve()), "weights_sha256": weight_hash,
        "latent_dim": latent_dim, "normal_train_ratio": TRAIN_RATIO, "split_seed": SPLIT_SEED,
        "test_balance": args.test_balance, "training_rows": len(train_errors),
        "normal_test_rows": int(np.sum(labels == 0)), "attack_test_rows": int(np.sum(labels == 1)),
        "percentile": args.percentile, "threshold": threshold,
        "threshold_source": "normal training errors; anomaly iff error > threshold",
        "loader": "utils.get_datasets.get_datasets_ciciot",
        "loader_source_sha256": hashlib.sha256(inspect.getsource(get_datasets_ciciot).encode()).hexdigest(),
        "data_sha256": data_hashes, "feature_order": features,
        "training_normal_rows_overlap_test": False,
        "attack_sampling": "class-stratified seed123 subset of merged attacks; no attacks used in RNEP training",
        "metrics": rows[0],
    }
    (output / f"{stem}_metrics.json").write_text(json.dumps(metadata, indent=2) + "\n")
    _plot_and_print_cm(labels, predicted_labels, output / f"{stem}_cm.png",
                       ["Normal", "Anomaly"], f"CICIoT2023 (P={args.percentile})")
    fpr, tpr, _ = roc_curve(labels, errors)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, label=f"AUC = {rows[0]['ROC_AUC']:.4f}")
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="CICIoT2023")
    ax.legend(); ax.grid(alpha=.25); fig.tight_layout()
    fig.savefig(output / f"{stem}_roc.png", dpi=180); plt.close(fig)
    plot_error_distributions(errors[labels == 0], error_by_attack, threshold,
                             stem, output / f"{stem}_distribution.png")
    print(text, flush=True)
    print(f"ROC AUC: {rows[0]['ROC_AUC']:.6f}\nSaved results: {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
