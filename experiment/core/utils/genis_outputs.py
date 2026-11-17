"""Dataset/model output files shared by GeNIS benchmark runners."""
import json
import os
from pathlib import Path
import shutil
import tempfile

os.environ.setdefault("MPLBACKEND", "Agg")


def save_confusion_matrix(matrix, destination, model_name=None, tag="", dataset_name="GeNIS"):
    """Render an existing matrix with the model's explicit display name."""
    import numpy as np
    import matplotlib.pyplot as plt
    matrix = np.asarray(matrix)
    title = f"{model_name} Confusion Matrix - {dataset_name}" if model_name else f"Confusion Matrix - {dataset_name}"
    if dataset_name == "GeNIS" and model_name in {"RNEP", "TAAE"} and not tag:
        title = "Confusion Matrix (GeNIS)"
    if tag:
        evaluation = {"cls": "Classification", "recon": "Reconstruction"}.get(tag, tag)
        title += f" ({evaluation})"
    if dataset_name == "GeNIS" and model_name in {"RNEP", "TAAE"} and not tag:
        # Match the original TAAE evaluation plots (8x6 heatmap with colorbar).
        import seaborn as sns
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Predicted Normal", "Predicted Anomaly"],
                    yticklabels=["Actual Normal", "Actual Anomaly"], ax=ax)
        ax.set(title=title, xlabel="Predicted", ylabel="Actual")
        fig.tight_layout()
        fig.savefig(destination, dpi=300)
        plt.close(fig)
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(matrix, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                    color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Normal", "Anomaly"],
           yticklabels=["Normal", "Anomaly"], xlabel="Predicted", ylabel="True", title=title)
    fig.tight_layout()
    fig.savefig(destination, dpi=200)
    plt.close(fig)


def save_detection_outputs(destination, labels, scores, threshold, attack_labels,
                           source_ids=None, tag="", percentile=None, score_kind="reconstruction_error",
                           model_name=None, dataset_name="GeNIS"):
    import numpy as np
    import pandas as pd
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
    import matplotlib.pyplot as plt
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    labels, scores = np.asarray(labels), np.asarray(scores)
    attack_labels = np.asarray(attack_labels)
    if labels.ndim != 1 or scores.shape != labels.shape or attack_labels.shape != labels.shape:
        raise ValueError("Scores, labels and attack labels must be aligned vectors")
    if set(np.unique(labels)) != {0, 1} or not np.isfinite(scores).all() or not np.isfinite(threshold):
        raise ValueError("Evaluation requires both classes and finite scores/threshold")
    if np.any(scores < 0):
        raise ValueError("Anomaly scores must be nonnegative")
    stem = dataset_name + (f"_{tag}" if tag else "")
    predicted = (scores > threshold).astype(int)
    report = classification_report(labels, predicted, labels=[0, 1],
                                   target_names=["Normal", "Anomaly"], output_dict=True, zero_division=0)
    matrix = confusion_matrix(labels, predicted, labels=[0, 1])
    metrics = {"dataset": dataset_name, "percentile": percentile, "threshold": float(threshold),
               "score_kind": score_kind, "roc_auc": float(roc_auc_score(labels, scores)),
               "classification_report": report, "confusion_matrix": matrix.tolist()}
    (destination / f"{stem}_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    text = f"Percentile: {percentile}\nThreshold: {threshold:.15g}\nROC AUC: {metrics['roc_auc']:.6f}\n\n"
    text += classification_report(labels, predicted, labels=[0, 1], target_names=["Normal", "Anomaly"],
                                  digits=6, zero_division=0)
    (destination / f"{stem}_report.txt").write_text(text)
    columns = {"attack": attack_labels, "y_true": labels, "anomaly_score": scores, "y_pred": predicted}
    if source_ids is not None:
        if len(source_ids) != len(labels):
            raise ValueError("Source IDs do not match evaluation rows")
        columns = {"source_id": source_ids, **columns}
    pd.DataFrame(columns).to_csv(destination / f"{stem}_predictions.csv", index=False)
    pd.DataFrame({"y_true": labels, "anomaly_score": scores}).to_csv(
        destination / f"{stem}_roc_data.csv", index=False)
    rows = []
    for name in ["ALL", *sorted(set(attack_labels[labels == 1]))]:
        mask = np.ones(len(labels), dtype=bool) if name == "ALL" else ((labels == 0) | (attack_labels == name))
        r = classification_report(labels[mask], predicted[mask], labels=[0, 1],
                                  target_names=["Normal", "Anomaly"], output_dict=True, zero_division=0)
        rows.append({"File": name, "Samples": int(mask.sum()), "Accuracy": r["accuracy"],
                     "Precision": r["Anomaly"]["precision"], "Recall": r["Anomaly"]["recall"],
                     "F1": r["Anomaly"]["f1-score"], "Pred_Normal": int((predicted[mask] == 0).sum()),
                     "Pred_Anomaly": int((predicted[mask] == 1).sum()),
                     "Normal": int((labels[mask] == 0).sum()), "Anomaly": int((labels[mask] == 1).sum()),
                     "Normal_Precision": r["Normal"]["precision"], "Normal_Recall": r["Normal"]["recall"],
                     "Macro_F1": r["macro avg"]["f1-score"],
                     "ROC_AUC": float(roc_auc_score(labels[mask], scores[mask])),
                     "Percentile": percentile, "Threshold": float(threshold)})
    pd.DataFrame(rows).to_csv(destination / f"{stem}_summary.csv", index=False)
    save_confusion_matrix(matrix, destination / f"{stem}_cm.png", model_name, tag, dataset_name)
    fpr, tpr, _ = roc_curve(labels, scores)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, label=f"AUC={metrics['roc_auc']:.4f}")
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title=f"ROC Curve - {dataset_name}")
    ax.legend(); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(destination / f"{stem}_roc.png", dpi=200); plt.close(fig)
    if score_kind == "reconstruction_error":
        positive = scores[scores > 0]
        floor = float(positive.min() / 2) if len(positive) else 1e-12
        values = np.log10(np.maximum(scores, floor))
        cut = np.log10(max(threshold, floor))
        xlabel = "log10(Reconstruction MSE)"
    else:
        values, cut, xlabel = scores, threshold, "Attack probability"
    low, high = float(values.min()), float(values.max())
    if low == high:
        low, high = low - .5, high + .5
    bins = np.linspace(low, high, 81)
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.hist(values[labels == 0], bins=bins, density=True, color="green", alpha=.3, label="Normal")
    attacks = sorted(set(attack_labels[labels == 1]))
    for name, color in zip(attacks, plt.cm.turbo(np.linspace(.03, .97, len(attacks)))):
        ax.hist(values[(labels == 1) & (attack_labels == name)], bins=bins, density=True,
                histtype="step", color=color, label=name)
    ax.axvline(cut, color="blue", linestyle="--", label=f"Threshold ({threshold:.6g})")
    ax.set(xlabel=xlabel, ylabel="Density", title=f"{dataset_name} score distribution", yscale="log")
    ax.legend(fontsize=8); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(destination / f"{stem}_distribution.png", dpi=200); plt.close(fig)
    return metrics


def publish_rnep_outputs(run_dir, destination):
    """Publish one completed variant using the existing dataset/model layout."""
    import numpy as np
    run_dir, destination = Path(run_dir), Path(destination)
    if not (run_dir / "fedsad_weights.h5").is_file():
        raise FileNotFoundError(f"Trained weights not found: {run_dir / 'fedsad_weights.h5'}")
    config = json.loads((run_dir / "model_config.json").read_text())
    dataset_name = config.get("dataset", "GeNIS")
    selected = json.loads((run_dir / "selected_threshold.json").read_text())
    prepared = Path(config["prepared_data"])
    with np.load(run_dir / "scores.npz") as z:
        labels, scores = z["y_test"], z["test_errors"]
    with np.load(prepared / "source_rows.npz") as z:
        attack_labels, ids = z["test_labels"], z["test_ids"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".genis_export_", dir=destination.parent) as temp:
        stage = Path(temp)
        metrics = save_detection_outputs(stage, labels, scores, selected["threshold"], attack_labels,
                                         ids, percentile=selected["percentile"], model_name="RNEP",
                                         dataset_name=dataset_name)
        expected = selected["test_metrics"]["confusion_matrix"]
        if metrics["confusion_matrix"] != expected:
            raise ValueError("Exported metrics do not match the selected threshold")
        copies = {
            run_dir / "fedsad_weights.h5": f"rnep_frame_{dataset_name.lower()}_weights.h5",
            run_dir / "model_config.json": "model_config.json",
            run_dir / "selected_threshold.json": "selected_threshold.json",
            run_dir / "history.csv": "rnep_history.csv",
            run_dir / "history.png": "rnep_history.png",
            run_dir / "client_evaluations.jsonl": "client.txt",
            prepared / "preprocessing.joblib": "preprocessing.joblib",
            prepared / "preprocessing_report.json": "preprocessing_report.json",
        }
        for source, name in copies.items():
            if source.is_file():
                shutil.copy2(source, stage / name)
        shutil.copy2(stage / f"{dataset_name}_report.txt", stage / "rnep_server.txt")
        rounds = run_dir / "round_validation_metrics.jsonl"
        if rounds.exists():
            with (stage / "rnep_server.txt").open("a") as stream:
                for line in rounds.read_text().splitlines():
                    row = json.loads(line)
                    report = row["classification_report"]
                    stream.write(f"\n[Round {row['round']}] Validation evaluation\n"
                                 f"Accuracy: {report['accuracy']:.6f}; AUROC: {row['roc_auc']:.6f}\n")
                    for label in ["Normal", "Anomaly"]:
                        r = report[label]
                        stream.write(f"{label}: precision={r['precision']:.6f} recall={r['recall']:.6f} "
                                     f"F1={r['f1-score']:.6f} support={r['support']:.0f}\n")
        manifest = {"source_run": str(run_dir.resolve()), "model_variant": config["taae_version"],
                    "prepared_data": str(prepared), "percentile": selected["percentile"],
                    "threshold": selected["threshold"], "selection_set": "validation",
                    "files": sorted(path.name for path in stage.iterdir())}
        (stage / "output_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        destination.mkdir(parents=True, exist_ok=True)
        for path in sorted(stage.iterdir(), key=lambda p: p.name == "output_manifest.json"):
            os.replace(path, destination / path.name)
    return destination
