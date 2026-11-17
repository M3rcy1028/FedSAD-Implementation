"""Publication plots from saved predictions; no model or dataset reload required."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

DISPLAY = {"kdd99": "KDD99", "nsl-kdd": "NSL-KDD", "insdn": "InSDN",
           "cic2018": "CSE-CIC-IDS2018", "unsw-nb15": "UNSW-NB15",
           "genis": "GeNIS", "gotham": "Gotham", "itu-ulak": "ITU/ULAK"}


def render(output, frame, config):
    output = Path(output)
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    dataset = DISPLAY.get(config["dataset"], config["dataset"])
    truth, scores, predicted = frame.y_true.to_numpy(), frame.score.to_numpy(), frame.y_pred.to_numpy()
    matrix = confusion_matrix(truth, predicted, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Predicted Normal", "Predicted Anomaly"],
                yticklabels=["Actual Normal", "Actual Anomaly"])
    ax.set(title=f"Confusion Matrix ({dataset})", xlabel="Predicted", ylabel="Actual")
    fig.tight_layout(); fig.savefig(figures / "confusion_matrix.png", dpi=300); plt.close(fig)
    fpr, tpr, _ = roc_curve(truth, scores)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc_score(truth, scores):.4f}")
    ax.plot([0, 1], [0, 1], color="navy", linestyle="--")
    ax.set(title=f"ROC Curve ({dataset})", xlabel="False Positive Rate (FPR)",
           ylabel="True Positive Rate (TPR)", xlim=(0, 1), ylim=(0, 1.05))
    ax.legend(loc="lower right"); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(figures / "roc.png", dpi=300); plt.close(fig)
    if "reconstruction_error" in frame:
        errors = frame.reconstruction_error.to_numpy()
        floor = errors[errors > 0].min() / 2 if np.any(errors > 0) else 1e-12
        logged = np.log10(np.maximum(errors, floor))
        lo, hi = float(logged.min()), float(logged.max())
        bins = np.linspace(lo - .01 if lo == hi else lo, hi + .01 if lo == hi else hi, 81)
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.hist(logged[truth == 0], bins=bins, density=True, alpha=.35, color="green", label="Normal")
        names = sorted(set(frame.loc[frame.y_true == 1, "attack"]))
        for name, color in zip(names, plt.cm.turbo(np.linspace(.03, .97, len(names)))):
            mask = (truth == 1) & (frame.attack.to_numpy() == name)
            ax.hist(logged[mask], bins=bins, density=True, histtype="step", color=color, label=name)
        threshold_file = output / "metadata/reconstruction_threshold.json"
        if threshold_file.exists():
            threshold = json.loads(threshold_file.read_text())["threshold"]
            ax.axvline(np.log10(max(threshold, floor)), color="blue", linestyle="--", label=f"Threshold ({threshold:.6g})")
        ax.set(title=f"Reconstruction Error ({dataset})", xlabel="log10(Reconstruction MSE)",
               ylabel="Density", yscale="log")
        ax.grid(alpha=.25); ax.legend(fontsize=8, ncol=2 if len(names) > 12 else 1)
        fig.tight_layout(); fig.savefig(figures / "reconstruction_error.png", dpi=300); plt.close(fig)
    attacks = pd.read_csv(output / "tables/per_attack.csv")
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.bar(attacks.attack, attacks.recall, color="tab:green")
    ax.set(title=f"Recall per Attack Type ({dataset})", ylabel="Recall", ylim=(0, 1.2))
    ax.set_title(ax.get_title(), fontsize=18, pad=18)
    ax.set_ylabel("Recall", fontsize=16)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=14)
    ax.tick_params(axis="y", labelsize=13)
    fig.tight_layout(); fig.savefig(figures / "per_attack_recall.png", dpi=300); plt.close(fig)
    rounds_file = output / "metadata/round_metrics.json"
    if rounds_file.exists():
        records = json.loads(rounds_file.read_text())
        flat = []
        for row in records:
            item = {key: value for key, value in row.items() if not isinstance(value, (list, dict))}
            if "classification_report" in row:
                report = row["classification_report"]
                item.update(accuracy=report["accuracy"], macro_f1=report["macro avg"]["f1-score"],
                            normal_recall=report["Normal"]["recall"], attack_recall=report["Anomaly"]["recall"])
            flat.append(item)
        pd.DataFrame(flat).to_csv(output / "tables/round_metrics.csv", index=False)
        fig, ax = plt.subplots(figsize=(8, 5))
        if all("macro_f1" in row for row in flat):
            for key, label in [("accuracy", "Accuracy"), ("macro_f1", "Macro F1"),
                               ("normal_recall", "Normal recall"), ("attack_recall", "Attack recall")]:
                ax.plot([row["round"] for row in flat], [row[key] for row in flat], marker="o", label=label)
            ax.set_ylabel("Validation metric"); ax.legend()
        else:
            ax.plot([row["round"] for row in records], [row["mean_validation_score"] for row in records], marker="o")
            ax.set_ylabel("Mean validation reconstruction error")
        ax.set(title=f"Validation by round ({dataset})", xlabel="Server round")
        ax.grid(alpha=.25)
        fig.tight_layout(); fig.savefig(figures / "round_metrics.png", dpi=300); plt.close(fig)


def compare(runs, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    rows = []
    fig, ax = plt.subplots(figsize=(8, 7))
    for run in runs:
        run = Path(run).resolve()
        config = json.loads((run / "config.json").read_text())
        metrics = json.loads((run / "metadata/metrics.json").read_text())
        frame = pd.read_csv(run / "tables/predictions.csv")
        fpr, tpr, _ = roc_curve(frame.y_true, frame.score)
        label = f"{config['model']} / {DISPLAY.get(config['dataset'], config['dataset'])} / {config['clients']} clients"
        ax.plot(fpr, tpr, label=f"{label} (AUC = {metrics['roc_auc']:.4f})")
        report = metrics["classification_report"]
        rows.append({"run": str(run), "model": config["model"], "dataset": config["dataset"],
                     "clients": config["clients"], "accuracy": report["accuracy"],
                     "precision": report["Anomaly"]["precision"], "recall": report["Anomaly"]["recall"],
                     "f1": report["Anomaly"]["f1-score"], "macro_f1": report["macro avg"]["f1-score"],
                     "roc_auc": metrics["roc_auc"]})
    ax.plot([0, 1], [0, 1], color="navy", linestyle="--")
    ax.set(title="ROC Curves", xlabel="False Positive Rate (FPR)", ylabel="True Positive Rate (TPR)",
           xlim=(0, 1), ylim=(0, 1.05))
    ax.grid(alpha=.25); ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(destination / "roc_comparison.png", dpi=300); plt.close(fig)
    pd.DataFrame(rows).to_csv(destination / "metrics_comparison.csv", index=False)
