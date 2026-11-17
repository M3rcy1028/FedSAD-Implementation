"""Plot six datasets' saved TAAE anomaly scores without reevaluating models."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


ROOT = Path(__file__).resolve().parents[3]
ROC_DIR = ROOT / "results"
SOURCES = [
    ("InSDN", ROC_DIR / "InSDN" / "InSDN_roc_data.csv", "tab:blue"),
    ("KDD99", ROC_DIR / "KDD99" / "KDD99_roc_data.csv", "tab:green"),
    ("UNSW-NB15", ROC_DIR / "UNSW-NB15" / "UNSW-NB15_roc_data.csv", "tab:brown"),
    ("CSE-CIC-IDS2018", ROC_DIR / "CSE-CIC-IDS2018" / "CSE-CIC-IDS2018_roc_data.csv", "tab:gray"),
    ("NSL-KDD", ROC_DIR / "NSL-KDD" / "NSL-KDD_roc_data.csv", "tab:cyan"),
    ("GeNIS", ROOT / "results" / "GeNIS" / "taae" / "GeNIS_roc_data.csv", "tab:purple"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROC_DIR / "combined_roc_curves_6datasets.png")
    args = parser.parse_args()
    rows = []
    fig, ax = plt.subplots(figsize=(8, 7))
    for name, source, color in SOURCES:
        data = pd.read_csv(source, usecols=["y_true", "anomaly_score"])
        labels = data["y_true"].to_numpy()
        scores = data["anomaly_score"].to_numpy()
        if set(np.unique(labels)) != {0, 1} or not np.isfinite(scores).all():
            raise ValueError(f"Invalid binary labels or anomaly scores: {source}")
        fpr, tpr, _ = roc_curve(labels, scores)
        area = roc_auc_score(labels, scores)
        ax.plot(fpr, tpr, color=color, linewidth=1.8,
                label=f"{name} (AUC = {area:.4f})")
        rows.append({"dataset": name, "auc": area, "samples": len(data),
                     "normal": int((labels == 0).sum()),
                     "anomaly": int((labels == 1).sum()), "source": str(source)})
    ax.plot([0, 1], [0, 1], color="navy", linestyle="--", linewidth=1.3)
    ax.set(xlim=(0, 1), ylim=(0, 1.05), xlabel="False Positive Rate (FPR)",
           ylabel="True Positive Rate (TPR)", title="ROC Curves")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
    ax.set_axisbelow(True)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, facecolor="white")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(args.output.with_suffix(".csv"), index=False)
    print(pd.DataFrame(rows)[["dataset", "auc", "samples"]].to_string(index=False))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
