"""Render CNN-LSTM, AE-LSTM and TAAE attack recall comparisons."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DATASETS = {
    "kdd99": "KDD99",
    "insdn": "InSDN",
    "cse": "CSE-CIC-IDS2018",
    "unsw": "UNSW-NB15",
    "genis": "GeNIS",
}
MODELS = ("CNN-LSTM", "AE-LSTM", "TAAE (Ours)")


def load_genis():
    """Read classification predictions and verify identical evaluation rows."""
    reference = None
    recalls = {}
    for model, folder, stem in [
        ("CNN-LSTM", "cnnlstm", "GeNIS"),
        ("AE-LSTM", "aelstm", "GeNIS_cls"),
        ("TAAE (Ours)", "taae", "GeNIS"),
    ]:
        directory = ROOT / "results" / "GeNIS" / folder
        predictions = pd.read_csv(directory / f"{stem}_predictions.csv")
        if predictions["source_id"].duplicated().any():
            raise ValueError(f"Duplicate evaluation rows for {model}")
        predictions = predictions.set_index("source_id").sort_index()
        truth = predictions[["attack", "y_true"]]
        if reference is None:
            reference = truth
        elif not reference.equals(truth):
            raise ValueError(f"Evaluation rows or labels differ for {model}")
        if not predictions["y_pred"].isin([0, 1]).all():
            raise ValueError(f"Invalid predictions for {model}")
        attacks = predictions[predictions["y_true"] == 1].groupby("attack")["y_pred"]
        recall = attacks.mean().sort_index()
        summary = pd.read_csv(directory / f"{stem}_summary.csv").set_index("File")
        if not np.allclose(recall, summary.loc[recall.index, "Recall"], atol=1e-12, rtol=0):
            raise ValueError(f"Recall differs from the saved summary for {model}")
        recalls[model] = recall
    frame = pd.DataFrame(recalls).rename_axis("Attack_Type").reset_index()
    frame["Support"] = attacks.size().sort_index().to_numpy()
    return frame


def plot_recall(frame, dataset, destination, legend=False):
    values = frame[list(MODELS)].to_numpy(dtype=float)
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError(f"Invalid recall values for {dataset}")
    fig, ax = plt.subplots(figsize=(16, 6))
    x = np.arange(len(frame))
    width = 0.25
    for offset, model in zip((-width, 0, width), MODELS):
        bars = ax.bar(x + offset, frame[model], width, label=model)
        for bar in bars:
            height = bar.get_height()
            if abs(height - 1.0) > 1e-12:
                label = f"{height:.4f}".rstrip("0").rstrip(".")
                ax.annotate(label, (bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=11, rotation=45)
    ax.set_xticks(x)
    ax.set_xticklabels(frame["Attack_Type"], rotation=30, ha="right")
    ax.set_ylabel("Recall", fontsize=16)
    ax.set_title(f"Recall per Attack Type ({dataset})", fontsize=18, pad=18)
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=13)
    ax.set_ylim(0, 1.2)
    if legend:
        ax.legend(loc="upper right")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=300)
    plt.close(fig)
    frame.to_csv(destination.with_suffix(".csv"), index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=list(DATASETS), default="genis")
    parser.add_argument("--all", action="store_true", help="Render all five datasets")
    parser.add_argument("--output", type=Path, help="PNG path for a single dataset")
    parser.add_argument("--output-dir", type=Path, help="Override the per-dataset result directories")
    parser.add_argument("--legend", action="store_true", help="Show the otherwise hidden legend")
    args = parser.parse_args()
    if args.all and args.output:
        parser.error("Use --output-dir with --all")
    for key in DATASETS if args.all else [args.dataset]:
        name = DATASETS[key]
        frame = (load_genis() if key == "genis" else
                 pd.read_csv(ROOT / "plot_data/recall" / f"{name}_recall.csv"))
        folder = "UNSW_NB15" if key == "unsw" else name
        directory = args.output_dir or ROOT / "results" / folder
        output = args.output or directory / f"{name}_recall.png"
        plot_recall(frame, name, output, legend=args.legend)
        print(f"Saved: {output}")


if __name__ == "__main__":
    main()
