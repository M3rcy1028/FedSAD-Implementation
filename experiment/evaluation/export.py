"""Final evaluation, threshold provenance and standard outputs."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

from experiment.artifacts import dump
from experiment.evaluation.metrics import report, score, choose_threshold


def save_predictions(config, labels, scores, predicted, source_ids, attacks, errors=None):
    output = Path(config["output"])
    frame = pd.DataFrame({"source_id": source_ids, "attack": attacks, "y_true": labels,
                          "score": scores, "y_pred": predicted})
    if errors is not None:
        frame["reconstruction_error"] = errors
    frame.to_csv(output / "tables/predictions.csv", index=False)
    frame[["y_true", "score"]].to_csv(output / "tables/roc_data.csv", index=False)
    per_attack = []
    for name in sorted(set(attacks[labels == 1])):
        mask = (labels == 1) & (attacks == name)
        per_attack.append({"attack": name, "support": int(mask.sum()),
                           "detected": int(predicted[mask].sum()), "recall": float(predicted[mask].mean())})
    pd.DataFrame(per_attack).to_csv(output / "tables/per_attack.csv", index=False)
    from experiment.plotting.figures import render
    render(output, frame, config)


def evaluate(model, bundle, config):
    output = Path(config["output"])
    kind = config["model"]
    train_scores, train_errors = score(model, bundle["X_train"][bundle["y_train"] == 0], kind)
    val_scores, _ = score(model, bundle["X_val"], kind)
    threshold, selection = choose_threshold(config, train_scores, val_scores, bundle["y_val"])
    dump(output / "metadata/threshold.json", selection)
    test_scores, errors = score(model, bundle["X_test"], kind)
    metrics = report(bundle["y_test"], test_scores, threshold)
    metrics.update(dataset=config["dataset"], model=kind, threshold_selection=selection,
                   score_kind="reconstruction_error" if kind == "taae" else "classifier_probability")
    dump(output / "metadata/metrics.json", metrics)
    if errors is not None:
        reconstruction_threshold = threshold if kind == "taae" else float(np.percentile(train_errors, config["percentile"]))
        dump(output / "metadata/reconstruction_threshold.json", {"threshold": reconstruction_threshold,
             "percentile": selection.get("percentile", config["percentile"]), "selection_set": "normal training errors"})
        if kind == "aelstm":
            dump(output / "metadata/reconstruction_metrics.json", report(bundle["y_test"], errors, reconstruction_threshold))
    rows = []
    for name in ["Normal", "Anomaly", "macro avg", "weighted avg"]:
        rows.append({"class": name, **metrics["classification_report"][name]})
    pd.DataFrame(rows).to_csv(output / "tables/metrics.csv", index=False)
    save_predictions(config, bundle["y_test"], test_scores, (test_scores > threshold).astype(int),
                     bundle["test_ids"], bundle["test_labels"], errors)
    summary = metrics["classification_report"]["Anomaly"]
    print(f"FINAL {kind}/{config['dataset']}: Precision={summary['precision']:.4f} "
          f"Recall={summary['recall']:.4f} F1={summary['f1-score']:.4f} "
          f"Accuracy={metrics['classification_report']['accuracy']:.4f}", flush=True)
    return metrics
