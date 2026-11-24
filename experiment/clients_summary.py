"""Build a GeNIS client-count table using the recorded study configuration."""
import argparse
import csv
import io
import json
import os
from pathlib import Path

POLICY = "validation_best_macro_f1"
COLUMNS = (
    ("clients", "Clients"), ("percentile", "Percentile"),
    ("accuracy", "Accuracy"), ("attack_precision", "Precision"),
    ("attack_recall", "Recall"), ("attack_f1", "F1-score"),
    ("macro_f1", "Macro F1"), ("roc_auc", "ROC-AUC"),
)


def atomic_text(path, text):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


def render(rows):
    lines = ["| " + " | ".join(label for _, label in COLUMNS) + " |",
             "| " + " | ".join("---:" for _ in COLUMNS) + " |"]
    for row in sorted(rows, key=lambda value: int(value["clients"])):
        values = [str(int(row[key])) if key == "clients" else
                  f"{float(row[key]):.2f}" if key == "percentile" else
                  f"{float(row[key]):.4f}" for key, _ in COLUMNS]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def collect(root):
    config = json.loads((root / "experiment_config.json").read_text())
    counts = tuple(config["client_counts"])
    epochs = config["client_epochs"]
    states = {count: json.loads((root / f"clients_{count}/status.json").read_text())
              for count in counts}
    failed = [count for count, state in states.items() if state["stage"] == "failed"]
    if failed:
        return {"stage": "failed", "clients": failed,
                "message": f"GeNIS client comparison: runs for {failed} clients failed, so the final table is incomplete. Status files: {root}"}
    completed = [count for count, state in states.items() if state["stage"] == "complete"]
    if len(completed) != len(counts):
        return {"stage": "waiting", "completed_clients": completed}
    assert config["server_rounds"] == 10 and epochs > 0
    rows = list(csv.DictReader((root / "client_comparison.csv").open()))
    rows = [row for row in rows if row["threshold_policy"] == POLICY
            and row["evaluation_set"] == "test" and int(row["round"]) == 10]
    # The collector may be updating its CSV as the last status changes.
    if len(rows) != len(counts) or sorted(int(row["clients"]) for row in rows) != list(counts):
        return {"stage": "waiting", "completed_clients": completed, "reason": "final CSV is being updated"}
    for row in rows:
        count = int(row["clients"])
        verification = json.loads((root / f"clients_{count}/run/previous/verification.json").read_text())
        assert verification["rounds_completed"] == 10
        assert verification["successful_client_fits"] == count * 10
        selected = json.loads((root / f"clients_{count}/selected_threshold.json").read_text())
        report = selected["test_metrics"]["classification_report"]
        assert report["Normal"]["support"] == report["Anomaly"]["support"] == 4373
        assert abs(float(row["accuracy"]) - report["accuracy"]) < 1e-12
        for source_key, metric in [("attack_precision", "precision"), ("attack_recall", "recall"),
                                   ("attack_f1", "f1-score")]:
            assert abs(float(row[source_key]) - report["Anomaly"][metric]) < 1e-12
    rows.sort(key=lambda row: int(row["clients"]))
    message = (f"GeNIS **{'/'.join(map(str, counts))}-client runs are complete.**\n\n"
               + render(rows) + "\n\n"
               f"All runs use {epochs} local epochs, 10 server rounds, seed 123 and the same 8,746 test flows (4,373 normal / 4,373 attack). "
               "Thresholds are selected by validation macro F1; precision, recall and F1-score are for the attack (Anomaly) class.")
    markdown = root / "final_comparison_table.md"
    table = root / "final_comparison_table.csv"
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=[key for key, _ in COLUMNS], lineterminator="\n")
    writer.writeheader()
    writer.writerows({key: row[key] for key, _ in COLUMNS} for row in rows)
    atomic_text(table, stream.getvalue())
    atomic_text(markdown, message + "\n")
    return {"stage": "complete", "message": message, "markdown": str(markdown), "csv": str(table)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mark-delivered", action="store_true")
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    if args.mark_delivered:
        path = root / "chat_notifications.json"
        config = json.loads(path.read_text())
        config["final_table_delivered"] = True
        atomic_text(path, json.dumps(config, indent=2) + "\n")
        return
    print(json.dumps(collect(root), ensure_ascii=False))


if __name__ == "__main__":
    main()
