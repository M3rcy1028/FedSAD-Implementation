"""Build the requested final six-count GeNIS table from completed study runs."""
import argparse
import csv
import io
import json
import os
from pathlib import Path

COUNTS = (5, 10, 15, 20, 25, 30)
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
    states = {count: json.loads((root / f"clients_{count}/status.json").read_text())
              for count in COUNTS}
    failed = [count for count, state in states.items() if state["stage"] == "failed"]
    if failed:
        return {"stage": "failed", "clients": failed,
                "message": f"GeNIS client 비교 실험 중 {failed} clients 실험이 실패해 최종 비교표를 아직 완성할 수 없어. 상태 파일: {root}"}
    completed = [count for count, state in states.items() if state["stage"] == "complete"]
    if len(completed) != len(COUNTS):
        return {"stage": "waiting", "completed_clients": completed}
    config = json.loads((root / "experiment_config.json").read_text())
    assert config["server_rounds"] == 10 and config["client_epochs"] == 70
    rows = list(csv.DictReader((root / "client_comparison.csv").open()))
    rows = [row for row in rows if row["threshold_policy"] == POLICY
            and row["evaluation_set"] == "test" and int(row["round"]) == 10]
    # The collector may be updating its CSV as the last status changes.
    if len(rows) != len(COUNTS) or sorted(int(row["clients"]) for row in rows) != list(COUNTS):
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
    message = ("GeNIS의 **5·10·15·20·25·30 clients 실험이 모두 완료됐어.**\n\n"
               + render(rows) + "\n\n"
               "모두 로컬 70 epochs·서버 10 rounds·seed 123, 같은 테스트 8,746개(정상·공격 각 4,373개) 기준이야. "
               "임계값은 검증 macro F1으로 선택했고, Precision·Recall·F1-score는 공격(Anomaly) 클래스 기준이야.")
    markdown = root / "final_comparison_table.md"
    table = root / "final_comparison_table.csv"
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=[key for key, _ in COLUMNS])
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
