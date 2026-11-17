"""Run the existing GeNIS RNEP experiment at the configured client counts."""
import argparse
import ast
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[3]
BACKEND = ROOT / "experiment/core"
COUNTS = (5, 10, 15, 20, 25, 30)


def now():
    return datetime.now(ZoneInfo("Asia/Seoul")).isoformat()


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, default=str) + "\n")
    os.replace(temp, path)


def read_rounds(path):
    if not path.exists():
        return []
    lines = path.read_text().splitlines()
    records = []
    for index, line in enumerate(lines):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise
    return records


def metrics_row(metrics):
    report = metrics["classification_report"]
    return {"accuracy": report["accuracy"], "macro_f1": report["macro avg"]["f1-score"],
            "normal_precision": report["Normal"]["precision"],
            "normal_recall": report["Normal"]["recall"],
            "normal_f1": report["Normal"]["f1-score"],
            "attack_precision": report["Anomaly"]["precision"],
            "attack_recall": report["Anomaly"]["recall"],
            "attack_f1": report["Anomaly"]["f1-score"], "roc_auc": metrics["roc_auc"],
            "threshold": metrics["threshold"]}


def collect_rounds(folder, count, percentile):
    import pandas as pd
    rows = []
    worker = folder / "run/previous"
    for record in read_rounds(worker / "round_validation_metrics.jsonl"):
        row = {"clients": count, "round": record["round"], "evaluation_set": "validation",
               "threshold_policy": "fixed_training_percentile", "percentile": percentile,
               **metrics_row(record)}
        rows.append(row)
        if record["round"]:
            round_dir = folder / "rounds" / f"round_{record['round']:02d}"
            dump(round_dir / "validation_metrics.json", {**row, "confusion_matrix": record["confusion_matrix"]})
            checkpoint = worker / "checkpoints" / f"round_{record['round']:02d}.h5"
            link = round_dir / "checkpoint.h5"
            if checkpoint.exists() and not link.exists():
                link.symlink_to(os.path.relpath(checkpoint, round_dir))
    if rows:
        pd.DataFrame(rows).to_csv(folder / "round_metrics.csv", index=False)
    return rows


def collect_final(folder, count, percentile):
    import pandas as pd
    selected = folder / "selected_threshold.json"
    baseline = folder / "run/previous/baseline_metrics.json"
    if not selected.exists() or not baseline.exists():
        return []
    chosen = json.loads(selected.read_text())
    fixed = json.loads(baseline.read_text())
    rows = [
        {"clients": count, "round": 10, "evaluation_set": "test",
         "threshold_policy": "validation_best_macro_f1", "percentile": chosen["percentile"],
         **metrics_row(chosen["test_metrics"])},
        {"clients": count, "round": 10, "evaluation_set": "test",
         "threshold_policy": "fixed_training_percentile", "percentile": percentile,
         **metrics_row(fixed)},
    ]
    pd.DataFrame(rows).to_csv(folder / "metrics_summary.csv", index=False)
    return rows


def collect_results(destination, config, plots=False):
    import pandas as pd
    round_rows, final_rows = [], []
    for count in config["client_counts"]:
        folder = destination / f"clients_{count}"
        round_rows.extend(collect_rounds(folder, count, config["percentile"]))
        final_rows.extend(collect_final(folder, count, config["percentile"]))
    round_frame, final_frame = pd.DataFrame(round_rows), pd.DataFrame(final_rows)
    if round_rows:
        round_frame.to_csv(destination / "round_comparison.csv", index=False)
    if final_rows:
        final_frame.to_csv(destination / "client_comparison.csv", index=False)
    if plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if round_rows:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
            for count, group in round_frame.groupby("clients"):
                for ax, metric, title in zip(axes, ["accuracy", "macro_f1"],
                                             ["Validation accuracy", "Validation macro F1"]):
                    ax.plot(group["round"], group[metric], marker="o", markersize=3,
                            label=f"{count} clients")
                    ax.set(title=title, xlabel="Server round", ylabel="Score", ylim=(0, 1.02))
                    ax.grid(alpha=.25)
                    ax.legend()
                one, ax = plt.subplots(figsize=(7, 4.5))
                for metric, label in [("accuracy", "Accuracy"), ("macro_f1", "Macro F1"),
                                       ("normal_recall", "Normal recall"), ("attack_recall", "Attack recall")]:
                    ax.plot(group["round"], group[metric], marker="o", markersize=3, label=label)
                ax.set(title=f"GeNIS: {count} clients (validation, percentile {config['percentile']:g})",
                       xlabel="Server round", ylabel="Score", ylim=(0, 1.02))
                ax.grid(alpha=.25)
                ax.legend()
                one.tight_layout()
                one.savefig(destination / f"clients_{count}/round_metrics.png", dpi=300)
                plt.close(one)
            fig.suptitle(f"GeNIS client-count comparison (fixed percentile {config['percentile']:g})")
            fig.tight_layout()
            fig.savefig(destination / "round_comparison.png", dpi=300)
            plt.close(fig)
        if final_rows:
            fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
            for policy, group in final_frame.groupby("threshold_policy"):
                label = "Validation-selected threshold" if policy.startswith("validation") else f"Fixed percentile {config['percentile']:g}"
                for ax, metric, title in zip(axes, ["accuracy", "macro_f1", "attack_f1"],
                                             ["Test accuracy", "Test macro F1", "Test attack F1"]):
                    ax.plot(group["clients"], group[metric], marker="o", label=label)
                    ax.set(title=title, xlabel="Number of clients", ylabel="Score", ylim=(0, 1.02))
                    ax.set_xticks(config["client_counts"])
                    ax.grid(alpha=.25)
                    ax.legend(fontsize=8)
            fig.suptitle("GeNIS: final results after 10 server rounds")
            fig.tight_layout()
            fig.savefig(destination / "client_comparison.png", dpi=300)
            plt.close(fig)
    return len(round_rows), len(final_rows)


def prepare(destination):
    import numpy as np
    reference = json.loads((ROOT / "results/GeNIS/taae/model_config.json").read_text())
    args = reference["arguments"]
    prepared = Path(reference["prepared_data"])
    data_hash = digest(prepared / "prepared_data.npz")
    if data_hash != reference["prepared_data_sha256"]:
        raise ValueError("The existing GeNIS split changed after the reference experiment")
    report = json.loads((prepared / "preprocessing_report.json").read_text())
    if report["preprocessor_sha256"] != digest(BACKEND / "utils/genis.py"):
        raise ValueError("GeNIS preprocessing code differs from the prepared split")
    partitions = {}
    with np.load(prepared / "prepared_data.npz") as z:
        for count in COUNTS:
            lengths = [len(chunk) for chunk in np.array_split(z["X_train"], count)]
            if min(lengths) < 2:
                raise ValueError("Too few training rows per client")
            partitions[str(count)] = lengths
        shapes = {key: list(z[key].shape) for key in z.files}
        for split in ["val", "test"]:
            if np.bincount(z[f"y_{split}"]).tolist()[0] != np.bincount(z[f"y_{split}"]).tolist()[1]:
                raise ValueError("Expected balanced evaluation data")
    with np.load(prepared / "source_rows.npz") as z:
        for key in ["ids", "groups"]:
            sets = [set(z[f"{part}_{key}"]) for part in ["train", "val", "test"]]
            if any(sets[i] & sets[j] for i, j in [(0, 1), (0, 2), (1, 2)]):
                raise ValueError(f"Overlapping evaluation {key}")
    source_files = next(ast.literal_eval(node.value) for node in
                        ast.parse((BACKEND / "utils/genis_experiment.py").read_text()).body
                        if isinstance(node, ast.Assign) and
                        any(isinstance(target, ast.Name) and target.id == "SOURCE_FILES" for target in node.targets))
    snapshot = destination / "source_snapshot"
    hashes = {}
    for name in source_files:
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(BACKEND / name, target)
        hashes[name] = digest(target)
    config = {"dataset": "GeNIS", "client_counts": list(COUNTS), "server_rounds": 10,
              "taae_version": "previous", "latent_dim": reference["latent_dim"],
              "client_epochs": args["client_epochs"], "batch_size": args["batch_size"],
              "learning_rate": args["learning_rate"], "dropout_rate": args["dropout_rate"],
              "random_seed": args["random_seed"], "percentile": args["percentile"],
              "gpu_per_client": args["gpu_per_client"], "flow_interval": args["flow_interval"],
              "prepared_data": str(prepared), "prepared_data_sha256": data_hash,
              "source_snapshot": str(snapshot), "source_sha256": hashes, "data_shapes": shapes,
              "client_training_rows": partitions, "created_at": now(),
              "partition": "same shuffled normal rows split with numpy.array_split",
              "round_evaluation": "validation only, fixed normal training percentile",
              "final_threshold_selection": "validation macro F1; test used only for final reporting",
              "reference_model_config": str(ROOT / "results/GeNIS/taae/model_config.json")}
    for count in COUNTS:
        folder = destination / f"clients_{count}"
        folder.mkdir()
        dump(folder / "status.json", {"stage": "queued", "clients": count})
    dump(destination / "experiment_config.json", config)
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT / "results/clients")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--wait-lock", action="store_true",
                        help="Wait for the current sweep, then resume any queued client counts")
    args = parser.parse_args()
    destination = args.output_root.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    lock = (destination / ".sweep.lock").open("a")
    if args.wait_lock:
        print(f"[{now()}] Waiting for the existing sweep lock: {destination}", flush=True)
    fcntl.flock(lock, fcntl.LOCK_EX | (0 if args.wait_lock else fcntl.LOCK_NB))
    config_path = destination / "experiment_config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else prepare(destination)
    if args.collect_only:
        collect_results(destination, config, plots=True)
        return
    if args.plan_only:
        print(json.dumps(config, indent=2), flush=True)
        return
    for key, value in {"MPLBACKEND": "Agg", "PYTHONUNBUFFERED": "1",
                       "PYTHONDONTWRITEBYTECODE": "1", "TF_FORCE_GPU_ALLOW_GROWTH": "true",
                       "TF_NUM_INTRAOP_THREADS": "2", "TF_NUM_INTEROP_THREADS": "1",
                       "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "1",
                       "RAY_worker_niceness": "0", "CUDA_VISIBLE_DEVICES": "0,1,2"}.items():
        os.environ.setdefault(key, value)
    status = {"stage": "running", "started_at": now(), "completed_clients": []}
    dump(destination / "status.json", status)
    try:
        for count in config["client_counts"]:
            folder = destination / f"clients_{count}"
            worker = folder / "run/previous"
            if (worker / "verification.json").exists() and (folder / "GeNIS_metrics.json").exists():
                status["completed_clients"].append(count)
                continue
            if (folder / "run").exists():
                raise FileExistsError(f"An incomplete run already exists at {folder / 'run'}; inspect its logs before restarting")
            for name, expected in config["source_sha256"].items():
                if digest(Path(config["source_snapshot"]) / name) != expected:
                    raise ValueError(f"Frozen experiment source changed: {name}")
            if digest(Path(config["prepared_data"]) / "prepared_data.npz") != config["prepared_data_sha256"]:
                raise ValueError("The shared prepared data changed")
            command = [sys.executable, "-u", str(Path(config["source_snapshot"]) / "train_taae.py"),
                       "--eval_dataset", "genis", "--taae_version", "previous",
                       "--client_nums", str(count), "--server_rounds", "10",
                       "--genis_prepared", config["prepared_data"],
                       "--genis_latent_dim", str(config["latent_dim"]),
                       "--output_dir", str(folder / "run"), "--result_path", str(folder)]
            for name in ["client_epochs", "batch_size", "learning_rate", "dropout_rate",
                         "random_seed", "percentile", "gpu_per_client", "flow_interval"]:
                command.extend([f"--{name}", str(config[name])])
            dump(folder / "launch_config.json", {"command": command, "created_at": now()})
            print(f"[{now()}] Starting {count} clients, 10 rounds, {config['client_epochs']} local epochs", flush=True)
            start = time.monotonic()
            with (folder / "driver.log").open("w") as log:
                process = subprocess.Popen(command, cwd=config["source_snapshot"], stdout=log,
                                           stderr=subprocess.STDOUT)
                current = {"stage": "training", "clients": count, "pid": process.pid,
                           "started_at": now(), "completed_rounds": 0}
                dump(folder / "status.json", current)
                status.update(active_clients=count, child_pid=process.pid)
                last_rows = -1
                while process.poll() is None:
                    rounds = read_rounds(worker / "round_validation_metrics.jsonl")
                    current["completed_rounds"] = max((row["round"] for row in rounds), default=0)
                    current["elapsed_seconds"] = round(time.monotonic() - start, 1)
                    current["updated_at"] = now()
                    dump(folder / "status.json", current)
                    dump(destination / "status.json", {**status, **{"active_run": current}})
                    if len(rounds) != last_rows:
                        collect_results(destination, config, plots=True)
                        print(f"[{now()}] {count} clients: round {current['completed_rounds']}/10", flush=True)
                        last_rows = len(rounds)
                    time.sleep(30)
                code = process.returncode
            if code:
                raise RuntimeError(f"{count}-client run failed with exit code {code}; see {folder / 'driver.log'}")
            verification = json.loads((worker / "verification.json").read_text())
            if verification["rounds_completed"] != 10 or verification["successful_client_fits"] != count * 10:
                raise ValueError("Incomplete requested client fits or rounds")
            collect_results(destination, config, plots=True)
            dump(folder / "status.json", {"stage": "complete", "clients": count,
                 "completed_rounds": 10, "elapsed_seconds": round(time.monotonic() - start, 1),
                 "finished_at": now(), "verification": verification})
            status["completed_clients"].append(count)
            print(f"[{now()}] Finished {count} clients", flush=True)
        collect_results(destination, config, plots=True)
        status.update(stage="complete", finished_at=now(), active_clients=None, child_pid=None)
        dump(destination / "status.json", status)
        print(f"All {len(config['client_counts'])} client-count experiments completed: {destination}", flush=True)
    except Exception as error:
        status.update(stage="failed", error=str(error), updated_at=now())
        dump(destination / "status.json", status)
        print(f"FAILED: {error}", flush=True)
        raise


if __name__ == "__main__":
    main()
