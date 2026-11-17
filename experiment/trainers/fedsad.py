"""FedSAD adapter: GSAD normal statistics for the chosen window, then GSAD + TAAE fusion.

TAAE itself is trained with `run.py --model taae`; FedSAD reuses those weights
(by default the archived weights in results/) and only fits the GSAD statistics.
"""
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

from experiment.artifacts import dump

UNIFIED = Path("/data/SDP_Dataset/Unified_model")
DATASETS = {
    "cic2018": {"key": "cic", "graph_dir": UNIFIED / "cic_graph", "normal_file": "CIC_ae_normal.csv"},
    "unsw-nb15": {"key": "unsw", "graph_dir": UNIFIED / "unsw_graph", "normal_file": "UNSW_NB15_normal.csv"},
    "genis": {"key": "genis_frame", "graph_dir": Path("/data/SDP_Dataset/GeNIS/processed/genis_frame_graph"),
              "normal_file": "GeNIS_frame_normal.csv"},
}


def train(config):
    import json
    from experiment.backends import CORE, load
    spec = DATASETS[config["dataset"]]
    output = Path(config["output"])
    tables, figures, weights = output / "tables", output / "figures", output / "weights"
    # --data-dir points to the folder that holds <name>_rnep/ and <name>_graph/ (CIC2018/UNSW-NB15)
    base = Path(config["data_dir"]) if config.get("data_dir") else UNIFIED
    graph_dir = base / spec["graph_dir"].name if config.get("data_dir") else spec["graph_dir"]
    stats = weights / "normal_stats.pkl"
    print(f"[FedSAD] GSAD statistics: window={config['time_window']}, data={graph_dir}", flush=True)
    load("gsad").train_single_model(SimpleNamespace(time_window=config["time_window"]),
                                    str(graph_dir / spec["normal_file"]), str(stats))
    command = [sys.executable, "-u", str(CORE / "fedsad.py"), "--eval_dataset", spec["key"],
               "--base_data_path", str(base), "--time_window", config["time_window"],
               "--gsad_model", str(stats), "--result_path", str(tables),
               "--threshold_std", str(config["threshold_std"]), "--episode_gap", config["episode_gap"]]
    if config.get("taae_weights"):
        command += ["--taae_model", str(config["taae_weights"])]
    dump(output / "metadata/fedsad_command.json", {"command": command})
    # Relative TAAE weight paths in fedsad.py point into the project's results/
    subprocess.run(command, cwd=config["project_root"], check=True)
    if (tables / "fedsad_cm.png").exists():
        shutil.move(str(tables / "fedsad_cm.png"), str(figures / "confusion_matrix.png"))
    metrics = json.loads((tables / "window_metrics.json").read_text())
    return {"classification_report": {"accuracy": metrics["accuracy"]}, **metrics}
