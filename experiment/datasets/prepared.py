"""Reuse the tested GeNIS/Gotham preprocessing implementations."""
import json
import hashlib
from pathlib import Path

import joblib
import numpy as np

from experiment.backends import load
from experiment.artifacts import sha256


def prepare(config):
    dataset = config["dataset"]
    base = Path(config["data_dir"]) if config.get("data_dir") else (
        Path(config["data_root"]) / "GeNIS" if dataset == "genis" else Path("/data/SDP_Dataset/Gotham"))
    if config.get("prepared"):
        prepared = Path(config["prepared"]).resolve()
    else:
        source_key = hashlib.sha256(str(base.expanduser().resolve()).encode()).hexdigest()[:12]
        prepared = (Path(config["cache_root"]) / dataset / source_key /
                    f"flow{config['flow_interval']}_seed{config['seed']}_max{config['max_train_samples']}")
    if not (prepared / "prepared_data.npz").exists():
        if dataset == "genis":
            module = load("utils.genis")
            module.prepare_genis(base, prepared, flow_interval=config["flow_interval"],
                                 seed=config["seed"], max_train_samples=config["max_train_samples"])
        else:
            module = load("utils.gotham")
            module.prepare_gotham(base, prepared, seed=config["seed"],
                                  max_train_samples=config["max_train_samples"],
                                  eval_normal_samples=25000, client_nums=config["clients"])
    report = json.loads((prepared / "preprocessing_report.json").read_text())
    if report["seed"] != config["seed"]:
        raise ValueError("Prepared data seed differs from the requested seed")
    module = load("utils.genis" if dataset == "genis" else "utils.gotham")
    if report.get("preprocessor_sha256") != sha256(Path(module.__file__)):
        raise ValueError("Prepared data do not match the preserved preprocessing source")
    if dataset == "genis" and report["flow_interval_seconds"] != config["flow_interval"]:
        raise ValueError("Prepared data flow interval differs")
    if config["model"] != "taae":
        sequence = load("utils.genis_sequences")
        prepared = sequence.prepare_sequence_data(prepared, seed=config["seed"],
                   attack_fraction=config["attack_fraction"] if config.get("attack_fraction") is not None else .5,
                   split_mode="shared", max_normal_samples=config["max_train_samples"])
        prepared = Path(prepared)
        report = json.loads((prepared / "preprocessing_report.json").read_text())
    with np.load(prepared / "prepared_data.npz") as z:
        arrays = {key: z[key] for key in ["X_train", "X_val", "y_val", "X_test", "y_test"]}
        arrays["y_train"] = z["y_train"] if "y_train" in z else np.zeros(len(arrays["X_train"]), dtype=np.int64)
    with np.load(prepared / "source_rows.npz") as z:
        for key in ["train_ids", "val_ids", "test_ids", "test_labels"]:
            arrays[key] = z[key]
    report = {**report, "prepared_data": str(prepared),
              "prepared_data_sha256": sha256(prepared / "prepared_data.npz"),
              "validation": "independent held-out connection/window groups"}
    return arrays, joblib.load(prepared / "preprocessing.joblib"), report
