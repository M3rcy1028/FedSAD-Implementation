"""Dataset adaptation and storage of the exact arrays used by an experiment."""
from pathlib import Path
import joblib
import numpy as np

from experiment.artifacts import dump, sha256


def prepare(config):
    if config["dataset"] in {"genis", "gotham"}:
        from experiment.datasets.prepared import prepare as loader
    elif config["dataset"] == "itu-ulak":
        return prepare_itu(config)
    else:
        from experiment.datasets.tabular import prepare as loader
    bundle, pipeline, report = loader(config)
    return store(config, bundle, pipeline, report)


def store(config, bundle, pipeline, report):
    runtime, output = Path(config["runtime"]), Path(config["output"])
    # Legacy numeric datasets use exactly the existing client-side validation tail.
    # Its rows are not fitted by that client's optimizer and never enter final test.
    if "X_val" not in bundle:
        parts = np.array_split(np.arange(len(bundle["X_train"])), config["clients"])
        fraction = .1 if config["model"] == "cnnlstm" else .2
        indices = np.concatenate([part[int(len(part) * (1 - fraction)):] for part in parts])
        bundle["X_val"], bundle["y_val"] = bundle["X_train"][indices], bundle["y_train"][indices]
        bundle["val_ids"] = bundle["train_ids"][indices]
    if min(len(bundle["X_train"]) // config["clients"], len(bundle["X_val"]), len(bundle["X_test"])) < 2:
        raise ValueError("Too few rows for the requested client count")
    if set(bundle["train_ids"].astype(str)) & set(bundle["test_ids"].astype(str)):
        raise ValueError("Training input rows overlap the final test")
    if not all(np.isfinite(bundle[key]).all() for key in ["X_train", "X_val", "X_test"]):
        raise ValueError("Nonfinite model inputs")
    path = runtime / "prepared_data.npz"
    np.savez_compressed(path, **bundle)
    joblib.dump(pipeline, output / "preprocessing.joblib")
    report.update(input_shape=list(bundle["X_train"].shape[1:]),
                  training_rows=len(bundle["X_train"]), validation_rows=len(bundle["X_val"]),
                  test_rows=len(bundle["X_test"]),
                  normal_test_rows=int((bundle["y_test"] == 0).sum()),
                  attack_test_rows=int((bundle["y_test"] == 1).sum()))
    dump(output / "metadata/data.json", report)
    config.update(prepared_cache=str(path), prepared_cache_sha256=sha256(path),
                  input_shape=report["input_shape"], independent_validation="independent" in report["validation"])
    dump(output / "config.json", config)
    return bundle


def prepare_itu(config):
    from experiment.backends import load
    folder = Path(config["data_dir"]) if config.get("data_dir") else Path(config["data_root"]) / "ITU_SDN_ULAK/raw"
    train, test, y_test, metadata = load("taae_current").get_datasets_zenodo(
        folder, random_seed=config["seed"], max_train_samples=config["max_train_samples"],
        balance_test=True, return_metadata=True)
    ids = metadata["test_source_rows"]
    bundle = {"X_train": train, "y_train": np.zeros(len(train), dtype=np.int64),
              "X_test": test, "y_test": y_test.astype(np.int64),
              "train_ids": np.array([f"Train_ULAK.csv:{i}" for i in metadata["train_source_rows"]]),
              "test_ids": np.array([f"Test_ULAK.csv:{i}" for i in ids]),
              "test_labels": np.where(y_test == 0, "Normal", "Anomaly")}
    report = {**metadata["report"], "validation": "per-client internal validation tail"}
    return store(config, bundle, metadata, report)
