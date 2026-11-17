"""Numeric CSV adapters preserving legacy split, scaling and sequence recipes."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle

from experiment.artifacts import sha256

FILES = {"kdd99": ("KDD99_normal.csv", "KDD99_anomaly.csv"),
         "nsl-kdd": ("KDD_normal.csv", "KDD_anomaly.csv"),
         "insdn": ("InSDN_normal.csv", "InSDN_anomaly.csv"),
         "cic2018": ("CIC_ae_normal.csv", "CIC_ae_anomaly.csv"),
         "unsw-nb15": ("UNSW_NB15_normal.csv", "UNSW_NB15_anomaly.csv")}


def directory(config):
    if config.get("data_dir"):
        return Path(config["data_dir"]).expanduser().resolve()
    base = Path(config["data_root"])
    return {"kdd99": base / "KDD99", "nsl-kdd": base / "NSL-KDD",
            "insdn": base / "InSDN/attack_data",
            "cic2018": Path("/data/SDP_Dataset/Unified_model/cic_rnep"),
            "unsw-nb15": Path("/data/SDP_Dataset/Unified_model/unsw_rnep")}[config["dataset"]]


def sequence(values, shape, preserve_truncation=False):
    width = int(np.prod(shape))
    if values.shape[1] > width:
        if not preserve_truncation:
            raise ValueError("Sequence shape would truncate input features")
        values = values[:, :width]
    padded = np.pad(values, ((0, 0), (0, width - values.shape[1])))
    return padded.reshape(len(values), *shape).astype(np.float32)


def attack_names(folder, normal, attacks, dataset):
    """Subtype labels are metadata only; ambiguous feature matches remain explicit."""
    names = np.full(len(attacks), "Anomaly", dtype=object)
    coverage = {"source": "binary CSV", "known_rows": 0, "ambiguous_rows": 0}
    encoded = folder / "KDD99_encoded.csv"
    if dataset == "kdd99" and encoded.exists():
        frame = pd.read_csv(encoded, low_memory=False)
        label = next((key for key in ["label", "Label"] if key in frame), None)
        if label and set(normal.columns).issubset(frame.columns):
            frame = frame[~frame[label].astype(str).str.lower().isin(["normal", "normal."])]
            values = frame[normal.columns].astype(np.float64)
            keys = pd.util.hash_pandas_object(values, index=False)
            groups = pd.DataFrame({"key": keys.to_numpy(), "label": frame[label].to_numpy()}).groupby("key").label
            unique = groups.nunique()
            mapping = groups.first()
            mapping.loc[unique > 1] = "Ambiguous attack"
            matched = pd.util.hash_pandas_object(attacks.astype(np.float64), index=False).map(mapping)
            names = matched.fillna("Anomaly").to_numpy(dtype=str)
            coverage = {"source": str(encoded), "sha256": sha256(encoded),
                        "known_rows": int(matched.notna().sum()),
                        "ambiguous_rows": int((names == "Ambiguous attack").sum())}
    return names.astype(str), coverage


def prepare(config):
    folder = directory(config)
    normal_name, attack_name = FILES[config["dataset"]]
    paths = [folder / normal_name, folder / attack_name]
    frames = [pd.read_csv(path, low_memory=False) for path in paths]
    features = [key for key in frames[0].columns if key.lower() not in {"label", "attack", "attack_cat"}]
    if any(not set(features).issubset(frame.columns) for frame in frames):
        raise ValueError("Normal/attack feature schemas differ")
    if config["dataset"] == "kdd99" and frames[0].columns.tolist() != frames[1].columns.tolist():
        raise ValueError("Use the matching 115-feature KDD99 normal/anomaly CSVs")
    strict_numeric = config["dataset"] in {"kdd99", "nsl-kdd"}
    normal, attacks = [frame[features].apply(pd.to_numeric, errors="raise" if strict_numeric else "coerce")
                       for frame in frames]
    if config["dataset"] not in {"kdd99", "nsl-kdd"}:
        normal, attacks = [frame.replace([np.inf, -np.inf], np.nan).fillna(0)
                           for frame in [normal, attacks]]
        if config["dataset"] == "cic2018" and config["model"] == "taae":
            normal, attacks = [frame.clip(lower=0) for frame in [normal, attacks]]
    if not all(np.isfinite(frame.to_numpy()).all() for frame in [normal, attacks]):
        raise ValueError("Nonfinite numeric data; use the cleaned dataset CSVs")
    names, subtype_report = attack_names(folder, normal, attacks, config["dataset"])
    ratio = config.get("normal_train_ratio")
    if ratio is None:
        ratio = .5 if (config["dataset"] in {"kdd99", "nsl-kdd"} or
                       (config["dataset"] == "insdn" and config["model"] == "aelstm")) else .8
    log_cols = []
    normal = shuffle(normal, random_state=config["seed"])
    cut = int(len(normal) * ratio)
    normal_train, normal_test = normal.iloc[:cut], normal.iloc[cut:]
    scaler = MinMaxScaler().fit(normal_train)
    train = scaler.transform(normal_train)
    attack_order = np.arange(len(attacks))
    if config["dataset"] == "cic2018" and config["model"] == "taae":
        attack_order = shuffle(attack_order, random_state=config["seed"])
        attacks, names = attacks.iloc[attack_order], names[attack_order]
    normal_test_values, attack_values = scaler.transform(normal_test), scaler.transform(attacks)
    y_train = np.zeros(len(train), dtype=np.int64)
    train_ids = np.array([f"{normal_name}:{index}" for index in normal_train.index])
    attack_fraction = config.get("attack_fraction")
    if attack_fraction is None:
        if config["model"] == "aelstm":
            attack_fraction = .2 if config["dataset"] in {"nsl-kdd", "cic2018"} else .05
        else:
            attack_fraction = .4 if config["dataset"] == "insdn" else .2
    attack_cut = int(len(attacks) * attack_fraction) if config["model"] != "taae" else 0
    if attack_cut:
        train = np.concatenate([train, attack_values[:attack_cut]])
        y_train = np.r_[y_train, np.ones(attack_cut, dtype=np.int64)]
        train_ids = np.r_[train_ids, [f"{attack_name}:{i}" for i in attack_order[:attack_cut]]]
        train, y_train, train_ids = shuffle(train, y_train, train_ids, random_state=config["seed"])
    test = np.concatenate([normal_test_values, attack_values[attack_cut:]])
    y_test = np.r_[np.zeros(len(normal_test_values), dtype=np.int64),
                   np.ones(len(attacks) - attack_cut, dtype=np.int64)]
    test_ids = np.r_[[f"{normal_name}:{index}" for index in normal_test.index],
                     [f"{attack_name}:{i}" for i in attack_order[attack_cut:]]]
    labels = np.r_[np.full(len(normal_test_values), "Normal"), names[attack_cut:]]
    test, y_test, test_ids, labels = shuffle(test, y_test, test_ids, labels, random_state=config["seed"])
    if set(train_ids) & set(test_ids):
        raise ValueError("Training rows overlap final test rows")
    if config["model"] != "taae":
        shapes = {"kdd99": (10, 13 if config["model"] == "aelstm" else 12),
                  "nsl-kdd": (10, 12), "insdn": (12, 7), "cic2018": (10, 8),
                  "unsw-nb15": (6, 7)}
        shape = shapes[config["dataset"]]
        legacy_truncation = config["dataset"] == "unsw-nb15"
        train, test = sequence(train, shape, legacy_truncation), sequence(test, shape, legacy_truncation)
    pipeline = {"scaler": scaler, "feature_order": features, "log1p_columns": log_cols,
                "normal_train_ratio": ratio, "seed": config["seed"]}
    report = {"dataset": config["dataset"], "input_features": len(features), "normal_train_ratio": ratio,
              "training_rows": len(train), "normal_training_rows": int((y_train == 0).sum()),
              "attack_training_rows": int((y_train == 1).sum()),
              "attack_train_fraction": attack_fraction if config["model"] != "taae" else 0,
              "test_rows": len(test), "normal_test_rows": int((y_test == 0).sum()),
              "attack_test_rows": int((y_test == 1).sum()), "raw_sha256": {str(p): sha256(p) for p in paths},
              "validation": "per-client internal validation tail; no final test rows used",
              "threshold_selection": "fixed normal training-error percentile",
              "subtype_labels": subtype_report, "test_row_overlap": False}
    if config["model"] != "taae":
        report["sequence_shape"] = list(shape)
        report["model_feature_order"] = features[:int(np.prod(shape))]
        report["source_feature_truncation_preserved"] = len(features) > int(np.prod(shape))
    return {"X_train": train, "y_train": y_train, "train_ids": train_ids,
            "X_test": test, "y_test": y_test, "test_ids": test_ids, "test_labels": labels}, pipeline, report

