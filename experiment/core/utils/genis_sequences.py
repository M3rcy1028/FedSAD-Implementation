"""GeNIS supervised inputs for AE-LSTM/CNN-LSTM, with group-safe holdouts."""
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from utils.genis import file_sha256, split_groups, balance

DEFAULT_PREPARED = Path.home() / "IDS_Datasets/GeNIS/processed/scenarios10_seed123_group_holdout_capped_v2"


def capped_attack_sample(frame, budget, seed):
    if budget <= 0:
        return frame.iloc[:0].copy()
    if len(frame) <= budget:
        return frame.sample(frac=1, random_state=seed)
    counts = frame.SubCategoryLabel.value_counts().sort_index()
    lo, hi = 0, budget
    while lo < hi:
        cap = (lo + hi + 1) // 2
        if np.minimum(counts.to_numpy(), cap).sum() <= budget:
            lo = cap
        else:
            hi = cap - 1
    amounts = np.minimum(counts.to_numpy(), lo)
    eligible = np.flatnonzero(counts.to_numpy() > amounts)
    remainder = budget - int(amounts.sum())
    amounts[np.random.default_rng(seed).permutation(eligible)[:remainder]] += 1
    return pd.concat([frame[frame.SubCategoryLabel == label].sample(n=int(n), random_state=seed + i)
                      for i, (label, n) in enumerate(zip(counts.index, amounts))])


def build_sequence_data(prepared, seed=123, attack_fraction=.5, max_normal_samples=None,
                        timesteps=10, split_mode="shared"):
    if not 0 <= attack_fraction < 1 or timesteps < 2 or split_mode not in {"shared", "new"}:
        raise ValueError("Invalid sequence split, timesteps, or training attack fraction")
    prepared = Path(prepared)
    base_report = json.loads((prepared / "preprocessing_report.json").read_text())
    base_pipeline = joblib.load(prepared / "preprocessing.joblib")
    with np.load(prepared / "source_rows.npz") as z:
        source = {key: z[key] for key in z.files}
    with np.load(prepared / "prepared_data.npz") as z:
        base_y_val, base_y_test = z["y_val"], z["y_test"]
    numeric, categorical = base_pipeline["numeric_features"], base_pipeline["categorical_features"]
    train_ids = source["train_ids"]
    if max_normal_samples is not None:
        train_ids = train_ids[:max_normal_samples]
    wanted = set(train_ids) | set(source["val_ids"]) | set(source["test_ids"])
    schema = ["FlowID", "StartTime", "LastTime", "BinaryLabel", "CategoryLabel", "SubCategoryLabel"]
    frames, seen, raw_hashes = [], set(), {}
    for name, expected_hash in base_report["source_sha256"].items():
        path = Path(name)
        if file_sha256(path) != expected_hash:
            raise ValueError(f"GeNIS raw CSV changed since the base split: {path}")
        raw_hashes[name] = expected_hash
        frame = pd.read_csv(path, usecols=schema + numeric + categorical, low_memory=False)
        frame["source_id"] = [f"{path.name}:{i}" for i in range(len(frame))]
        frame["BinaryLabel"] = pd.to_numeric(frame.BinaryLabel, errors="raise").astype(np.int64)
        for column in numeric:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(np.float64)
        frame[numeric] = frame[numeric].replace([np.inf, -np.inf], np.nan)
        for column in categorical:
            frame[column] = frame[column].fillna("__MISSING__").astype(str)
        hashes = pd.util.hash_pandas_object(frame[schema + numeric + categorical], index=False).to_numpy()
        keep = ~np.isin(hashes, np.fromiter(seen, dtype=np.uint64)) if seen else np.ones(len(frame), dtype=bool)
        keep |= frame.source_id.isin(wanted).to_numpy()
        seen.update(hashes.tolist())
        frame = frame.loc[keep].copy()
        frame["group_id"] = frame.FlowID.astype(str)
        portless = frame.Proto.isin(["icmp", "ipv6-icmp", "arp"])
        times = pd.to_numeric(frame.loc[portless, "StartTime"], errors="coerce")
        if times.isna().any():
            times = pd.to_timedelta(frame.loc[portless, "StartTime"].astype(str)).dt.total_seconds()
        frame.loc[portless, "group_id"] += "|window=" + (times // base_report["flow_interval_seconds"]).astype(np.int64).astype(str)
        frames.append(frame)
    raw = pd.concat(frames, ignore_index=True).set_index("source_id", drop=False)
    normal_train = raw.loc[train_ids].copy()
    if not (normal_train.BinaryLabel == 0).all():
        raise ValueError("The base normal training IDs contain attacks")
    attacks = raw[raw.BinaryLabel == 1]
    if split_mode == "shared":
        val, test = raw.loc[source["val_ids"]].copy(), raw.loc[source["test_ids"]].copy()
        np.testing.assert_array_equal(val.BinaryLabel.to_numpy(), base_y_val)
        np.testing.assert_array_equal(test.BinaryLabel.to_numpy(), base_y_test)
        held_groups = set(source["val_groups"]) | set(source["test_groups"])
        attack_candidates = attacks[~attacks.group_id.isin(held_groups)]
    else:
        val_normal = raw.loc[source["val_ids"][base_y_val == 0]].copy()
        test_normal = raw.loc[source["test_ids"][base_y_test == 0]].copy()
        forced_val = attacks[attacks.group_id.isin(val_normal.group_id)]
        forced_test = attacks[attacks.group_id.isin(test_normal.group_id)]
        remaining = attacks[~attacks.group_id.isin(set(val_normal.group_id) | set(test_normal.group_id))]
        attack_candidates, hold = split_groups(remaining, .5, seed + 11)
        val_attack, test_attack = split_groups(hold, .5, seed + 12)
        val = balance(pd.concat([val_normal, forced_val, val_attack]), seed + 13)
        test = balance(pd.concat([test_normal, forced_test, test_attack]), seed + 14)
    budget = int(len(normal_train) * attack_fraction / (1 - attack_fraction))
    selected_attacks = capped_attack_sample(attack_candidates, budget, seed + 15)
    train = pd.concat([normal_train, selected_attacks]).sample(frac=1, random_state=seed)
    groups = [set(part.group_id) for part in [train, val, test]]
    if any(groups[i] & groups[j] for i, j in [(0, 1), (0, 2), (1, 2)]):
        raise ValueError("Sequence training/evaluation connection groups overlap")
    ids = [set(part.source_id) for part in [train, val, test]]
    if any(ids[i] & ids[j] for i, j in [(0, 1), (0, 2), (1, 2)]):
        raise ValueError("Sequence training/evaluation source rows overlap")
    # These models learn from labeled normal AND attack training examples.
    # Fit their transforms on that training set, keeping their sigmoid decoder
    # targets in [0,1]. Validation/test never participate in fitting.
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = MinMaxScaler()
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
    train_numeric = scaler.fit_transform(imputer.fit_transform(train[numeric]))
    train_categorical = encoder.fit_transform(train[categorical])
    flat_dim = train_numeric.shape[1] + train_categorical.shape[1]
    features = math.ceil(flat_dim / timesteps)

    def pack(values):
        if values.shape[1] < timesteps * features:
            values = np.pad(values, ((0, 0), (0, timesteps * features - values.shape[1])))
        return values.astype(np.float32).reshape(len(values), timesteps, features)

    def transform(part):
        values = np.concatenate([scaler.transform(imputer.transform(part[numeric])),
                                 encoder.transform(part[categorical])], axis=1)
        return pack(values)

    arrays = {"X_train": pack(np.concatenate([train_numeric, train_categorical], axis=1)),
              "y_train": train.BinaryLabel.to_numpy(dtype=np.int64),
              "X_val": transform(val), "y_val": val.BinaryLabel.to_numpy(dtype=np.int64),
              "X_test": transform(test), "y_test": test.BinaryLabel.to_numpy(dtype=np.int64)}
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("Nonfinite sequence data after preprocessing")
    sources = {}
    for name, part in [("train", train), ("val", val), ("test", test)]:
        for suffix, column in [("ids", "source_id"), ("groups", "group_id"), ("labels", "SubCategoryLabel")]:
            sources[f"{name}_{suffix}"] = part[column].to_numpy(dtype=str)
    eval_attacks = set(val[val.BinaryLabel == 1].SubCategoryLabel) | set(test[test.BinaryLabel == 1].SubCategoryLabel)
    report = {"dataset": "GeNIS", "split_mode": split_mode, "base_prepared_data": str(prepared.resolve()),
              "same_evaluation_rows_as_rnep": split_mode == "shared", "seed": seed,
              "train_normal_rows": len(normal_train), "train_attack_rows": len(selected_attacks),
              "train_attack_fraction_requested": attack_fraction,
              "train_class_counts": train.SubCategoryLabel.value_counts().to_dict(),
              "validation_class_counts": val.SubCategoryLabel.value_counts().to_dict(),
              "test_class_counts": test.SubCategoryLabel.value_counts().to_dict(),
              "unseen_attack_types_at_training": sorted(eval_attacks - set(selected_attacks.SubCategoryLabel)),
              "candidate_attack_class_counts": attack_candidates.SubCategoryLabel.value_counts().to_dict(),
              "flat_input_dim": flat_dim, "timesteps": timesteps, "features": features,
              "padding_features": timesteps * features - flat_dim,
              "reshape": "pack features of each individual flow; no cross-flow temporal window or truncation",
              "preprocessing": "median + MinMax + one-hot fitted only on labeled normal/attack training rows",
              "validation_test_used_for_fitting": False, "group_overlap": False, "row_overlap": False,
              "source_sha256": raw_hashes}
    pipeline = {"imputer": imputer, "scaler": scaler, "encoder": encoder,
                "numeric_features": numeric, "categorical_features": categorical,
                "timesteps": timesteps, "features": features}
    return arrays, sources, pipeline, report


def prepare_sequence_data(prepared=DEFAULT_PREPARED, seed=123, attack_fraction=.5,
                          max_normal_samples=None, timesteps=10, split_mode="shared"):
    prepared = Path(prepared)
    fingerprint = {"base_data_sha256": file_sha256(prepared / "prepared_data.npz"),
                   "source_rows_sha256": file_sha256(prepared / "source_rows.npz"),
                   "loader_sha256": file_sha256(Path(__file__)), "seed": seed,
                   "attack_fraction": attack_fraction, "max_normal_samples": max_normal_samples,
                   "timesteps": timesteps, "split_mode": split_mode}
    key = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()[:16]
    destination = prepared.parent / f"sequences_{key}"
    with (prepared.parent / f".sequences_{key}.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not (destination / "manifest.json").exists():
            arrays, sources, pipeline, report = build_sequence_data(
                prepared, seed, attack_fraction, max_normal_samples, timesteps, split_mode)
            with tempfile.TemporaryDirectory(prefix=".sequence_prepare_", dir=prepared.parent) as tmp:
                stage = Path(tmp) / "data"
                stage.mkdir()
                np.savez_compressed(stage / "prepared_data.npz", **arrays)
                np.savez_compressed(stage / "source_rows.npz", **sources)
                joblib.dump(pipeline, stage / "preprocessing.joblib")
                (stage / "preprocessing_report.json").write_text(json.dumps(report, indent=2) + "\n")
                (stage / "manifest.json").write_text(json.dumps(fingerprint, indent=2) + "\n")
                os.replace(stage, destination)
    return destination


def load_genis_sequences(prepared=DEFAULT_PREPARED, seed=123, attack_fraction=.5,
                         max_normal_samples=None, timesteps=10, split_mode="shared", return_metadata=False):
    path = prepare_sequence_data(prepared, seed, attack_fraction, max_normal_samples, timesteps, split_mode)
    with np.load(path / "prepared_data.npz") as z:
        data = tuple(z[key] for key in ["X_train", "y_train", "X_test", "y_test"])
    if return_metadata:
        metadata = json.loads((path / "preprocessing_report.json").read_text())
        metadata["prepared_data"] = str(path.resolve())
        return (*data, metadata)
    return data
