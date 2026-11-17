"""Shared, train-only preprocessing for the two GeNIS RNEP experiments."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b""):
            digest.update(block)
    return digest.hexdigest()


def split_groups(frame, test_size, seed):
    """Keep every connection/window group in the same partition."""
    counts_by_label = frame.groupby(["group_id", "SubCategoryLabel"], sort=True).size()
    groups = (counts_by_label.rename("count").reset_index()
              .sort_values("count", ascending=False, kind="stable")
              .drop_duplicates("group_id").set_index("group_id")["SubCategoryLabel"].sort_index())
    counts = groups.value_counts()
    if len(groups) < 2 or counts.min() < 2:
        raise ValueError(f"Insufficient independent flow groups per class: {counts.to_dict()}")
    left, right = train_test_split(
        groups.index.to_numpy(), test_size=test_size, random_state=seed,
        stratify=groups.to_numpy(),
    )
    return frame[frame.group_id.isin(left)], frame[frame.group_id.isin(right)]


def balance(frame, seed):
    normal = frame[frame.BinaryLabel == 0]
    attack = frame[frame.BinaryLabel == 1]
    count = min(len(normal), len(attack))
    if count < attack.SubCategoryLabel.nunique() or not count:
        raise ValueError("Too few evaluation rows for balanced attack-class sampling")
    if len(normal) > count:
        normal = normal.sample(n=count, random_state=seed)
    if len(attack) > count:
        # Reuse the class-cap strategy from the earlier CICIoT experiments:
        # large floods cannot consume the entire balanced evaluation budget.
        counts = attack.SubCategoryLabel.value_counts().sort_index()
        lower, upper = 0, count
        while lower < upper:
            middle = (lower + upper + 1) // 2
            if np.minimum(counts.to_numpy(), middle).sum() <= count:
                lower = middle
            else:
                upper = middle - 1
        allocation = np.minimum(counts.to_numpy(), lower)
        remainder = count - int(allocation.sum())
        eligible = np.flatnonzero(counts.to_numpy() > allocation)
        eligible = np.random.default_rng(seed).permutation(eligible)
        allocation[eligible[:remainder]] += 1
        attack = pd.concat([
            attack[attack.SubCategoryLabel == label].sample(n=int(budget), random_state=seed + i)
            for i, (label, budget) in enumerate(zip(counts.index, allocation))
        ])
    return pd.concat([normal, attack]).sample(frac=1, random_state=seed)


def load_genis(base_dir, flow_interval=10, seed=123, max_train_samples=150000, layout="scenarios"):
    """Build disjoint normal-only train and balanced validation/test sets.

    Use a single flow interval, raw published feature tables, and groups defined
    by the publisher's FlowID. Normal groups split 80/10/10; attack groups split
    50/50 into validation/test. Identifiers, clocks, exporter bookkeeping and
    address-wide connection counts are excluded from model inputs. Signed
    numeric measurements (notably PCRatio) are retained. No log transform or
    supervised feature selection is used.
    """
    base_dir = Path(base_dir)
    if layout not in {"flows", "scenarios"}:
        raise ValueError("GeNIS layout must be flows or scenarios")
    folder = (base_dir / "raw" / ("3-scenarios" if layout == "scenarios" else "2-flows")
              / f"{layout}-{flow_interval}-sec")
    paths = sorted(folder.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No GeNIS CSV files in {folder}")
    # The official description column contains unquoted commas; only the first
    # two fields define the schema and are required here.
    with (base_dir / "raw" / "0-info" / "genis-features.csv").open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.reader(stream)
        next(reader)
        feature_info = pd.DataFrame([row[:2] for row in reader if len(row) >= 2],
                                    columns=["FeatureName", "FeatureType"])
    excluded = {
        "Rank", "Seq", "AutoId", "Offset", "SrcTCPBase", "DstTCPBase",
        "sIpId", "dIpId", "SRange", "ERange", "IdleTime", "Ssaddr", "Sdaddr",
        "srcUdata", "dstUdata",
    }
    numeric = [row.FeatureName for row in feature_info.itertuples()
               if row.FeatureType.lower() in {"integer", "continuous", "countinuous"}
               and row.FeatureName not in excluded]
    categorical = ["Proto", "Flgs", "Dir", "State", "Cause", "TcpOpt"]
    schema = ["FlowID", "StartTime", "LastTime", "BinaryLabel", "CategoryLabel", "SubCategoryLabel"]
    frames = []
    seen = set()
    cross_scenario_duplicates = 0
    hashes = {}
    for path in paths:
        frame = pd.read_csv(path, usecols=schema + numeric + categorical, low_memory=False)
        frame["source_id"] = [f"{path.name}:{i}" for i in range(len(frame))]
        if frame.FlowID.isna().any() or frame.SubCategoryLabel.isna().any():
            raise ValueError(f"Missing flow IDs or labels in {path}")
        frame["BinaryLabel"] = pd.to_numeric(frame.BinaryLabel, errors="raise").astype(int)
        if not set(frame.BinaryLabel.unique()) <= {0, 1}:
            raise ValueError(f"Unexpected labels in {path}")
        for column in numeric:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(np.float64)
        frame[numeric] = frame[numeric].replace([np.inf, -np.inf], np.nan)
        for column in categorical:
            frame[column] = frame[column].fillna("__MISSING__").astype(str)
        frame["group_id"] = frame.FlowID.astype(str)
        portless = frame.Proto.isin(["icmp", "ipv6-icmp", "arp"])
        # ICMP has no transport connection: its whole attack capture has one
        # FlowID. Keep each exporter-sized time block together instead.
        raw_times = frame.loc[portless, "StartTime"]
        seconds = pd.to_numeric(raw_times, errors="coerce")
        if seconds.isna().any():
            seconds = pd.to_timedelta(raw_times.astype(str), errors="raise").dt.total_seconds()
        blocks = (seconds // flow_interval).astype(np.int64).astype(str)
        frame.loc[portless, "group_id"] += "|window=" + blocks
        if layout == "scenarios":
            record_hashes = pd.util.hash_pandas_object(frame[schema + numeric + categorical], index=False).to_numpy()
            repeated = np.asarray([value in seen for value in record_hashes])
            cross_scenario_duplicates += int(repeated.sum())
            seen.update(record_hashes.tolist())
            frame = frame.loc[~repeated]
        frames.append(frame)
        hashes[str(path.resolve())] = file_sha256(path)
    frame = pd.concat(frames, ignore_index=True)
    original_count = len(frame) + cross_scenario_duplicates
    # Equal behavioral vectors can be distinct packets/transactions. Retain
    # their multiplicity instead of silently deduplicating selected features.
    original_counts = frame.SubCategoryLabel.value_counts().to_dict()
    normal = frame[frame.BinaryLabel == 0]
    attack = frame[frame.BinaryLabel == 1]
    normal_in_attack_groups = normal[normal.group_id.isin(attack.group_id)]
    pure_normal = normal[~normal.group_id.isin(attack.group_id)]
    train, holdout = split_groups(pure_normal, .2, seed)
    holdout = pd.concat([holdout, normal_in_attack_groups])
    val_normal, test_normal = split_groups(holdout, .5, seed + 1)
    # A five-tuple shared with any normal partition is assigned exclusively to
    # that partition; attack records sharing normal-training IDs cannot be held out.
    shared_train = attack.group_id.isin(train.group_id)
    excluded_attack_rows = int(shared_train.sum())
    if excluded_attack_rows:
        raise AssertionError("Attack groups entered the normal training partition")
    forced_val = attack[attack.group_id.isin(val_normal.group_id)]
    forced_test = attack[attack.group_id.isin(test_normal.group_id)]
    remaining = attack[~attack.group_id.isin(holdout.group_id)]
    val_attack, test_attack = split_groups(remaining, .5, seed + 2)
    val = balance(pd.concat([val_normal, val_attack, forced_val]), seed + 3)
    test = balance(pd.concat([test_normal, test_attack, forced_test]), seed + 4)
    train = train.sample(frac=1, random_state=seed)
    if max_train_samples is not None:
        train = train.iloc[:max_train_samples]
    group_sets = [set(part.group_id) for part in [train, val, test]]
    if any(group_sets[i] & group_sets[j] for i, j in [(0, 1), (0, 2), (1, 2)]):
        raise AssertionError("Flow groups overlap between train, validation, and test")
    all_missing = [column for column in numeric if train[column].isna().all()]
    numeric = [column for column in numeric if column not in all_missing]
    imputer = SimpleImputer(strategy="median")
    scaler = MinMaxScaler()
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
    numeric_train = scaler.fit_transform(imputer.fit_transform(train[numeric]))
    categorical_train = encoder.fit_transform(train[categorical])

    def transform(part):
        return np.concatenate([
            scaler.transform(imputer.transform(part[numeric])),
            encoder.transform(part[categorical]),
        ], axis=1).astype(np.float32)

    arrays = {
        "X_train": np.concatenate([numeric_train, categorical_train], axis=1).astype(np.float32),
        "X_val": transform(val), "y_val": val.BinaryLabel.to_numpy(dtype=np.int64),
        "X_test": transform(test), "y_test": test.BinaryLabel.to_numpy(dtype=np.int64),
    }
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("Nonfinite model inputs after preprocessing")
    sources = {}
    for name, part in [("train", train), ("val", val), ("test", test)]:
        sources[f"{name}_ids"] = part.source_id.to_numpy(dtype=str)
        sources[f"{name}_groups"] = part.group_id.to_numpy(dtype=str)
        sources[f"{name}_labels"] = part.SubCategoryLabel.to_numpy(dtype=str)
    report = {
        "dataset": "GeNIS", "record": "https://zenodo.org/records/14919237",
        "preprocessor_sha256": file_sha256(Path(__file__)),
        "flow_interval_seconds": flow_interval, "seed": seed, "input_dim": arrays["X_train"].shape[1],
        "layout": layout, "original_rows": original_count,
        "duplicate_records_reused_across_scenarios_removed": cross_scenario_duplicates,
        "within_capture_equal_feature_vectors_retained": True,
        "original_class_counts": original_counts, "source_sha256": hashes,
        "split": "stratified connection/window groups: normal 80/10/10; attacks validation/test only; no overlapping groups",
        "portless_grouping": "ICMP, IPv6-ICMP, ARP: FlowID + exporter-sized StartTime block; TCP/UDP: entire FlowID",
        "attack_sampling": "common class cap min(available, cap), allocating the balanced budget across all attack types",
        "normal_training_rows": len(train), "validation_rows": len(val), "test_rows": len(test),
        "validation_class_counts": val.SubCategoryLabel.value_counts().to_dict(),
        "test_class_counts": test.SubCategoryLabel.value_counts().to_dict(),
        "normal_validation_rows": int((val.BinaryLabel == 0).sum()),
        "attack_validation_rows": int((val.BinaryLabel == 1).sum()),
        "normal_test_rows": int((test.BinaryLabel == 0).sum()),
        "attack_test_rows": int((test.BinaryLabel == 1).sum()),
        "attack_rows_excluded_due_to_training_group_overlap": excluded_attack_rows,
        "normal_rows_held_out_due_to_shared_attack_group": len(normal_in_attack_groups),
        "numeric_features": numeric, "categorical_features": categorical,
        "expanded_features": numeric + encoder.get_feature_names_out(categorical).tolist(),
        "all_missing_training_features_removed": all_missing,
        "excluded_numeric_identifiers_and_bookkeeping": sorted(excluded),
        "preprocessing": "normal-train-only median imputation, MinMax, and categorical one-hot encoding; no clipping/log transform",
        "signed_values_retained": True,
        "normal_training_range": [float(arrays["X_train"].min()), float(arrays["X_train"].max())],
        "unknown_categories": {name: {
            column: int((~part[column].isin(categories)).sum())
            for column, categories in zip(categorical, encoder.categories_)
        } for name, part in [("validation", val), ("test", test)]},
        "group_overlap": False,
    }
    # Report equal feature vectors without using test values to change features.
    train_hashes = pd.util.hash_pandas_object(pd.DataFrame(arrays["X_train"]), index=False).to_numpy()
    for name in ["val", "test"]:
        hashes_part = pd.util.hash_pandas_object(pd.DataFrame(arrays[f"X_{name}"]), index=False).to_numpy()
        report[f"{name}_vectors_equal_to_training_vectors"] = int(np.isin(hashes_part, train_hashes).sum())
    pipeline = {"imputer": imputer, "scaler": scaler, "encoder": encoder,
                "numeric_features": numeric, "categorical_features": categorical}
    return arrays, sources, pipeline, report


def prepare_genis(base_dir, destination, **kwargs):
    import joblib
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise FileExistsError(f"Prepared data directory is not empty: {destination}")
    arrays, sources, pipeline, report = load_genis(base_dir, **kwargs)
    np.savez_compressed(destination / "prepared_data.npz", **arrays)
    np.savez_compressed(destination / "source_rows.npz", **sources)
    joblib.dump(pipeline, destination / "preprocessing.joblib")
    (destination / "preprocessing_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
