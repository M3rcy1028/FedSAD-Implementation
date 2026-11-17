"""Bounded-memory Gotham packet preparation with globally disjoint connections."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from utils.genis import file_sha256

NUMERIC = ["frame.len", "ip.flags", "ip.ttl", "ip.proto", "ip.tos", "tcp.srcport", "tcp.dstport",
           "tcp.flags", "tcp.window_size_value", "tcp.window_size_scalefactor", "tcp.pdu.size",
           "udp.srcport", "udp.dstport"]
CATEGORICAL = ["frame.protocols"]
GROUP_COLUMNS = ["frame.time", "ip.src", "ip.dst", "eth.src", "eth.dst", "ip.proto",
                 "tcp.srcport", "tcp.dstport", "udp.srcport", "udp.dstport", "label"]
UNKNOWN = {"", "unknown", "nan", "unlabeled", "unlabelled"}
_SAMPLE_CONTEXT = None


def connection_ids(frame):
    """TCP/UDP: bidirectional IP/port tuple + capture day; others: 30s block."""
    proto = pd.to_numeric(frame["ip.proto"], errors="coerce").fillna(-1).astype(int).astype(str)
    day = frame["frame.time"].str.extract(r"^([A-Za-z]+\s+\d+,\s+\d{4})", expand=False)
    if day.isna().any():
        raise ValueError("Unexpected Gotham timestamp format")
    source = frame["ip.src"].fillna(frame["eth.src"]).fillna("__MISSING__")
    target = frame["ip.dst"].fillna(frame["eth.dst"]).fillna("__MISSING__")
    sport = pd.to_numeric(frame["tcp.srcport"].fillna(frame["udp.srcport"]), errors="coerce").fillna(-1).astype(int).astype(str)
    dport = pd.to_numeric(frame["tcp.dstport"].fillna(frame["udp.dstport"]), errors="coerce").fillna(-1).astype(int).astype(str)
    first, second = source + ":" + sport, target + ":" + dport
    lower = first.where(first <= second, second)
    upper = second.where(first <= second, first)
    key = day + "|" + proto + "|" + lower + "|" + upper
    portless = ~proto.isin(["6", "17"])
    clock = frame.loc[portless, "frame.time"].str.extract(r"(\d{2}):(\d{2}):(\d{2})").astype(int)
    blocks = ((clock[0] * 3600 + clock[1] * 60 + clock[2]) // 30).astype(str)
    key.loc[portless] += "|window=" + blocks
    return key


def allocations(counts, budget, seed):
    """Max-min class/device caps, retaining small strata without replacement."""
    counts = pd.Series(counts, dtype=np.int64).sort_index()
    if counts.empty or budget <= 0:
        return {}
    budget = min(int(counts.sum()), budget)
    low, high = 0, budget
    while low < high:
        middle = (low + high + 1) // 2
        if np.minimum(counts.to_numpy(), middle).sum() <= budget:
            low = middle
        else:
            high = middle - 1
    amounts = np.minimum(counts.to_numpy(), low)
    eligible = np.flatnonzero(counts.to_numpy() > amounts)
    remainder = budget - int(amounts.sum())
    amounts[np.random.default_rng(seed).permutation(eligible)[:remainder]] += 1
    return {str(k): int(n) for k, n in zip(counts.index, amounts) if n}


def group_assignments(counts, seed):
    records = counts.reset_index(name="rows")
    records["attack"] = records.label.ne("Benign")
    dominant = (records.sort_values(["attack", "rows", "device"], ascending=[False, False, True], kind="stable")
                .drop_duplicates("group").set_index("group"))
    rng = np.random.default_rng(seed)
    assigned = {}
    for device, group in dominant[~dominant.attack].groupby("device", sort=True):
        ids = rng.permutation(sorted(group.index))
        n = len(ids)
        val = max(1, int(n * .1)) if n >= 3 else 0
        test = max(1, int(n * .1)) if n >= 2 else 0
        for name, selected in [("val", ids[:val]), ("test", ids[val:val + test]), ("train", ids[val + test:])]:
            assigned.update(dict.fromkeys(selected, name))
    for label, group in dominant[dominant.attack].groupby("label", sort=True):
        ids = rng.permutation(sorted(group.index))
        mid = len(ids) // 2
        assigned.update(dict.fromkeys(ids[:mid], "val"))
        assigned.update(dict.fromkeys(ids[mid:], "test"))
    records["partition"] = records.group.map(assigned)
    return assigned, records


def numeric_values(frame, columns):
    values = {}
    for column in columns:
        text = frame[column].astype("string").str.split(",").str[0]
        if column in {"ip.flags", "tcp.flags"}:
            values[column] = text.map(lambda s: int(s, 16) if isinstance(s, str) and s.startswith("0x") else pd.to_numeric(s, errors="coerce"))
        else:
            values[column] = pd.to_numeric(text, errors="coerce")
    return pd.DataFrame(values, index=frame.index).astype(float).replace([np.inf, -np.inf], np.nan)


def _scan_file(job):
    path, expected_hash, chunksize, cache = job
    actual = file_sha256(path)
    if actual != expected_hash:
        raise ValueError(f"CSV changed after download: {path}")
    if cache.is_file():
        return joblib.load(cache)
    summaries, original, unknown_counts = [], Counter(), Counter()
    for frame in pd.read_csv(path, usecols=GROUP_COLUMNS, dtype=str, chunksize=chunksize):
        frame["label"] = frame.label.fillna("").str.strip()
        original.update(frame.label.value_counts().to_dict())
        unknown = frame.label.str.lower().isin(UNKNOWN)
        unknown_counts.update(frame.loc[unknown, "label"].value_counts().to_dict())
        frame = frame.loc[~unknown].copy()
        if frame.empty:
            continue
        frame["group"], frame["device"] = connection_ids(frame), path.stem
        summaries.append(frame.groupby(["group", "device", "label"], sort=False).size())
    counts = pd.concat(summaries).groupby(level=[0, 1, 2], sort=True).sum()
    result = (str(path.resolve()), actual, counts, original, unknown_counts)
    cache.parent.mkdir(parents=True, exist_ok=True)
    part = cache.with_name(cache.name + f".{os.getpid()}.part")
    joblib.dump(result, part)
    os.replace(part, cache)
    return result


def _sample_file(job):
    path, file_index = job
    assigned, budgets, seed, chunksize = _SAMPLE_CONTEXT
    device, row_start, pools = path.stem, 0, {}
    columns = list(dict.fromkeys(GROUP_COLUMNS + NUMERIC + CATEGORICAL))
    rng = np.random.default_rng(seed + 1000 + file_index)
    for frame in pd.read_csv(path, usecols=columns, dtype=str, chunksize=chunksize):
        frame["_row"] = np.arange(row_start, row_start + len(frame))
        row_start += len(frame)
        frame["_priority"] = rng.random(len(frame))
        frame["label"] = frame.label.fillna("").str.strip()
        frame = frame.loc[~frame.label.str.lower().isin(UNKNOWN)].copy()
        if frame.empty:
            continue
        frame["group"] = connection_ids(frame)
        frame["_partition"] = frame.group.map(assigned)
        for part in ["train", "val", "test"]:
            portion = frame[frame._partition == part]
            for label, rows in portion.groupby("label", sort=False):
                category, stratum = ("normal", device) if label == "Benign" else ("attack", label)
                cap = budgets.get((part, category), {}).get(stratum, 0)
                if not cap:
                    continue
                key = (part, category, stratum)
                sampled = rows.nsmallest(cap, "_priority").copy()
                sampled["device"] = device
                sampled["source_id"] = path.name + ":" + sampled._row.astype(str)
                if key in pools:
                    sampled = pd.concat([pools[key], sampled], ignore_index=True).nsmallest(cap, "_priority")
                pools[key] = sampled
    return path.name, pools


def _jobs(function, jobs, workers):
    if workers == 1:
        yield from map(function, jobs)
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("fork")) as pool:
            futures = [pool.submit(function, job) for job in jobs]
            for future in as_completed(futures):
                yield future.result()


def prepare_gotham(base_dir, destination, seed=123, max_train_samples=150000,
                   eval_normal_samples=25000, client_nums=15, chunksize=200000, workers=4):
    base_dir, destination = Path(base_dir), Path(destination)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Prepared directory is not empty: {destination}")
    paths = sorted((base_dir / "raw" / "processed").glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No official Gotham CSVs under {base_dir / 'raw/processed'}")
    manifest = json.loads((base_dir / "download_manifest.json").read_text())
    expected = {Path(m["name"]).name: m for m in manifest["members"] if m["name"].endswith(".csv")}
    if set(expected) != {p.name for p in paths}:
        raise ValueError("CSV inventory does not match the verified official download")
    summaries, hashes, original_counts, unknown_counts = [], {}, Counter(), Counter()
    code_hash = file_sha256(Path(__file__))
    scan_jobs = [(p, expected[p.name]["sha256"], chunksize,
                  base_dir / "processed" / "scan_cache" / f"{p.stem}_{code_hash[:16]}.joblib") for p in paths]
    for i, (name, actual, counts, original, unknown) in enumerate(_jobs(_scan_file, scan_jobs, workers)):
        hashes[name] = actual
        summaries.append(counts)
        original_counts.update(original)
        unknown_counts.update(unknown)
        print(f"[Gotham scan {i+1}/{len(paths)}] {Path(name).name}", flush=True)
    counts = pd.concat(summaries).groupby(level=[0, 1, 2], sort=True).sum()
    assigned, records = group_assignments(counts, seed)
    budgets = {}
    for part in ["train", "val", "test"]:
        normal = records[(records.partition == part) & ~records.attack].groupby("device").rows.sum()
        attack = records[(records.partition == part) & records.attack].groupby("label").rows.sum()
        budget = max_train_samples if part == "train" else min(eval_normal_samples, int(normal.sum()), int(attack.sum()))
        budgets[(part, "normal")] = allocations(normal, budget, seed)
        if part != "train":
            budgets[(part, "attack")] = allocations(attack, budget, seed + 1)
            if budget == 0:
                raise ValueError(f"No balanced {part} data after connection grouping")
    print("[Gotham sampling]", {str(k): sum(v.values()) for k, v in budgets.items()}, flush=True)
    global _SAMPLE_CONTEXT
    # Build the indexed mapper once. Rebuilding it from a large dict in every
    # chunk would make sampling quadratic in the number of connections.
    _SAMPLE_CONTEXT = (pd.Series(assigned), budgets, seed, chunksize)
    pools = {}
    for i, (name, sampled_pools) in enumerate(_jobs(_sample_file, [(p, i) for i, p in enumerate(paths)], workers)):
        for key, sampled in sampled_pools.items():
            cap = budgets[(key[0], key[1])][key[2]]
            if key in pools:
                sampled = pd.concat([pools[key], sampled], ignore_index=True).nsmallest(cap, "_priority")
            pools[key] = sampled
        print(f"[Gotham sample {i+1}/{len(paths)}] {name}", flush=True)
    _SAMPLE_CONTEXT = None
    frames = {part: pd.concat([v for k, v in sorted(pools.items()) if k[0] == part], ignore_index=True)
              .sample(frac=1, random_state=seed).reset_index(drop=True) for part in ["train", "val", "test"]}
    for left, right in [("train", "val"), ("train", "test"), ("val", "test")]:
        if set(frames[left].group) & set(frames[right].group):
            raise ValueError("Connection overlap between partitions")
        if set(frames[left].source_id) & set(frames[right].source_id):
            raise ValueError("Source row overlap between partitions")
    train = frames["train"]
    if not train.label.eq("Benign").all():
        raise ValueError("Attacks in normal-only training data")
    device_counts = train.device.value_counts()
    if len(device_counts) < client_nums or len(train) < 2 * client_nums:
        raise ValueError("Too few normal training devices/rows for the requested clients")
    loads, device_clients = np.zeros(client_nums, dtype=int), {}
    for device, count in sorted(device_counts.items(), key=lambda item: (-item[1], item[0])):
        cid = int(loads.argmin())
        device_clients[device] = cid
        loads[cid] += count
    if loads.min() < 2:
        raise ValueError("Some device clients have insufficient rows for local validation")
    values = numeric_values(train, NUMERIC)
    all_missing = values.columns[values.isna().all()].tolist()
    numeric = [c for c in NUMERIC if c not in all_missing]
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = MinMaxScaler()
    scaler.fit(imputer.fit_transform(values[numeric]))
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
    encoder.fit(train[CATEGORICAL].fillna("__MISSING__"))
    arrays, source_rows = {}, {}
    for part, frame in frames.items():
        arrays[f"X_{part}"] = np.concatenate([
            scaler.transform(imputer.transform(numeric_values(frame, numeric))),
            encoder.transform(frame[CATEGORICAL].fillna("__MISSING__"))], axis=1).astype(np.float32)
        if part != "train":
            arrays[f"y_{part}"] = frame.label.ne("Benign").to_numpy(dtype=np.int64)
        for suffix, column in [("ids", "source_id"), ("groups", "group"), ("labels", "label"), ("devices", "device")]:
            source_rows[f"{part}_{suffix}"] = frame[column].to_numpy(dtype=str)
    arrays["train_client_ids"] = train.device.map(device_clients).to_numpy(dtype=np.int64)
    if not all(np.isfinite(a).all() for a in arrays.values()):
        raise ValueError("Nonfinite model inputs")
    report = {"dataset": "Gotham", "record": "https://zenodo.org/records/14502760", "seed": seed,
              "preprocessor_sha256": file_sha256(Path(__file__)), "source_sha256": hashes,
              "unit": "official packet CSV row; no CICFlowMeter conversion or flow aggregation",
              "grouping": "global bidirectional IP/port/protocol/capture-day groups; portless uses 30s blocks",
              "normal_split": "connection groups per device approximately 80/10/10; attack groups validation/test only",
              "sampling": "uniform row priorities without replacement; max-min device caps for normal, class caps for attacks",
              "original_class_counts": dict(original_counts), "unknown_rows_excluded": dict(unknown_counts),
              "normal_training_rows": len(train), "validation_rows": len(frames["val"]), "test_rows": len(frames["test"]),
              "input_dim": arrays["X_train"].shape[1], "numeric_features": numeric, "categorical_features": CATEGORICAL,
              "excluded_features": ["frame.time", "eth.src", "eth.dst", "ip.src", "ip.dst", "ip.checksum", "tcp.checksum", "tcp.options", "label", "device"],
              "all_missing_normal_training_features_removed": all_missing,
              "preprocessing": "normal-training-only median, MinMax, and one-hot; no clipping/log transform",
              "client_devices": {str(cid): sorted(d for d, c in device_clients.items() if c == cid) for cid in range(client_nums)},
              "client_training_rows": loads.tolist(), "training_devices": len(device_clients),
              "source_device_files": len(paths), "group_overlap": False, "row_overlap": False,
              "validation_test_used_for_fitting": False}
    for part, frame in frames.items():
        report[f"{part}_class_counts"] = frame.label.value_counts().to_dict()
    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination / "prepared_data.npz", **arrays)
    np.savez_compressed(destination / "source_rows.npz", **source_rows)
    joblib.dump({"imputer": imputer, "scaler": scaler, "encoder": encoder,
                 "numeric_features": numeric, "categorical_features": CATEGORICAL}, destination / "preprocessing.joblib")
    (destination / "preprocessing_report.json").write_text(json.dumps(report, indent=2) + "\n")
    (destination / "verification.json").write_text(json.dumps({"normal_only_training": True,
        "disjoint_source_rows": True, "disjoint_connections": True, "train_only_fitting": True,
        "validation_balanced": bool(np.bincount(arrays["y_val"])[0] == np.bincount(arrays["y_val"])[1]),
        "test_balanced": bool(np.bincount(arrays["y_test"])[0] == np.bincount(arrays["y_test"])[1])}, indent=2) + "\n")
    return report
