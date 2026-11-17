"""GeNIS preprocessing for FedSAD and the supervised baselines.

Both outputs reuse the flows of the RNEP "frame" training run (<training_dir>):
  prepared_data.npz          99-d vectors (imputed / MinMax-scaled / one-hot, group-based split)
  source_rows.npz            source id of every row ("scenario-N-xxx.csv:<row>") and its label
  preprocessing_report.json  feature lists

Raw StartTime is time-of-day only and every scenario starts at 09:00, so each scenario gets
its own synthetic date (scenario N -> 0N/01/2025) and windows never mix scenarios.

Subcommands
  frame     Row-aligned RNEP/GSAD CSVs for FedSAD (--eval_dataset genis_frame)
              <out_dir>/genis_frame_rnep : the 99-d prepared vectors (FedSAD "prescaled")
              <out_dir>/genis_frame_graph: Timestamp + Dst Port of the same flows
            Normal = X_train (split=train) + normal part of X_test (split=test);
            attacks = attack part of X_test, one file per class. Validation rows are unused.

  baseline  One raw-flow CSV for MV-IDS / MM-FewShot-IDS (experiment/_legacy/scripts/run_genis_baselines.py)
              split=train : normal training rows + validation rows (normal + attack)
              split=test  : exactly the FedSAD test rows
            Columns: source_id, Timestamp, split, part, label (0/1), attack_cat,
            the 67 numeric + 6 categorical features of the RNEP frame pipeline.
            Numeric NaNs are kept (each baseline imputes); hex ports are converted to int.

Usage:
    python experiment/_legacy/scripts/preprocess_genis.py frame
    python experiment/_legacy/scripts/preprocess_genis.py baseline
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

DEFAULT_TRAINING_DIR = "/data/SDP_Dataset/GeNIS/processed/training_20261110_170017_516719"
DEFAULT_RAW_DIR = "/data/SDP_Dataset/GeNIS/raw/3-scenarios/scenarios-10-sec"
DEFAULT_OUT_DIR = "/data/SDP_Dataset/GeNIS/processed"

NORMAL_LABELS = {"user", "background", "admin"}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def load_report(training_dir):
    with open(os.path.join(training_dir, "preprocessing_report.json")) as f:
        return json.load(f)


def split_source_id(sid):
    """'scenario-N-xxx.csv:<row>' -> ('scenario-N-xxx.csv', row)"""
    fname, row = sid.rsplit(":", 1)
    return fname, int(row)


def scenario_date(fname):
    """Synthetic per-scenario date (dayfirst), e.g. scenario-3-... -> '03/01/2025'."""
    return f"{int(fname.split('-')[1]):02d}/01/2025"


def read_raw_rows(raw_dir, fname, rows, columns):
    """Read `columns` (as str) of the given positional rows of one raw scenario CSV."""
    raw = pd.read_csv(os.path.join(raw_dir, fname), usecols=columns, dtype=str)
    return raw.iloc[rows]


def to_num(series, missing=np.nan):
    """Numeric conversion with hex support ('0x..'). Values that are neither stay `missing`."""
    out = pd.to_numeric(series, errors="coerce")
    hex_mask = out.isna() & series.astype(str).str.startswith("0x")
    if hex_mask.any():
        out[hex_mask] = series[hex_mask].map(lambda v: float(int(str(v), 16)))
    return out.fillna(missing) if not pd.isna(missing) else out


# ---------------------------------------------------------------------------
# frame: FedSAD inputs
# ---------------------------------------------------------------------------
def build_frame(args):
    z = np.load(os.path.join(args.training_dir, "prepared_data.npz"))
    s = np.load(os.path.join(args.training_dir, "source_rows.npz"))
    features = load_report(args.training_dir)["expanded_features"]
    assert len(features) == z["X_train"].shape[1]

    # Raw lookup: Timestamp / Dst Port for every referenced row
    needed = {}
    for part in ("train", "test"):
        for sid in s[f"{part}_ids"]:
            fname, row = split_source_id(sid)
            needed.setdefault(fname, set()).add(row)

    lookup = {}
    for fname, rows in sorted(needed.items()):
        sub = read_raw_rows(args.raw_dir, fname, sorted(rows), ["StartTime", "Dport"])
        date = scenario_date(fname)
        # hex ports (0x..) -> int, missing ports (ARP/ICMP) -> 0
        ports = to_num(sub["Dport"], missing=0.0)
        for r, st, port in zip(sub.index, sub["StartTime"], ports):
            lookup[f"{fname}:{r}"] = (f"{date} {st}", float(port))
        print(f"[raw] {fname}: {len(rows)} rows")

    def meta(ids):
        ts, port = zip(*(lookup[i] for i in ids))
        return pd.DataFrame({"Timestamp": ts, "Dst Port": port})

    rnep_dir = os.path.join(args.out_dir, "genis_frame_rnep")
    graph_dir = os.path.join(args.out_dir, "genis_frame_graph")
    os.makedirs(rnep_dir, exist_ok=True)
    os.makedirs(graph_dir, exist_ok=True)

    # Normal: train + normal test
    test_labels = s["test_labels"]
    test_is_normal = np.isin(test_labels, list(NORMAL_LABELS))
    assert set(s["train_labels"]) <= NORMAL_LABELS
    assert (test_is_normal == (z["y_test"] == 0)).all()

    X_norm = np.vstack([z["X_train"], z["X_test"][test_is_normal]])
    ids_norm = np.concatenate([s["train_ids"], s["test_ids"][test_is_normal]])
    split = ["train"] * len(z["X_train"]) + ["test"] * int(test_is_normal.sum())

    pd.DataFrame(X_norm, columns=features).to_csv(
        os.path.join(rnep_dir, "GeNIS_frame_normal.csv"), index=False)
    g = meta(ids_norm)
    g["split"] = split
    g.to_csv(os.path.join(graph_dir, "GeNIS_frame_normal.csv"), index=False)
    print(f"[normal] train={len(z['X_train'])} test={int(test_is_normal.sum())}")

    # Attacks: one file per class
    for cls in sorted(set(test_labels[~test_is_normal])):
        m = test_labels == cls
        pd.DataFrame(z["X_test"][m], columns=features).to_csv(
            os.path.join(rnep_dir, f"GeNIS_frame_anomaly_{cls}.csv"), index=False)
        meta(s["test_ids"][m]).to_csv(
            os.path.join(graph_dir, f"GeNIS_frame_anomaly_{cls}.csv"), index=False)
        print(f"[anomaly] {cls}: {int(m.sum())}")

    print(f"Saved to {rnep_dir} and {graph_dir}")


# ---------------------------------------------------------------------------
# baseline: raw-flow CSV for MV-IDS / MM-FewShot-IDS
# ---------------------------------------------------------------------------
def build_baseline(args):
    s = np.load(os.path.join(args.training_dir, "source_rows.npz"))
    rep = load_report(args.training_dir)
    num_cols, cat_cols = rep["numeric_features"], rep["categorical_features"]

    parts = []
    for part, split in (("train", "train"), ("val", "train"), ("test", "test")):
        ids, labels = s[f"{part}_ids"], s[f"{part}_labels"]
        parts.append(pd.DataFrame({"source_id": ids, "attack_cat": labels, "split": split, "part": part}))
    meta = pd.concat(parts, ignore_index=True)
    meta["file"] = meta["source_id"].str.rsplit(":", n=1).str[0]
    meta["row"] = meta["source_id"].str.rsplit(":", n=1).str[1].astype(int)

    frames = []
    for fname, grp in meta.groupby("file"):
        sub = read_raw_rows(args.raw_dir, fname, grp["row"].values,
                            ["StartTime"] + num_cols + cat_cols).reset_index(drop=True)
        out = pd.DataFrame({"source_id": grp["source_id"].values})
        out["Timestamp"] = f"{scenario_date(fname)} " + sub["StartTime"]
        for c in num_cols:
            out[c] = to_num(sub[c]).values
        for c in cat_cols:
            out[c] = sub[c].fillna("__MISSING__").values
        frames.append(out)
        print(f"[raw] {fname}: {len(grp)} rows")

    feat = pd.concat(frames, ignore_index=True).set_index("source_id")
    df = meta.join(feat, on="source_id")
    df["label"] = (~df["attack_cat"].isin(NORMAL_LABELS)).astype(int)
    df = df[["source_id", "Timestamp", "split", "part", "label", "attack_cat"] + num_cols + cat_cols]

    out_csv = args.out_csv or os.path.join(args.out_dir, "genis_baseline", "GeNIS_baseline.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(df.groupby(["part", "label"]).size())
    print(f"Saved {len(df)} rows to {out_csv}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", choices=["frame", "baseline"], help="Which data to build")
    ap.add_argument("--training_dir", default=DEFAULT_TRAINING_DIR)
    ap.add_argument("--raw_dir", default=DEFAULT_RAW_DIR)
    ap.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--out_csv", default=None,
                    help="baseline only: output CSV (default <out_dir>/genis_baseline/GeNIS_baseline.csv)")
    args = ap.parse_args()

    if args.target == "frame":
        build_frame(args)
    else:
        build_baseline(args)


if __name__ == "__main__":
    main()
