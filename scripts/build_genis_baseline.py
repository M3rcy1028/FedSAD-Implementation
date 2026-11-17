"""Raw-flow GeNIS CSV for the supervised baselines (MV-IDS, MM-FewShot-IDS).

Uses the same flows as the FedSAD "frame" data (scripts/build_genis_frame.py), but with
the raw Argus columns so each baseline can run its own preprocessing:

  split=train : normal training rows + validation rows (normal + attack)  -> training pool
  split=test  : exactly the FedSAD test rows (normal 4373 + attack 4373)

Columns: Timestamp (same synthetic per-scenario date as the GSAD CSVs), split, label (0/1),
attack_cat (GeNIS class), the 67 numeric + 6 categorical features of the RNEP frame pipeline.
Numeric NaNs are kept (each baseline imputes); hex ports are converted to int.

Usage:
    python scripts/build_genis_baseline.py
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--training_dir",
                default="/data/SDP_Dataset/GeNIS/processed/training_20261110_170017_516719")
ap.add_argument("--raw_dir", default="/data/SDP_Dataset/GeNIS/raw/3-scenarios/scenarios-10-sec")
ap.add_argument("--out_csv", default="/data/SDP_Dataset/GeNIS/processed/genis_baseline/GeNIS_baseline.csv")
args = ap.parse_args()

NORMAL_LABELS = {"user", "background", "admin"}

s = np.load(os.path.join(args.training_dir, "source_rows.npz"))
with open(os.path.join(args.training_dir, "preprocessing_report.json")) as f:
    rep = json.load(f)
num_cols, cat_cols = rep["numeric_features"], rep["categorical_features"]

parts = []
for part, split in (("train", "train"), ("val", "train"), ("test", "test")):
    ids, labels = s[f"{part}_ids"], s[f"{part}_labels"]
    parts.append(pd.DataFrame({"source_id": ids, "attack_cat": labels, "split": split, "part": part}))
meta = pd.concat(parts, ignore_index=True)
meta["file"] = meta["source_id"].str.rsplit(":", n=1).str[0]
meta["row"] = meta["source_id"].str.rsplit(":", n=1).str[1].astype(int)


def to_num(series):
    out = pd.to_numeric(series, errors="coerce")
    hex_mask = out.isna() & series.astype(str).str.startswith("0x")
    if hex_mask.any():
        out[hex_mask] = series[hex_mask].map(lambda v: float(int(str(v), 16)))
    return out


frames = []
for fname, grp in meta.groupby("file"):
    scen_no = int(fname.split("-")[1])
    raw = pd.read_csv(os.path.join(args.raw_dir, fname), usecols=["StartTime"] + num_cols + cat_cols,
                      dtype=str)
    sub = raw.iloc[grp["row"].values].reset_index(drop=True)
    out = pd.DataFrame({"source_id": grp["source_id"].values})
    out["Timestamp"] = f"{scen_no:02d}/01/2025 " + sub["StartTime"]
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

os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
df.to_csv(args.out_csv, index=False)
print(df.groupby(["part", "label"]).size())
print(f"Saved {len(df)} rows to {args.out_csv}")
