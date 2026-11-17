"""Build row-aligned RNEP/GSAD CSVs for GeNIS from the RNEP "frame" training pipeline.

rnep_frame_genis_weights.h5 was trained on <training_dir>/prepared_data.npz (99 features,
already imputed/MinMax-scaled/one-hot encoded, group-based split). To evaluate FedSAD with
that model we reuse exactly those rows:

  RNEP CSV  : the 99-d prepared vectors (no further scaling in FedSAD, see "prescaled")
  GSAD CSV  : Timestamp + Dst Port of the same flows, looked up in the raw scenario CSVs
              through source_rows.npz ("scenario-N-xxx.csv:<row>")

Raw StartTime is time-of-day only and every scenario starts at 09:00, so each scenario gets
its own synthetic date (scenario N -> 2025-01-0N) so windows never mix scenarios.

Normal rows: X_train (split=train) + normal part of X_test (split=test).
Attack rows: attack part of X_test, one file per attack class.
Validation rows are not used.

Usage:
    python scripts/build_genis_frame.py
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
ap.add_argument("--out_dir", default="/data/SDP_Dataset/GeNIS/processed")
args = ap.parse_args()

NORMAL_LABELS = {"user", "background", "admin"}

z = np.load(os.path.join(args.training_dir, "prepared_data.npz"))
s = np.load(os.path.join(args.training_dir, "source_rows.npz"))
with open(os.path.join(args.training_dir, "preprocessing_report.json")) as f:
    features = json.load(f)["expanded_features"]
assert len(features) == z["X_train"].shape[1]

# ---- raw lookup: StartTime / Dport for every referenced row ----
needed = {}
for part in ("train", "test"):
    for sid in s[f"{part}_ids"]:
        fname, row = sid.rsplit(":", 1)
        needed.setdefault(fname, set()).add(int(row))

lookup = {}
for fname, rows in sorted(needed.items()):
    scen_no = int(fname.split("-")[1])
    raw = pd.read_csv(os.path.join(args.raw_dir, fname), usecols=["StartTime", "Dport"],
                      dtype={"StartTime": str, "Dport": str})
    sub = raw.iloc[sorted(rows)]
    date = f"0{scen_no}/01/2025" if scen_no < 10 else f"{scen_no}/01/2025"
    for r, st, dp in zip(sub.index, sub["StartTime"], sub["Dport"]):
        port = pd.to_numeric(dp, errors="coerce")
        if pd.isna(port):  # hex ports (0x..) or missing (ARP/ICMP)
            try:
                port = float(int(str(dp), 16))
            except (TypeError, ValueError):
                port = 0.0
        lookup[f"{fname}:{r}"] = (f"{date} {st}", port)
    print(f"[raw] {fname}: {len(rows)} rows")


def meta(ids):
    ts, port = zip(*(lookup[i] for i in ids))
    return pd.DataFrame({"Timestamp": ts, "Dst Port": port})


rnep_dir = os.path.join(args.out_dir, "genis_frame_rnep")
graph_dir = os.path.join(args.out_dir, "genis_frame_graph")
os.makedirs(rnep_dir, exist_ok=True)
os.makedirs(graph_dir, exist_ok=True)

# ---- normal: train + normal test ----
test_labels = s["test_labels"]
test_is_normal = np.isin(test_labels, list(NORMAL_LABELS))
assert set(s["train_labels"]) <= NORMAL_LABELS
assert (test_is_normal == (z["y_test"] == 0)).all()

X_norm = np.vstack([z["X_train"], z["X_test"][test_is_normal]])
ids_norm = np.concatenate([s["train_ids"], s["test_ids"][test_is_normal]])
split = ["train"] * len(z["X_train"]) + ["test"] * int(test_is_normal.sum())

pd.DataFrame(X_norm, columns=features).to_csv(os.path.join(rnep_dir, "GeNIS_frame_normal.csv"), index=False)
g = meta(ids_norm)
g["split"] = split
g.to_csv(os.path.join(graph_dir, "GeNIS_frame_normal.csv"), index=False)
print(f"[normal] train={len(z['X_train'])} test={int(test_is_normal.sum())}")

# ---- attacks: one file per class ----
for cls in sorted(set(test_labels[~test_is_normal])):
    m = test_labels == cls
    pd.DataFrame(z["X_test"][m], columns=features).to_csv(
        os.path.join(rnep_dir, f"GeNIS_frame_anomaly_{cls}.csv"), index=False)
    meta(s["test_ids"][m]).to_csv(os.path.join(graph_dir, f"GeNIS_frame_anomaly_{cls}.csv"), index=False)
    print(f"[anomaly] {cls}: {int(m.sum())}")

print(f"Saved to {rnep_dir} and {graph_dir}")
