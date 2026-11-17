"""Train/test MV-IDS and MM-FewShot-IDS on GeNIS with the same test flows as FedSAD.

Data: scripts/build_genis_baseline.py
  split=train : training pool (normal train rows + validation normal/attack rows)
  split=test  : FedSAD test rows (normal 4373 + attack 4373)

MV-IDS        : trained on the whole training pool (original hyper-parameters).
MM-FewShot-IDS: few-shot, k_train flows per attack type + k_train * n_types normal flows
                sampled from the training pool (attack_balanced_split rule), seed 42.

Reports
  flow level   : every test flow
  window level : test flows grouped into the same 1-min windows FedSAD uses (normal and
                 attack flows windowed separately); a window is Anomaly if
                 any flow (rule=any) / at least half the flows (rule=majority) are predicted Anomaly.

Usage:
    python scripts/run_genis_baselines.py --models mvids mmfewshotsids
"""
import argparse
import os
import sys
import time
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import classification_report, confusion_matrix  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

META_COLS = ["source_id", "Timestamp", "split", "part", "label", "attack_cat"]

ap = argparse.ArgumentParser()
ap.add_argument("--data_path", default="/data/SDP_Dataset/GeNIS/processed/genis_baseline/GeNIS_baseline.csv")
ap.add_argument("--models", nargs="+", default=["mvids", "mmfewshotsids"])
ap.add_argument("--result_root", default="results/GeNIS/frame")
ap.add_argument("--time_window", default="1min")
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--seed", type=int, default=42)
# MV-IDS (models/model_mvids.py defaults)
ap.add_argument("--mv_epochs", type=int, default=100)
ap.add_argument("--mv_batch_size", type=int, default=64)
ap.add_argument("--mv_lr", type=float, default=1e-4)
# MM-FewShot-IDS (models/model_mmfewshotsids.py defaults)
ap.add_argument("--mm_epochs", type=int, default=40)
ap.add_argument("--mm_batch_size", type=int, default=32)
ap.add_argument("--mm_lr", type=float, default=5e-4)
ap.add_argument("--mm_k_train", type=int, default=15)
ap.add_argument("--kg_topk", type=int, default=50)
ap.add_argument("--kg_thresh", type=float, default=0.3)
args = ap.parse_args()


# ---------------------------------------------------------------- reports
def report_block(title, y_true, y_pred):
    rep = classification_report(y_true, y_pred, labels=[0, 1], target_names=["Normal", "Anomaly"],
                                digits=4, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    cm = (f"\n🧾 [Confusion Matrix]\n                Predicted Normal  Predicted Anomaly\n"
          f"Actual Normal   {tn:>16}  {fp:>17}\nActual Anomaly  {fn:>16}  {tp:>17}\n")
    return f"\n📊 [{title}]\n\n{rep}{cm}"


def window_labels(test_df, pred, rule):
    df = test_df[["Timestamp", "label"]].copy()
    df["pred"] = pred
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], dayfirst=True)
    y_true, y_pred = [], []
    for lab in (0, 1):  # FedSAD windows normal and attack flows separately
        part = df[df["label"] == lab].set_index("Timestamp").sort_index()
        for _, w in part.groupby(pd.Grouper(freq=args.time_window)):
            if w.empty:
                continue
            frac = w["pred"].mean()
            y_pred.append(int(frac > 0) if rule == "any" else int(frac >= 0.5))
            y_true.append(lab)
    return np.array(y_true), np.array(y_pred)


def save_reports(name, out_dir, test_df, pred, train_info, infer_ms):
    os.makedirs(out_dir, exist_ok=True)
    y = test_df["label"].to_numpy()
    text = f"===== {name} on GeNIS (frame test rows) =====\n{train_info}\n"
    text += f"Inference: {infer_ms:.4f} ms/flow\n"
    text += report_block(f"Flow-level Classification Report (n={len(y)})", y, pred)
    for rule in ("any", "majority"):
        wt, wp = window_labels(test_df, pred, rule)
        text += report_block(f"{args.time_window} Window Classification Report (rule={rule}, "
                             f"windows normal={int((wt == 0).sum())} anomaly={int((wt == 1).sum())})", wt, wp)
    print(text)
    with open(os.path.join(out_dir, f"{name.lower()}_server.txt"), "w") as f:
        f.write(text)
    out = test_df[META_COLS].copy()
    out["pred"] = pred
    out.to_csv(os.path.join(out_dir, "test_predictions.csv"), index=False)


# ---------------------------------------------------------------- data
df = pd.read_csv(args.data_path, low_memory=False)
pool_df = df[df["split"] == "train"].reset_index(drop=True)
test_df = df[df["split"] == "test"].reset_index(drop=True)
print(f"[INFO] pool={len(pool_df)} (attack={int(pool_df.label.sum())}) | test={len(test_df)} "
      f"(attack={int(test_df.label.sum())})")


# ---------------------------------------------------------------- MV-IDS
def run_mvids():
    from models.MVIDS.load_data import MultiViewDataset, VIEW_SCHEMAS, build_model
    from models.MVIDS.preprocess import MultiViewPreprocessor
    from models.MVIDS.train import train_loop

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = os.path.join(args.result_root, "mvids")
    os.makedirs(out_dir, exist_ok=True)

    feat_cols = [c for c in df.columns if c not in META_COLS]
    pre = MultiViewPreprocessor(VIEW_SCHEMAS["genis"])
    tr_views = pre.fit_transform(pool_df[feat_cols])
    te_views = pre.transform(test_df[feat_cols])
    view_lengths = [v.shape[1] for v in tr_views]
    print(f"[MV-IDS] view_lengths={view_lengths}")

    dl_tr = DataLoader(MultiViewDataset(tr_views, pool_df["label"].to_numpy(np.float32)),
                       batch_size=args.mv_batch_size, shuffle=True)
    dl_te = DataLoader(MultiViewDataset(te_views, test_df["label"].to_numpy(np.float32)),
                       batch_size=args.mv_batch_size, shuffle=False)

    cfg = SimpleNamespace(device=args.device, lr=args.mv_lr, epochs=args.mv_epochs, result_path=out_dir)
    model = build_model(view_lengths)
    save_path = os.path.join(out_dir, "mvids_weights.pt")
    model = train_loop(model, dl_tr, None, cfg, save_path=save_path)
    torch.save({"model_state_dict": model.state_dict(), "view_lengths": view_lengths,
                "args": vars(cfg), "preprocessor": pre}, save_path)

    model.eval()
    probs = []
    t0 = time.perf_counter()
    with torch.no_grad():
        for xb, _ in dl_te:
            probs.append(torch.sigmoid(model([x.to(args.device) for x in xb])).cpu().numpy())
    infer_ms = (time.perf_counter() - t0) * 1000 / len(test_df)
    pred = (np.concatenate(probs) >= 0.5).astype(int)
    info = (f"train rows={len(pool_df)} (normal={int((pool_df.label == 0).sum())}, "
            f"attack={int(pool_df.label.sum())}) | epochs={args.mv_epochs} lr={args.mv_lr}")
    save_reports("MV-IDS", out_dir, test_df, pred, info, infer_ms)


# ---------------------------------------------------------------- MM-FewShot-IDS
def nf_adapter(d):
    """Map Argus columns onto the NetFlow names build_packet_sequence_tensor reads."""
    z = d.fillna(0)
    return pd.DataFrame({
        "IN_PKTS": z["SrcPkts"], "OUT_PKTS": z["DstPkts"],
        "IN_BYTES": z["SrcBytes"], "OUT_BYTES": z["DstBytes"],
        "SRC_TO_DST_IAT_AVG": z["SIntPkt"], "SRC_TO_DST_IAT_STDDEV": z["SrcJitter"],
        "MIN_TTL": np.minimum(z["sTtl"], z["dTtl"]), "MAX_TTL": np.maximum(z["sTtl"], z["dTtl"]),
        "TCP_WIN_MAX_IN": z["SrcWin"], "TCP_WIN_MAX_OUT": z["DstWin"],
        "SRC_TO_DST_AVG_THROUGHPUT": z["SrcLoad"], "DST_TO_SRC_AVG_THROUGHPUT": z["DstLoad"],
        "RETRANSMITTED_IN_PKTS": z["SrcRetra"], "RETRANSMITTED_OUT_PKTS": z["DstRetra"],
        "LONGEST_FLOW_PKT": np.maximum(z["sMaxPktSz"], z["dMaxPktSz"]),
        "SHORTEST_FLOW_PKT": np.minimum(z["sMinPktSz"], z["dMinPktSz"]),
        "MIN_IP_PKT_LEN": np.minimum(z["sMinPktSz"], z["dMinPktSz"]),
        "MAX_IP_PKT_LEN": np.maximum(z["sMaxPktSz"], z["dMaxPktSz"]),
    })


def run_mmfewshot():
    from models.MMFEWSHOTSIDS.load_data import DATASET_PRESETS, MultimodalDataset, build_graph_A
    from models.MMFEWSHOTSIDS.multimodal_ids import MultiModalIDS
    from models.MMFEWSHOTSIDS.traffic_feature_graph import build_packet_sequence_tensor
    from models.MMFEWSHOTSIDS.train import train_self

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = os.path.join(args.result_root, "mmfewshotsids")
    os.makedirs(out_dir, exist_ok=True)
    preset = DATASET_PRESETS["genis"]

    # few-shot sampling from the training pool (attack_balanced_split rule)
    rng = np.random.default_rng(args.seed)
    k = args.mm_k_train
    att = pool_df[pool_df.label == 1]
    types = sorted(t for t, n in att["attack_cat"].value_counts().items() if n >= k)
    skipped = sorted(set(att["attack_cat"]) - set(types))
    idx = []
    for t in types:
        idx += rng.choice(att.index[att.attack_cat == t].to_numpy(), k, replace=False).tolist()
    idx += rng.choice(pool_df.index[pool_df.label == 0].to_numpy(), k * len(types), replace=False).tolist()
    shot_df = pool_df.loc[sorted(idx)]
    print(f"[MM-FewShot] few-shot train={len(shot_df)} ({k}/attack type x {len(types)} types "
          f"+ {k * len(types)} normal); skipped types (<{k} samples): {skipped}")

    num_cols = preset["s_cont_candidates"]
    sub = pd.concat([shot_df, test_df], ignore_index=True)
    sub[num_cols] = sub[num_cols].fillna(0.0)
    sub_csv = os.path.join(out_dir, "fewshot_train_test.csv")
    sub.to_csv(sub_csv, index=False)
    n_train = len(shot_df)

    adapter_csv = os.path.join(out_dir, "nf_adapter.csv")
    nf_adapter(sub).to_csv(adapter_csv, index=False)
    tensor = build_packet_sequence_tensor(adapter_csv, os.path.join(out_dir, "traffic_tensor.npy"))
    tensor = np.nan_to_num(tensor)

    ds = MultimodalDataset(csv_path=sub_csv, tensor_data=tensor, label_col="label",
                           attack_col="attack_cat", dataset_type="genis")
    # label-free feature correlation from the whole training pool (no test rows)
    A = build_graph_A(pool_df[preset["g_candidates"]].fillna(0.0), preset["g_candidates"],
                      topk=args.kg_topk, thresh=args.kg_thresh).to(args.device)
    tr_idx, te_idx = list(range(n_train)), list(range(n_train, len(sub)))
    ds.fit_transforms(tr_idx)

    model = MultiModalIDS(
        num_cont=ds.num_cont, disc_cardinalities=ds.cardinalities, num_classes=2,
        fusion_dim=128, fusion_type="hif", transformer_layers=2, transformer_heads=4,
        g_dropout=0.1, s_dropout=0.1, fusion_dropout=0.1, g_in_dim=A.shape[0], A_tensor=A,
    ).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.mm_lr)
    cfg = SimpleNamespace(device=args.device, epochs=args.mm_epochs, result_path=out_dir, dataset="genis")
    train_self(cfg, model, DataLoader(Subset(ds, tr_idx), batch_size=args.mm_batch_size, shuffle=True), opt)
    model.load_state_dict(torch.load(os.path.join(out_dir, "mmfewshotsids_genis.pt"), map_location=args.device))

    model.eval()
    preds = []
    dl_te = DataLoader(Subset(ds, te_idx), batch_size=args.mm_batch_size, shuffle=False)
    t0 = time.perf_counter()
    with torch.no_grad():
        for g, cont, disc, _ in dl_te:
            logits = model(g.to(args.device), cont.to(args.device), disc.to(args.device))
            preds.append(logits.argmax(1).cpu().numpy())
    infer_ms = (time.perf_counter() - t0) * 1000 / len(te_idx)
    pred = np.concatenate(preds)
    info = (f"few-shot train rows={n_train} (k={k} per attack type, {len(types)} types, "
            f"skipped={skipped}) | epochs={args.mm_epochs} lr={args.mm_lr}")
    save_reports("MM-FewShot-IDS", out_dir, test_df, pred, info, infer_ms)


if "mvids" in args.models:
    run_mvids()
if "mmfewshotsids" in args.models:
    run_mmfewshot()
