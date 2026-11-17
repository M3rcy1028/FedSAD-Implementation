"""Detection-window-size sweep for FedSAD (GSAD + TAAE + DS fusion).

For every (dataset, window) pair this script
  1. rebuilds the GSAD normal statistics with that window (they depend on the window size),
  2. runs fedsad.py with the same window,
  3. collects window_metrics.json into a summary CSV.

Usage (from the repository root):
    python experiment/_legacy/scripts/run_window_sweep.py --datasets cic unsw --windows 30s 1min 2min 4min
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = str(Path(__file__).resolve().parents[3])
BACKEND = os.path.join(ROOT, "experiment", "_legacy")
sys.path.insert(0, BACKEND)

import pandas as pd  # noqa: E402
from gsad import train_single_model  # noqa: E402

DATASETS = {
    "cic": {"name": "CSE-CIC-IDS2018", "graph_dir": "cic_graph", "normal_file": "CIC_ae_normal.csv"},
    "unsw": {"name": "UNSW_NB15", "graph_dir": "unsw_graph", "normal_file": "UNSW_NB15_normal.csv"},
}


def window_tag(w):
    return w.replace(".", "p")


def flatten(m):
    return {
        "dataset": m["dataset"],
        "time_window": m["time_window"],
        "window_s": m["window_seconds"],
        "threshold_std": m["threshold_std"],
        "n_normal_win": m["n_windows"]["normal"],
        "n_anomaly_win": m["n_windows"]["anomaly"],
        "tn": m["confusion"]["tn"], "fp": m["confusion"]["fp"],
        "fn": m["confusion"]["fn"], "tp": m["confusion"]["tp"],
        "precision": m["precision"], "recall": m["recall"], "f1": m["f1"], "fpr": m["fpr"],
        "infer_ms_win_mean": m["inference_ms_per_window"]["mean"],
        "infer_ms_win_std": m["inference_ms_per_window"]["std"],
        "infer_ms_win_p95": m["inference_ms_per_window"]["p95"],
        "infer_ms_per_flow": m["inference_ms_per_flow"],
        "flows_win_median": m["flows_per_window"]["all"]["median"],
        "flows_win_mean": m["flows_per_window"]["all"]["mean"],
        "flows_win_normal_median": m["flows_per_window"]["normal"]["median"],
        "flows_win_anomaly_median": m["flows_per_window"]["anomaly"]["median"],
        "win_delay_s_mean": m["window_detection_delay_s"]["mean"],
        "win_delay_s_median": m["window_detection_delay_s"]["median"],
        "episodes": m["episodes"]["n"],
        "episodes_detected": m["episodes"]["detected"],
        "ttd_s_mean": m["episodes"]["time_to_detect_s"]["mean"],
        "ttd_s_median": m["episodes"]["time_to_detect_s"]["median"],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["cic", "unsw"], choices=list(DATASETS))
    p.add_argument("--windows", nargs="+", default=["30s", "1min", "2min", "4min"])
    p.add_argument("--base_data_path", default="/data/SDP_Dataset/Unified_model")
    p.add_argument("--threshold_std_cic", type=float, default=0.25)
    # 5 reproduces the reported UNSW-NB15 FedSAD result (results/UNSW_NB15/fedsad/fedsad_server.txt)
    p.add_argument("--threshold_std_unsw", type=float, default=5.0)
    p.add_argument("--episode_gap", default="5min")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--result_root", default=os.path.join(ROOT, "outputs"),
                   help="Output root; pass results to collect the archived window study")
    p.add_argument("--skip_existing", action="store_true",
                   help="Reuse window_metrics.json if it already exists")
    args = p.parse_args()

    rows = []
    for ds in args.datasets:
        cfg = DATASETS[ds]
        thr = getattr(args, f"threshold_std_{ds}")
        normal_file = os.path.join(args.base_data_path, cfg["graph_dir"], cfg["normal_file"])
        for w in args.windows:
            out_dir = os.path.join(args.result_root, cfg["name"], "window_sweep", window_tag(w))
            os.makedirs(out_dir, exist_ok=True)
            metrics_path = os.path.join(out_dir, "window_metrics.json")

            if not (args.skip_existing and os.path.exists(metrics_path)):
                print(f"\n===== {cfg['name']} | window={w} | threshold_std={thr} =====", flush=True)
                stats_path = os.path.join(out_dir, "normal_stats.pkl")
                train_single_model(SimpleNamespace(time_window=w), normal_file, stats_path)

                report = os.path.join(out_dir, "fedsad_server.txt")
                if os.path.exists(report):
                    os.remove(report)
                cmd = [args.python, os.path.join(BACKEND, "fedsad.py"),
                       "--eval_dataset", ds,
                       "--base_data_path", args.base_data_path,
                       "--time_window", w,
                       "--gsad_model", stats_path,
                       "--result_path", out_dir,
                       "--threshold_std", str(thr),
                       "--episode_gap", args.episode_gap]
                subprocess.run(cmd, cwd=ROOT, check=True)

            with open(metrics_path) as f:
                rows.append(flatten(json.load(f)))

    summary = pd.DataFrame(rows)
    out_csv = os.path.join(args.result_root, "window_sweep_summary.csv")
    if os.path.exists(out_csv):
        # keep rows of datasets/windows not re-run this time
        old = pd.read_csv(out_csv)
        key = set(zip(summary["dataset"], summary["time_window"]))
        old = old[[(d, w) not in key for d, w in zip(old["dataset"], old["time_window"])]]
        summary = pd.concat([old, summary], ignore_index=True)
    summary.to_csv(out_csv, index=False, float_format="%.4f")
    with pd.option_context("display.max_columns", None, "display.width", 200,
                           "display.float_format", "{:.4f}".format):
        print(summary[["dataset", "time_window", "precision", "recall", "f1", "fpr",
                       "infer_ms_win_mean", "flows_win_median", "win_delay_s_mean", "ttd_s_mean"]])
    print(f"\nSaved summary to {out_csv}")


if __name__ == "__main__":
    main()
