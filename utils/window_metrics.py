"""Metrics for the detection-window-size experiment (FedSAD window sweep).

All metrics treat Anomaly as the positive class and are computed per window,
which is the unit FedSAD makes a decision on.
"""
import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset


def _describe(values):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"mean": None, "std": None, "median": None, "p95": None, "min": None, "max": None}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _split_episodes(timestamps, gap):
    """Split sorted anomaly timestamps into episodes separated by more than `gap`."""
    ts = pd.Series(pd.DatetimeIndex(timestamps).dropna().sort_values())
    if ts.empty:
        return []
    new_episode = ts.diff() > pd.Timedelta(gap)
    episode_id = new_episode.cumsum()
    return [(g.iloc[0], g.iloc[-1]) for _, g in ts.groupby(episode_id)]


def compute_window_metrics(results, time_window, episode_gap="5min", anomaly_timestamps=None):
    """
    results: list of per-window dicts from fedsad.evaluate (needs window_time, pred, label,
             n_rows, first_ts, infer_ms)
    time_window: pandas offset alias used for windowing
    anomaly_timestamps: flow timestamps of the anomaly test stream (for episode time-to-detect)
    """
    window = pd.Timedelta(to_offset(time_window))
    df = pd.DataFrame(results)
    y_true = (df["label"] != "Normal").astype(int).to_numpy()
    y_pred = df["pred"].astype(int).to_numpy()

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0

    # Inference time per window (GSAD + TAAE + DS fusion, measured end to end)
    infer_ms = df["infer_ms"].to_numpy(dtype=float)
    total_rows = int(df["n_rows"].sum())

    # Window-level detection delay for correctly detected anomaly windows:
    # time from the first flow in the window until the window closes, plus inference.
    window_end = pd.to_datetime(df["window_time"]) + window
    wait_s = (window_end - pd.to_datetime(df["first_ts"])).dt.total_seconds().to_numpy()
    tp_mask = (y_true == 1) & (y_pred == 1)
    window_delay_s = wait_s[tp_mask] + infer_ms[tp_mask] / 1000.0

    # Episode-level time-to-detect: from the first flow of an attack episode until the end
    # of the first anomaly window inside that episode flagged as anomalous (+ its inference).
    episodes = []
    if anomaly_timestamps is not None:
        anom = df[df["label"] != "Normal"].copy()
        anom["window_time"] = pd.to_datetime(anom["window_time"])
        anom = anom.sort_values("window_time")
        for start, end in _split_episodes(anomaly_timestamps, episode_gap):
            in_ep = anom[(anom["window_time"] + window > start) & (anom["window_time"] <= end)]
            hit = in_ep[in_ep["pred"] == 1]
            if hit.empty:
                episodes.append({"start": start, "end": end, "detected": False, "ttd_s": None})
                continue
            first = hit.iloc[0]
            ttd = (first["window_time"] + window - start).total_seconds() + first["infer_ms"] / 1000.0
            episodes.append({"start": start, "end": end, "detected": True, "ttd_s": float(ttd)})
    ttd_values = [e["ttd_s"] for e in episodes if e["detected"]]

    return {
        "time_window": time_window,
        "window_seconds": window.total_seconds(),
        "n_windows": {"normal": int((y_true == 0).sum()), "anomaly": int((y_true == 1).sum())},
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "accuracy": (tp + tn) / len(df) if len(df) else 0.0,
        "inference_ms_per_window": _describe(infer_ms),
        "inference_ms_per_flow": float(infer_ms.sum() / total_rows) if total_rows else None,
        "flows_per_window": {
            "all": _describe(df["n_rows"]),
            "normal": _describe(df.loc[y_true == 0, "n_rows"]),
            "anomaly": _describe(df.loc[y_true == 1, "n_rows"]),
        },
        "window_detection_delay_s": _describe(window_delay_s),
        "episode_gap": episode_gap,
        "episodes": {
            "n": len(episodes),
            "detected": len(ttd_values),
            "detection_rate": len(ttd_values) / len(episodes) if episodes else None,
            "time_to_detect_s": _describe(ttd_values),
        },
    }


def _fmt(v, nd=4):
    return "n/a" if v is None else f"{v:.{nd}f}"


def format_window_metrics(m):
    lines = [
        f"\n🪟 [Window Metrics] time_window={m['time_window']} ({m['window_seconds']:.0f}s)",
        f"Windows: normal={m['n_windows']['normal']} anomaly={m['n_windows']['anomaly']} | "
        f"TN={m['confusion']['tn']} FP={m['confusion']['fp']} FN={m['confusion']['fn']} TP={m['confusion']['tp']}",
        f"Precision={_fmt(m['precision'])} Recall={_fmt(m['recall'])} F1={_fmt(m['f1'])} FPR={_fmt(m['fpr'])}",
        f"Inference per window: {_fmt(m['inference_ms_per_window']['mean'], 3)} ± "
        f"{_fmt(m['inference_ms_per_window']['std'], 3)} ms (p95 {_fmt(m['inference_ms_per_window']['p95'], 3)}) | "
        f"per flow: {_fmt(m['inference_ms_per_flow'], 5)} ms",
        f"Flows per window (median): all={_fmt(m['flows_per_window']['all']['median'], 0)} "
        f"normal={_fmt(m['flows_per_window']['normal']['median'], 0)} "
        f"anomaly={_fmt(m['flows_per_window']['anomaly']['median'], 0)}",
        f"Window detection delay (TP windows): mean={_fmt(m['window_detection_delay_s']['mean'], 2)}s "
        f"median={_fmt(m['window_detection_delay_s']['median'], 2)}s",
        f"Episodes (gap>{m['episode_gap']}): {m['episodes']['detected']}/{m['episodes']['n']} detected | "
        f"time-to-detect mean={_fmt(m['episodes']['time_to_detect_s']['mean'], 2)}s "
        f"median={_fmt(m['episodes']['time_to_detect_s']['median'], 2)}s",
    ]
    return "\n".join(lines)
