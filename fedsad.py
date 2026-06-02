import os
import pickle

import numpy as np
import networkx as nx
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import sys

import argparse as _ap
_pre = _ap.ArgumentParser(add_help=False)
_pre.add_argument("--eval_dataset", type=str, default="cic")
_pre.add_argument("--base_data_path", type=str, default="/data/SDP_Dataset/Unified_model")
_pre.add_argument("--n_samples", type=int, default=150_000)
_pre.add_argument("--batch_size", type=int, default=16)
_pre.add_argument("--dropout_rate", type=float, default=0.1)
_pre.add_argument("--percentile", type=int, default=82)
_pre.add_argument("--random_seed", type=int, default=123)
_pre.add_argument("--client_nums", type=int, default=15)
_pre.add_argument("--client_epochs", type=int, default=30)
_pre.add_argument("--set_verbose", type=int, default=2)
_pre.add_argument("--server_rounds", type=int, default=10)
_pre.add_argument("--threshold_std", type=float, default=0.25) # unsw 5
_known, _ = _pre.parse_known_args()
sys.argv = [sys.argv[0]]

from taae import TransformerAAE
from gsad import create_gsad_from_window, extract_gsad_features
from utils.get_datasets import sanitize_numeric, fedsad_data_preprocessing
import argparse
from utils.metrics import save_report, plt_confusion_matrix, print_latency, timed_step, GLOBAL_TIMER

# GPU Configuration
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2" 
TF_DEVICE='/CPU:0'

# Global constants for evaluation
TIME_WINDOW = '1T'
RANDOM_STATE = 123
THRESHOLD_PERCENTILE = 82  

# Global model and scaler instances
SCALER = MinMaxScaler()
TAAE_FEATURE_COLUMNS = []
TRAIN_NORMAL_SCALED = None
GSAD_MODEL=None
TAAE_MODEL = None 
TAAE_THRESHOLD = None  

def build_and_load_taae(n_features: int,taae_model_path: str):
    """Load simplified TAAE model"""
    global TAAE_MODEL

    try:
        with tf.device(TF_DEVICE):
            model = TransformerAAE(input_dim=n_features)
            dummy_input = tf.zeros((1, n_features), dtype=tf.float32)
            dummy_prior = tf.zeros((1, 1), dtype=tf.float32)
            _ = model(dummy_input, prior_labels=dummy_prior, training=False)
            model.load_weights(taae_model_path)
        TAAE_MODEL = model
        print(f"[INFO] TAAE model loaded successfully")
    except Exception as e:
        print(f"[WARN] Failed to load RNEP model: {e}")
        TAAE_MODEL = None

def gsad_model_predict(feat):
    """Compute DS mass based on graph feature deviation."""
    considered = 0
    violations = 0

    for name, val in feat.items():
        if name not in GSAD_MODEL.columns:
            continue
        considered += 1
        mean = float(GSAD_MODEL.loc["mean", name])
        std = float(GSAD_MODEL.loc["std", name])

        lower = mean - args.threshold_std * std
        upper = mean + args.threshold_std * std

        if not (lower <= val <= upper):
            violations += 1

    # If no valid features are considered, return default mass
    if considered == 0:
        return {"normal": 0.4, "anomaly": 0.2, "uncertain": 0.4}

    violation_ratio = violations / considered
    violation_ratio = float(np.clip(violation_ratio, 0.0, 1.0))

    normal_prob = 1.0 - violation_ratio
    anomaly_prob = violation_ratio

    # Uncertainty increases when ratio is near 0.5
    uncertainty = float(np.clip(
        0.15 + 0.6 * min(violation_ratio, 1.0 - violation_ratio),
        0.15, 0.6
    ))
    confidence = 1.0 - uncertainty

    normal_mass = confidence * normal_prob
    anomaly_mass = confidence * anomaly_prob
    uncertain_mass = uncertainty

    # Reduce reliability for small graphs
    small_graph = feat.get("num_nodes", 0) < 2 or feat.get("num_edges", 0) == 0
    reliability = confidence * (0.5 + 0.5 * violation_ratio)

    if small_graph:
        reliability *= 0.4

    reliability = float(np.clip(reliability, 0.1, 0.85))

    normal_mass *= reliability
    anomaly_mass *= reliability
    uncertain_mass = 1.0 - reliability + uncertain_mass * reliability

    return {
        "normal": float(np.clip(normal_mass, 1e-6, 1.0)),
        "anomaly": float(np.clip(anomaly_mass, 1e-6, 1.0)),
        "uncertain": float(np.clip(uncertain_mass, 1e-6, 1.0))
    }

def taae_model_predict(window_df):
    """Improved TAAE prediction - continuous probability estimation"""

    # Align columns with training features
    window_df = window_df.reindex(columns=TAAE_FEATURE_COLUMNS, fill_value=0)
    window_df = sanitize_numeric(window_df)
    arr = SCALER.transform(window_df.values)

    with tf.device(TF_DEVICE):
        reconstructed = TAAE_MODEL(arr, training=False)

    reconstruction_errors = np.mean(np.square(arr - reconstructed), axis=1)
    exceed = reconstruction_errors > TAAE_THRESHOLD
    ratio = float(np.sum(exceed)) / len(reconstruction_errors)

    ratio = float(np.clip(ratio, 0.0, 1.0))

    normal_prob = 1.0 - ratio
    anomaly_prob = ratio

    # Uncertainty increases near decision boundary
    uncertainty = float(np.clip(
        0.2 + 0.4 * min(ratio, 1.0 - ratio),
        0.2, 0.4
    ))
    confidence = 1.0 - uncertainty

    normal_mass = confidence * normal_prob
    anomaly_mass = confidence * anomaly_prob
    uncertain_mass = uncertainty

    return {
        "normal": float(np.clip(normal_mass, 1e-6, 1.0)),
        "anomaly": float(np.clip(anomaly_mass, 1e-6, 1.0)),
        "uncertain": float(np.clip(uncertain_mass, 1e-6, 1.0))
    }


def normalize_mass(mass):
    """Normalize Dempster-Shafer mass values to sum to 1."""
    normal = max(float(mass.get("normal", 0.0)), 0.0)
    anomaly = max(float(mass.get("anomaly", 0.0)), 0.0)
    uncertain = max(float(mass.get("uncertain", 0.0)), 0.0)
    total = normal + anomaly + uncertain
    if total <= 0:
        return {"normal": 0.4, "anomaly": 0.2, "uncertain": 0.4}
    return {
        "normal": normal / total,
        "anomaly": anomaly / total,
        "uncertain": uncertain / total
    }

def combine_dempster_shafer(m1, m2):
    """Dempster-Shafer combination (binary hypothesis + uncertainty)."""

    m1_norm = normalize_mass(m1)
    m2_norm = normalize_mass(m2)

    K = (
        m1_norm["normal"] * m2_norm["anomaly"] +
        m1_norm["anomaly"] * m2_norm["normal"]
    )

    # If conflict is too high, return default mass
    if K >= 1.0 - 1e-9:
        return {"normal": 0.4, "anomaly": 0.2, "uncertain": 0.4}

    denom = 1.0 - K

    normal = (
        m1_norm["normal"] * m2_norm["normal"] +
        m1_norm["normal"] * m2_norm["uncertain"] +
        m1_norm["uncertain"] * m2_norm["normal"]
    ) / denom

    anomaly = (
        m1_norm["anomaly"] * m2_norm["anomaly"] +
        m1_norm["anomaly"] * m2_norm["uncertain"] +
        m1_norm["uncertain"] * m2_norm["anomaly"]
    ) / denom

    uncertain = (m1_norm["uncertain"] * m2_norm["uncertain"]) / denom

    return {
        "normal": float(normal),
        "anomaly": float(anomaly),
        "uncertain": float(uncertain)
    }


@timed_step("GSAD")
def gsad_pipeline(df_window):
    G = create_gsad_from_window(df_window)
    feat = extract_gsad_features(G)
    return gsad_model_predict(feat)


@timed_step("TAAE")
def taae_pipeline(df_window):
    return taae_model_predict(df_window)


@timed_step("Fusion")
def fusion_pipeline(m_gsad, m_taae):
    return combine_dempster_shafer(m_gsad, m_taae)


def evaluate(df_gsad, df_taae, label,
             gsad_pipeline,
             taae_pipeline,
             fusion_pipeline):

    gb_gsad = df_gsad.groupby(pd.Grouper(freq=TIME_WINDOW))

    taae_idx = 0
    total_taae_rows = len(df_taae)

    y_true, y_pred = [], []
    results = []

    # tqdm 추가
    for window_time, df_gsad_window in tqdm(gb_gsad, desc=f"{label} Test"):
        if df_gsad_window.empty:
            continue

        # GSAD
        m_gsad = gsad_pipeline(df_gsad_window)

        # TAAE window
        size = len(df_gsad_window)
        if taae_idx + size <= total_taae_rows:
            df_taae_window = df_taae.iloc[taae_idx:taae_idx + size].copy()
            taae_idx += size
        else:
            remain = (taae_idx + size) - total_taae_rows
            df_taae_window = pd.concat([
                df_taae.iloc[taae_idx:],
                df_taae.iloc[:remain]
            ], ignore_index=True)
            taae_idx = remain

        m_taae = taae_pipeline(df_taae_window)

        # Fusion
        m_comb = fusion_pipeline(m_gsad, m_taae)
        pred = 1 if m_comb["anomaly"] > m_comb["normal"] else 0

        y_pred.append(pred)
        y_true.append(0 if label == "Normal" else 1)

        results.append({
            "window_time": window_time,
            "m_gsad": m_gsad,
            "m_taae": m_taae,
            "m_comb": m_comb,
            "pred": pred,
            "label": label
        })

    return y_pred, y_true, results

def calculate_taae_threshold():
    """Calculate TAAE threshold dynamically using training data"""

    global TAAE_THRESHOLD

    if TAAE_MODEL is not None and TRAIN_NORMAL_SCALED is not None:
        print(f"[INFO] Calculating TAAE threshold using {THRESHOLD_PERCENTILE}th percentile...")

        try:
            with tf.device(TF_DEVICE):
                train_reconstructed = TAAE_MODEL(TRAIN_NORMAL_SCALED, training=False)

            train_errors = np.mean(
                np.square(TRAIN_NORMAL_SCALED - train_reconstructed),
                axis=1
            )

            TAAE_THRESHOLD = np.percentile(train_errors, THRESHOLD_PERCENTILE)

            print(f"[INFO] TAAE threshold set to {TAAE_THRESHOLD:.8f} ({THRESHOLD_PERCENTILE}th percentile)")

        except Exception as e:
            print(f"[WARN] Failed to calculate TAAE threshold: {e}")
            TAAE_THRESHOLD = 0.00008
            print(f"[INFO] Using fallback TAAE threshold: {TAAE_THRESHOLD}")

    else:
        TAAE_THRESHOLD = 0.00008
        print(f"[INFO] Using default RNEP threshold: {TAAE_THRESHOLD}")

if __name__ == "__main__":
    args = _known

    dataset_configs = {
        "cic": {
            "taae_dir": f"{args.base_data_path}/cic_rnep",
            "gsad_dir": f"{args.base_data_path}/cic_graph",
            "taae_model": "results/CSE-CIC-IDS2018/taae/taae_cic_weights.h5",
            "gsad_model": "results/CSE-CIC-IDS2018/gsad/normal_stats.pkl",
            "result_path": "results/CSE-CIC-IDS2018/fedsad",
            "normal_file": "CIC_ae_normal.csv", 
            "anomaly_prefix": "CIC_anomaly_ae_"
        },
        "unsw": {
            "taae_dir": f"{args.base_data_path}/unsw_rnep",
            "gsad_dir": f"{args.base_data_path}/unsw_graph",
            "taae_model": "results/UNSW_NB15/taae/taae_unsw_weights.h5",
            "gsad_model": "results/UNSW_NB15/gsad/normal_stats.pkl",
            "result_path": "results/UNSW_NB15/fedsad",
            "normal_file": "UNSW_NB15_normal.csv", 
            "anomaly_prefix": "UNSW_NB15_anomaly_"
        }
    }

    config = dataset_configs.get(args.eval_dataset.lower())
    if not config:
        print(f"[ERROR] Dataset '{args.eval_dataset}' is not supported.")
        exit(1)

    os.makedirs(config["result_path"], exist_ok=True)
    model_path = os.path.join(config["result_path"], "fedsad.pkl")
    matrix_path = os.path.join(config["result_path"], "fedsad_cm.png")
    report_path = os.path.join(config["result_path"], "fedsad_server.txt")

    # TAAE data path
    taae_normal_file = os.path.join(config["taae_dir"], config["normal_file"])
    taae_anomaly_files = [
        os.path.join(config["taae_dir"], f) for f in os.listdir(config["taae_dir"]) 
        if f.startswith(config["anomaly_prefix"]) and f.endswith(".csv")
    ]

    # GSAD data path
    gsad_normal_file = os.path.join(config["gsad_dir"], config["normal_file"])
    gsad_anomaly_files = [
        os.path.join(config["gsad_dir"], f) for f in os.listdir(config["gsad_dir"]) 
        if f.startswith(config["anomaly_prefix"]) and f.endswith(".csv")
    ]

    # Load models
    with open(config["gsad_model"], "rb") as f:
        GSAD_MODEL = pickle.load(f)

    gsad_normal_test, gsad_anomaly_test, taae_normal_test, taae_anomaly_test, TRAIN_NORMAL_SCALED= fedsad_data_preprocessing( taae_normal_file,
                                                                                                                    taae_anomaly_files,
                                                                                                                    gsad_normal_file,
                                                                                                                    gsad_anomaly_files,
                                                                                                                    SCALER)
    TAAE_FEATURE_COLUMNS=taae_normal_test.columns.tolist()

    # Load TAAE model
    build_and_load_taae(len(TAAE_FEATURE_COLUMNS), config["taae_model"])

    # Dynamically calculate threshold
    calculate_taae_threshold()

    GLOBAL_TIMER.reset()

    # Evaluate normal data
    normal_pred, normal_test, normal_results =evaluate(
        gsad_normal_test, taae_normal_test, "Normal",
        gsad_pipeline,
        taae_pipeline,
        fusion_pipeline
    )

    # Evaluate anomaly data
    anomaly_pred, anomaly_test, anomaly_results =evaluate(
        gsad_anomaly_test, taae_anomaly_test, "Anomaly",
        gsad_pipeline,
        taae_pipeline,
        fusion_pipeline
    )

    print_latency("Overall")

    y_pred = normal_pred + anomaly_pred
    y_true = normal_test + anomaly_test
    results = normal_results + anomaly_results

    save_report(y_true, y_pred, report_path)
    plt_confusion_matrix(y_true, y_pred, matrix_path)

    with open(model_path, "wb") as f:
        pickle.dump(results, f)

    print(f"\nSaved ensemble results to {model_path}")

