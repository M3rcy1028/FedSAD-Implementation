import os
import pickle
import time
from datetime import datetime
from typing import List, Optional

import numpy as np
import networkx as nx
import pandas as pd
from tqdm import tqdm
import tensorflow as tf

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle

# Custom module imports
from tensorflow.keras.models import load_model
from taae import TransformerAAE
from gsad import create_gsad_from_window, extract_gsad_features
from utils import fedsad_data_preprocessing, sanitize_numeric, plt_confusion_matrix, save_report
from arguments import get_args, dataset_configs
args = get_args()

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


def evaluate(df_gsad, df_taae, label):
    """실시간성 측정을 위해 파이프라인별 시간 측정 로직이 추가된 버전"""
    gb_gsad = df_gsad.groupby(pd.Grouper(freq=TIME_WINDOW))

    taae_idx = 0
    total_taae_rows = len(df_taae)
    results = []
    y_true, y_pred = [], []

    # --- 시간 측정을 위한 리스트 ---
    gsad_times = []
    taae_times = []
    fusion_times = []
    total_pipeline_times = []

    for window_time, df_gsad_window in tqdm(gb_gsad, desc=f"{label} Test"):
        if df_gsad_window.empty:
            continue

        # 전체 파이프라인 시작 시간
        start_total = time.perf_counter()

        # 1. GSAD 파이프라인 (특징 추출 + 예측)
        G = create_gsad_from_window(df_gsad_window)
        feat = extract_gsad_features(G)
        start_gsad = time.perf_counter()
        m_gsad = gsad_model_predict(feat)
        end_gsad = time.perf_counter()
        
        # 2. TAAE 파이프라인 (윈도우 샘플링 + 예측)
        gsad_window_size = len(df_gsad_window)
        if taae_idx + gsad_window_size <= total_taae_rows:
            df_taae_window = df_taae.iloc[taae_idx:taae_idx + gsad_window_size].copy()
            taae_idx += gsad_window_size
        else:
            remaining = (taae_idx + gsad_window_size) - total_taae_rows
            df_taae_part1 = df_taae.iloc[taae_idx:].copy()
            df_taae_part2 = df_taae.iloc[:remaining].copy()
            df_taae_window = pd.concat([df_taae_part1, df_taae_part2], ignore_index=True)
            taae_idx = remaining
        start_taae = time.perf_counter()
        m_taae = taae_model_predict(df_taae_window)
        end_taae = time.perf_counter()

        # 3. DS Fusion (결합 + 최종 결정)
        start_fusion = time.perf_counter()
        m_comb = combine_dempster_shafer(m_gsad, m_taae)
        pred = 1 if m_comb["anomaly"] > m_comb["normal"] else 0
        end_fusion = time.perf_counter()

        end_total = time.perf_counter()

        # 각 단계별 소요 시간 기록
        gsad_times.append(end_gsad - start_gsad)
        taae_times.append(end_taae - start_taae)
        fusion_times.append(end_fusion - start_fusion)
        total_pipeline_times.append(end_total - start_total)

        y_pred.append(pred)
        y_true.append(0 if label == "Normal" else 1)
        results.append({
            "window_time": window_time,
            "m_gsad": m_gsad, "m_taae": m_taae, "m_comb": m_comb,
            "pred": pred, "label": label
        })

    # --- 결과 출력 (Revision용 통계) ---
    avg_total = np.mean(total_pipeline_times) * 1000
    std_total = np.std(total_pipeline_times) * 1000 # 표준편차 계산 (ms 단위)
    
    print(f"\n⏱️  [{label}] Detailed Latency Analysis (per {TIME_WINDOW} window)")
    print(f"  - GSAD Step:    {np.mean(gsad_times)*1000:.4f} ± {np.std(gsad_times)*1000:.4f} ms")
    print(f"  - TAAE Step:    {np.mean(taae_times)*1000:.4f} ± {np.std(taae_times)*1000:.4f} ms")
    print(f"  - DS Fusion:      {np.mean(fusion_times)*1000:.4f} ± {np.std(fusion_times)*1000:.4f} ms")
    print("-" * 60)
    # 전체 파이프라인 평균 ± 표준편차
    print(f"  - Total Pipeline Time: {avg_total:.4f} ± {std_total:.4f} ms") 
    print(f"  - Throughput:          {1/(avg_total/1000):.2f} windows/sec")

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

    # args = get_args()

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

    # Evaluate normal data
    normal_pred, normal_test, normal_results = evaluate(
        gsad_normal_test, taae_normal_test, "Normal"
    )

    # Evaluate anomaly data
    anomaly_pred, anomaly_test, anomaly_results = evaluate(
        gsad_anomaly_test, taae_anomaly_test, "Anomaly"
    )

    y_pred = normal_pred + anomaly_pred
    y_true = normal_test + anomaly_test
    results = normal_results + anomaly_results

    save_report(y_true, y_pred, report_path)
    plt_confusion_matrix(y_true, y_pred, matrix_path)

    with open(model_path, "wb") as f:
        pickle.dump(results, f)

    print(f"\nSaved ensemble results to {model_path}")

