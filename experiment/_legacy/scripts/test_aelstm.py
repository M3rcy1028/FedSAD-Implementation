import os
import sys
import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score, 
    confusion_matrix, roc_curve, auc
)
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import seaborn as sns
import random
from models.model_aelstm import AE_LSTM
from AE_LSTM import get_datasets_ciciot_semi

# --------------------------------------------------
# helper functions (same as TAAE script)
# --------------------------------------------------

def _clean_dataframe(df):
    """Drops 'Label'/'label' columns, replaces 'inf'/'nan' values with 0, and clips large values."""
    if 'Label' in df.columns:
        df = df.drop(columns=['Label'])
    if 'label' in df.columns:
        df = df.drop(columns=['label'])
        
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    df = np.clip(df, -1e6, 1e6) 
    return df

def _plot_and_print_cm(y_test, y_pred, save_path, labels, title):
    """Prints the Confusion Matrix as a DataFrame and saves it as a seaborn heatmap."""
    cm = confusion_matrix(y_test, y_pred)
    
    try:
        tn, fp, fn, tp = cm.ravel()
        cm_table = pd.DataFrame(
            [[tn, fp], [fn, tp]],
            index=[f'Actual {labels[0]}', f'Actual {labels[1]}'],
            columns=[f'Predicted {labels[0]}', f'Predicted {labels[1]}']
        )
        print("\n[Confusion Matrix]")
        print(cm_table)
    except ValueError:
        print(f"\n[Confusion Matrix] (Raw)\n{cm}")

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=[f'Predicted {labels[0]}', f'Predicted {labels[1]}'],
                yticklabels=[f'Actual {labels[0]}', f'Actual {labels[1]}'])
    plt.title(title)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def reshape_for_sequence(X, timesteps=10, features=12):
    """
    Reshapes 2D (N, F) data into 3D (N, T, F) sequence data (with padding/truncation)
    (helper function from main_aelstm.py)
    """
    n_samples, n_feats = X.shape
    target_len = timesteps * features
    
    if n_feats < target_len:
        pad = np.zeros((n_samples, target_len - n_feats))
        X = np.concatenate([X, pad], axis=1)
    elif n_feats > target_len:
        X = X[:, :target_len]
        
    return X.reshape(-1, timesteps, features)

# --------------------------------------------------
# dataset config (same as TAAE script)
# --------------------------------------------------
DATASET_CONFIG = {
    "KDD99": {
        "base_dir": "./KDD99/KDD99_split",
        "normal_file": "KDD99_normal.csv",
        "anomaly_prefix": "KDD99_anomaly_",
        "merged_anomaly_file": "KDD99_anomaly.csv",
        "attack_map": {
            0: "back", 1: "buffer_overflow", 2: "ftp_write", 3: "guess_passwd",
            4: "imap", 5: "ipsweep", 6: "land", 7: "loadmodule", 8: "multihop",
            9: "neptune", 10: "nmap", 11: "perl", 12: "phf", 13: "portsweep",
            14: "rootkit", 15: "satan", 16: "spy", 17: "warezclient", 18: "warezmaster"
        }
    },
    "CSE-CIC-IDS2018": {
        "base_dir": "./CIC2018/ae_datas_sampled",
        "normal_file": "CIC_ae_normal.csv",
        "anomaly_prefix": "CIC_anomaly_ae_",
        "merged_anomaly_file": "CIC_ae_anomaly.csv",
        "attack_map": {
            1: "DDOS attack-HOIC", 2: "DDoS attacks-LOIC-HTTP", 3: "DoS attacks-Hulk",
            4: "Bot", 5: "FTP-BruteForce", 6: "SSH-Bruteforce", 7: "Infiltration",
            8: "DoS attacks-SlowHTTPTest", 9: "DoS attacks-GoldenEye", 10: "DoS attacks-Slowloris",
            11: "DDOS attack-LOIC-UDP", 12: "Brute Force -Web", 13: "Brute Force -XSS", 14: "SQL Injection"
        }
    },
    "InSDN": {
        "base_dir": "./InSDN/ae_datas",
        "normal_file": "InSDN_normal.csv", # 83 features
        "anomaly_prefix": "InSDN_anomaly_",
        "merged_anomaly_file": "InSDN_anomaly.csv",
        "attack_map": {
            0: "BFA", 1: "BOTNET", 2: "DDoS", 3: "DoS",
            4: "Probe", 5: "U2R", 6: "Web-Attack"
        }
    },
    "CICIoT2023": {
        "base_dir": "./CICIoT2023/ciciot_rnep",
        "normal_file": "CICIoT_ae_normal.csv", # 78 features (same as CSE-CIC-IDS2018)
        "anomaly_prefix": "CICIoT_anomaly_ae_",
        "merged_anomaly_file": "CICIoT_ae_anomaly.csv",
        "attack_map": {
            1: "DDoS-UDP_Flood", 2: "DDoS-TCP_Flood", 3: "DDoS-PSHACK_Flood", 4: "DDoS-SYN_Flood",
            5: "DDoS-RSTFINFlood", 6: "DDoS-SynonymousIP_Flood", 7: "DoS-UDP_Flood",
            8: "DoS-TCP_Flood", 9: "DoS-SYN_Flood", 10: "Mirai-udpplain", 11: "VulnerabilityScan",
            12: "DDoS-UDP_Fragmentation", 13: "DDoS-ACK_Fragmentation", 14: "DNS_Spoofing",
            15: "Recon-HostDiscovery", 16: "Recon-OSScan", 17: "Recon-PortScan", 18: "DoS-HTTP_Flood",
            19: "DDoS-HTTP_Flood", 20: "DDoS-SlowLoris", 21: "DictionaryBruteForce",
            22: "BrowserHijacking", 23: "CommandInjection", 24: "SqlInjection", 25: "XSS",
            26: "Backdoor_Malware", 27: "Uploading_Attack"
        }
    },
    "UNSW_NB15": {
        "base_dir": "./UNSW_NB15/ae_datas",
        "normal_file": "UNSW_NB15_normal.csv", # 42 features
        "anomaly_prefix": "UNSW_NB15_anomaly_",
        "merged_anomaly_file": "UNSW_NB15_anomaly.csv",
        "attack_map": {
            0: "analysis", 1: "backdoor", 2: "dos", 3: "exploits",
            4: "fuzzers", 5: "generic", 6: "Web-reconnaissance",
            6: "shellcode", 7: "worms"
        }
    }
}

# --------------------------------------------------
# AE-LSTM evaluation function by type
# --------------------------------------------------
def evaluate_aelstm_by_type(model, dataset_name, model_params, train_split_ratio=0.8, percentile=95, beta=0.5):
    """
    Evaluates AE-LSTM model performance by type using two methods: (1) reconstruction error and (2) classifier probability.

    :param model: trained AE-LSTM model
    :param dataset_name: one of "KDD99", "CSE-CIC-IDS2018", "InSDN"
    :param model_params: dictionary for model input (e.g., {'timesteps': 10, 'features': 12})
    :param train_split_ratio: ratio for scaler training and normal test set split
    :param percentile: percentile for computing the reconstruction error threshold
    """
    
    # load config
    try:
        config = DATASET_CONFIG[dataset_name]
    except KeyError:
        print(f"Error: No config found for dataset '{dataset_name}'")
        return

    timesteps = model_params['timesteps']
    features = model_params['features']
    
    base_dir = config["base_dir"]
    normal_path = os.path.join(base_dir, config["normal_file"])
    anomaly_prefix = config["anomaly_prefix"]
    merged_file = config["merged_anomaly_file"]
    attack_map = config["attack_map"]
    
    # result save path
    save_dir = "./ae_lstm_eval"
    os.makedirs(save_dir, exist_ok=True)

    # search anomaly file list
    anomaly_files = sorted([
        f for f in os.listdir(base_dir)
        if f.startswith(anomaly_prefix) and f.endswith(".csv")
    ])
    merged_path = os.path.join(base_dir, merged_file)
    if os.path.exists(merged_path):
        anomaly_files = [merged_file] + anomaly_files
    
    print(f"\nEvaluating AE-LSTM for dataset: {dataset_name.upper()}")
    print(f"Found {len(anomaly_files)} anomaly datasets for evaluation")

    results = []

    # load, clean, and split normal data (80% for scaler training / 20% for testing)
    df_normal = pd.read_csv(normal_path)
    df_normal = shuffle(df_normal, random_state=42)
    split_point = int(len(df_normal) * train_split_ratio)
    df_normal_train = df_normal.iloc[:split_point] # for scaler training
    df_normal_test = df_normal.iloc[split_point:] # for actual testing

    df_normal_train = _clean_dataframe(df_normal_train)
    df_normal_test = _clean_dataframe(df_normal_test)
    
    print(f"Normal data: Train(for scaler)={len(df_normal_train):,}, Test(for eval)={len(df_normal_test):,}")

    # normalization (Scaler)
    scaler = MinMaxScaler()
    X_train_flat = scaler.fit_transform(df_normal_train.values) 
    X_normal_test_flat = scaler.transform(df_normal_test.values)
    
    # Reshape (Normal Test)
    X_normal_test_seq = reshape_for_sequence(X_normal_test_flat, timesteps, features)
    y_normal_test = np.zeros(len(X_normal_test_seq))

    # compute threshold
    # (Reconstruction) based on training data (normal 80%)
    X_train_seq = reshape_for_sequence(X_train_flat, timesteps, features)
    preds_train_dict = model.predict(X_train_seq, verbose=0)
    recon_train = preds_train_dict["decoded"]
    # MSE for 3D (N, T, F) data uses axis=(1, 2)
    train_errors = np.mean(np.square(X_train_seq - recon_train), axis=(1, 2))
    threshold_recon = np.percentile(train_errors, percentile)
    
    # (Classifier) fixed threshold
    threshold_cls = 0.5
    print(f"\nRecon Threshold ({percentile}th percentile): {threshold_recon:.6f}")
    print(f"Classifier Threshold (fixed): {threshold_cls}")

    recon_errors_by_attack = {} # store errors for visualization
    cls_probs_by_attack = {}    # store probabilities for visualization
    numeric_keys = []

    # evaluate per anomaly file
    for file in anomaly_files:
        anomaly_path = os.path.join(base_dir, file)
        df_anomaly = pd.read_csv(anomaly_path)
        df_anomaly = _clean_dataframe(df_anomaly)

        if df_anomaly.empty:
            print(f"\nWarning: '{file}' is empty or became empty after cleaning. Skipping.")
            continue

        # scale and reshape anomaly data
        X_anomaly_flat = scaler.transform(df_anomaly.values)
        X_anomaly_seq = reshape_for_sequence(X_anomaly_flat, timesteps, features)
        y_anomaly = np.ones(len(X_anomaly_seq))
        
        # construct test set (Normal-Test + Anomaly)
        X_test_seq = np.concatenate([X_normal_test_seq, X_anomaly_seq])
        y_test = np.concatenate([y_normal_test, y_anomaly])

        # predict (model output is a dictionary)
        preds_test_dict = model.predict(X_test_seq, verbose=0)
        recon_test = preds_test_dict["decoded"]
        probs_test = preds_test_dict["pred"].flatten() # probability of shape (N,)

        # --- [Method 1: Reconstruction error-based] ---
        test_errors = np.mean(np.square(X_test_seq - recon_test), axis=(1, 2))
        y_pred_recon = (test_errors > threshold_recon).astype(int)

        acc_recon = accuracy_score(y_test, y_pred_recon)
        f1_recon = f1_score(y_test, y_pred_recon, zero_division=0)
        prec_recon = precision_score(y_test, y_pred_recon, zero_division=0)
        rec_recon = recall_score(y_test, y_pred_recon, zero_division=0)
        pred_normal_recon = int(np.sum(y_pred_recon == 0))
        pred_anomaly_recon = int(np.sum(y_pred_recon == 1))

        # --- [Method 2: Classifier probability-based] ---
        y_pred_cls = (probs_test > threshold_cls).astype(int)

        acc_cls = accuracy_score(y_test, y_pred_cls)
        f1_cls = f1_score(y_test, y_pred_cls, zero_division=0)
        prec_cls = precision_score(y_test, y_pred_cls, zero_division=0)
        rec_cls = recall_score(y_test, y_pred_cls, zero_division=0)
        pred_normal_cls = int(np.sum(y_pred_cls == 0))
        pred_anomaly_cls = int(np.sum(y_pred_cls == 1))

        # --- [Method 3: Fusion-based] ---
        beta = 0.5  # weighted ratio of reconstruction error and classification probability (0.5 = equal weight)

        # scale reconstruction error to [0,1] range
        e_min, e_max = np.min(test_errors), np.max(test_errors)
        e_scaled = (test_errors - e_min) / (e_max - e_min + 1e-12)
        
        # compute fusion score
        fusion_score = beta * e_scaled + (1 - beta) * probs_test
        
        # set threshold (default: 0.5)
        threshold_fusion = 0.5
        y_pred_fusion = (fusion_score > threshold_fusion).astype(int)

        acc_fusion = accuracy_score(y_test, y_pred_fusion)
        f1_fusion = f1_score(y_test, y_pred_fusion, zero_division=0)
        prec_fusion = precision_score(y_test, y_pred_fusion, zero_division=0)
        rec_fusion = recall_score(y_test, y_pred_fusion, zero_division=0)

        # --- [output] ---
        print(f"\n{file}")
        print(f"  Samples: {len(X_test_seq)} (Normal: {len(y_normal_test)}, Anomaly: {len(y_anomaly)})")
        print(f"  [Recon] Acc={acc_recon:.6f}, F1={f1_recon:.6f} | Pred Cnts: N={pred_normal_recon:,}, A={pred_anomaly_recon:,}")
        print(f"  [Class] Acc={acc_cls:.6f}, F1={f1_cls:.6f} | Pred Cnts: N={pred_normal_cls:,}, A={pred_anomaly_cls:,}")
        print(f"  [Fusion β={beta}] Acc={acc_fusion:.6f}, F1={f1_fusion:.6f} | Prec={prec_fusion:.6f}, Rec={rec_fusion:.6f}")

        # generate Confusion Matrix and ROC Curve for merged file
        if file == merged_file:
            # === CM (Recon) ===
            cm_save_path_r = os.path.join(save_dir, f"{dataset_name}_recon_cm.png")
            title_r = f'AE-LSTM (Recon) CM - {dataset_name.upper()} (P={percentile})'
            _plot_and_print_cm(y_test, y_pred_recon, cm_save_path_r, ['Normal', 'Anomaly'], title_r)
            print(f"  Saved Recon CM -> {cm_save_path_r}")
            
            # === CM (Class) ===
            cm_save_path_c = os.path.join(save_dir, f"{dataset_name}_cls_cm.png")
            title_c = f'AE-LSTM (Classifier) CM - {dataset_name.upper()}'
            _plot_and_print_cm(y_test, y_pred_cls, cm_save_path_c, ['Normal', 'Anomaly'], title_c)
            print(f"  Saved Classifier CM -> {cm_save_path_c}")

            # === ROC (Recon) ===
            fpr_r, tpr_r, _ = roc_curve(y_test, test_errors)
            roc_auc_r = auc(fpr_r, tpr_r)
            roc_save_path_r = os.path.join(save_dir, f"{dataset_name}_recon_roc.png")
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr_r, tpr_r, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc_r:.4f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.title(f'AE-LSTM (Recon) ROC Curve - {dataset_name.upper()}')
            plt.xlabel('False Positive Rate (FPR)')
            plt.ylabel('True Positive Rate (TPR)')
            plt.legend(loc="lower right")
            plt.grid(True, linestyle="--", alpha=0.4)
            plt.tight_layout()
            plt.savefig(roc_save_path_r, dpi=300)
            plt.close()
            print(f"  Saved Recon ROC -> {roc_save_path_r} (AUC = {roc_auc_r:.4f})")
            
            # === ROC (Class) ===
            fpr_c, tpr_c, _ = roc_curve(y_test, probs_test)
            roc_auc_c = auc(fpr_c, tpr_c)
            roc_save_path_c = os.path.join(save_dir, f"{dataset_name}_cls_roc.png")
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr_c, tpr_c, color='darkgreen', lw=2, label=f'ROC curve (AUC = {roc_auc_c:.4f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.title(f'AE-LSTM (Classifier) ROC Curve - {dataset_name.upper()}')
            plt.xlabel('False Positive Rate (FPR)')
            plt.ylabel('True Positive Rate (TPR)')
            plt.legend(loc="lower right")
            plt.grid(True, linestyle="--", alpha=0.4)
            plt.tight_layout()
            plt.savefig(roc_save_path_c, dpi=300)
            plt.close()
            print(f"  Saved Classifier ROC -> {roc_save_path_c} (AUC = {roc_auc_c:.4f})")

        # save results
        results.append({
            "File": file, "Samples": len(X_test_seq), 
            "Acc_Recon": acc_recon, "Prec_Recon": prec_recon, "Rec_Recon": rec_recon, "F1_Recon": f1_recon,
            "Pred_N_Recon": pred_normal_recon, "Pred_A_Recon": pred_anomaly_recon,
            "Acc_Cls": acc_cls, "Prec_Cls": prec_cls, "Rec_Cls": rec_cls, "F1_Cls": f1_cls,
            "Acc_Fusion": acc_fusion, "Prec_Fusion": prec_fusion,
            "Rec_Fusion": rec_fusion, "F1_Fusion": f1_fusion,
            "Pred_N_Cls": pred_normal_cls, "Pred_A_Cls": pred_anomaly_cls
        })

        # store data for visualization (excluding merged file)
        if file != merged_file:
            try:
                num_str = file.replace(anomaly_prefix, "").replace(".csv", "")
                attack_num = int(num_str)
                attack_label = attack_map.get(attack_num, f"attack_{attack_num}")
                
                # predict anomaly data only
                preds_anomaly_dict = model.predict(X_anomaly_seq, verbose=0)
                
                # store reconstruction errors
                errors_anomaly_only = np.mean(np.square(X_anomaly_seq - preds_anomaly_dict["decoded"]), axis=(1,2))
                recon_errors_by_attack[attack_num] = (attack_label, errors_anomaly_only)
                
                # store classifier probabilities
                probs_anomaly_only = preds_anomaly_dict["pred"].flatten()
                cls_probs_by_attack[attack_num] = (attack_label, probs_anomaly_only)

                numeric_keys.append(attack_num)
            except Exception as e:
                print(f"Warning: Could not parse attack number from '{file}'. Skipping for plot. Error: {e}")

    # save result summary
    df = pd.DataFrame(results)
    summary_save_path = os.path.join(save_dir, f"{dataset_name}_summary.csv")
    
    df.to_csv(summary_save_path, index=False)
    print(f"\nSaved summary -> {summary_save_path}")
    print(df.round(6))

    # Reconstruction Error histogram by attack type
    if recon_errors_by_attack:
        plot_save_path_r = os.path.join(save_dir, f"{dataset_name}_recon_distribution.png")
        plt.figure(figsize=(12, 7))
        
        preds_normal_dict = model.predict(X_normal_test_seq, verbose=0)
        errors_normal_test = np.mean(np.square(X_normal_test_seq - preds_normal_dict["decoded"]), axis=(1,2))
        
        plt.hist(errors_normal_test, bins=200, alpha=0.6, label="Normal", color="green", density=True)

        numeric_keys = sorted(set(numeric_keys))
        n_attacks = len(numeric_keys)
        palette = plt.cm.tab20(np.linspace(0, 1, max(3, n_attacks)))

        for i, atk_num in enumerate(numeric_keys):
            atk_name, errs = recon_errors_by_attack[atk_num]
            plt.hist(errs, bins=200, alpha=0.5, label=f"{atk_name}", color=palette[i % 20], density=True)

        plt.axvline(threshold_recon, color="blue", linestyle="--", label=f"Threshold ({threshold_recon:.6f})")
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Reconstruction Error (log scale)")
        plt.ylabel("Density (log scale)")
        plt.title(f"AE-LSTM Recon Error Distribution - ({dataset_name.upper()})")
        plt.legend(fontsize=8, loc="upper right", ncol=1)
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig(plot_save_path_r, dpi=300)
        plt.close()
        print(f"Saved Recon distribution plot -> {plot_save_path_r}")

    # Classifier Probability histogram by attack type
    if cls_probs_by_attack:
        plot_save_path_c = os.path.join(save_dir, f"{dataset_name}_cls_distribution.png")
        plt.figure(figsize=(12, 7))
        
        preds_normal_dict = model.predict(X_normal_test_seq, verbose=0)
        probs_normal_test = preds_normal_dict["pred"].flatten()
        
        plt.hist(probs_normal_test, bins=100, alpha=0.6, label="Normal", color="green", density=True, range=(0,1))

        numeric_keys = sorted(set(numeric_keys))
        n_attacks = len(numeric_keys)
        palette = plt.cm.gist_rainbow(np.linspace(0, 1, max(3, n_attacks))) 

        for i, atk_num in enumerate(numeric_keys):
            atk_name, probs = cls_probs_by_attack[atk_num]
            plt.hist(probs, bins=100, alpha=0.5, label=f"{atk_name}", color=palette[i % n_attacks], density=True, range=(0,1))

        plt.axvline(threshold_cls, color="blue", linestyle="--", label=f"Threshold ({threshold_cls})")
        plt.xlabel("Model Output Probability (0.0=Normal, 1.0=Anomaly)")
        plt.ylabel("Density")
        plt.title(f"AE-LSTM Classifier Probability Distribution - ({dataset_name.upper()})")
        plt.legend(fontsize=8, loc="upper right", ncol=1)
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.xlim(0.0, 1.0)
        plt.tight_layout()
        plt.savefig(plot_save_path_c, dpi=300)
        plt.close()
        print(f"Saved Classifier distribution plot -> {plot_save_path_c}")


# --------------------------------------------------
# model load and evaluation config (based on main_aelstm.py)
# --------------------------------------------------

def evaluate_ciciot_aelstm(model, base_dir, save_dir, random_seed=123,
                         percentile=85, batch_size=512):
    """Evaluate the exact held-out half from the CICIoT training loader."""
    if not 0 <= percentile <= 100 or batch_size < 1:
        raise ValueError("percentile must be in [0, 100] and batch_size must be positive")
    X_train, y_train, X_test, y_test, split = get_datasets_ciciot_semi(
        base_dir=str(base_dir), random_seed=random_seed,
        train_ratio=0.5, anomaly_ratio=0.5, return_metadata=True,
    )
    if np.intersect1d(split["train_indices"], split["test_indices"]).size:
        raise ValueError("Training and evaluation source rows overlap")

    print("[CICIoT2023] Calibrating on normal-train rows...", flush=True)
    normal_train = X_train[y_train == 0]
    decoded_train = model.predict(normal_train, batch_size=batch_size, verbose=0)["decoded"]
    normal_errors = np.mean(np.square(normal_train - decoded_train), axis=(1, 2))
    threshold = float(np.percentile(normal_errors, percentile))
    del X_train, normal_train, decoded_train

    print("[CICIoT2023] Evaluating held-out normal and attack rows...", flush=True)
    outputs = model.predict(X_test, batch_size=batch_size, verbose=0)
    errors = np.mean(np.square(X_test - outputs["decoded"]), axis=(1, 2))
    probabilities = outputs["pred"].reshape(-1)
    if not (np.isfinite(errors).all() and np.isfinite(probabilities).all()):
        raise ValueError("Model produced non-finite evaluation scores")
    predictions = {"Recon": (errors > threshold).astype(int),
                   "Cls": (probabilities > 0.5).astype(int)}
    scores = {"Recon": errors, "Cls": probabilities}

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    attack_map = pd.read_csv(Path(base_dir).parent / "ciciot_attack_map.csv")
    attack_ids = split["test_attack_ids"]
    rows = []
    groups = [(0, "All", np.ones(len(y_test), dtype=bool))]
    groups.extend((int(row.id), row[1], (y_test == 0) | (attack_ids == row.id))
                  for row in attack_map.itertuples(index=False))
    for attack_id, name, mask in groups:
        truth = y_test[mask]
        result = {"Attack_ID": attack_id, "Attack": name, "Samples": int(mask.sum()),
                  "Normal": int((truth == 0).sum()), "Anomaly": int((truth == 1).sum())}
        for method, predicted in predictions.items():
            fpr, tpr, _ = roc_curve(truth, scores[method][mask])
            result.update({
                f"Acc_{method}": accuracy_score(truth, predicted[mask]),
                f"Prec_{method}": precision_score(truth, predicted[mask], zero_division=0),
                f"Rec_{method}": recall_score(truth, predicted[mask], zero_division=0),
                f"F1_{method}": f1_score(truth, predicted[mask], zero_division=0),
                f"AUC_{method}": auc(fpr, tpr),
            })
        rows.append(result)
    summary = pd.DataFrame(rows)
    summary.to_csv(save_dir / "CICIoT2023_summary.csv", index=False)
    pd.DataFrame({
        "flow_index": split["test_indices"], "attack_id": attack_ids,
        "y_true": y_test, "reconstruction_error": errors,
        "classifier_probability": probabilities,
        "pred_recon": predictions["Recon"], "pred_cls": predictions["Cls"],
    }).to_csv(save_dir / "CICIoT2023_predictions.csv", index=False)
    metadata = {"dataset": "CICIoT2023", "random_seed": random_seed,
                "normal_train_ratio": 0.5, "attack_train_ratio": 0.5,
                "train_rows": len(y_train), "test_rows": len(y_test),
                "normal_test_rows": int((y_test == 0).sum()),
                "attack_test_rows": int((y_test == 1).sum()),
                "percentile": percentile, "reconstruction_threshold": threshold,
                "classifier_threshold": 0.5,
                "threshold_source": "normal-train reconstruction errors",
                "source_rows_disjoint": True,
                "overall": rows[0]}
    (save_dir / "CICIoT2023_metrics.json").write_text(json.dumps(metadata, indent=2) + "\n")
    for method, predicted in predictions.items():
        label = method.lower()
        _plot_and_print_cm(y_test, predicted, str(save_dir / f"CICIoT2023_{label}_cm.png"),
                           ["Normal", "Anomaly"], f"AE-LSTM ({method}) - CICIoT2023")
        fpr, tpr, _ = roc_curve(y_test, scores[method])
        plt.figure(figsize=(7, 5))
        plt.plot(fpr, tpr, label=f"AUC = {auc(fpr, tpr):.4f}")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_dir / f"CICIoT2023_{label}_roc.png", dpi=200)
        plt.close()
    print(summary.round(6).to_string(index=False))
    print(f"Results saved to: {save_dir}", flush=True)
    return summary, metadata


MODEL_EVAL_CONFIG = {
    "GeNIS": {"weights": "results/GeNIS/aelstm/ae_lstm_weights.h5"},
    "KDD99": {
        # KDD99 is reshaped to (10, 13) in main_aelstm.py
        "model_params": {"timesteps": 10, "features": 13},
        "weights": "results/KDD99/aelstm/ae_lstm_weights.h5" # KDD99 trained weights
    },
    "CSE-CIC-IDS2018": {
        # CIC is reshaped to (10, 4) in main_aelstm.py
        "model_params": {"timesteps": 10, "features": 8},
        "weights": "results/CSE-CIC-IDS2018/aelstm/ae_lstm_weights.h5" # (modify if path differs)
    },
    "InSDN": {
         "model_params": {"timesteps": 12, "features": 7},
         "weights": "results/InSDN/aelstm/ae_lstm_weights.h5" # (modify if path differs)
    },
    "CICIoT2023": {
        # 78 features, same reshape as CIC2018 (10, 8) -> 80 (padding)
        "model_params": {"timesteps": 10, "features": 8},
        "weights": "results/CICIoT2023/aelstm/ae_lstm_weights.h5"
    },
    "UNSW_NB15": { # 85
        "model_params": {"timesteps": 6, "features": 7},
        "weights": "results/UNSW_NB15/aelstm/ae_lstm_weights.h5" # (modify if path differs)
    }
}

# --------------------------------------------------
# run evaluation
# --------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate saved AE-LSTM weights")
    parser.add_argument("--eval_dataset", default="CICIoT2023",
                        choices=["ciciot", "genis", *MODEL_EVAL_CONFIG])
    parser.add_argument("--base_data_path", default="/data/SDP_Dataset/Unified_model")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--result_path", default=None)
    parser.add_argument("--random_seed", type=int, default=123)
    parser.add_argument("--percentile", type=float, default=None,
                        help="Reconstruction percentile; GeNIS uses the saved training setting, others use 85")
    parser.add_argument("--batch_size", type=int, default=512)
    args = parser.parse_args()
    dataset = {"ciciot": "CICIoT2023", "genis": "GeNIS"}.get(args.eval_dataset, args.eval_dataset)
    config = MODEL_EVAL_CONFIG[dataset]
    weights_path = (Path(args.weights).expanduser().resolve() if args.weights
                    else PROJECT_ROOT / config["weights"])
    if not weights_path.is_file():
        parser.error(f"Weight file not found: {weights_path}")
    if dataset == "GeNIS":
        from utils.genis_sequence_experiment import evaluate_saved_genis
        return evaluate_saved_genis("aelstm", weights_path, args.result_path,
                                    args.percentile, args.batch_size)
    if args.percentile is None:
        args.percentile = 85
    params = config["model_params"]
    model = AE_LSTM(input_dim=params["features"], timesteps=params["timesteps"],
                    features=params["features"])
    model(tf.zeros((1, params["timesteps"], params["features"])))
    model.load_weights(str(weights_path))
    print(f"Loaded weights: {weights_path}", flush=True)
    if dataset == "CICIoT2023":
        save_dir = (Path(args.result_path).expanduser().resolve() if args.result_path
                    else PROJECT_ROOT / "results/CICIoT2023/aelstm/eval")
        evaluate_ciciot_aelstm(
            model, Path(args.base_data_path) / "ciciot_rnep", save_dir,
            random_seed=args.random_seed, percentile=args.percentile,
            batch_size=args.batch_size,
        )
    else:
        evaluate_aelstm_by_type(model, dataset, params, train_split_ratio=0.8,
                               percentile=args.percentile)


if __name__ == "__main__":
    main()
