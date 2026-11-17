import argparse
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.utils import shuffle
import tensorflow as tf
from taae import TransformerAAE  # this model class is required
from utils.taae_checkpoint import resolve_latent_dim
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------
# data cleaning helper function
# --------------------------------------------------
def _clean_dataframe(df):
    """Drops 'Label'/'label' columns, replaces 'inf'/'nan' values with 0, and clips large values."""
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    # # clip excessively large values (characteristic of InSDN, CIC datasets)
    # df = np.clip(df, -1e6, 1e6) 
    return df

# --------------------------------------------------
# Confusion Matrix helper function
# --------------------------------------------------
def _plot_and_print_cm(y_test, y_pred, save_path, labels, title):
    """Prints the Confusion Matrix as a DataFrame and saves it as a seaborn heatmap."""
    cm = confusion_matrix(y_test, y_pred)
    
    # print as text (DataFrame)
    try:
        # assume cm.ravel() returns 4 values (tn, fp, fn, tp)
        tn, fp, fn, tp = cm.ravel()
        cm_table = pd.DataFrame(
            [[tn, fp], [fn, tp]],
            index=[f'Actual {labels[0]}', f'Actual {labels[1]}'],
            columns=[f'Predicted {labels[0]}', f'Predicted {labels[1]}']
        )
        print("\n[Confusion Matrix]")
        print(cm_table)
    except ValueError:
        # when not a 2x2 matrix (e.g., only one class predicted)
        print(f"\n[Confusion Matrix] (Raw)\n{cm}")

    # save as image (Heatmap)
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

def plot_error_distributions(normal_errors, error_by_attack, threshold, dataset_name, save_path):
    """Use shared bins in log10(MSE), without changing scores or predictions."""
    normal = np.asarray(normal_errors, dtype=float)
    attack_keys = sorted(error_by_attack)
    attacks = [np.asarray(error_by_attack[k][1], dtype=float) for k in attack_keys]
    arrays = [normal] + attacks
    if not attacks or any(a.ndim != 1 or not len(a) or not np.isfinite(a).all() or np.any(a < 0) for a in arrays):
        raise ValueError("Plotting requires finite, nonnegative score vectors for Normal and attacks")
    combined = np.concatenate(arrays)
    positive = combined[combined > 0]
    floor = float(positive.min() / 2) if len(positive) else 1e-12
    logged = [np.log10(np.maximum(a, floor)) for a in arrays]
    low, high = min(a.min() for a in logged), max(a.max() for a in logged)
    if low == high:
        low, high = low - 0.5, high + 0.5
    bins = np.linspace(low, high, 201)
    threshold_log = np.log10(max(float(threshold), floor))
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.hist(logged[0], bins=bins, density=True, histtype="stepfilled",
            alpha=0.3, color="green", label="Normal")
    palette = plt.cm.turbo(np.linspace(0.03, 0.97, len(attacks)))
    for i, key in enumerate(attack_keys):
        ax.hist(logged[i + 1], bins=bins, density=True, histtype="step",
                linewidth=1, color=palette[i], label=error_by_attack[key][0])
    ax.axvline(threshold_log, color="blue", linestyle="--", label=f"Threshold ({threshold:.6g})")
    ax.set_yscale("log")
    ax.set(xlabel="log10(Reconstruction MSE)", ylabel="Density per unit log10(MSE)",
           title=f"Reconstruction Error - {dataset_name.upper()} (shared log bins)")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

    # The aggregate view and empirical CDF avoid the occlusion of 27 classes.
    attack = np.concatenate(attacks)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for values, color, label in [(normal, "#16a34a", "Normal"), (attack, "#dc2626", "All attacks")]:
        axes[0].hist(np.log10(np.maximum(values, floor)), bins=bins, density=True,
                     alpha=0.4, color=color, label=label)
        axes[1].plot(np.log10(np.maximum(np.sort(values), floor)),
                     np.arange(1, len(values) + 1) / len(values), color=color, label=label)
    for ax in axes:
        ax.axvline(threshold_log, color="blue", linestyle="--", label="Threshold")
        ax.set_xlabel("log10(Reconstruction MSE)")
        ax.grid(alpha=0.25)
        ax.legend()
    axes[0].set_ylabel("Density per unit log10(MSE)")
    axes[0].set_title("Same scores, shared log bins")
    axes[1].set(ylabel="Fraction of samples with score <= x", ylim=(0, 1), title="Empirical CDF")
    fig.suptitle(dataset_name)
    fig.tight_layout()
    overview = save_path.with_name(save_path.stem + "_overview.png")
    fig.savefig(overview, dpi=180)
    plt.close(fig)
    return {"binning": "shared equal-width bins in log10(MSE)", "bins": 200,
            "zero_score_plot_floor": floor, "overview": str(overview)}


# --------------------------------------------------
# dataset-specific config
# --------------------------------------------------
DATASET_CONFIG = {
    "NSL-KDD": {
        "base_dir": "./NSL-KDD",
        "normal_file": "KDD_normal.csv",
        "anomaly_prefix": "KDD_anomaly_",
        "merged_anomaly_file": "KDD_anomaly.csv",
        "plot_save_path": "./KDD99_distribution.png",
        "attack_map": {
            0: "back", 1: "buffer_overflow", 2: "ftp_write", 3: "guess_passwd",
            4: "imap", 5: "ipsweep", 6: "land", 7: "loadmodule", 8: "multihop",
            9: "neptune", 10: "nmap", 11: "perl", 12: "phf", 13: "portsweep",
            14: "rootkit", 15: "satan", 16: "spy", 17: "warezclient", 18: "warezmaster"
        }
    },
    "KDD99": {
        "base_dir": "./KDD99/KDD99_split",
        "normal_file": "KDD99_normal.csv",
        "anomaly_prefix": "KDD99_anomaly_",
        "merged_anomaly_file": "KDD99_anomaly.csv",
        "plot_save_path": "./KDD99_distribution.png",
        "attack_map": {
            0: "back", 1: "buffer_overflow", 2: "ftp_write", 3: "guess_passwd",
            4: "imap", 5: "ipsweep", 6: "land", 7: "loadmodule", 8: "multihop",
            9: "neptune", 10: "nmap", 11: "perl", 12: "phf", 13: "portsweep",
            14: "rootkit", 15: "satan", 16: "spy", 17: "warezclient", 18: "warezmaster"
        }
    },
    "InSDN": {
        "base_dir": "./InSDN/ae_datas",
        "normal_file": "InSDN_normal.csv",
        "anomaly_prefix": "InSDN_anomaly_",
        "merged_anomaly_file": "InSDN_anomaly.csv",
        "plot_save_path": "./InSDN_distribution.png",
        "attack_map": {
            0: "BFA (BruteForce)",
            1: "BOTNET",
            2: "DDoS",
            3: "DoS",
            4: "Probe",
            5: "U2R",
            6: "Web-Attack"
        }
    },
    "CSE-CIC-IDS2018": {
        "base_dir": "./CIC2018/ae_datas_sampled",
        "normal_file": "CIC_ae_normal.csv",
        "anomaly_prefix": "CIC_anomaly_ae_", # individual file prefix
        "merged_anomaly_file": "CIC_ae_anomaly.csv", # merged file name
        "plot_save_path": "./CSE-CIC-IDS2018_distribution.png",
        "attack_map": {
            1: "DDOS attack-HOIC",
            2: "DDoS attacks-LOIC-HTTP",
            3: "DoS attacks-Hulk",
            4: "Bot",
            5: "FTP-BruteForce",
            6: "SSH-Bruteforce",
            7: "Infiltration",
            8: "DoS attacks-SlowHTTPTest",
            9: "DoS attacks-GoldenEye",
            10: "DoS attacks-Slowloris",
            11: "DDOS attack-LOIC-UDP",
            12: "Brute Force -Web",
            13: "Brute Force -XSS",
            14: "SQL Injection"
        }
    },
    "CICIoT2023": {
        "base_dir": "/data/SDP_Dataset/Unified_model/ciciot_rnep",
        "normal_file": "CICIoT_ae_normal.csv", # 78 features (same as CSE-CIC-IDS2018)
        "anomaly_prefix": "CICIoT_anomaly_ae_",
        "merged_anomaly_file": "CICIoT_ae_anomaly.csv",
        "plot_save_path": "./CICIoT2023_distribution.png",
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
        "normal_file": "UNSW_NB15_normal.csv", 
        "anomaly_prefix": "UNSW_NB15_anomaly_",
        "merged_anomaly_file": "UNSW_NB15_anomaly.csv",
        "plot_save_path": "./UNSW_NB15_distribution.png",
        "attack_map": {
            0: "analysis", 1: "backdoor", 2: "dos", 3: "exploits",
            4: "fuzzers", 5: "generic", 6: "Web-reconnaissance",
            6: "shellcode", 7: "worms"
        }
    }
}

# --------------------------------------------------
# general-purpose evaluation function
# --------------------------------------------------
def evaluate_dataset(model, dataset_name, percentile, train_split_ratio=0.8,
                     base_dir=None, result_path=None):
    """
    Evaluates the TAAE model's anomaly detection performance on the specified dataset.

    :param model: trained TAAE model
    :param dataset_name: one of "KDD99", "InSDN", "CSE-CIC-IDS2018"
    :param percentile: percentile for threshold computation (e.g., 90)
    :param train_split_ratio: ratio for splitting normal data into train/test (e.g., 0.8)
    """
    
    # load config
    try:
        config = DATASET_CONFIG[dataset_name]
    except KeyError:
        print(f"Error: No config found for dataset '{dataset_name}'")
        return

    base_dir = Path(base_dir or config["base_dir"])
    if not base_dir.is_absolute():
        base_dir = PROJECT_ROOT / base_dir
    output_dir = Path(result_path) if result_path is not None else Path.cwd()
    output_dir.mkdir(parents=True, exist_ok=True)
    normal_path = os.path.join(base_dir, config["normal_file"])
    anomaly_prefix = config["anomaly_prefix"]
    merged_file = config["merged_anomaly_file"]
    attack_map = config["attack_map"]
    plot_save_path = config["plot_save_path"]
    if result_path is not None:
        plot_save_path = output_dir / f"{dataset_name}_distribution.png"

    # search anomaly file list
    anomaly_files = sorted([
        f for f in os.listdir(base_dir)
        if f.startswith(anomaly_prefix) and f.endswith(".csv")
    ])
    
    # prepend merged file to the list
    merged_path = os.path.join(base_dir, merged_file)
    if os.path.exists(merged_path):
        anomaly_files = [merged_file] + anomaly_files
    
    print(f"\nEvaluating dataset: {dataset_name.upper()}")
    print(f"Found {len(anomaly_files)} anomaly datasets for evaluation")

    results = []

    # load, clean, and split normal data
    df_normal = pd.read_csv(normal_path)
    df_normal = shuffle(df_normal, random_state=123) # InSDN
    # df_normal = df_normal.sample(frac=1, random_state=48).reset_index(drop=True) # KDD99, NSL-KDD
    split_point = int(len(df_normal) * train_split_ratio)
    df_normal_train = df_normal[:split_point]
    df_normal_test = df_normal.iloc[split_point:]

    df_normal_train = _clean_dataframe(df_normal_train)
    df_normal_test = _clean_dataframe(df_normal_test)

    print(f"Train size: {len(df_normal_train):,}, Test size: {len(df_normal_test):,}")

    # normalization (Scaler)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(df_normal_train.values)
    X_normal_test = scaler.transform(df_normal_test.values)

    # compute threshold
    preds_train = model.predict(X_train, verbose=0)
    train_errors = np.mean(np.square(X_train - preds_train), axis=1)
    threshold = np.percentile(train_errors, percentile)
    # threshold = 0.000069
    print(f"\nThreshold ({percentile}th percentile): {threshold:.20f}")

    error_by_attack = {} # store errors for visualization
    numeric_keys = []

    # evaluate per anomaly file
    for file in anomaly_files:
        anomaly_path = os.path.join(base_dir, file)
        df_anomaly = pd.read_csv(anomaly_path)
        df_anomaly = _clean_dataframe(df_anomaly) # apply cleaning

        if "label" in df_anomaly.columns:
            df_anomaly = df_anomaly.drop(columns=["label"])

        X_anomaly = scaler.transform(df_anomaly.values)
        
        # construct test set (Normal-Test + Anomaly)
        X_test = np.concatenate([X_normal_test, X_anomaly])
        y_test = np.concatenate([np.zeros(len(X_normal_test)), np.ones(len(X_anomaly))])

        # prediction and error computation
        preds_test = model.predict(X_test, verbose=0)
        test_errors = np.mean(np.square(X_test - preds_test), axis=1)
        y_pred = (test_errors > threshold).astype(int)

        # compute metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        pred_normal = int(np.sum(y_pred == 0))
        pred_anomaly = int(np.sum(y_pred == 1))

        print(f"\n{file}")
        print(f"Samples: {len(X_test)} | Accuracy={acc:.6f}, Precision={prec:.6f}, Recall={rec:.6f}, F1={f1:.6f}")

        # generate Confusion Matrix and ROC Curve for merged file
        if file == merged_file:

            # === Confusion Matrix ===
            cm_save_path = output_dir / f"{dataset_name}_cm.png"
            title = f'Confusion Matrix - {dataset_name.upper()} (P={percentile})'
            
            _plot_and_print_cm(
                y_test, 
                y_pred, 
                cm_save_path, 
                labels=['Normal', 'Anomaly'], 
                title=title
            )
            print(f"Saved confusion matrix -> {cm_save_path}")

            # === ROC Curve ===
            # (y_test is 0/1, test_errors is the score (reconstruction error))
            fpr, tpr, _ = roc_curve(y_test, test_errors)
            roc_auc = auc(fpr, tpr)
            
            roc_save_path = output_dir / f"{dataset_name}_roc.png"
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, 
                     label=f'ROC curve (AUC = {roc_auc:.4f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate (FPR)')
            plt.ylabel('True Positive Rate (TPR)')
            plt.title(f'ROC Curve - {dataset_name.upper()}')
            plt.legend(loc="lower right")
            plt.grid(True, linestyle="--", alpha=0.4)
            plt.tight_layout()
            plt.savefig(roc_save_path, dpi=300)
            plt.close()
            
            print(f"Saved ROC curve -> {roc_save_path} (AUC = {roc_auc:.4f})")
            
            try:
                df_roc_data = pd.DataFrame({
                    'y_true': y_test,
                    'anomaly_score': test_errors
                })
                roc_data_save_path = output_dir / f"{dataset_name}_roc_data.csv"
                df_roc_data.to_csv(roc_data_save_path, index=False)
                print(f"Saved ROC curve raw data -> {roc_data_save_path} (rows: {len(df_roc_data)})")
            except Exception as e:
                print(f"Warning: FAILED to save ROC raw data. Error: {e}")

            results.append({
                "File": file, "Samples": len(X_test), "Accuracy": acc,
                "Precision": prec, "Recall": rec, "F1": f1,
                "Pred_Normal": pred_normal, "Pred_Anomaly": pred_anomaly
            })
        
        # store data for visualization (excluding merged file)
        if file != merged_file:
            try:
                num_str = file.replace(anomaly_prefix, "").replace(".csv", "")
                attack_num = int(num_str)
                
                # compute errors for anomaly data only (excluding X_normal_test)
                preds_anomaly_only = model.predict(X_anomaly, verbose=0)
                errors_anomaly_only = np.mean(np.square(X_anomaly - preds_anomaly_only), axis=1)

                attack_label = attack_map.get(attack_num, f"attack_{attack_num}")
                error_by_attack[attack_num] = (attack_label, errors_anomaly_only)
                numeric_keys.append(attack_num)
            except Exception as e:
                print(f"Warning: Could not parse attack number from '{file}'. Skipping for plot. Error: {e}")
    
    # save results
    df = pd.DataFrame(results)
    summary_save_path = output_dir / f"{dataset_name}_summary.csv"
    
    df.to_csv(summary_save_path, index=False)
    print(f"\nSaved summary -> {summary_save_path}")
    print(df.round(6))

    # Reconstruction Error histogram by attack type
    if error_by_attack:
        # compute Normal (Test) errors
        preds_normal_test = model.predict(X_normal_test, verbose=0)
        errors_normal_test = np.mean(np.square(X_normal_test - preds_normal_test), axis=1)
        plot_error_distributions(errors_normal_test, error_by_attack, threshold,
                                 dataset_name, plot_save_path)
        print(f"Saved distribution plot -> {plot_save_path}")
        

# --------------------------------------------------
# model load config (trained model class 'TransformerAAE' required)
# --------------------------------------------------

MODEL_CONFIG = {
    "NSL-KDD": {
        "input_dim": 119,
        "weights": "results/NSL-KDD/rnep/fedsad_nsl_weights.h5"
    }, # P= 98
    "KDD99": {
        "input_dim": 115,
        "weights": "results/KDD99/fedsad/fedsad_kdd_weights.h5"
    }, # P= 95
    "InSDN": {
        "input_dim": 83,
        "weights": "results/InSDN/fedsad/fedsad_insdn_weights.h5"
    }, # P= 82
    "CSE-CIC-IDS2018": {
        "input_dim": 78,
        "weights": "results/CSE-CIC-IDS2018/taae/taae_cic_weights.h5"
    }, # P= 90, 0.000069
    "CICIoT2023": {
        "input_dim": 78,
        "weights": "results/CICIoT2023/taae/fedsad_weights.h5"
    }, # P=80.57
    "UNSW_NB15": {
        "input_dim": 43,
        "weights": "results/UNSW_NB15/taae/taae_unsw_weights.h5"
    } # P= 90
}

# --------------------------------------------------
# run evaluation
# --------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate saved TAAE weights")
    parser.add_argument("--eval_dataset", default="CICIoT2023",
                        choices=list(MODEL_CONFIG) + ["ciciot"])
    parser.add_argument("--percentile", type=float, default=84)
    parser.add_argument("--train_ratio", type=float, default=None)
    parser.add_argument("--base_dir", type=Path, default=None)
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--latent_dim", type=int, default=None,
                        help="Override latent dimension; otherwise use checkpoint metadata or legacy 64")
    parser.add_argument("--result_path", type=Path, default=None)
    parser.add_argument("--check_only", action="store_true",
                        help="Check imports, paths, weights and a forward pass without full evaluation")
    args = parser.parse_args()
    dataset = "CICIoT2023" if args.eval_dataset == "ciciot" else args.eval_dataset
    train_ratio = args.train_ratio if args.train_ratio is not None else (0.5 if dataset == "CICIoT2023" else 0.8)
    if not 0 < train_ratio < 1 or not 0 <= args.percentile <= 100:
        parser.error("train_ratio must be in (0, 1) and percentile must be in [0, 100]")
    weights_path = args.weights or Path(MODEL_CONFIG[dataset]["weights"])
    if not weights_path.is_absolute():
        weights_path = PROJECT_ROOT / weights_path
    base_dir = args.base_dir or Path(DATASET_CONFIG[dataset]["base_dir"])
    if not base_dir.is_absolute():
        base_dir = PROJECT_ROOT / base_dir
    normal_path = base_dir / DATASET_CONFIG[dataset]["normal_file"]
    if not weights_path.is_file():
        raise FileNotFoundError(f"Weight file not found: {weights_path}; specify --weights")
    if not normal_path.is_file():
        raise FileNotFoundError(f"Normal dataset not found: {normal_path}; specify --base_dir")

    input_dim = MODEL_CONFIG[dataset]["input_dim"]
    latent_dim = resolve_latent_dim(weights_path, args.latent_dim)
    model = TransformerAAE(input_dim=input_dim, latent_dim=latent_dim)
    model(tf.zeros((1, input_dim)), prior_labels=tf.zeros((1, 1)))
    model.load_weights(str(weights_path))
    print(f"Loaded pre-trained weights from {weights_path}")
    print(f"Model latent dimension: {latent_dim}")
    print(f"Dataset: {base_dir}; normal train ratio: {train_ratio}; percentile: {args.percentile}")
    if args.check_only:
        output = model(tf.zeros((2, input_dim)), training=False).numpy()
        if output.shape != (2, input_dim) or not np.isfinite(output).all():
            raise RuntimeError("Loaded model returned invalid outputs")
        print(f"Check complete: imports, dataset path, saved weights, output shape {output.shape}")
        return
    result_path = args.result_path
    if result_path is None and dataset == "CICIoT2023":
        result_path = PROJECT_ROOT / "results/CICIoT2023/taae/eval"
    model.summary()
    evaluate_dataset(model, dataset, args.percentile, train_split_ratio=train_ratio,
                     base_dir=base_dir, result_path=result_path)
    print(f"\n--- Evaluation Complete for {dataset} (Percentile = {args.percentile}) ---")


if __name__ == "__main__":
    main()
