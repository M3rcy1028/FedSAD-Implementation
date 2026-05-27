import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.utils import shuffle
import tensorflow as tf
from taae import TransformerAAE  # this model class is required
import matplotlib.pyplot as plt
import seaborn as sns

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

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
def evaluate_dataset(model, dataset_name, percentile, train_split_ratio=0.8):
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

    base_dir = config["base_dir"]
    normal_path = os.path.join(base_dir, config["normal_file"])
    anomaly_prefix = config["anomaly_prefix"]
    merged_file = config["merged_anomaly_file"]
    attack_map = config["attack_map"]
    plot_save_path = config["plot_save_path"]

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
            cm_save_path = f"./{dataset_name}_cm.png"
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
            
            roc_save_path = f"./{dataset_name}_roc.png"
            
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
                roc_data_save_path = f"./{dataset_name}_roc_data.csv"
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
    summary_save_path = f"./{dataset_name}_summary.csv"
    
    df.to_csv(summary_save_path, index=False)
    print(f"\nSaved summary -> {summary_save_path}")
    print(df.round(6))

    # Reconstruction Error histogram by attack type
    if error_by_attack:
        plt.figure(figsize=(12, 7))

        # compute Normal (Test) errors
        preds_normal_test = model.predict(X_normal_test, verbose=0)
        errors_normal_test = np.mean(np.square(X_normal_test - preds_normal_test), axis=1)

        # Normal histogram
        plt.hist(errors_normal_test, bins=200, alpha=0.6, label="Normal", color="green", density=True)

        # histogram by attack type
        numeric_keys = sorted(set(numeric_keys))
        n_attacks = len(numeric_keys)
        palette = plt.cm.tab20(np.linspace(0, 1, max(3, n_attacks)))

        for i, atk_num in enumerate(numeric_keys):
            atk_name, errs = error_by_attack[atk_num]
            plt.hist(errs, bins=200, alpha=0.5, label=f"{atk_name}", color=palette[i % 20], density=True)

        # Threshold
        plt.axvline(threshold, color="blue", linestyle="--", label=f"Threshold ({threshold:.6f})")

        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Reconstruction Error (log scale)")
        plt.ylabel("Density (log scale)")
        plt.title(f"Reconstruction Error - ({dataset_name.upper()})")
        plt.legend(fontsize=8, loc="upper right", ncol=1)
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig(plot_save_path, dpi=300)
        plt.close()

        print(f"Saved distribution plot -> {plot_save_path}")
        

# --------------------------------------------------
# model load config (trained model class 'TransformerAAE' required)
# --------------------------------------------------

# if the 'model_taae_rnep' module is missing, this will not run;
# for actual use, import the file where TransformerAAE is defined.
try:
    from taae import TransformerAAE
except ImportError:
    print("="*50)
    print("WARNING: 'model_taae_rnep' module not found.")
    print("The 'TransformerAAE' model class is required for actual evaluation.")
    print("Temporarily creating a tf.keras.Model for script execution testing.")
    print("="*50)

    # temporary model (for script execution testing)
    def create_dummy_model(input_dim):
        inp = tf.keras.Input(shape=(input_dim,))
        x = tf.keras.layers.Dense(int(input_dim/2), activation='relu')(inp)
        out = tf.keras.layers.Dense(input_dim, activation='sigmoid')(x)
        model = tf.keras.Model(inputs=inp, outputs=out)
        # add dummy function to match TAAE model's predict signature
        original_predict = model.predict
        model.predict = lambda x, verbose=0, prior_labels=None: original_predict(x, verbose=verbose)
        return model
    
    # replace TransformerAAE class with temporary model creation function
    TransformerAAE = lambda input_dim: create_dummy_model(input_dim)


MODEL_CONFIG = {
    "NSL-KDD": {
        "input_dim": 119,
        "weights": "Results/NSL-KDD/rnep/rnep_aae_transformer_weights.h5"
    }, # P= 98
    "KDD99": {
        "input_dim": 115,
        "weights": "Results/KDD99/rnep/rnep_frame_kdd_weights.h5"
    }, # P= 95
    "InSDN": {
        "input_dim": 83,
        "weights": "Results/InSDN/rnep/rnep_frame_insdn_weights.h5"
    }, # P= 82
    "CSE-CIC-IDS2018": {
        "input_dim": 78,
        "weights": "Results/CSE-CIC-IDS2018/rnep/rnep_frame_cic_weights.h5"
    }, # P= 90, 0.000069
    "UNSW_NB15": {
        "input_dim": 43,
        "weights": "Results/UNSW_NB15/rnep/rnep_frame_unsw_weights.h5"
    } # P= 90
}

# --------------------------------------------------
# run evaluation
# --------------------------------------------------
if __name__ == "__main__":

    # --- Select the dataset to run here ---
    DATASET_TO_RUN = "KDD99"
    # (options: "KDD99", "CSE-CIC-IDS2018", "InSDN")
    # -----------------------------------------

    PERCENTILE = 90
    
    # load config for selected dataset
    if DATASET_TO_RUN not in MODEL_CONFIG:
        print(f"Error: No model config found for '{DATASET_TO_RUN}' in MODEL_CONFIG.")
    else:
        config = MODEL_CONFIG[DATASET_TO_RUN]
        input_dim = config["input_dim"]
        weights_path = config["weights"]

        # build model
        model = TransformerAAE(input_dim=input_dim)
        # TAAE model may require prior_labels for build (based on original code)
        try:
             _ = model(tf.zeros((1, input_dim)), prior_labels=tf.zeros((1, 1)))
        except Exception as e:
            print(f"Model build with prior_labels failed, trying without: {e}")
            try:
                # attempt build without prior_labels as a fallback
                 _ = model.build(input_shape=(None, input_dim))
            except Exception as e2:
                 print(f"Model build failed: {e2}")
                 # build가 안되면 predict에서 오류날 수 있음
                 pass

        # load weights
        if os.path.exists(weights_path):
            model.load_weights(weights_path)
            print(f"Loaded pre-trained weights from {weights_path}")
        else:
            print(f"WARNING: Weight file not found at {weights_path}. The model has not been trained.")
        model.summary()
        # run evaluation
        evaluate_dataset(
            model=model,
            dataset_name=DATASET_TO_RUN,
            percentile=PERCENTILE
        )

        print(f"\n--- Evaluation Complete for {DATASET_TO_RUN} (Percentile = {PERCENTILE}) ---")