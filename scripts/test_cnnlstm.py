## Test CNN-LSTM

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score, 
    confusion_matrix, roc_curve, auc
)
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import seaborn as sns
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

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

# --------------------------------------------------
# CNN-LSTM model and utilities
# --------------------------------------------------

@tf.keras.utils.register_keras_serializable(package="Custom")
class CNN_LSTM(tf.keras.Model):
    """
    CNN-LSTM model architecture (same as model_cnnlstm.py)
    """
    def __init__(self, timesteps=10, features=12, cnn_filters=64, lstm_units=128):
        super().__init__()
        self.timesteps = timesteps
        self.features = features
        self.conv1 = layers.Conv1D(cnn_filters, 3, padding="same", activation="relu")
        self.conv2 = layers.Conv1D(cnn_filters, 3, padding="same", activation="relu")
        self.pool = layers.MaxPooling1D(pool_size=2)
        self.dropout_cnn = layers.Dropout(0.25)
        self.flatten = layers.Flatten()
        self.lstm = layers.LSTM(lstm_units, return_sequences=False)
        self.fc1 = layers.Dense(
            128,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(0.1)
        )
        self.dropout_fc = layers.Dropout(0.5)
        self.output_layer = layers.Dense(1, activation="sigmoid")

    def call(self, inputs, training=None):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.dropout_cnn(x, training=training)
        x = self.flatten(x)
        x = tf.expand_dims(x, axis=1)
        x = self.lstm(x)
        x = self.fc1(x)
        x = self.dropout_fc(x, training=training)
        return self.output_layer(x)

def reshape_for_sequence(X, timesteps=10, features=12):
    """
    Reshapes 2D (N, F) data into 3D (N, T, F) sequence data (with padding)
    """
    n_samples, n_feats = X.shape
    target_len = timesteps * features
    
    if n_feats < target_len:
        pad = np.zeros((n_samples, target_len - n_feats))
        X = np.concatenate([X, pad], axis=1)
    elif n_feats > target_len:
        X = X[:, :target_len]
        
    return X.reshape(-1, timesteps, features)

DATASET_CONFIG = {
    "KDD99": {
        "base_dir": "./KDD99/KDD99_split",
        "normal_file": "KDD99_normal.csv",
        "anomaly_prefix": "KDD99_anomaly_",
        "merged_anomaly_file": "KDD99_anomaly.csv",
        "plot_save_path": "./cnn_lstm/KDD99_cnn_lstm_distribution.png",
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
        "plot_save_path": "./cnn_lstm/CIC_cnn_lstm_distribution.png",
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
        "plot_save_path": "./cnn_lstm/InSDN_cnn_lstm_distribution.png",
        "attack_map": {
            0: "BFA", 1: "BOTNET", 2: "DDoS", 3: "DoS",
            4: "Probe", 5: "U2R", 6: "Web-Attack"
        }
    },
    "UNSW_NB15": {
        "base_dir": "./UNSW_NB15/ae_datas",
        "normal_file": "UNSW_NB15_normal.csv", # 42 features
        "anomaly_prefix": "UNSW_NB15_anomaly_",
        "merged_anomaly_file": "UNSW_NB15_anomaly.csv",
        "plot_save_path": "./cnn_lstm/UNSW_NB15_cnn_lstm_distribution.png",
        "attack_map": {
            0: "analysis", 1: "backdoor", 2: "dos", 3: "exploits",
            4: "fuzzers", 5: "generic", 6: "Web-reconnaissance",
            6: "shellcode", 7: "worms"
        }
    }
}

# --------------------------------------------------
# CNN-LSTM evaluation function by type
# --------------------------------------------------
def evaluate_cnn_lstm_by_type(model, dataset_name, model_params, train_split_ratio=0.8):
    """
    Evaluates CNN-LSTM model performance by type, borrowing the structure of the TAAE evaluation script.

    :param model: trained CNN-LSTM model
    :param dataset_name: one of "KDD99", "CSE-CIC-IDS2018", "InSDN"
    :param model_params: dictionary for model input (e.g., {'timesteps': 10, 'features': 12})
    :param train_split_ratio: ratio for scaler training and normal test set split
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
    merged_path = os.path.join(base_dir, merged_file)
    if os.path.exists(merged_path):
        anomaly_files = [merged_file] + anomaly_files
    
    print(f"\nEvaluating CNN-LSTM for dataset: {dataset_name.upper()}")
    print(f"Found {len(anomaly_files)} anomaly datasets for evaluation")

    results = []

    # load, clean, and split normal data (80% for scaler training / 20% for testing)
    df_normal = pd.read_csv(normal_path)
    df_normal = shuffle(df_normal, random_state=123)
    split_point = int(len(df_normal) * train_split_ratio)
    df_normal_train = df_normal.iloc[:split_point] # for scaler training
    df_normal_test = df_normal.iloc[split_point:] # for actual testing

    df_normal_train = _clean_dataframe(df_normal_train)
    df_normal_test = _clean_dataframe(df_normal_test)
    
    print(f"Normal data: Train(for scaler)={len(df_normal_train):,}, Test(for eval)={len(df_normal_test):,}")

    # normalization (Scaler)
    scaler = MinMaxScaler()
    # fit scaler based on training data (normal 80%)
    scaler.fit(df_normal_train.values) 
    
    # scale and reshape normal *test* (20%) data
    X_normal_test_flat = scaler.transform(df_normal_test.values)
    X_normal_test_seq = reshape_for_sequence(X_normal_test_flat, **model_params)
    y_normal_test = np.zeros(len(X_normal_test_seq))

    # set threshold
    threshold = 0.5
    print(f"\nThreshold (fixed for CNN-LSTM): {threshold}")

    probs_by_attack = {} # store probabilities for visualization
    numeric_keys = []

    # evaluate per anomaly file
    for file in anomaly_files:
        anomaly_path = os.path.join(base_dir, file)
        df_anomaly = pd.read_csv(anomaly_path)
        df_anomaly = _clean_dataframe(df_anomaly) # apply cleaning

        if df_anomaly.empty:
            print(f"\nWarning: '{file}' is empty or became empty after cleaning. Skipping.")
            continue

        # scale and reshape anomaly data
        X_anomaly_flat = scaler.transform(df_anomaly.values)
        X_anomaly_seq = reshape_for_sequence(X_anomaly_flat, **model_params)
        y_anomaly = np.ones(len(X_anomaly_seq))
        
        # construct test set (Normal-Test + Anomaly)
        X_test = np.concatenate([X_normal_test_seq, X_anomaly_seq])
        y_test = np.concatenate([y_normal_test, y_anomaly])

        # prediction and probability computation
        test_probs = model.predict(X_test, verbose=0).reshape(-1) # probability of shape (N,)
        y_pred = (test_probs > threshold).astype(int)

        # compute metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        # *** [fix] count predictions ***
        pred_normal = int(np.sum(y_pred == 0))
        pred_anomaly = int(np.sum(y_pred == 1))
        # **************************

        print(f"\n{file}")
        print(f"Samples: {len(X_test)} (Normal: {len(y_normal_test)}, Anomaly: {len(y_anomaly)})")
        print(f"Accuracy={acc:.6f}, Precision={prec:.6f}, Recall={rec:.6f}, F1={f1:.6f}")
        
        # *** [fix] print prediction counts ***
        print(f"Predicted counts: Normal={pred_normal:,}, Anomaly={pred_anomaly:,}")
        # **************************

        # generate Confusion Matrix and ROC Curve for merged file
        if file == merged_file:
            # === Confusion Matrix ===
            cm_save_path = f"./cnn_lstm/{dataset_name}_cnn_lstm_cm.png"
            title = f'CNN-LSTM CM - {dataset_name.upper()}'
            _plot_and_print_cm(y_test, y_pred, cm_save_path, ['Normal', 'Anomaly'], title)
            print(f"Saved confusion matrix -> {cm_save_path}")

            # === ROC Curve ===
            fpr, tpr, _ = roc_curve(y_test, test_probs)
            roc_auc = auc(fpr, tpr)
            
            roc_save_path = f"./cnn_lstm/{dataset_name}_cnn_lstm_roc.png"
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.title(f'CNN-LSTM ROC Curve - {dataset_name.upper()}')
            plt.xlabel('False Positive Rate (FPR)')
            plt.ylabel('True Positive Rate (TPR)')
            plt.legend(loc="lower right")
            plt.grid(True, linestyle="--", alpha=0.4)
            plt.tight_layout()
            plt.savefig(roc_save_path, dpi=300)
            plt.close()
            
            print(f"Saved ROC curve -> {roc_save_path} (AUC = {roc_auc:.4f})")

        # *** [fix] add prediction counts to results ***
        results.append({
            "File": file, "Samples": len(X_test), "Accuracy": acc,
            "Precision": prec, "Recall": rec, "F1": f1,
            "Pred_Normal": pred_normal, "Pred_Anomaly": pred_anomaly
        })
        # ********************************

        # store data for visualization (excluding merged file)
        if file != merged_file:
            try:
                num_str = file.replace(anomaly_prefix, "").replace(".csv", "")
                attack_num = int(num_str)
                
                probs_anomaly_only = model.predict(X_anomaly_seq, verbose=0).reshape(-1)

                attack_label = attack_map.get(attack_num, f"attack_{attack_num}")
                probs_by_attack[attack_num] = (attack_label, probs_anomaly_only)
                numeric_keys.append(attack_num)
            except Exception as e:
                print(f"Warning: Could not parse attack number from '{file}'. Skipping for plot. Error: {e}")

    # save result summary
    df = pd.DataFrame(results)
    summary_save_path = f"./cnn_lstm/{dataset_name}_cnn_lstm_summary.csv"
    
    df.to_csv(summary_save_path, index=False)
    print(f"\nSaved summary -> {summary_save_path}")
    print(df.round(6))

    # output probability histogram by attack type
    if probs_by_attack:
        plt.figure(figsize=(12, 7))
        
        # compute Normal (Test) probabilities
        probs_normal_test = model.predict(X_normal_test_seq, verbose=0).reshape(-1)
        
        plt.hist(probs_normal_test, bins=100, alpha=0.6, label="Normal", color="green", density=True, range=(0,1))

        # histogram by attack type
        numeric_keys = sorted(set(numeric_keys))
        n_attacks = len(numeric_keys)
        palette = plt.cm.gist_rainbow(np.linspace(0, 1, max(3, n_attacks))) 

        for i, atk_num in enumerate(numeric_keys):
            atk_name, probs = probs_by_attack[atk_num]
            plt.hist(probs, bins=100, alpha=0.5, label=f"{atk_name}", color=palette[i % n_attacks], density=True, range=(0,1))

        plt.axvline(threshold, color="blue", linestyle="--", label=f"Threshold ({threshold})")

        plt.xlabel("Model Output Probability (0.0=Normal, 1.0=Anomaly)")
        plt.ylabel("Density")
        plt.title(f"CNN-LSTM Output Probability Distribution - ({dataset_name.upper()})")
        plt.legend(fontsize=8, loc="upper right", ncol=1)
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.xlim(0.0, 1.0)
        plt.tight_layout()
        plt.savefig(plot_save_path, dpi=300)
        plt.close()

        print(f"Saved distribution plot -> {plot_save_path}")

# --------------------------------------------------
# model load and evaluation config
# --------------------------------------------------

# model parameters and weight paths trained in main_cnnlstm_fl.py
MODEL_EVAL_CONFIG = {
    "KDD99": {
        "model_params": {"timesteps": 10, "features": 12},
        "weights": "./Results/KDD99/cnnlstm/cnn_lstm_weights.h5" # KDD99 trained weights
    },
    "CSE-CIC-IDS2018": {
        "model_params": {"timesteps": 10, "features": 8},
        "weights": "Results/CSE-CIC-IDS2018/cnnlstm/cnn_lstm_weights.h5" # (modify if path differs)
    },
    "InSDN": {
        # use 83-feature csv if trained with (10, 9) etc. for 83 features
        "model_params": {"timesteps": 12, "features": 7}, # 83 -> 90 (padding)
        "weights": "Results/InSDN/cnnlstm/cnn_lstm_weights.h5" # (modify if path differs)
    },
    "UNSW_NB15": {
        # use 83-feature csv if trained with (10, 9) etc. for 83 features
        "model_params": {"timesteps": 6, "features": 7}, # 83 -> 90 (padding)
        "weights": "Results/UNSW_NB15/cnnlstm/cnn_lstm_weights.h5" # (modify if path differs)
    }
}

# --------------------------------------------------
# run evaluation
# --------------------------------------------------
if __name__ == "__main__":
    
    # --- Select the dataset to run here ---
    DATASET_TO_RUN = "CSE-CIC-IDS2018" 
    # (options: "KDD99", "CSE-CIC-IDS2018", "InSDN")
    # -----------------------------------------

    # create output directory for CNN-LSTM results
    os.makedirs("./cnn_lstm", exist_ok=True)

    # load config for selected dataset
    if DATASET_TO_RUN not in MODEL_EVAL_CONFIG:
        print(f"Error: No model config found for '{DATASET_TO_RUN}' in MODEL_EVAL_CONFIG.")
    else:
        eval_cfg = MODEL_EVAL_CONFIG[DATASET_TO_RUN]
        model_params = eval_cfg["model_params"]
        weights_path = eval_cfg["weights"]

        # build model
        model = CNN_LSTM(**model_params)
        dummy_input_shape = (1, model_params["timesteps"], model_params["features"])
        _ = model(tf.zeros(dummy_input_shape))
        model.compile(optimizer=Adam(0.0001), loss="binary_crossentropy", metrics=["accuracy"])

        # load weights
        if os.path.exists(weights_path):
            model.load_weights(weights_path)
            print(f"Loaded pre-trained weights from {weights_path}")
        else:
            print(f"WARNING: Weight file not found at {weights_path}. The model has not been trained.")
            exit()

        model.summary()
        exit()

        # run evaluation
        evaluate_cnn_lstm_by_type(
            model=model,
            dataset_name=DATASET_TO_RUN,
            model_params=model_params,
            train_split_ratio=0.8 # 80% for scaler training / 20% for testing
        )

        print(f"\n--- CNN-LSTM Evaluation Complete for {DATASET_TO_RUN} ---")

