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

# --------------------------------------------------
# 1. 헬퍼 함수 (TAAE 스크립트와 동일)
# --------------------------------------------------

def _clean_dataframe(df):
    """'Label'/'label' 컬럼을 삭제하고, 'inf'/'nan' 값을 0으로 대체하며, 큰 값을 clip합니다."""
    if 'Label' in df.columns:
        df = df.drop(columns=['Label'])
    if 'label' in df.columns:
        df = df.drop(columns=['label'])
        
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    df = np.clip(df, -1e6, 1e6) 
    return df

def _plot_and_print_cm(y_test, y_pred, save_path, labels, title):
    """Confusion Matrix를 DataFrame으로 출력하고, seaborn 히트맵으로 저장합니다."""
    cm = confusion_matrix(y_test, y_pred)
    
    try:
        tn, fp, fn, tp = cm.ravel()
        cm_table = pd.DataFrame(
            [[tn, fp], [fn, tp]],
            index=[f'Actual {labels[0]}', f'Actual {labels[1]}'],
            columns=[f'Predicted {labels[0]}', f'Predicted {labels[1]}']
        )
        print("\n🧾 [Confusion Matrix]")
        print(cm_table)
    except ValueError: 
        print(f"\n🧾 [Confusion Matrix] (Raw)\n{cm}")

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
# 2. CNN-LSTM 모델 및 유틸리티
# --------------------------------------------------

@tf.keras.utils.register_keras_serializable(package="Custom")
class CNN_LSTM(tf.keras.Model):
    """
    CNN-LSTM 모델 아키텍처 (model_cnnlstm.py와 동일)
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
    2D (N, F) 데이터를 3D (N, T, F) 시퀀스 데이터로 Reshape (패딩 포함)
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
# 3. 데이터셋 설정 (TAAE 스크립트와 동일)
# --------------------------------------------------
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
        "base_dir": "./CIC2018/ae_datas_all_features",
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
# 4. CNN-LSTM 유형별 평가 함수
# --------------------------------------------------
def evaluate_cnn_lstm_by_type(model, dataset_name, model_params, train_split_ratio=0.8):
    """
    TAAE 평가 스크립트의 구조를 차용하여 CNN-LSTM 모델의 유형별 성능을 평가합니다.
    
    :param model: 훈련된 CNN-LSTM 모델
    :param dataset_name: "KDD99", "CSE-CIC-IDS2018", "InSDN" 중 하나
    :param model_params: 모델 입력을 위한 딕셔너리 (예: {'timesteps': 10, 'features': 12})
    :param train_split_ratio: 스케일러 학습 및 정상 테스트셋 분리 비율
    """
    
    # 1. 설정 로드
    try:
        config = DATASET_CONFIG[dataset_name]
    except KeyError:
        print(f"❌ Error: No config found for dataset '{dataset_name}'")
        return

    base_dir = config["base_dir"]
    normal_path = os.path.join(base_dir, config["normal_file"])
    anomaly_prefix = config["anomaly_prefix"]
    merged_file = config["merged_anomaly_file"]
    attack_map = config["attack_map"]
    plot_save_path = config["plot_save_path"] # 저장 경로 변경

    # 2. Anomaly 파일 리스트 탐색
    anomaly_files = sorted([
        f for f in os.listdir(base_dir)
        if f.startswith(anomaly_prefix) and f.endswith(".csv")
    ])
    merged_path = os.path.join(base_dir, merged_file)
    if os.path.exists(merged_path):
        anomaly_files = [merged_file] + anomaly_files
    
    print(f"\nEvaluating CNN-LSTM for dataset: {dataset_name.upper()}")
    print(f"📊 Found {len(anomaly_files)} anomaly datasets for evaluation")

    results = []

    # 3. 정상 데이터 로드, 클리닝, 분할 (스케일러 훈련용 80% / 테스트용 20%)
    df_normal = pd.read_csv(normal_path)
    df_normal = shuffle(df_normal, random_state=123)
    split_point = int(len(df_normal) * train_split_ratio)
    df_normal_train = df_normal.iloc[:split_point] # 스케일러 훈련용
    df_normal_test = df_normal.iloc[split_point:] # 실제 테스트용

    df_normal_train = _clean_dataframe(df_normal_train)
    df_normal_test = _clean_dataframe(df_normal_test)
    
    print(f"✅ Normal data: Train(for scaler)={len(df_normal_train):,}, Test(for eval)={len(df_normal_test):,}")

    # 4. 정규화 (Scaler)
    scaler = MinMaxScaler()
    # 훈련 데이터(정상 80%) 기준으로 스케일러 피팅
    scaler.fit(df_normal_train.values) 
    
    # 정상 *테스트*(20%) 데이터를 스케일링 및 Reshape
    X_normal_test_flat = scaler.transform(df_normal_test.values)
    X_normal_test_seq = reshape_for_sequence(X_normal_test_flat, **model_params)
    y_normal_test = np.zeros(len(X_normal_test_seq))

    # 5. 임계값(Threshold) 설정
    threshold = 0.5
    print(f"\n📏 Threshold (fixed for CNN-LSTM): {threshold}")

    probs_by_attack = {} # 시각화를 위한 확률 저장
    numeric_keys = []

    # 6. 각 Anomaly 파일별 평가
    for file in anomaly_files:
        anomaly_path = os.path.join(base_dir, file)
        df_anomaly = pd.read_csv(anomaly_path)
        df_anomaly = _clean_dataframe(df_anomaly) # 클리닝 적용

        if df_anomaly.empty:
            print(f"\n⚠️ Warning: '{file}' is empty or became empty after cleaning. Skipping.")
            continue

        # 비정상 데이터를 스케일링 및 Reshape
        X_anomaly_flat = scaler.transform(df_anomaly.values)
        X_anomaly_seq = reshape_for_sequence(X_anomaly_flat, **model_params)
        y_anomaly = np.ones(len(X_anomaly_seq))
        
        # 테스트셋 구성 (Normal-Test + Anomaly)
        X_test = np.concatenate([X_normal_test_seq, X_anomaly_seq])
        y_test = np.concatenate([y_normal_test, y_anomaly])

        # 예측 및 확률 계산
        test_probs = model.predict(X_test, verbose=0).reshape(-1) # (N,) shape의 확률
        y_pred = (test_probs > threshold).astype(int)

        # Metrics 계산
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        # *** [수정] 예측 개수 계산 ***
        pred_normal = int(np.sum(y_pred == 0))
        pred_anomaly = int(np.sum(y_pred == 1))
        # **************************

        print(f"\n🚨 {file}")
        print(f"Samples: {len(X_test)} (Normal: {len(y_normal_test)}, Anomaly: {len(y_anomaly)})")
        print(f"Accuracy={acc:.6f}, Precision={prec:.6f}, Recall={rec:.6f}, F1={f1:.6f}")
        
        # *** [수정] 예측 개수 출력 ***
        print(f"Predicted counts: Normal={pred_normal:,}, Anomaly={pred_anomaly:,}")
        # **************************

        # 6-1. 합본(Merged) 파일인 경우 Confusion Matrix 및 ROC Curve 생성
        if file == merged_file:
            # === Confusion Matrix ===
            cm_save_path = f"./cnn_lstm/{dataset_name}_cnn_lstm_cm.png"
            title = f'CNN-LSTM CM - {dataset_name.upper()}'
            _plot_and_print_cm(y_test, y_pred, cm_save_path, ['Normal', 'Anomaly'], title)
            print(f"📈 Saved confusion matrix → {cm_save_path}")

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
            
            print(f"📈 Saved ROC curve → {roc_save_path} (AUC = {roc_auc:.4f})")

        # *** [수정] 결과에 예측 개수 추가 ***
        results.append({
            "File": file, "Samples": len(X_test), "Accuracy": acc,
            "Precision": prec, "Recall": rec, "F1": f1,
            "Pred_Normal": pred_normal, "Pred_Anomaly": pred_anomaly
        })
        # ********************************

        # 7. 시각화용 데이터 저장 (합본 파일 제외)
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

    # 8. 결과 요약 저장
    df = pd.DataFrame(results)
    summary_save_path = f"./cnn_lstm/{dataset_name}_cnn_lstm_summary.csv"
    
    df.to_csv(summary_save_path, index=False)
    print(f"\n📁 Saved summary → {summary_save_path}")
    print(df.round(6))

    # 9. 🔥 공격 유형별 출력 확률 히스토그램
    if probs_by_attack:
        plt.figure(figsize=(12, 7))
        
        # Normal (Test) 확률 계산
        probs_normal_test = model.predict(X_normal_test_seq, verbose=0).reshape(-1)
        
        plt.hist(probs_normal_test, bins=100, alpha=0.6, label="Normal", color="green", density=True, range=(0,1))

        # 공격별 히스토그램
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

        print(f"📊 Saved distribution plot → {plot_save_path}")

# --------------------------------------------------
# 5. 모델 로드 및 평가 설정
# --------------------------------------------------

# main_cnnlstm_fl.py에서 훈련시킨 모델의 파라미터 및 가중치 경로
MODEL_EVAL_CONFIG = {
    "KDD99": {
        "model_params": {"timesteps": 10, "features": 12},
        "weights": "./Results/KDD99/cnnlstm/cnn_lstm_weights.h5" # KDD99 훈련 가중치
    },
    "CSE-CIC-IDS2018": {
        "model_params": {"timesteps": 10, "features": 8},
        "weights": "./cnn_lstm/cnn_lstm_weights.h5" # (경로가 다르다면 수정)
    },
    "InSDN": {
        # 83 피처를 (10, 9) 등으로 훈련했다면 83 피처 csv 사용
        "model_params": {"timesteps": 12, "features": 7}, # 83 -> 90 (패딩)
        "weights": "Results/InSDN/cnnlstm/cnn_lstm_weights.h5" # (경로가 다르다면 수정)
    },
    "UNSW_NB15": {
        # 83 피처를 (10, 9) 등으로 훈련했다면 83 피처 csv 사용
        "model_params": {"timesteps": 6, "features": 7}, # 83 -> 90 (패딩)
        "weights": "Results/UNSW_NB15/cnnlstm/cnn_lstm_weights.h5" # (경로가 다르다면 수정)
    }
}

# --------------------------------------------------
# 6. 평가 실행
# --------------------------------------------------
if __name__ == "__main__":
    
    # --- ⚠️ 여기서 실행할 데이터셋을 선택하세요 ---
    DATASET_TO_RUN = "UNSW_NB15" 
    # (옵션: "KDD99", "CSE-CIC-IDS2018", "InSDN")
    # -----------------------------------------

    # CNN-LSTM 결과물 저장 디렉토리 생성
    os.makedirs("./cnn_lstm", exist_ok=True)

    # 선택된 데이터셋의 설정 로드
    if DATASET_TO_RUN not in MODEL_EVAL_CONFIG:
        print(f"❌ Error: '{DATASET_TO_RUN}'에 대한 모델 설정이 MODEL_EVAL_CONFIG에 없습니다.")
    else:
        eval_cfg = MODEL_EVAL_CONFIG[DATASET_TO_RUN]
        model_params = eval_cfg["model_params"]
        weights_path = eval_cfg["weights"]

        # 모델 빌드
        model = CNN_LSTM(**model_params)
        dummy_input_shape = (1, model_params["timesteps"], model_params["features"])
        _ = model(tf.zeros(dummy_input_shape))
        model.compile(optimizer=Adam(0.0001), loss="binary_crossentropy", metrics=["accuracy"])

        # 가중치 로드
        if os.path.exists(weights_path):
            model.load_weights(weights_path)
            print(f"✅ Loaded pre-trained weights from {weights_path}")
        else:
            print(f"⚠️ WARNING: Weight file not found at {weights_path}. 모델이 훈련되지 않았습니다.")
            exit()

        # 평가 실행
        evaluate_cnn_lstm_by_type(
            model=model,
            dataset_name=DATASET_TO_RUN,
            model_params=model_params,
            train_split_ratio=0.8 # 스케일러 훈련용 80% / 테스트용 20%
        )

        print(f"\n--- CNN-LSTM Evaluation Complete for {DATASET_TO_RUN} ---")

