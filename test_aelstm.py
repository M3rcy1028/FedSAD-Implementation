import os
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
from model_aelstm import AE_LSTM

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

def reshape_for_sequence(X, timesteps=10, features=12):
    """
    2D (N, F) 데이터를 3D (N, T, F) 시퀀스 데이터로 Reshape (패딩/절단)
    (main_aelstm.py의 헬퍼 함수)
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
# 4. AE-LSTM 유형별 평가 함수
# --------------------------------------------------
def evaluate_aelstm_by_type(model, dataset_name, model_params, train_split_ratio=0.8, percentile=95, beta=0.5):
    """
    AE-LSTM 모델의 유형별 성능을 (1)재구성 오류, (2)분류기 확률 두 가지로 평가합니다.
    
    :param model: 훈련된 AE-LSTM 모델
    :param dataset_name: "KDD99", "CSE-CIC-IDS2018", "InSDN" 중 하나
    :param model_params: 모델 입력을 위한 딕셔너리 (예: {'timesteps': 10, 'features': 12})
    :param train_split_ratio: 스케일러 학습 및 정상 테스트셋 분리 비율
    :param percentile: 재구성 오류 임계값 계산을 위한 백분위수
    """
    
    # 1. 설정 로드
    try:
        config = DATASET_CONFIG[dataset_name]
    except KeyError:
        print(f"❌ Error: No config found for dataset '{dataset_name}'")
        return

    timesteps = model_params['timesteps']
    features = model_params['features']
    
    base_dir = config["base_dir"]
    normal_path = os.path.join(base_dir, config["normal_file"])
    anomaly_prefix = config["anomaly_prefix"]
    merged_file = config["merged_anomaly_file"]
    attack_map = config["attack_map"]
    
    # 결과 저장 경로
    save_dir = "./ae_lstm_eval"
    os.makedirs(save_dir, exist_ok=True)

    # 2. Anomaly 파일 리스트 탐색
    anomaly_files = sorted([
        f for f in os.listdir(base_dir)
        if f.startswith(anomaly_prefix) and f.endswith(".csv")
    ])
    merged_path = os.path.join(base_dir, merged_file)
    if os.path.exists(merged_path):
        anomaly_files = [merged_file] + anomaly_files
    
    print(f"\nEvaluating AE-LSTM for dataset: {dataset_name.upper()}")
    print(f"📊 Found {len(anomaly_files)} anomaly datasets for evaluation")

    results = []

    # 3. 정상 데이터 로드, 클리닝, 분할 (스케일러 훈련용 80% / 테스트용 20%)
    df_normal = pd.read_csv(normal_path)
    df_normal = shuffle(df_normal, random_state=42)
    split_point = int(len(df_normal) * train_split_ratio)
    df_normal_train = df_normal.iloc[:split_point] # 스케일러 훈련용
    df_normal_test = df_normal.iloc[split_point:] # 실제 테스트용

    df_normal_train = _clean_dataframe(df_normal_train)
    df_normal_test = _clean_dataframe(df_normal_test)
    
    print(f"✅ Normal data: Train(for scaler)={len(df_normal_train):,}, Test(for eval)={len(df_normal_test):,}")

    # 4. 정규화 (Scaler)
    scaler = MinMaxScaler()
    X_train_flat = scaler.fit_transform(df_normal_train.values) 
    X_normal_test_flat = scaler.transform(df_normal_test.values)
    
    # Reshape (Normal Test)
    X_normal_test_seq = reshape_for_sequence(X_normal_test_flat, timesteps, features)
    y_normal_test = np.zeros(len(X_normal_test_seq))

    # 5. 임계값(Threshold) 계산
    # 5-1. (Reconstruction) 훈련 데이터(정상 80%) 기준
    X_train_seq = reshape_for_sequence(X_train_flat, timesteps, features)
    preds_train_dict = model.predict(X_train_seq, verbose=0)
    recon_train = preds_train_dict["decoded"]
    # 3D (N, T, F) 데이터의 MSE는 axis=(1, 2)
    train_errors = np.mean(np.square(X_train_seq - recon_train), axis=(1, 2))
    threshold_recon = np.percentile(train_errors, percentile)
    
    # 5-2. (Classifier) 고정 임계값
    threshold_cls = 0.5
    print(f"\n📏 Recon Threshold ({percentile}th percentile): {threshold_recon:.6f}")
    print(f"📏 Classifier Threshold (fixed): {threshold_cls}")

    recon_errors_by_attack = {} # 시각화를 위한 오류 저장
    cls_probs_by_attack = {}    # 시각화를 위한 확률 저장
    numeric_keys = []

    # 6. 각 Anomaly 파일별 평가
    for file in anomaly_files:
        anomaly_path = os.path.join(base_dir, file)
        df_anomaly = pd.read_csv(anomaly_path)
        df_anomaly = _clean_dataframe(df_anomaly)

        if df_anomaly.empty:
            print(f"\n⚠️ Warning: '{file}' is empty or became empty after cleaning. Skipping.")
            continue

        # 비정상 데이터 스케일링 및 Reshape
        X_anomaly_flat = scaler.transform(df_anomaly.values)
        X_anomaly_seq = reshape_for_sequence(X_anomaly_flat, timesteps, features)
        y_anomaly = np.ones(len(X_anomaly_seq))
        
        # 테스트셋 구성 (Normal-Test + Anomaly)
        X_test_seq = np.concatenate([X_normal_test_seq, X_anomaly_seq])
        y_test = np.concatenate([y_normal_test, y_anomaly])

        # 예측 (모델 출력은 딕셔너리)
        preds_test_dict = model.predict(X_test_seq, verbose=0)
        recon_test = preds_test_dict["decoded"]
        probs_test = preds_test_dict["pred"].flatten() # (N,) shape의 확률

        # --- [방법 1: 재구성 오류 기반] ---
        test_errors = np.mean(np.square(X_test_seq - recon_test), axis=(1, 2))
        y_pred_recon = (test_errors > threshold_recon).astype(int)

        acc_recon = accuracy_score(y_test, y_pred_recon)
        f1_recon = f1_score(y_test, y_pred_recon, zero_division=0)
        prec_recon = precision_score(y_test, y_pred_recon, zero_division=0)
        rec_recon = recall_score(y_test, y_pred_recon, zero_division=0)
        pred_normal_recon = int(np.sum(y_pred_recon == 0))
        pred_anomaly_recon = int(np.sum(y_pred_recon == 1))

        # --- [방법 2: 분류기 확률 기반] ---
        y_pred_cls = (probs_test > threshold_cls).astype(int)

        acc_cls = accuracy_score(y_test, y_pred_cls)
        f1_cls = f1_score(y_test, y_pred_cls, zero_division=0)
        prec_cls = precision_score(y_test, y_pred_cls, zero_division=0)
        rec_cls = recall_score(y_test, y_pred_cls, zero_division=0)
        pred_normal_cls = int(np.sum(y_pred_cls == 0))
        pred_anomaly_cls = int(np.sum(y_pred_cls == 1))

         # --- [방법 3: Fusion 기반] ---
        beta = 0.5  # 🔹 재구성 오류와 분류 확률의 가중 비율 (0.5 = 동일 비중)
        
        # 재구성 오류를 [0,1] 구간으로 스케일링
        e_min, e_max = np.min(test_errors), np.max(test_errors)
        e_scaled = (test_errors - e_min) / (e_max - e_min + 1e-12)
        
        # Fusion 점수 계산
        fusion_score = beta * e_scaled + (1 - beta) * probs_test
        
        # 임계값 설정 (기본: 0.5)
        threshold_fusion = 0.5
        y_pred_fusion = (fusion_score > threshold_fusion).astype(int)

        acc_fusion = accuracy_score(y_test, y_pred_fusion)
        f1_fusion = f1_score(y_test, y_pred_fusion, zero_division=0)
        prec_fusion = precision_score(y_test, y_pred_fusion, zero_division=0)
        rec_fusion = recall_score(y_test, y_pred_fusion, zero_division=0)

        # --- [출력] ---
        print(f"\n🚨 {file}")
        print(f"  Samples: {len(X_test_seq)} (Normal: {len(y_normal_test)}, Anomaly: {len(y_anomaly)})")
        print(f"  [Recon] Acc={acc_recon:.6f}, F1={f1_recon:.6f} | Pred Cnts: N={pred_normal_recon:,}, A={pred_anomaly_recon:,}")
        print(f"  [Class] Acc={acc_cls:.6f}, F1={f1_cls:.6f} | Pred Cnts: N={pred_normal_cls:,}, A={pred_anomaly_cls:,}")
        print(f"  [Fusion β={beta}] Acc={acc_fusion:.6f}, F1={f1_fusion:.6f} | Prec={prec_fusion:.6f}, Rec={rec_fusion:.6f}")

        # 6-1. 합본(Merged) 파일인 경우 Confusion Matrix 및 ROC Curve 생성
        if file == merged_file:
            # === CM (Recon) ===
            cm_save_path_r = os.path.join(save_dir, f"{dataset_name}_recon_cm.png")
            title_r = f'AE-LSTM (Recon) CM - {dataset_name.upper()} (P={percentile})'
            _plot_and_print_cm(y_test, y_pred_recon, cm_save_path_r, ['Normal', 'Anomaly'], title_r)
            print(f"  📈 Saved Recon CM → {cm_save_path_r}")
            
            # === CM (Class) ===
            cm_save_path_c = os.path.join(save_dir, f"{dataset_name}_cls_cm.png")
            title_c = f'AE-LSTM (Classifier) CM - {dataset_name.upper()}'
            _plot_and_print_cm(y_test, y_pred_cls, cm_save_path_c, ['Normal', 'Anomaly'], title_c)
            print(f"  📈 Saved Classifier CM → {cm_save_path_c}")

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
            print(f"  📈 Saved Recon ROC → {roc_save_path_r} (AUC = {roc_auc_r:.4f})")
            
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
            print(f"  📈 Saved Classifier ROC → {roc_save_path_c} (AUC = {roc_auc_c:.4f})")

        # 결과 저장
        results.append({
            "File": file, "Samples": len(X_test_seq), 
            "Acc_Recon": acc_recon, "Prec_Recon": prec_recon, "Rec_Recon": rec_recon, "F1_Recon": f1_recon,
            "Pred_N_Recon": pred_normal_recon, "Pred_A_Recon": pred_anomaly_recon,
            "Acc_Cls": acc_cls, "Prec_Cls": prec_cls, "Rec_Cls": rec_cls, "F1_Cls": f1_cls,
            "Acc_Fusion": acc_fusion, "Prec_Fusion": prec_fusion,
            "Rec_Fusion": rec_fusion, "F1_Fusion": f1_fusion,
            "Pred_N_Cls": pred_normal_cls, "Pred_A_Cls": pred_anomaly_cls
        })

        # 7. 시각화용 데이터 저장 (합본 파일 제외)
        if file != merged_file:
            try:
                num_str = file.replace(anomaly_prefix, "").replace(".csv", "")
                attack_num = int(num_str)
                attack_label = attack_map.get(attack_num, f"attack_{attack_num}")
                
                # Anomaly 데이터만 예측
                preds_anomaly_dict = model.predict(X_anomaly_seq, verbose=0)
                
                # 재구성 오류 저장
                errors_anomaly_only = np.mean(np.square(X_anomaly_seq - preds_anomaly_dict["decoded"]), axis=(1,2))
                recon_errors_by_attack[attack_num] = (attack_label, errors_anomaly_only)
                
                # 분류기 확률 저장
                probs_anomaly_only = preds_anomaly_dict["pred"].flatten()
                cls_probs_by_attack[attack_num] = (attack_label, probs_anomaly_only)

                numeric_keys.append(attack_num)
            except Exception as e:
                print(f"Warning: Could not parse attack number from '{file}'. Skipping for plot. Error: {e}")

    # 8. 결과 요약 저장
    df = pd.DataFrame(results)
    summary_save_path = os.path.join(save_dir, f"{dataset_name}_summary.csv")
    
    df.to_csv(summary_save_path, index=False)
    print(f"\n📁 Saved summary → {summary_save_path}")
    print(df.round(6))

    # 9-1. 🔥 공격 유형별 Reconstruction Error 히스토그램
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
        print(f"📊 Saved Recon distribution plot → {plot_save_path_r}")

    # 9-2. 🔥 공격 유형별 Classifier Probability 히스토그램
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
        print(f"📊 Saved Classifier distribution plot → {plot_save_path_c}")


# --------------------------------------------------
# 5. 모델 로드 및 평가 설정 (main_aelstm.py 기준)
# --------------------------------------------------

MODEL_EVAL_CONFIG = {
    "KDD99": {
        # main_aelstm.py에서 KDD99는 (10, 13)으로 reshape
        "model_params": {"timesteps": 10, "features": 13},
        "weights": "Results/KDD99/aelstm/ae_lstm_weights.h5" # KDD99 훈련 가중치
    },
    "CSE-CIC-IDS2018": {
        # main_aelstm.py에서 CIC는 (10, 4)로 reshape
        "model_params": {"timesteps": 10, "features": 8},
        "weights": "Results/CSE-CIC-IDS2018/aelstm/ae_lstm_weights.h5" # (경로가 다르다면 수정)
    },
    "InSDN": {
         "model_params": {"timesteps": 12, "features": 7},
         "weights": "Results/InSDN/aelstm/ae_lstm_weights.h5" # (경로가 다르다면 수정)
    },
    "UNSW_NB15": { # 85
        "model_params": {"timesteps": 6, "features": 7},
        "weights": "Results/UNSW_NB15/aelstm/ae_lstm_weights.h5" # (경로가 다르다면 수정)
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

    # --- ⚠️ 재구성 오류 임계값 백분위수 ---
    PERCENTILE = 85
    # ---------------------------------
    
    # 선택된 데이터셋의 설정 로드
    if DATASET_TO_RUN not in MODEL_EVAL_CONFIG:
        print(f"❌ Error: '{DATASET_TO_RUN}'에 대한 모델 설정이 MODEL_EVAL_CONFIG에 없습니다.")
    else:
        eval_cfg = MODEL_EVAL_CONFIG[DATASET_TO_RUN]
        model_params = eval_cfg["model_params"]
        weights_path = eval_cfg["weights"]

        # 모델 빌드
        model = AE_LSTM(
            input_dim=model_params['features'],
            timesteps=model_params['timesteps'],
            features=model_params['features']
        )
        
        # 모델 컴파일 (loss 정의 필요)
        model.compile(
            optimizer='adam', 
            loss={"decoded": "mse", "pred": "binary_crossentropy"}
        )

        # 모델 빌드 (더미 입력)
        dummy_input_shape = (1, model_params["timesteps"], model_params["features"])
        _ = model(tf.zeros(dummy_input_shape))

        # 가중치 로드
        if os.path.exists(weights_path):
            model.load_weights(weights_path)
            print(f"✅ Loaded pre-trained weights from {weights_path}")
        else:
            print(f"⚠️ WARNING: Weight file not found at {weights_path}. 모델이 훈련되지 않았습니다.")
            exit()

        # 평가 실행
        evaluate_aelstm_by_type(
            model=model,
            dataset_name=DATASET_TO_RUN,
            model_params=model_params,
            train_split_ratio=0.8, # 스케일러 훈련용 80% / 테스트용 20%
            percentile=PERCENTILE,
            beta=0.9
        )

        print(f"\n--- AE-LSTM Evaluation Complete for {DATASET_TO_RUN} (Percentile = {PERCENTILE}) ---")
