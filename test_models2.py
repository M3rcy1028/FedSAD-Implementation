import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.utils import shuffle
import tensorflow as tf
from model_taae_rnep import TransformerAAE  # 이 모델 클래스가 필요합니다.
import matplotlib.pyplot as plt
import seaborn as sns

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# --------------------------------------------------
# 데이터 클리닝 헬퍼 함수
# --------------------------------------------------
def _clean_dataframe(df):
    """'Label'/'label' 컬럼을 삭제하고, 'inf'/'nan' 값을 0으로 대체하며, 큰 값을 clip합니다."""
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    # 너무 큰 값 잘라내기 (InSDN, CIC 데이터셋의 특성)
    df = np.clip(df, -1e6, 1e6) 
    return df

# --------------------------------------------------
# Confusion Matrix 헬퍼 함수
# --------------------------------------------------
def _plot_and_print_cm(y_test, y_pred, save_path, labels, title):
    """Confusion Matrix를 DataFrame으로 출력하고, seaborn 히트맵으로 저장합니다."""
    cm = confusion_matrix(y_test, y_pred)
    
    # 1. 텍스트(DataFrame)로 출력
    try:
        # cm.ravel()이 (tn, fp, fn, tp) 4개 값을 반환한다고 가정
        tn, fp, fn, tp = cm.ravel()
        cm_table = pd.DataFrame(
            [[tn, fp], [fn, tp]],
            index=[f'Actual {labels[0]}', f'Actual {labels[1]}'],
            columns=[f'Predicted {labels[0]}', f'Predicted {labels[1]}']
        )
        print("\n🧾 [Confusion Matrix]")
        print(cm_table)
    except ValueError: 
        # 2x2 행렬이 아닌 경우 (예: 한 클래스만 예측된 경우)
        print(f"\n🧾 [Confusion Matrix] (Raw)\n{cm}")

    # 2. 이미지(Heatmap)로 저장
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
# 데이터셋별 설정
# --------------------------------------------------
DATASET_CONFIG = {
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
        "base_dir": "./CIC2018/ae_datas_all_features",
        "normal_file": "CIC_ae_normal.csv",
        "anomaly_prefix": "CIC_anomaly_ae_", # 개별 파일 접두사
        "merged_anomaly_file": "CIC_ae_anomaly.csv", # 합본 파일 이름
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
    }
}

# --------------------------------------------------
# 범용 평가 함수
# --------------------------------------------------
def evaluate_dataset(model, dataset_name, percentile, train_split_ratio=0.8):
    """
    지정된 데이터셋에 대해 TAAE 모델의 이상 탐지 성능을 평가합니다.

    :param model: 훈련된 TAAE 모델
    :param dataset_name: "KDD99", "InSDN", "CSE-CIC-IDS2018" 중 하나
    :param percentile: 임계값 계산을 위한 백분위수 (예: 90)
    :param train_split_ratio: 정상 데이터를 훈련/테스트로 나눌 비율 (예: 0.8)
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
    plot_save_path = config["plot_save_path"]

    # 2. Anomaly 파일 리스트 탐색
    anomaly_files = sorted([
        f for f in os.listdir(base_dir)
        if f.startswith(anomaly_prefix) and f.endswith(".csv")
    ])
    
    # 합본 파일(Merged)을 리스트 맨 앞에 추가
    merged_path = os.path.join(base_dir, merged_file)
    if os.path.exists(merged_path):
        anomaly_files = [merged_file] + anomaly_files
    
    print(f"\nEvaluating dataset: {dataset_name.upper()}")
    print(f"📊 Found {len(anomaly_files)} anomaly datasets for evaluation")

    results = []

    # 3. 정상 데이터 로드, 클리닝 및 분할
    df_normal = pd.read_csv(normal_path)
    df_normal = shuffle(df_normal, random_state=0) # InSDN
    # df_normal = df_normal.sample(frac=1, random_state=48).reset_index(drop=True) # KDD99, NSL-KDD
    split_point = int(len(df_normal) * train_split_ratio)
    df_normal_train = df_normal.iloc[:split_point]
    df_normal_test = df_normal.iloc[split_point:]

    df_normal_train = _clean_dataframe(df_normal_train)
    df_normal_test = _clean_dataframe(df_normal_test)

    # train_split_ratio = 0.8 # CIC2018

    # 🔹 1. 정상 데이터 전체를 하나의 feature 기준으로 정렬 (예: 재구성 오차나 합계 등)
    # 만약 특정 컬럼 없으면 평균값으로 스코어를 만들어서 기준 삼기
    # score = df_normal.mean(axis=1)   # 각 row의 평균값 (정상성 정도 대략 반영)

    # # 🔹 2. score 오름차순 정렬 후 뒤쪽(상위 80%) 선택
    # df_normal['score'] = score
    # df_normal_sorted = df_normal.sort_values(by='score').reset_index(drop=True)

    # split_point = int(len(df_normal_sorted) * (1 - train_split_ratio))
    # df_normal_train = df_normal_sorted.iloc[split_point:]   # 상위 80%
    # df_normal_test = df_normal_sorted.iloc[:split_point]    # 하위 20%

    # # 🔹 3. score 컬럼 제거 (필요시)
    # df_normal_train = df_normal_train.drop(columns=['score'])
    # df_normal_test = df_normal_test.drop(columns=['score'])
    # # ***************

    df_normal_train = _clean_dataframe(df_normal_train)
    df_normal_test = _clean_dataframe(df_normal_test)

    print(f"✅ Train size: {len(df_normal_train):,}, Test size: {len(df_normal_test):,}")

    # print(f"Normal data split: Train={len(df_normal_train)}, Test={len(df_normal_test)}")

    # 4. 정규화 (Scaler)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(df_normal_train.values)
    X_normal_test = scaler.transform(df_normal_test.values)

    # 5. 임계값(Threshold) 계산
    preds_train = model.predict(X_train, verbose=0)
    train_errors = np.mean(np.square(X_train - preds_train), axis=1)
    threshold = np.percentile(train_errors, percentile)
    threshold = 0.003020
    print(f"\n📏 Threshold ({percentile}th percentile): {threshold:.6f}")

    error_by_attack = {} # 시각화를 위한 오류 저장
    numeric_keys = []

    # 6. 각 Anomaly 파일별 평가
    for file in anomaly_files:
        anomaly_path = os.path.join(base_dir, file)
        df_anomaly = pd.read_csv(anomaly_path)
        df_anomaly = _clean_dataframe(df_anomaly) # 클리닝 적용

        if "label" in df_anomaly.columns:
            df_anomaly = df_anomaly.drop(columns=["label"])

        X_anomaly = scaler.transform(df_anomaly.values)
        
        # 테스트셋 구성 (Normal-Test + Anomaly)
        X_test = np.concatenate([X_normal_test, X_anomaly])
        y_test = np.concatenate([np.zeros(len(X_normal_test)), np.ones(len(X_anomaly))])

        # 예측 및 오류 계산
        preds_test = model.predict(X_test, verbose=0)
        test_errors = np.mean(np.square(X_test - preds_test), axis=1)
        y_pred = (test_errors > threshold).astype(int)

        # Metrics 계산
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        pred_normal = int(np.sum(y_pred == 0))
        pred_anomaly = int(np.sum(y_pred == 1))

        print(f"\n🚨 {file}")
        print(f"Samples: {len(X_test)} | Accuracy={acc:.6f}, Precision={prec:.6f}, Recall={rec:.6f}, F1={f1:.6f}")

        # 6-1. 합본(Merged) 파일인 경우 Confusion Matrix 및 ROC Curve 생성
        if file == merged_file:
            
            # === Confusion Matrix (기존 로직) ===
            cm_save_path = f"./{dataset_name}_cm.png"
            title = f'Confusion Matrix - {dataset_name.upper()} (P={percentile})'
            
            _plot_and_print_cm(
                y_test, 
                y_pred, 
                cm_save_path, 
                labels=['Normal', 'Anomaly'], 
                title=title
            )
            print(f"📈 Saved confusion matrix → {cm_save_path}")

            # === ROC Curve (📌 신규 추가) ===
            # (y_test는 0/1, test_errors는 점수(reconstruction error)임)
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
            
            print(f"📈 Saved ROC curve → {roc_save_path} (AUC = {roc_auc:.4f})")

        results.append({
            "File": file, "Samples": len(X_test), "Accuracy": acc,
            "Precision": prec, "Recall": rec, "F1": f1,
            "Pred_Normal": pred_normal, "Pred_Anomaly": pred_anomaly
        })

        # 7. 시각화용 데이터 저장 (합본 파일 제외)
        if file != merged_file:
            try:
                num_str = file.replace(anomaly_prefix, "").replace(".csv", "")
                attack_num = int(num_str)
                
                # Anomaly 데이터만(X_normal_test 제외)의 오류 계산
                preds_anomaly_only = model.predict(X_anomaly, verbose=0)
                errors_anomaly_only = np.mean(np.square(X_anomaly - preds_anomaly_only), axis=1)

                attack_label = attack_map.get(attack_num, f"attack_{attack_num}")
                error_by_attack[attack_num] = (attack_label, errors_anomaly_only)
                numeric_keys.append(attack_num)
            except Exception as e:
                print(f"Warning: Could not parse attack number from '{file}'. Skipping for plot. Error: {e}")

    # 8. 결과 저장
    df = pd.DataFrame(results)
    summary_save_path = f"./{dataset_name}_summary.csv"
    
    df.to_csv(summary_save_path, index=False)
    print(f"\n📁 Saved summary → {summary_save_path}")
    print(df.round(6))

    # 9. 🔥 공격 유형별 Reconstruction Error 히스토그램
    if error_by_attack:
        plt.figure(figsize=(12, 7))
        
        # Normal (Test) 오류 계산
        preds_normal_test = model.predict(X_normal_test, verbose=0)
        errors_normal_test = np.mean(np.square(X_normal_test - preds_normal_test), axis=1)
        
        # Normal 히스토그램
        plt.hist(errors_normal_test, bins=200, alpha=0.6, label="Normal", color="green", density=True)

        # 공격별 히스토그램
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

        print(f"📊 Saved distribution plot → {plot_save_path}")

# --------------------------------------------------
# 모델 로드 설정 (훈련된 모델 클래스 'TransformerAAE'가 필요합니다)
# --------------------------------------------------

# 'model_taae_rnep' 모듈이 없으면 실행되지 않으므로, 
# 실제 사용 시에는 TransformerAAE 클래스가 정의된 파일을 import 해야 합니다.
try:
    from model_taae_rnep import TransformerAAE
except ImportError:
    print("="*50)
    print("⚠️ WARNING: 'model_taae_rnep' 모듈을 찾을 수 없습니다.")
    print("실제 평가를 위해서는 'TransformerAAE' 모델 클래스가 필요합니다.")
    print("임시로 tf.keras.Model을 생성하여 스크립트 실행 테스트를 진행합니다.")
    print("="*50)
    
    # 임시 모델 (스크립트 실행 테스트용)
    def create_dummy_model(input_dim):
        inp = tf.keras.Input(shape=(input_dim,))
        x = tf.keras.layers.Dense(int(input_dim/2), activation='relu')(inp)
        out = tf.keras.layers.Dense(input_dim, activation='sigmoid')(x)
        model = tf.keras.Model(inputs=inp, outputs=out)
        # TAAE 모델의 predict 시그니처와 맞추기 위해 더미 함수 추가
        original_predict = model.predict
        model.predict = lambda x, verbose=0, prior_labels=None: original_predict(x, verbose=verbose)
        return model
    
    # TransformerAAE 클래스를 임시 모델 생성 함수로 대체
    TransformerAAE = lambda input_dim: create_dummy_model(input_dim)


MODEL_CONFIG = {
    "KDD99": {
        "input_dim": 115,
        "weights": "Results/KDD99/rnep/rnep_frame_aae_transformer_weights.h5"
    },
    "InSDN": {
        "input_dim": 83,
        "weights": "Results/InSDN/rnep/rnep_frame_aae_transformer_weights.h5"
        # "weights": "rnep_frame_revised2/rnep_frame_aae_transformer_weights.h5"
    },
    "CSE-CIC-IDS2018": {
        "input_dim": 78,
        "weights": "rnep_cic2018/rnep_frame_aae_transformer_weights.h5"
    }
}

# --------------------------------------------------
# 평가 실행
# --------------------------------------------------
if __name__ == "__main__":
    
    # --- ⚠️ 여기서 실행할 데이터셋을 선택하세요 ---
    DATASET_TO_RUN = "CSE-CIC-IDS2018" 
    # (옵션: "KDD99", "InSDN", "CSE-CIC-IDS2018")
    # -----------------------------------------

    PERCENTILE = 75
    
    # 선택된 데이터셋의 설정 로드
    if DATASET_TO_RUN not in MODEL_CONFIG:
        print(f"❌ Error: '{DATASET_TO_RUN}'에 대한 모델 설정이 MODEL_CONFIG에 없습니다.")
    else:
        config = MODEL_CONFIG[DATASET_TO_RUN]
        input_dim = config["input_dim"]
        weights_path = config["weights"]

        # 모델 빌드
        model = TransformerAAE(input_dim=input_dim)
        # TAAE 모델은 build를 위해 prior_labels가 필요할 수 있음 (원본 코드 기준)
        try:
             _ = model(tf.zeros((1, input_dim)), prior_labels=tf.zeros((1, 1)))
        except Exception as e:
            print(f"Model build with prior_labels failed, trying without: {e}")
            try:
                # prior_labels가 없는 경우를 대비한 빌드 시도
                 _ = model.build(input_shape=(None, input_dim))
            except Exception as e2:
                 print(f"Model build failed: {e2}")
                 # build가 안되면 predict에서 오류날 수 있음
                 pass

        # 가중치 로드
        if os.path.exists(weights_path):
            model.load_weights(weights_path)
            print(f"✅ Loaded pre-trained weights from {weights_path}")
        else:
            print(f"⚠️ WARNING: Weight file not found at {weights_path}. 모델이 훈련되지 않았습니다.")

        # 평가 실행
        evaluate_dataset(
            model=model,
            dataset_name=DATASET_TO_RUN,
            percentile=PERCENTILE
        )

        print(f"\n--- Evaluation Complete for {DATASET_TO_RUN} (Percentile = {PERCENTILE}) ---")