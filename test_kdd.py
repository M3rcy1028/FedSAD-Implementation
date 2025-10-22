import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import tensorflow as tf
from model_taae_rnep import TransformerAAE

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# --------------------------------------------------
# KDD99 Dataset Evaluation Function
# --------------------------------------------------
def evaluate_taae_kdd99(model, base_dir="./KDD99/KDD99_split", features=117, percentile=95):
    normal_path = os.path.join(base_dir, "KDD99_normal.csv")

    # anomaly_x.csv 리스트 추출
    anomaly_files = sorted([
        f for f in os.listdir(base_dir)
        if f.startswith("KDD99_anomaly_") and f.endswith(".csv")
    ])

    # ✅ 전체 anomaly 합본 파일도 함께 평가
    anomaly_all_path = os.path.join(base_dir, "KDD99_anomaly.csv")
    if os.path.exists(anomaly_all_path):
        anomaly_files = ["KDD99_anomaly.csv"] + anomaly_files

    print(f"\n📊 Found {len(anomaly_files)} anomaly datasets for KDD99 evaluation")

    results = []

    # ✅ 정상 데이터 로드 및 절반 분리 (Train / Test)
    df_normal = pd.read_csv(normal_path)
    df_normal = df_normal.sample(frac=1, random_state=123).reset_index(drop=True)
    mid = len(df_normal) // 2
    df_normal_train = df_normal.iloc[:mid]
    df_normal_test = df_normal.iloc[mid:]

    # 정규화 fit은 Train(normal)으로만 수행
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(df_normal_train)
    X_normal_test = scaler.transform(df_normal_test)

    # ✅ Train(normal) reconstruction error → threshold 계산
    preds_train = model.predict(X_train, verbose=0)
    train_errors = np.mean(np.square(X_train - preds_train), axis=1)
    threshold = np.percentile(train_errors, percentile)
    print(f"\n📏 Threshold ({percentile}th percentile): {threshold:.6f}")

    # --------------------------------------------------
    # 각 anomaly_i.csv 에 대해 평가
    # --------------------------------------------------
    for file in anomaly_files:
        anomaly_path = os.path.join(base_dir, file)
        df_anomaly = pd.read_csv(anomaly_path)
        df_anomaly = df_anomaly.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)

        # ✅ label 컬럼이 있으면 제거
        if "label" in df_anomaly.columns:
            df_anomaly = df_anomaly.drop(columns=["label"])

        X_anomaly = scaler.transform(df_anomaly)
        X_test = np.concatenate([X_normal_test, X_anomaly])
        y_test = np.concatenate([np.zeros(len(X_normal_test)), np.ones(len(X_anomaly))])

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

        results.append({
            "File": file,
            "Samples": len(X_test),
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1": f1,
            "Pred_Normal": pred_normal,
            "Pred_Anomaly": pred_anomaly
        })

    # --------------------------------------------------
    # 결과 저장
    # --------------------------------------------------
    df = pd.DataFrame(results)
    save_path = os.path.join(base_dir, f"evaluation_mixed_summary_p{percentile}.csv")
    df.to_csv(save_path, index=False)
    print(f"\n📁 Saved summary → {save_path}")
    print(df.round(6))


# --------------------------------------------------
# 모델 로드 및 평가 실행
# --------------------------------------------------
if __name__ == "__main__":
    input_dim = 115  # ✅ KDD99 feature 수
    model = TransformerAAE(input_dim=input_dim)
    _ = model(tf.zeros((1, input_dim)), prior_labels=tf.zeros((1, 1)))  # build

    PRETRAIN_PATH = "Results/KDD99/rnep/rnep_frame_aae_transformer_weights.h5"
    model.load_weights(PRETRAIN_PATH)
    print(f"✅ Loaded pre-trained weights from {PRETRAIN_PATH}")

    PERCENTILE = 96
    evaluate_taae_kdd99(
        model=model,
        base_dir="./KDD99/KDD99_split",
        features=input_dim,
        percentile=PERCENTILE
    )

    print("PERCENTILE =", PERCENTILE)
