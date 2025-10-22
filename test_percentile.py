import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, precision_score, recall_score, f1_score, accuracy_score
)
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler

from model_taae_rnep import TransformerAAE

# --------------------------------------------------
# 기본 설정
# --------------------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
WEIGHT_PATH = "Results/InSDN/rnep/rnep_frame_aae_transformer_weights.h5"
RANDOM_SEED = 123
PERCENTILES = [50, 55, 60, 65, 70, 75, 80]

# --------------------------------------------------
# 데이터셋 로드 함수 (Train: 정상, Test: 정상 절반 + 이상)
# --------------------------------------------------
def get_datasets_insdn(random_seed=RANDOM_SEED):
    np.random.seed(random_seed)
    random.seed(random_seed)

    normal_path = "./InSDN/ae_datas/InSDN_normal.csv"
    anomaly_path = "./InSDN/ae_datas/InSDN_anomaly.csv"

    df_normal = pd.read_csv(normal_path)
    df_anomaly = pd.read_csv(anomaly_path)
    print(f"✅ Loaded InSDN data → Normal: {df_normal.shape}, Anomaly: {df_anomaly.shape}")

    # ✅ 정상 데이터 절반 split → train / test
    df_normal = shuffle(df_normal, random_state=random_seed)
    mid_idx = len(df_normal) // 2
    df_normal_train = df_normal.iloc[:mid_idx]
    df_normal_test = df_normal.iloc[mid_idx:]

    # 숫자형 변환 + 결측치 처리
    df_normal_train = df_normal_train.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
    df_normal_test = df_normal_test.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
    df_anomaly = df_anomaly.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)

    # ✅ 정규화 (train 기준 fit)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(df_normal_train.values)

    df_test = pd.concat([df_normal_test, df_anomaly], ignore_index=True)
    X_test_scaled = scaler.transform(df_test.values)
    y_test = np.concatenate([np.zeros(len(df_normal_test)), np.ones(len(df_anomaly))])
    X_test_scaled, y_test = shuffle(X_test_scaled, y_test, random_state=random_seed)

    print(f"Train: {X_train_scaled.shape}, Test: {X_test_scaled.shape}")
    print(f"Normal train: {len(df_normal_train)}, Normal test: {len(df_normal_test)}, Anomaly: {len(df_anomaly)}")
    return X_train_scaled, X_test_scaled, y_test, df_normal_test, df_anomaly


# --------------------------------------------------
# 데이터 로드
# --------------------------------------------------
X_train_scaled, X_test_scaled, y_test, df_normal_test, df_anomaly = get_datasets_insdn(RANDOM_SEED)
input_dim = X_train_scaled.shape[1]

# --------------------------------------------------
# 모델 로드
# --------------------------------------------------
model = TransformerAAE(input_dim)
_ = model(tf.zeros((1, input_dim)), prior_labels=tf.zeros((1, 1)))  # build
model.load_weights(WEIGHT_PATH)
print(f"✅ Loaded weights from {WEIGHT_PATH}")

# --------------------------------------------------
# Train Reconstruction Error 계산 (Threshold 기준)
# --------------------------------------------------
X_train_pred = model.predict(X_train_scaled, verbose=0)
train_errors = np.mean(np.square(X_train_scaled - X_train_pred), axis=1)

# --------------------------------------------------
# Percentile 실험 (Train-normal 기반 threshold)
# --------------------------------------------------
results = []

for p in PERCENTILES:
    threshold = np.percentile(train_errors, p)

    # ✅ Test (normal + anomaly)에 적용
    X_pred = model.predict(X_test_scaled, verbose=0)
    errors = np.mean(np.square(X_test_scaled - X_pred), axis=1)
    y_pred = (errors > threshold).astype(int)

    # Metrics 계산
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    fpr, tpr, _ = roc_curve(y_test, errors)
    roc_auc = auc(fpr, tpr)

    results.append({
        "Percentile": p,
        "Threshold": threshold,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "AUC": roc_auc
    })

    print(f"\n[Percentile {p}]")
    print(f"Threshold={threshold:.6f}")
    print(f"Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}, AUC={roc_auc:.4f}")

# ✅ DataFrame 저장
df_res = pd.DataFrame(results)
os.makedirs("Results/InSDN", exist_ok=True)
df_res.to_csv("Results/InSDN/percentile_experiment_results.csv", index=False)
print("\n📁 Saved percentile_experiment_results.csv")
print(df_res.round(4))

# ✅ AUC 기준 최적 percentile 선택
best_idx = df_res["AUC"].idxmax()
best_row = df_res.loc[best_idx]
best_p = best_row["Percentile"]
best_auc = best_row["AUC"]
best_threshold = best_row["Threshold"]

print(f"\n🏆 Best Percentile: {best_p} (AUC={best_auc:.4f}, Threshold={best_threshold:.6f})")

# --------------------------------------------------
# 이후 부분: confusion matrix / ROC curve / report 등은 best_p 기준으로 유지
# --------------------------------------------------
threshold = best_threshold
X_pred = model.predict(X_test_scaled, verbose=0)
errors = np.mean(np.square(X_test_scaled - X_pred), axis=1)
y_pred = (errors > threshold).astype(int)

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Pred_Normal", "Pred_Anomaly"],
            yticklabels=["Actual_Normal", "Actual_Anomaly"])
plt.title(f"Confusion Matrix (InSDN, {best_p}th Percentile, AUC={best_auc:.4f})")
plt.tight_layout()
plt.savefig(f"Results/InSDN/cm_best_p{int(best_p)}.png", dpi=300)

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, errors)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(5, 4))
plt.plot(fpr, tpr, lw=2, label=f"AUC={roc_auc:.4f} (p={best_p})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title(f"ROC Curve (Best Percentile {best_p})")
plt.legend()
plt.tight_layout()
plt.savefig(f"Results/InSDN/roc_best_p{int(best_p)}.png", dpi=300)
print(f"✅ Saved best ROC & Confusion Matrix (p={best_p}, AUC={best_auc:.4f})")
