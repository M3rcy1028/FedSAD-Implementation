import numpy as np
import os
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from model_taae_rnep import TransformerAAE

os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 

# --------------------------------------------------
# 경로 및 파라미터
# --------------------------------------------------
WEIGHT_PATH = "Results/InSDN/rnep/rnep_frame_aae_transformer_weights.h5"
PERCENTILE = 70
RANDOM_SEED = 123
N_TRAIN = 5_000_000

# --------------------------------------------------
# 데이터 불러오기 (Normal 5M 학습, 나머지 Normal+Anomaly 테스트)
# --------------------------------------------------
def get_datasets_cic_with_labels(random_seed=123, n_train=5_000_000):
    # Normal
    df_normal = pd.read_csv("./CIC2018/ae_datas/CIC_ae_normal.csv")
    df_normal = df_normal.apply(pd.to_numeric, errors="coerce")
    df_normal = df_normal.replace([np.inf, -np.inf], np.nan).fillna(0)
    df_normal[df_normal < 0] = 0
    df_normal = shuffle(df_normal, random_state=random_seed)

    if len(df_normal) <= n_train:
        raise ValueError(f"Normal 데이터({len(df_normal)})가 {n_train}보다 적음!")

    df_normal_train = df_normal.iloc[:n_train].copy()
    df_normal_train["label"] = "Normal"
    df_normal_test = df_normal.iloc[n_train:].copy()
    df_normal_test["label"] = "Normal"

    # Anomaly
    anomaly_files = [
        ("DDOS_HOIC", "./CIC2018/ae_datas/CIC_anomaly_ae_1.csv"),
        ("DDoS_LOIC_HTTP", "./CIC2018/ae_datas/CIC_anomaly_ae_2.csv"),
        ("DoS_Hulk", "./CIC2018/ae_datas/CIC_anomaly_ae_3.csv"),
        ("Bot", "./CIC2018/ae_datas/CIC_anomaly_ae_4.csv"),
        ("FTP_BruteForce", "./CIC2018/ae_datas/CIC_anomaly_ae_5.csv"),
        ("SSH_BruteForce", "./CIC2018/ae_datas/CIC_anomaly_ae_6.csv"),
        ("Infiltration", "./CIC2018/ae_datas/CIC_anomaly_ae_7.csv"),
        ("DoS_SlowHTTPTest", "./CIC2018/ae_datas/CIC_anomaly_ae_8.csv"),
        ("DoS_GoldenEye", "./CIC2018/ae_datas/CIC_anomaly_ae_9.csv"),
        ("DoS_Slowloris", "./CIC2018/ae_datas/CIC_anomaly_ae_10.csv"),
        ("DDOS_LOIC_UDP", "./CIC2018/ae_datas/CIC_anomaly_ae_11.csv"),
        ("BruteForce_Web", "./CIC2018/ae_datas/CIC_anomaly_ae_12.csv"),
        ("BruteForce_XSS", "./CIC2018/ae_datas/CIC_anomaly_ae_13.csv"),
        ("SQL_Injection", "./CIC2018/ae_datas/CIC_anomaly_ae_14.csv"),
    ]

    df_anomaly_list = []
    for attack_name, f in anomaly_files:
        df = pd.read_csv(f)
        df = df.apply(pd.to_numeric, errors="coerce")
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
        df[df < 0] = 0
        df["label"] = attack_name
        df_anomaly_list.append(df)

    df_anomaly = pd.concat(df_anomaly_list, ignore_index=True)

    # Train/Test 분리
    df_train = df_normal_train
    df_test = pd.concat([df_normal_test, df_anomaly], ignore_index=True)

    # Feature / Label 분리
    X_train = df_train.drop(columns=["label"]).values
    X_test = df_test.drop(columns=["label"]).values
    y_test_labels = df_test["label"].values
    y_test_binary = (df_test["label"] != "Normal").astype(int)

    # Scaling (Normal Train 기준)
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"✅ Normal Train: {len(df_normal_train)}개")
    print(f"✅ Normal Test: {len(df_normal_test)}개")
    print(f"✅ Anomaly Test: {len(df_anomaly)}개")

    return X_train_scaled, X_test_scaled, y_test_binary, y_test_labels, scaler

# --------------------------------------------------
# 데이터 준비
# --------------------------------------------------
X_train_scaled, X_test_scaled, y_test, y_labels, scaler = get_datasets_cic_with_labels(
    random_seed=RANDOM_SEED, n_train=N_TRAIN
)
input_dim = X_train_scaled.shape[1]

# --------------------------------------------------
# 모델 초기화 & weight 로드
# --------------------------------------------------
model = TransformerAAE(input_dim)
_ = model(tf.zeros((1, input_dim)), prior_labels=tf.zeros((1,1)))  # D까지 build
model.load_weights(WEIGHT_PATH)
print(f"✅ Loaded weights from {WEIGHT_PATH}")

# --------------------------------------------------
# Threshold 계산 (Train Normal 기반)
# --------------------------------------------------
X_train_pred = model.predict(X_train_scaled, verbose=0)
train_errors = np.mean(np.square(X_train_scaled - X_train_pred), axis=1)
threshold = np.percentile(train_errors, PERCENTILE)
print(f"✅ Threshold set at {PERCENTILE}th percentile Threshold = {threshold:.6f}")

# --------------------------------------------------
# 전체 평가
# --------------------------------------------------
X_pred = model.predict(X_test_scaled, verbose=0)
errors = np.mean(np.square(X_test_scaled - X_pred), axis=1)
y_pred = (errors > threshold).astype(int)

report = classification_report(y_test, y_pred, target_names=["Normal","Anomaly"])
print("\n[전체 Classification Report]\n", report)

with open("report_cic.txt", "w") as f:
    f.write("[전체 Classification Report]\n")
    f.write(report)

# Confusion Matrix (이진)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Pred_Normal","Pred_Anomaly"],
            yticklabels=["Actual_Normal","Actual_Anomaly"])
plt.title("Confusion Matrix (Normal 5M train, rest test)")
plt.tight_layout()
plt.savefig("cm_cic.png", dpi=300)

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, errors)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(5,4))
plt.plot(fpr, tpr, lw=2, label=f"AUC={roc_auc:.4f}")
plt.plot([0,1],[0,1], linestyle="--", color="gray")
plt.xlabel("FPR"); plt.ylabel("TPR")
plt.title("ROC Curve (Normal 5M train, rest test)")
plt.legend(); plt.tight_layout()
plt.savefig("roc_cic.png", dpi=300)

# --------------------------------------------------
# 공격별 Confusion Matrix
# --------------------------------------------------
df_res = pd.DataFrame({
    "y_true": y_labels,
    "y_pred": np.where(y_pred == 0, "Normal", "Anomaly")
})
cm_attack = pd.crosstab(df_res["y_true"], df_res["y_pred"])

plt.figure(figsize=(10,12))
sns.heatmap(cm_attack, annot=True, fmt="d", cmap="Blues", cbar=False,
            annot_kws={"size":8})
plt.title("Confusion Matrix per Attack (Normal 5M train, rest test)")
plt.ylabel("Actual Class"); plt.xlabel("Prediction")
plt.tight_layout()
plt.savefig("cm_cic_per_attack.png", dpi=300)

print("✅ Saved report_cic.txt, cm_cic.png, roc_cic.png, cm_cic_per_attack.png")
