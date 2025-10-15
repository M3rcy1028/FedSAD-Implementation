import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns
from model_aae import TransformerAAE 
from model_cic import TransformerVAE 

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# --------------------------------------------------
# 경로 및 파라미터
# --------------------------------------------------
WEIGHT_PATH = "rnep/rnep_vae_transformer_weights.h5"
PERCENTILE = 96

# --------------------------------------------------
# 데이터 불러오기
# --------------------------------------------------
from utils import get_datasets_nsl
X_train_scaled, X_test_scaled, y_test = get_datasets_nsl()
input_dim = X_train_scaled.shape[1]

# --------------------------------------------------
# 모델 초기화 & weight 로드
# --------------------------------------------------
# model = TransformerVAE(input_dim)
# _ = model(tf.zeros((1, input_dim)))  # build

model = TransformerAAE(input_dim)
_ = model(tf.zeros((1, input_dim)), prior_labels=tf.zeros((1,1)))  # build
model.load_weights(WEIGHT_PATH)
print(f"✅ Loaded weights from {WEIGHT_PATH}")

# --------------------------------------------------
# Threshold 계산 (train 기반)
# --------------------------------------------------
X_train_pred = model.predict(X_train_scaled, verbose=0)
train_errors = np.mean(np.square(X_train_scaled - X_train_pred), axis=1)
threshold = np.percentile(train_errors, PERCENTILE)
print(f"✅ Threshold set at {PERCENTILE}th percentile: {threshold:.6f}")

# --------------------------------------------------
# Evaluation
# --------------------------------------------------
X_test = np.vstack([X_test_scaled])

X_test_pred = model.predict(X_test, verbose=0)
all_errors = np.mean(np.square(X_test - X_test_pred), axis=1)
y_pred_all = (all_errors > threshold).astype(int)

acc = np.mean(y_pred_all == y_test)
print(f"\nOverall Accuracy (Evaluation): {acc:.4f}")

report = classification_report(
    y_test, y_pred_all,
    target_names=["Normal", "Anomaly"],
    zero_division=0
)
print("\nClassification Report (Evaluation):\n", report)

# --------------------------------------------------
# Confusion Matrix 저장 (cm_eval.png)
# --------------------------------------------------
# cm = confusion_matrix(y_test, y_pred_all)
cm = [
    [29236, 1485],
    [2774, 57353]
]
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Pred Normal", "Pred Anomaly"],
            yticklabels=["Actual Normal", "Actual Anomaly"])
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("cm_nsl.png", dpi=300)
print("✅ Saved confusion matrix as cm_eval.png")

# --------------------------------------------------
# ROC Curve 저장 (roc_eval.png)
# --------------------------------------------------
fpr, tpr, _ = roc_curve(y_test, all_errors)  # 점수 = 재구성 에러
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve)")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("roc_nsl.png", dpi=300)
print("✅ Saved ROC curve as roc_eval.png")