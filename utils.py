'''
    import required libraries
    declare extra functions
'''
import sys
import os
from io import StringIO

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Optional

from sklearn.preprocessing import MinMaxScaler, RobustScaler 
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, roc_curve, roc_auc_score
from sklearn.utils import shuffle

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.saving import register_keras_serializable

import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.common import Scalar

from arguments import get_args
args = get_args()

# For datasets
def get_datasets_nsl(random_seed=args.random_seed):
    np.random.seed(random_seed)
    random.seed(random_seed)

    df_normal = pd.read_csv("./NSL-KDD/KDD_normal.csv")
    df_anomaly = pd.read_csv("./NSL-KDD/KDD_anomaly.csv")

    df_normal = shuffle(df_normal, random_state=random_seed)

    scaler = MinMaxScaler()
    mid_idx = len(df_normal) // 2
    df_normal_train = df_normal[:mid_idx]
    df_normal_test = df_normal[mid_idx:]

    X_train_scaled = scaler.fit_transform(df_normal_train)
    df_test = pd.concat([df_normal_test, df_anomaly], ignore_index=True)
    y_test = np.concatenate([np.zeros(len(df_normal_test)), np.ones(len(df_anomaly))])
    X_test_scaled, y_test = shuffle(scaler.transform(df_test), y_test, random_state=random_seed)

    return X_train_scaled, X_test_scaled, y_test


def get_datasets_cic(random_seed=args.random_seed):
    np.random.seed(random_seed)
    random.seed(random_seed)

    # ---------------------------
    # (0) [변경] 데이터 개수 설정
    # ---------------------------
    N_NORMAL_TRAIN = 2748235
    N_NORMAL_TEST = 2748235
    N_ANOMALY_TEST = 2748235

    # ---------------------------
    # (1) 정상 데이터 로드 & 정리
    # ---------------------------
    df_normal = pd.read_csv("./CIC2018/ae_datas/CIC_ae_normal.csv")
    df_normal = shuffle(df_normal, random_state=random_seed)
    df_normal = df_normal.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
    df_normal[df_normal < 0] = 0

    # ---------------------------
    # (2) 공격 데이터 로드 & 정리
    # ---------------------------
    anomaly_files = [f"./CIC2018/ae_datas/CIC_anomaly_ae_{i}.csv" for i in range(1, 15)]
    df_anomaly_list = [pd.read_csv(f) for f in anomaly_files]
    df_anomaly = pd.concat(df_anomaly_list, ignore_index=True)
    df_anomaly = shuffle(df_anomaly, random_state=random_seed)
    df_anomaly = df_anomaly.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
    df_anomaly[df_anomaly < 0] = 0

    # ---------------------------
    # (3) [변경] 요청사항에 맞게 데이터 분할
    # ---------------------------
    
    # 정상 데이터 분할
    total_normal_needed = N_NORMAL_TRAIN + N_NORMAL_TEST
    if len(df_normal) < total_normal_needed:
        raise ValueError(
            f"정상 데이터 개수({len(df_normal)})가 "
            f"필요한 총 개수({total_normal_needed})보다 부족합니다."
        )
    
    df_normal_train = df_normal.iloc[:N_NORMAL_TRAIN]
    df_normal_test = df_normal.iloc[N_NORMAL_TRAIN:total_normal_needed]

    # 이상 데이터 분할
    if len(df_anomaly) < N_ANOMALY_TEST:
        raise ValueError(
            f"이상 데이터 개수({len(df_anomaly)})가 "
            f"테스트에 필요한 개수({N_ANOMALY_TEST})보다 부족합니다."
        )

    df_anomaly_test = df_anomaly.iloc[:N_ANOMALY_TEST]
    
    print(f"정상 훈련 데이터: {len(df_normal_train)}개")
    print(f"정상 테스트 데이터: {len(df_normal_test)}개")
    print(f"이상 테스트 데이터: {len(df_anomaly_test)}개")

    # ---------------------------
    # (4) MinMax 정규화
    # ---------------------------
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(df_normal_train.values)

    # ---------------------------
    # (5) 레이블 생성 & 셔플
    # ---------------------------
    df_test = pd.concat([df_normal_test, df_anomaly_test], ignore_index=True)
    X_test = scaler.transform(df_test.values)

    y_test = np.concatenate([np.zeros(len(df_normal_test)), np.ones(len(df_anomaly_test))])
    
    X_test, y_test = shuffle(X_test, y_test, random_state=random_seed)

    return X_train, X_test, y_test


# def get_datasets_dapt(random_seed=args.random_seed):
#     np.random.seed(random_seed)
#     random.seed(random_seed)

#     df_normal = pd.read_csv("./5/dapt_normal.csv")
#     df_anomaly = pd.read_csv("./5/dapt_anomal.csv")

#     df_normal = shuffle(df_normal, random_state=random_seed)

#     scaler = MinMaxScaler()
#     mid_idx = len(df_normal) // 2
#     df_normal_train = df_normal[:mid_idx]
#     df_normal_test = df_normal[mid_idx:]

#     X_train_scaled = scaler.fit_transform(df_normal_train)
#     df_test = pd.concat([df_normal_test, df_anomaly], ignore_index=True)
#     y_test = np.concatenate([np.zeros(len(df_normal_test)), np.ones(len(df_anomaly))])
#     X_test_scaled, y_test = shuffle(scaler.transform(df_test), y_test, random_state=random_seed)

#     return X_train_scaled, X_test_scaled, y_test

# For evaluation
def plt_confusion_matrix(y_test, y_pred, 
                    save_path, labels=['Normal', 'Anomaly'], 
                    title="Confusion Matrix"):
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    cm_table = pd.DataFrame(
        [[tn, fp], [fn, tp]],
        index=['Actual Normal', 'Actual Anomaly'],
        columns=['Predicted Normal', 'Predicted Anomaly']
    )
    print("\n🧾 [Confusion Matrix]")
    print(cm_table)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=[f'Predicted {l}' for l in labels],
                yticklabels=[f'Actual {l}' for l in labels])
    plt.title(title)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_roc_curve(y_true, scores, roc_path="./roc_curve.png", title="ROC Curve"):
    """
    y_true: 실제 라벨 (0=Normal, 1=Anomaly)
    scores: 모델의 연속적인 점수 (ex. reconstruction error)
    roc_path: ROC curve 저장 경로
    """
    fpr, tpr, _ = roc_curve(y_true, scores)
    auc_score = roc_auc_score(y_true, scores)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(roc_path, dpi=300)
    plt.close()

    print(f"ROC curve saved to {roc_path}")
    return auc_score

def eval_server(model, X_train_scaled, X_test_scaled, y_test, result_path, matrix_path, roc_path):
    X_test_pred = model.predict(X_test_scaled, verbose=0)
    recon_errors = np.mean(np.square(X_test_scaled - X_test_pred), axis=1)

    X_train_pred = model.predict(X_train_scaled, verbose=0)
    train_errors = np.mean(np.square(X_train_scaled - X_train_pred), axis=1)

    threshold = np.percentile(train_errors, args.percentile)
    y_pred = (recon_errors > threshold).astype(int)
    print(f"[Threshold: {threshold}%]")

    server_report = classification_report(
        y_test, y_pred,
        target_names=["Normal", "Anomaly"],
        zero_division=0
    )
    
    print("\n📊 [Server Classification Report]\n")
    print(server_report)

    with open(result_path, "a") as f:
        f.write("\n📊 [Server Classification Report]\n")
        f.write(server_report + "\n")

    plt_confusion_matrix(y_test, y_pred, matrix_path)

    auc_score = plot_roc_curve(
                    y_test, recon_errors, 
                    roc_path=roc_path, 
                    title="Server ROC Curve"
                )
    
    return server_report
''' 
# CML 저장
def save_and_plot_history(history, csv_path, png_path):
    """
    Saves and plots the training and validation loss history.
    """
    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv(csv_path, index=False)
    print(f"\nTraining history saved to {csv_path}")
    
    plt.figure(figsize=(10, 6))
    plt.plot(hist_df['loss'], label='Training Loss')
    plt.plot(hist_df['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.savefig(png_path)
    plt.close()
    print(f"Loss plot saved to {png_path}")
'''

# FL 저장
def save_and_plot_history(history, csv_path, png_path):
    """
    Save and plot Flower simulation history (centralized loss).
    """
    # 중앙 손실 기록 가져오기
    rounds, losses = zip(*history.losses_centralized)

    # CSV 저장
    hist_df = pd.DataFrame({"round": rounds, "loss": losses})
    hist_df.to_csv(csv_path, index=False)
    print(f"\nTraining history saved to {csv_path}")

    # 그래프 저장
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, losses, marker="o", label="Centralized Loss")
    plt.title("Centralized Loss Over Rounds")
    plt.xlabel("Federated Rounds")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(png_path)
    plt.close()
    print(f"Loss plot saved to {png_path}")
