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
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.saving import register_keras_serializable

import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.common import Scalar, parameters_to_ndarrays, ndarrays_to_parameters

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

def get_datasets_kdd99(random_seed=args.random_seed):
    np.random.seed(random_seed)
    random.seed(random_seed)

    df_normal = pd.read_csv("./KDD99/KDD99_normal.csv")
    df_anomaly = pd.read_csv("./KDD99/KDD99_anomaly.csv")

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

# CIC
def get_datasets_cic(random_seed=args.random_seed):
    np.random.seed(random_seed)
    random.seed(random_seed)

    # ---------------------------
    # (1) 정상 데이터 로드 & 정리
    # ---------------------------
    normal_path = "./CIC2018/ae_datas_all_features/CIC_ae_normal.csv"
    df_normal = pd.read_csv(normal_path, low_memory=False)

    # 공통 피처만 사용 (파일에 존재하는 컬럼만 선택)

    # 클린업
    df_normal = df_normal.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
    df_normal[df_normal < 0] = 0
    df_normal = shuffle(df_normal, random_state=random_seed)

    # ---------------------------
    # (2) 이상 데이터 로드 & 정리
    # ---------------------------
    anomaly_files = [
        f"./CIC2018/ae_datas_all_features/CIC_anomaly_ae_{i}.csv" for i in range(1, 15)
    ]
    anomaly_dfs = []
    for path in anomaly_files:
        if os.path.exists(path):
            df_temp = pd.read_csv(path, low_memory=False)
            # use_cols = [c for c in existing_cols if c in df_temp.columns]
            # df_temp = df_temp[use_cols].copy()
            anomaly_dfs.append(df_temp)
        else:
            print(f"⚠️ Warning: {path} not found, skipping.")

    df_anomaly = pd.concat(anomaly_dfs, ignore_index=True)
    df_anomaly = shuffle(df_anomaly, random_state=random_seed)
    df_anomaly = df_anomaly.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
    df_anomaly[df_anomaly < 0] = 0

    # # ---------------------------
    # # (3) 정상 데이터 절반 분할
    # # ---------------------------
    # mid_idx = len(df_normal) // 2
    # df_normal_train = df_normal.iloc[:mid_idx]
    # df_normal_test = df_normal.iloc[mid_idx:]

    # print(f"정상 데이터 총 {len(df_normal)}개 → Train {len(df_normal_train)}, Test {len(df_normal_test)}")
    # print(f"이상 데이터 총 {len(df_anomaly)}개 (모두 테스트에 사용)")

    # # ---------------------------
    # # (4) MinMax 정규화
    # # ---------------------------
    # scaler = MinMaxScaler()
    # X_train = scaler.fit_transform(df_normal_train.values)

    # df_test = pd.concat([df_normal_test, df_anomaly], ignore_index=True)
    # X_test = scaler.transform(df_test.values)
    # y_test = np.concatenate([
    #     np.zeros(len(df_normal_test)),
    #     np.ones(len(df_anomaly))
    # ])

    # # ---------------------------
    # # (5) 셔플 & 리턴
    # # ---------------------------
    # X_test, y_test = shuffle(X_test, y_test, random_state=random_seed)

    # print(f"최종 Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    # print(f"y_test: Normal={np.sum(y_test==0)}, Anomaly={np.sum(y_test==1)}")

    # return X_train, X_test, y_test
    # ---------------------------
    # (3) 개수 제한 (각 150,000개)
    # ---------------------------
    n_samples = 150_000
    df_normal = df_normal.sample(n=min(len(df_normal), n_samples * 2), random_state=random_seed)
    df_anomaly = df_anomaly.sample(n=min(len(df_anomaly), n_samples), random_state=random_seed)

    mid_idx = len(df_normal) // 2
    df_normal_train = df_normal.iloc[:n_samples].copy() if len(df_normal) >= n_samples else df_normal.iloc[:mid_idx]
    df_normal_test = df_normal.iloc[-n_samples:].copy() if len(df_normal) >= n_samples * 2 else df_normal.iloc[mid_idx:]

    print(f"정상 데이터 총 {len(df_normal)}개 → Train {len(df_normal_train)}, Test {len(df_normal_test)}")
    print(f"이상 데이터 총 {len(df_anomaly)}개 (모두 테스트에 사용)")

    # ---------------------------
    # (4) MinMax 정규화
    # ---------------------------
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(df_normal_train.values)

    df_test = pd.concat([df_normal_test, df_anomaly], ignore_index=True)
    X_test = scaler.transform(df_test.values)
    y_test = np.concatenate([
        np.zeros(len(df_normal_test)),
        np.ones(len(df_anomaly))
    ])

    # ---------------------------
    # (5) 셔플 & 리턴
    # ---------------------------
    X_test, y_test = shuffle(X_test, y_test, random_state=random_seed)

    print(f"최종 Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"y_test: Normal={np.sum(y_test==0)}, Anomaly={np.sum(y_test==1)}")

    return X_train, X_test, y_test

def get_datasets_cic_val(random_seed=args.random_seed):
    np.random.seed(random_seed)
    random.seed(random_seed)

    # (피처 선택 부분은 주석 처리됨 - 원본 유지)
    
    # 1. 정상 데이터 로드
    normal_path = "./CIC2018/ae_datas_all_features/CIC_ae_normal.csv"
    df_normal = pd.read_csv(normal_path, low_memory=False)
    df_normal = df_normal.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
    df_normal[df_normal < 0] = 0
    df_normal = shuffle(df_normal, random_state=random_seed)

    # 2. 이상 데이터 로드
    anomaly_files = [
        f"./CIC2018/ae_datas_all_features/CIC_anomaly_ae_{i}.csv" for i in range(1, 15)
    ]
    anomaly_dfs = []
    for path in anomaly_files:
        if os.path.exists(path):
            df_temp = pd.read_csv(path, low_memory=False)
            anomaly_dfs.append(df_temp)
        else:
            print(f"⚠️ Warning: {path} not found, skipping.")
    
    df_anomaly = pd.concat(anomaly_dfs, ignore_index=True)
    df_anomaly = shuffle(df_anomaly, random_state=random_seed)
    df_anomaly = df_anomaly.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
    df_anomaly[df_anomaly < 0] = 0

    # 3. 개수 제한 (각 150,000개) 및 분할
    n_samples = 1_000_000
    df_normal = df_normal.sample(n=min(len(df_normal), n_samples * 2), random_state=random_seed)
    df_anomaly = df_anomaly.sample(n=min(len(df_anomaly), n_samples), random_state=random_seed)

    mid_idx = len(df_normal) // 2
    df_normal_train = df_normal.iloc[:n_samples].copy() if len(df_normal) >= n_samples else df_normal.iloc[:mid_idx]
    df_normal_test = df_normal.iloc[-n_samples:].copy() if len(df_normal) >= n_samples * 2 else df_normal.iloc[mid_idx:]

    print(f"정상 데이터 총 {len(df_normal)}개 → Train {len(df_normal_train)}, Test Pool {len(df_normal_test)}")
    print(f"이상 데이터 총 {len(df_anomaly)}개 (모두 Test Pool에 사용)")

    # 4. MinMax 정규화
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(df_normal_train.values) # 훈련셋 (정상 100%)

    # 테스트 풀 (정상 50% + 이상 100%)
    df_test_pool = pd.concat([df_normal_test, df_anomaly], ignore_index=True)
    X_test_pool = scaler.transform(df_test_pool.values)
    y_test_pool = np.concatenate([
        np.zeros(len(df_normal_test)),
        np.ones(len(df_anomaly))
    ])
    
    # 5. 셔플
    X_test_pool, y_test_pool = shuffle(X_test_pool, y_test_pool, random_state=random_seed)
    
    # 6. 🔽 [신규] 테스트 풀을 검증(Validation)셋과 최종 테스트(Test)셋으로 50:50 분할
    X_val, X_test, y_val, y_test = train_test_split(
        X_test_pool, 
        y_test_pool, 
        test_size=0.5, 
        random_state=random_seed,
        stratify=y_test_pool # 0/1 비율을 유지하며 분할
    )

    print(f"최종 Train shape: {X_train.shape}")
    print(f"최종 Val   shape: {X_val.shape}")
    print(f"최종 Test  shape: {X_test.shape}")
    print(f"y_val: Normal={np.sum(y_val==0)}, Anomaly={np.sum(y_val==1)}")
    print(f"y_test: Normal={np.sum(y_test==0)}, Anomaly={np.sum(y_test==1)}")

    # 7. 🔽 [수정] 5개 항목 반환
    return X_train, X_val, y_val, X_test, y_test

def get_datasets_cic_sam(random_seed=args.random_seed):
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    base_path = "./CIC2018/ae_datas_all_features/"

    # ---------------------------
    # (1) 정상 데이터 로드 & 샘플링 (Total: 200k -> Train: 160k, Test: 40k)
    # ---------------------------
    n_total_normal = 200_000
    n_train_normal = int(n_total_normal * 0.8) # 160,000
    n_test_normal = n_total_normal - n_train_normal # 40,000

    sampled_normal_path = os.path.join(base_path, "CIC_ae_normal_sam.csv")

    if os.path.exists(sampled_normal_path):
        # 1. 샘플링된 파일이 존재하면 바로 로드
        print(f"'{sampled_normal_path}' 파일 로드 중...")
        df_normal = pd.read_csv(sampled_normal_path, low_memory=False)
    
    else:
        # 2. 파일이 없으면 원본에서 새로 샘플링
        original_normal_path = os.path.join(base_path, "CIC_ae_normal.csv")
        print(f"'{sampled_normal_path}' 파일이 없습니다. '{original_normal_path}'에서 새로 샘플링합니다.")
        
        df_normal = pd.read_csv(original_normal_path, low_memory=False)

        # 클린업
        df_normal = df_normal.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
        df_normal[df_normal < 0] = 0
        
        # 20만 개 샘플링 (원본이 20만 개 미만이면 가능한 만큼만)
        df_normal = df_normal.sample(n=min(len(df_normal), n_total_normal), random_state=random_seed)
        
        # 다음 실행을 위해 저장
        df_normal.to_csv(sampled_normal_path, index=False)
        print(f"'{sampled_normal_path}' 에 샘플링된 정상 데이터 저장 완료.")

    # 훈련용(80%) / 테스트용(20%) 분할
    df_normal_train = df_normal.iloc[:n_train_normal].copy()
    df_normal_test = df_normal.iloc[n_train_normal:].copy()
    
    print(f"정상 데이터 총 {len(df_normal)}개 로드 → Train {len(df_normal_train)}개, Test {len(df_normal_test)}개")


    # ---------------------------
    # (2) 이상 데이터 로드 & 샘플링 (Test: 200k)
    # ---------------------------
    sampled_anomaly_path = os.path.join(base_path, "CIC_ae_anomaly_sam.csv")
    
    if os.path.exists(sampled_anomaly_path):
        # 1. 샘플링된 파일이 존재하면 바로 로드
        print(f"'{sampled_anomaly_path}' 파일 로드 중...")
        df_anomaly = pd.read_csv(sampled_anomaly_path, low_memory=False)
    
    else:
        # 2. 파일이 없으면 14개 파일에서 새로 샘플링
        print(f"'{sampled_anomaly_path}' 파일이 없습니다. 14개 파일에서 새로 샘플링합니다.")
        
        # 샘플링 계획 (총 200,000개)
        sampling_plan = {
            1: 36897,  # DDOS attack-HOIC
            2: 30990,  # DDoS attacks-LOIC-HTTP
            3: 24844,  # DoS attacks-Hulk
            4: 15392,  # Bot
            5: 10400,  # FTP-BruteForce
            6: 10088,  # SSH-Bruteforce
            7: 8709,   # Infilteration
            8: 7524,   # DoS attacks-SlowHTTPTest
            9: 41508,  # DoS attacks-GoldenEye (모두 사용)
            10: 10990, # DoS attacks-Slowloris (모두 사용)
            11: 1730,  # DDOS attack-LOIC-UDP (모두 사용)
            12: 611,   # Brute Force -Web (모두 사용)
            13: 230,   # Brute Force -XSS (모두 사용)
            14: 87     # SQL Injection (모두 사용)
        }
        
        anomaly_dfs = []
        for i in range(1, 15):
            path = os.path.join(base_path, f"CIC_anomaly_ae_{i}.csv")
            n_to_sample = sampling_plan[i]
            
            if os.path.exists(path):
                df_temp = pd.read_csv(path, low_memory=False)
                
                if len(df_temp) > n_to_sample:
                    df_temp = df_temp.sample(n=n_to_sample, random_state=random_seed)
                
                anomaly_dfs.append(df_temp)
            else:
                 print(f"⚠️ Warning: {path} not found, skipping.")

        print("이상 데이터 샘플링 완료. 병합 및 클린업 시작...")
        df_anomaly = pd.concat(anomaly_dfs, ignore_index=True)
        
        # 셔플 및 클린업
        df_anomaly = shuffle(df_anomaly, random_state=random_seed)
        df_anomaly = df_anomaly.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
        df_anomaly[df_anomaly < 0] = 0
        
        # 다음 실행을 위해 저장
        df_anomaly.to_csv(sampled_anomaly_path, index=False)
        print(f"'{sampled_anomaly_path}' 에 샘플링된 이상 데이터 저장 완료.")

    print(f"이상 데이터 총 {len(df_anomaly)}개 (모두 테스트에 사용)")

    # ---------------------------
    # (3) MinMax 정규화 (Autoencoder 학습 방식)
    # ---------------------------
    print("MinMaxScaler 정규화 중...")
    scaler = MinMaxScaler()
    
    # 훈련 데이터(정상)로 스케일러 학습
    X_train = scaler.fit_transform(df_normal_train.values)

    # 테스트 데이터(정상+이상) 구성 및 스케일러 적용
    df_test = pd.concat([df_normal_test, df_anomaly], ignore_index=True)
    X_test = scaler.transform(df_test.values)
    
    # 테스트 레이블 생성 (정상: 0, 이상: 1)
    y_test = np.concatenate([
        np.zeros(len(df_normal_test)),
        np.ones(len(df_anomaly))
    ])

    # ---------------------------
    # (4) 셔플 & 리턴
    # ---------------------------
    X_test, y_test = shuffle(X_test, y_test, random_state=random_seed)

    print(f"최종 Train shape: {X_train.shape} (정상 데이터만)")
    print(f"최종 Test shape: {X_test.shape}")
    print(f"y_test 구성: Normal={np.sum(y_test==0)}, Anomaly={np.sum(y_test==1)}")

    return X_train, X_test, y_test

# InSDN
def get_datasets_insdn(random_seed=args.random_seed):
    np.random.seed(random_seed)
    random.seed(random_seed)

    # ---------------------------
    # (1) Load dataset
    # ---------------------------
    normal_path = "./InSDN/raw_datas/InSDN_normal.csv"
    anomaly_path = "./InSDN/raw_datas/InSDN_anomaly.csv"

    df_normal = pd.read_csv(normal_path)
    df_anomaly = pd.read_csv(anomaly_path)

    print(f"✅ Loaded InSDN data → Normal: {df_normal.shape}, Anomaly: {df_anomaly.shape}")

    # ---------------------------
    # (2) Shuffle normal data
    # ---------------------------
    df_normal = shuffle(df_normal, random_state=random_seed)

    # ---------------------------
    # (3) Split train/test for normal
    # ---------------------------
    split_point = int(len(df_normal) * 0.8)
    df_normal_train = df_normal.iloc[:split_point]
    df_normal_test = df_normal.iloc[split_point:]

    # ---------------------------
    # (4) Preprocessing: numeric cleanup
    # ---------------------------
    df_normal_train = df_normal_train.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
    df_normal_test = df_normal_test.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
    df_anomaly = df_anomaly.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)

    def _clean(df):
        df = df.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
        # ✅ Drop last column
        return df.iloc[:, :-1]

    df_normal_train = _clean(df_normal_train)
    df_normal_test = _clean(df_normal_test)
    df_anomaly = _clean(df_anomaly)
    
    # ---------------------------
    # (5) Scaling (MinMax)
    # ---------------------------
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(df_normal_train.values)

    df_test = pd.concat([df_normal_test, df_anomaly], ignore_index=True)
    X_test_scaled = scaler.transform(df_test.values)

    # ---------------------------
    # (6) Labeling (0=Normal, 1=Anomaly)
    # ---------------------------
    y_test = np.concatenate([np.zeros(len(df_normal_test)), np.ones(len(df_anomaly))])

    # ---------------------------
    # (7) Shuffle test set
    # ---------------------------
    X_test_scaled, y_test = shuffle(X_test_scaled, y_test, random_state=random_seed)

    # ---------------------------
    # (8) Print summary
    # ---------------------------
    print(f"Train: {X_train_scaled.shape}, Test: {X_test_scaled.shape}, y_test: {y_test.shape}")
    print(f"Normal train: {len(df_normal_train)}, Normal test: {len(df_normal_test)}, Anomaly: {len(df_anomaly)}")

    return X_train_scaled, X_test_scaled, y_test

# UNSW_NB15
def get_datasets_unsw(random_seed=args.random_seed):
    np.random.seed(random_seed)
    random.seed(random_seed)

    # ---------------------------
    # (1) Load dataset
    # ---------------------------
    normal_path = "./UNSW_NB15/ae_datas/UNSW_NB15_normal.csv"
    anomaly_path = "./UNSW_NB15/ae_datas/UNSW_NB15_anomaly.csv"

    df_normal = pd.read_csv(normal_path)
    df_anomaly = pd.read_csv(anomaly_path)

    print(f"✅ Loaded UNSW_NB15 data → Normal: {df_normal.shape}, Anomaly: {df_anomaly.shape}")

    # ---------------------------
    # (2) Shuffle normal data
    # ---------------------------
    df_normal = shuffle(df_normal, random_state=random_seed)

    # ---------------------------
    # (3) Split train/test for normal
    # ---------------------------
    split_point = int(len(df_normal) * 0.8)
    df_normal_train = df_normal.iloc[:split_point]
    df_normal_test = df_normal.iloc[split_point:]

    # ---------------------------
    # (4) Preprocessing: numeric cleanup
    # ---------------------------
    df_normal_train = df_normal_train.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
    df_normal_test = df_normal_test.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
    df_anomaly = df_anomaly.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)

    def _clean(df):
        df = df.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
        # ✅ Drop last column
        return df.iloc[:, :-1]

    df_normal_train = _clean(df_normal_train)
    df_normal_test = _clean(df_normal_test)
    df_anomaly = _clean(df_anomaly)
    
    # ---------------------------
    # (5) Scaling (MinMax)
    # ---------------------------
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(df_normal_train.values)

    df_test = pd.concat([df_normal_test, df_anomaly], ignore_index=True)
    X_test_scaled = scaler.transform(df_test.values)

    # ---------------------------
    # (6) Labeling (0=Normal, 1=Anomaly)
    # ---------------------------
    y_test = np.concatenate([np.zeros(len(df_normal_test)), np.ones(len(df_anomaly))])

    # ---------------------------
    # (7) Shuffle test set
    # ---------------------------
    X_test_scaled, y_test = shuffle(X_test_scaled, y_test, random_state=random_seed)

    # ---------------------------
    # (8) Print summary
    # ---------------------------
    print(f"Train: {X_train_scaled.shape}, Test: {X_test_scaled.shape}, y_test: {y_test.shape}")
    print(f"Normal train: {len(df_normal_train)}, Normal test: {len(df_normal_test)}, Anomaly: {len(df_anomaly)}")

    return X_train_scaled, X_test_scaled, y_test

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
