from arguments import get_args
from utils import *
from model_aelstm import SaveEvaluationFedAvg, AE_LSTM, FLClient  # 비지도 AE-LSTM

# ----------------------------
# 경로 설정
# ----------------------------
os.makedirs("./ae_lstm", exist_ok=True)
WEIGHT_PATH = "./ae_lstm/ae_lstm_weights.h5"
RESULT_PATH = "./ae_lstm/ae_lstm_server.txt"
ROC_PATH = "./ae_lstm/ae_lstm_roc.png"

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ----------------------------
# 결과 파일 초기화
# ----------------------------
with open(RESULT_PATH, "w") as f:
    f.write("[Server Evaluation Report]\n")

def reshape_for_sequence(X, timesteps=10, features=2):
    n_samples, n_feats = X.shape
    if n_feats < timesteps * features:
        pad = np.zeros((n_samples, timesteps * features - n_feats))
        X = np.concatenate([X, pad], axis=1)
    X = X[:, :timesteps * features]
    return X.reshape(-1, timesteps, features)

# KDD99
def get_datasets_kdd99_semi(random_seed=86, anomaly_ratio=0.05):
    np.random.seed(random_seed)
    random.seed(random_seed)

    df_normal = pd.read_csv("./KDD99/KDD99_normal.csv")
    df_anomaly = pd.read_csv("./KDD99/KDD99_anomaly.csv")
    df_normal = shuffle(df_normal, random_state=random_seed)

    # split (50% train / 50% test)
    mid_idx = len(df_normal) // 2
    df_normal_train = df_normal[:mid_idx]
    df_normal_test = df_normal[mid_idx:]

    # scaling
    scaler = MinMaxScaler()
    X_normal_train = scaler.fit_transform(df_normal_train)
    X_normal_test = scaler.transform(df_normal_test)
    X_anomaly_all = scaler.transform(df_anomaly)

    # train anomaly ratio
    num_anomaly_for_train = int(len(X_anomaly_all) * anomaly_ratio)
    X_anomaly_train = X_anomaly_all[:num_anomaly_for_train]
    X_anomaly_test = X_anomaly_all[num_anomaly_for_train:]

    # train data & label
    X_train = np.concatenate([X_normal_train, X_anomaly_train], axis=0)
    y_train_cls = np.concatenate([
        np.zeros(len(X_normal_train)), np.ones(len(X_anomaly_train))
    ])

    # test data & label
    X_test = np.concatenate([X_normal_test, X_anomaly_test], axis=0)
    y_test = np.concatenate([
        np.zeros(len(X_normal_test)), np.ones(len(X_anomaly_test))
    ])

    X_train, y_train_cls = shuffle(X_train, y_train_cls, random_state=random_seed)
    X_test, y_test = shuffle(X_test, y_test, random_state=random_seed)

    return X_train, y_train_cls, X_test, y_test

# NSL-KDD
def get_datasets_nsl_semi(
    normal_csv="./NSL-KDD/KDD_normal.csv",
    anomaly_csv="./NSL-KDD/KDD_anomaly.csv",
    random_seed=42,
    anomaly_ratio=0.2,
    timesteps=10,
    features=12,
):
    """
    Semi-supervised NSL-KDD dataset for AE-LSTM
    ---------------------------------------------------------
    - 정상(normal): 절반 train / 절반 test
    - 이상(anomaly): 일부(anomaly_ratio)만 train에 포함
    - AE-LSTM 입력용 (reshape_for_sequence_nsl 적용)
    ---------------------------------------------------------
    Return:
        X_train_seq, y_train_cls, X_test_seq, y_test
    """
    np.random.seed(random_seed)
    random.seed(random_seed)

    # 1️⃣ Load data
    df_normal = pd.read_csv(normal_csv)
    df_anomaly = pd.read_csv(anomaly_csv)
    df_normal = shuffle(df_normal, random_state=random_seed)

    # 2️⃣ Split normal
    mid_idx = len(df_normal) // 2
    df_normal_train = df_normal.iloc[:mid_idx]
    df_normal_test = df_normal.iloc[mid_idx:]

    # 3️⃣ Scale
    scaler = MinMaxScaler()
    X_normal_train = scaler.fit_transform(df_normal_train)
    X_normal_test  = scaler.transform(df_normal_test)
    X_anomaly_all  = scaler.transform(df_anomaly)

    # 4️⃣ Split anomaly
    num_anomaly_to_add = int(len(X_anomaly_all) * anomaly_ratio)
    X_anomaly_train = X_anomaly_all[:num_anomaly_to_add]
    X_anomaly_test  = X_anomaly_all[num_anomaly_to_add:]

    # 5️⃣ Compose datasets
    X_train = np.concatenate([X_normal_train, X_anomaly_train], axis=0)
    y_train_cls = np.concatenate([
        np.zeros(len(X_normal_train)), np.ones(len(X_anomaly_train))
    ])
    X_test = np.concatenate([X_normal_test, X_anomaly_test], axis=0)
    y_test = np.concatenate([
        np.zeros(len(X_normal_test)), np.ones(len(X_anomaly_test))
    ])

    X_train, y_train_cls = shuffle(X_train, y_train_cls, random_state=random_seed)
    X_test,  y_test      = shuffle(X_test,  y_test,      random_state=random_seed)

    # 6️⃣ Reshape for AE-LSTM (3D input)
    X_train_seq = reshape_for_sequence_nsl(X_train, timesteps=timesteps, features=features)
    X_test_seq  = reshape_for_sequence_nsl(X_test, timesteps=timesteps, features=features)

    print(f"[NSL-KDD Semi] Train: {X_train_seq.shape}, Test: {X_test_seq.shape}")
    print(f"Anomaly ratio (train): {anomaly_ratio}")
    return X_train_seq, y_train_cls, X_test_seq, y_test

# CIC
import os
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle

def reshape_for_sequence(X, timesteps=10, features=2):
    """
    X: (n_samples, n_features_flat)
    -> (n_samples, timesteps, features)
    부족하면 0으로 패딩, 초과면 앞에서 절단
    """
    if X.ndim == 3:
        # 이미 LSTM 입력 형태
        return X
        
    n = timesteps * features
    n_feats = X.shape[1]
    if n_feats < n:
        pad = np.zeros((X.shape[0], n - n_feats), dtype=X.dtype)
        Xp = np.concatenate([X, pad], axis=1)
    else:
        Xp = X[:, :n]
    return Xp.reshape(-1, timesteps, features)

def get_datasets_cic_multi_semi(
    normal_csv="./CIC2018/ae_datas_all_features/CIC_ae_normal.csv",
    anomaly_pattern="./CIC2018/ae_datas_all_features/CIC_anomaly_ae_{}.csv",
    num_anomaly_files=14,
    random_seed=42,
    anomaly_ratio=0.2,
    timesteps=10,
    features=4,
):
    """
    Semi-supervised CIC-IDS2018 dataset for AE-LSTM
    - 정상: 50% train / 50% test
    - 이상: 여러 CSV 병합 후 anomaly_ratio 만큼만 train에 투입
    - 스케일러는 정상-train에만 fit, 나머지는 transform
    - 리셰이프는 (timesteps, features)로 패딩/절단 안전 처리
    Return:
        X_train_seq, y_train_cls, X_test_seq, y_test
    """
    np.random.seed(random_seed)
    random.seed(random_seed)

    # 1) InSDN 유사 피처(필요 시 조정)
    SELECTED_FEATURES = [
        "Flow Duration", "Tot Fwd Pkts", "Tot Bwd Pkts",
        "TotLen Fwd Pkts", "TotLen Bwd Pkts",
        "Fwd Pkt Len Max", "Fwd Pkt Len Mean",
        "Bwd Pkt Len Max", "Bwd Pkt Len Mean",
        "Flow Byts/s", "Flow Pkts/s",
        "Flow IAT Mean", "Flow IAT Std",
        "Fwd IAT Mean", "Fwd IAT Std",
        "Bwd IAT Mean", "Bwd IAT Std",
        "Pkt Len Min", "Pkt Len Max", "Pkt Len Mean", "Pkt Len Std", "Pkt Len Var",
        "FIN Flag Cnt", "SYN Flag Cnt", "RST Flag Cnt", "PSH Flag Cnt", "ACK Flag Cnt", "URG Flag Cnt",
        "Fwd Header Len", "Bwd Header Len",
        "Down/Up Ratio", "Pkt Size Avg",
        "Active Mean", "Active Std", "Idle Mean", "Idle Std",
    ]

    # 2) 정상 로드 + 안전 컬럼 선택
    df_normal_raw = pd.read_csv(normal_csv, low_memory=False)
    cols_exist = [c for c in SELECTED_FEATURES if c in df_normal_raw.columns]
    if len(cols_exist) == 0:
        raise ValueError("선택한 피처가 정상 CSV에 없습니다. 컬럼명을 확인하세요.")
    df_normal = df_normal_raw[cols_exist].copy()

    # 3) 이상 여러 파일 병합 + 동일 컬럼만 사용
    anomaly_dfs = []
    for i in range(1, num_anomaly_files + 1):
        path = anomaly_pattern.format(i)
        if os.path.exists(path):
            df_tmp = pd.read_csv(path, low_memory=False)
            use_cols = [c for c in cols_exist if c in df_tmp.columns]
            # 정상과 공통 컬럼만 사용
            df_tmp = df_tmp[use_cols].copy()
            anomaly_dfs.append(df_tmp)
        else:
            print(f"⚠️ Warning: {path} not found, skipping.")
    if not anomaly_dfs:
        raise ValueError("병합할 anomaly 파일이 없습니다.")
    df_anomaly = pd.concat(anomaly_dfs, axis=0, ignore_index=True)

    # 4) NaN/inf 처리
    for df in (df_normal, df_anomaly):
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)

    # 5) 정상 50/50 split
    df_normal = shuffle(df_normal, random_state=random_seed)
    mid_idx = len(df_normal) // 2
    df_normal_train = df_normal.iloc[:mid_idx]
    df_normal_test  = df_normal.iloc[mid_idx:]

    # 6) 스케일링 (정상-train에만 fit)
    scaler = MinMaxScaler()
    X_normal_train = scaler.fit_transform(df_normal_train.values)
    X_normal_test  = scaler.transform(df_normal_test.values)
    X_anomaly_all  = scaler.transform(df_anomaly.values)

    # 7) anomaly 일부만 학습 포함
    n_anom_train = int(len(X_anomaly_all) * anomaly_ratio)
    X_anomaly_train = X_anomaly_all[:n_anom_train]
    X_anomaly_test  = X_anomaly_all[n_anom_train:]

    # 8) 합치기 + 라벨
    X_train = np.concatenate([X_normal_train, X_anomaly_train], axis=0)
    y_train_cls = np.concatenate([
        np.zeros(len(X_normal_train), dtype=int),
        np.ones(len(X_anomaly_train), dtype=int),
    ])
    X_test = np.concatenate([X_normal_test, X_anomaly_test], axis=0)
    y_test = np.concatenate([
        np.zeros(len(X_normal_test), dtype=int),
        np.ones(len(X_anomaly_test), dtype=int),
    ])

    # 9) 셔플
    X_train, y_train_cls = shuffle(X_train, y_train_cls, random_state=random_seed)
    X_test,  y_test      = shuffle(X_test,  y_test,      random_state=random_seed)

    # 10) AE-LSTM 입력 리셰이프 (패딩/절단 안전)
    X_train_seq = reshape_for_sequence(X_train, timesteps=timesteps, features=features)
    X_test_seq  = reshape_for_sequence(X_test,  timesteps=timesteps, features=features)

    # 11) 로그
    print(f"✅ [CIC-IDS2018 Semi] Normal={len(df_normal)}, Anomaly={len(df_anomaly)}")
    print(f"Train: {X_train_seq.shape}, Test: {X_test_seq.shape} "
          f"(flatten={X_train.shape[1]} → {timesteps}x{features})")
    print(f"Anomaly ratio (train): {anomaly_ratio}")

    return X_train_seq, y_train_cls, X_test_seq, y_test

# InSDN
def get_datasets_insdn_semi(
    normal_csv="./InSDN/ae_datas/InSDN_normal_48.csv",
    anomaly_csv="./InSDN/ae_datas/InSDN_anomaly_48.csv",
    random_seed=86,
    anomaly_ratio=0.05,
):
    """
    Semi-supervised InSDN dataset for AE-LSTM
    ---------------------------------------------------------
    - 정상(normal) 데이터: 50% train, 50% test
    - 이상(anomaly) 데이터: 일부(anomaly_ratio)만 train에 혼합
    - AE-LSTM 입력용으로 정규화된 48 feature 벡터 반환
    ---------------------------------------------------------
    Return:
        X_train (normal + small anomaly)
        y_train_cls (0/1 for classifier head)
        X_test (normal + anomaly)
        y_test  (0/1 for evaluation)
    """
    np.random.seed(random_seed)
    random.seed(random_seed)

    # ---------------------------
    # (1) Load
    # ---------------------------
    df_normal = pd.read_csv(normal_csv, low_memory=False)
    df_anomaly = pd.read_csv(anomaly_csv, low_memory=False)

    # ---------------------------
    # (2) Clean
    # ---------------------------
    def _clean(df):
        return (
            df.apply(pd.to_numeric, errors="coerce")
              .replace([np.inf, -np.inf], np.nan)
              .fillna(0)
        )

    df_normal = _clean(df_normal)
    df_anomaly = _clean(df_anomaly)

    # ---------------------------
    # (3) Split normal 50/50
    # ---------------------------
    df_normal = shuffle(df_normal, random_state=random_seed)
    mid_idx = len(df_normal) // 2
    df_normal_train = df_normal.iloc[:mid_idx]
    df_normal_test = df_normal.iloc[mid_idx:]

    # ---------------------------
    # (4) Scaling
    # ---------------------------
    scaler = MinMaxScaler()
    X_normal_train = scaler.fit_transform(df_normal_train.values)
    X_normal_test  = scaler.transform(df_normal_test.values)
    X_anomaly_all  = scaler.transform(df_anomaly.values)

    # ---------------------------
    # (5) Split anomaly data
    # ---------------------------
    n_anom_train = int(len(X_anomaly_all) * anomaly_ratio)
    X_anomaly_train = X_anomaly_all[:n_anom_train]
    X_anomaly_test  = X_anomaly_all[n_anom_train:]

    # ---------------------------
    # (6) Compose datasets
    # ---------------------------
    X_train = np.concatenate([X_normal_train, X_anomaly_train], axis=0)
    y_train_cls = np.concatenate([
        np.zeros(len(X_normal_train)),
        np.ones(len(X_anomaly_train))
    ])

    X_test = np.concatenate([X_normal_test, X_anomaly_test], axis=0)
    y_test = np.concatenate([
        np.zeros(len(X_normal_test)),
        np.ones(len(X_anomaly_test))
    ])

    X_train, y_train_cls = shuffle(X_train, y_train_cls, random_state=random_seed)
    X_test,  y_test      = shuffle(X_test,  y_test,      random_state=random_seed)

    # ---------------------------
    # (7) Log info
    # ---------------------------
    print(f"[InSDN Semi] Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"y_train_cls: {y_train_cls.shape}, y_test: {y_test.shape}")
    print(f"Anomaly ratio (train): {anomaly_ratio}")

    return X_train, y_train_cls, X_test, y_test


# ----------------------------
# 메인 함수
# ----------------------------
def main():
    args = get_args()

    # 데이터 불러오기
    # X_train, y_train_cls, X_test, y_test = get_datasets_kdd99_semi()
    # X_train = reshape_for_sequence(X_train, timesteps=10, features=13) # KDD
    # X_test = reshape_for_sequence(X_test, timesteps=10, features=13)

    # X_train, y_train_cls, X_test, y_test = get_datasets_nslkdd_semi() 
    # X_train = reshape_for_sequence(X_train, timesteps=10, features=12) # NSL-KDD
    # X_test = reshape_for_sequence(X_test, timesteps=10, features=12)

    # X_train, y_train_cls, X_test, y_test = get_datasets_insdn_semi()
    # X_train = reshape_for_sequence(X_train, timesteps=12, features=4) # InSDN
    # X_test = reshape_for_sequence(X_test, timesteps=12, features=4)

    X_train, y_train_cls, X_test, y_test = get_datasets_cic_multi_semi(
        normal_csv="./CIC2018/ae_datas_all_features/CIC_ae_normal.csv",
        anomaly_pattern="./CIC2018/ae_datas_all_features/CIC_anomaly_ae_{}.csv",
        num_anomaly_files=14,
        anomaly_ratio=0.1,
        timesteps=10,
        features=4
    )
    X_train = reshape_for_sequence(X_train, timesteps=10, features=4) # CIC2018
    X_test = reshape_for_sequence(X_test, timesteps=10, features=4)
    
    
    # 클라이언트 분할 시 classifier 라벨도 같이 나눔
    client_data = np.array_split(X_train, args.client_nums)
    label_data = np.array_split(y_train_cls, args.client_nums)

    # ----------------------------
    # 서버 모델 정의 (비지도 AE-LSTM)
    # ----------------------------
    # model = AE_LSTM(timesteps=10, features=12) # NSL
    # _ = model(tf.zeros((1, 10, 12)))
    # model = AE_LSTM(input_dim=X_train.shape[-1], timesteps=10, features=13) # KDD
    # _ = model(tf.zeros((1, 10, 13))) # KDD99
    model = AE_LSTM(input_dim=X_train.shape[-1], timesteps=10, features=4) # CIC
    _ = model(tf.zeros((1, 10, 4)))
    # model = AE_LSTM(input_dim=X_train.shape[-1], timesteps=12, features=4) # KDD
    # _ = model(tf.zeros((1, 12, 4))) # InSDN
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss={"decoded": "mse", "pred": "binary_crossentropy"},
        loss_weights={"decoded": 1.0, "pred": 0.1},
        metrics={"pred": ["accuracy"]}
    )

    model.summary()

    # ----------------------------
    # 클라이언트 데이터 분할 (정상만)
    # ----------------------------
    client_data = np.array_split(X_train, args.client_nums)

    # ----------------------------
    # 서버 평가 콜백
    # ----------------------------
    def server_evaluate(server_round: int, parameters, config):
        weights = parameters_to_ndarrays(parameters)
        model.set_weights(weights)

        preds = model.predict(X_test, verbose=0)
        recon = preds["decoded"]
        cls_prob = preds["pred"].flatten()

        # ---------- Reconstruction 기반 ----------
        mse = np.mean(np.square(X_test - recon), axis=(1, 2))
        threshold = np.percentile(mse, 80)
        y_pred_recon = (mse > threshold).astype(int)

        # ---------- Classifier 기반 ----------
        y_pred_cls = (cls_prob > 0.5).astype(int)

        # ---------- Accuracy ----------
        acc_recon = np.mean(y_pred_recon == y_test)
        acc_cls = np.mean(y_pred_cls == y_test)

        print(f"\n[Server Round {server_round}] Recon-Acc={acc_recon:.4f}, Cls-Acc={acc_cls:.4f}, Th={threshold:.6f}")

        # ---------- Confusion Matrices ----------
        cm_recon = confusion_matrix(y_test, y_pred_recon)
        cm_cls = confusion_matrix(y_test, y_pred_cls)

        print("\n📊 [f-based Confusion Matrix]")
        print(cm_recon)
        print("\n📊 [Classifier-based Confusion Matrix]")
        print(cm_cls)

        # ---------- Reports ----------
        report_recon = classification_report(
            y_test, y_pred_recon,
            target_names=["Normal", "Anomaly"],
            zero_division=0
        )

        report_cls = classification_report(
            y_test, y_pred_cls,
            target_names=["Normal", "Anomaly"],
            zero_division=0
        )

        print("\n📊 [Reconstruction-based Report]")
        print(report_recon)
        print("📊 [Classifier-based Report]")
        print(report_cls)

        # ---------- 로그 파일 저장 ----------
        with open(RESULT_PATH, "a") as f:
            f.write(f"\n[Round {server_round}] Server Evaluation\n")
            f.write(f"Threshold={threshold:.6f}, Recon-Acc={acc_recon:.4f}, Cls-Acc={acc_cls:.4f}\n")

            f.write("[Reconstruction-based Confusion Matrix]\n")
            f.write(np.array2string(cm_recon) + "\n")
            f.write("[Classifier-based Confusion Matrix]\n")
            f.write(np.array2string(cm_cls) + "\n")

            f.write("[Reconstruction-based Report]\n" + report_recon + "\n")
            f.write("[Classifier-based Report]\n" + report_cls + "\n")

        return float(1 - acc_recon), {"acc_cls": acc_cls}



    # ----------------------------
    # 커스텀 FedAvg 전략
    # ----------------------------
    strategy = SaveEvaluationFedAvg(
        eval_server_args={
            "model": model,
            "X_test_scaled": X_test,
            "y_test": y_test,
            "result_path": RESULT_PATH,
        },
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=args.client_nums,
        min_evaluate_clients=args.client_nums,
        min_available_clients=args.client_nums,
        evaluate_fn=server_evaluate,
    )


    # ----------------------------
    # 클라이언트 함수
    # ----------------------------
    def client_fn(cid: str):
        cid_int = int(cid)
        # client_model = AE_LSTM(input_dim=X_train.shape[-1], timesteps=12, features=4)
        # _ = client_model(tf.zeros((1, 10, 13))) # KDD99
        # _ = model(tf.zeros((1, 12, 4))) # InSDN
        client_model = AE_LSTM(input_dim=X_train.shape[-1], timesteps=10, features=4)
        _ = model(tf.zeros((1, 10, 4))) # CIC

        return FLClient(
            cid=cid_int,
            model=client_model,
            X_train=client_data[cid_int],
            y_train_cls=label_data[cid_int],
            X_test=X_test,
            y_test=y_test
        )


    # ----------------------------
    # 연합 학습 시작
    # ----------------------------
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=args.client_nums,
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=args.server_rounds),
        client_resources={"num_cpus": 1},
        ray_init_args={"include_dashboard": False, "ignore_reinit_error": True},
    )

    # ----------------------------
    # 최종 평가
    # ----------------------------
    recon = model.predict(X_test, verbose=0)
    mse = np.mean(np.square(X_test - recon), axis=(1, 2))
    threshold = np.percentile(mse, 95)
    y_pred = (mse > threshold).astype(int)
    acc = np.mean(y_pred == y_test)

    # ----------------------------
    # 최종 모델 저장
    # ----------------------------
    if hasattr(strategy, "final_parameters") and strategy.final_parameters is not None:
        final_weights = parameters_to_ndarrays(strategy.final_parameters)
        model.set_weights(final_weights)

    save_and_plot_history(
        history,
        csv_path="./ae_lstm/ae_lstm_history",
        png_path="./ae_lstm/ae_lstm_history.png",
    )

    print(f"\n[Final] Accuracy={acc:.4f}, Threshold={threshold:.6f}")
    model.save_weights(WEIGHT_PATH)
    print(f"✅ Model weights saved to {WEIGHT_PATH}")


if __name__ == "__main__":
    main()
