from arguments import get_args
from utils import *
from model_cnnlstm import SaveEvaluationFedAvg, CNN_LSTM, FLClient  # 커스텀 FedAvg, CNN-LSTM, Client

# ----------------------------
# 경로 설정
# ----------------------------
os.makedirs("./cnn_lstm", exist_ok=True)
WEIGHT_PATH = "./cnn_lstm/cnn_lstm_weights.h5"
MATRIX_PATH = "./cnn_lstm/cnn_lstm_cm.png"
RESULT_PATH = "./cnn_lstm/cnn_lstm_server.txt"
ROC_PATH = "./cnn_lstm/cnn_lstm_roc.png"

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


# ----------------------------
# CNN 입력용 reshape 함수
# ----------------------------
def reshape_for_sequence_nsl(X, timesteps=10, features=12):
    n_samples, n_feats = X.shape
    if n_feats < timesteps * features:
        pad = np.zeros((n_samples, timesteps * features - n_feats))
        X = np.concatenate([X, pad], axis=1)
    X = X[:, :timesteps * features]
    return X.reshape(-1, timesteps, features)


# ----------------------------
# NSL-KDD 지도학습 데이터셋 로더
# ----------------------------
def get_datasets_nsl_supervised(random_seed=42, anomaly_ratio=0.2, timesteps=10, features=12):
    np.random.seed(random_seed)
    random.seed(random_seed)

    df_normal = pd.read_csv("./NSL-KDD/KDD_normal.csv")
    df_anomaly = pd.read_csv("./NSL-KDD/KDD_anomaly.csv")
    df_normal = shuffle(df_normal, random_state=random_seed)

    scaler = MinMaxScaler()
    mid_idx = len(df_normal) // 2
    df_normal_train = df_normal.iloc[:mid_idx]
    df_normal_test = df_normal.iloc[mid_idx:]

    X_normal_train = scaler.fit_transform(df_normal_train)
    X_normal_test = scaler.transform(df_normal_test)

    X_anomaly_all = scaler.transform(df_anomaly)
    num_anomaly_to_add = int(len(X_anomaly_all) * anomaly_ratio)
    X_anomaly_train = X_anomaly_all[:num_anomaly_to_add]
    X_anomaly_test = X_anomaly_all[num_anomaly_to_add:]

    X_train_supervised = np.concatenate([X_normal_train, X_anomaly_train], axis=0)
    y_train_supervised = np.concatenate(
        [np.zeros(len(X_normal_train)), np.ones(len(X_anomaly_train))], axis=0
    )

    X_test_supervised = np.concatenate([X_normal_test, X_anomaly_test], axis=0)
    y_test_supervised = np.concatenate(
        [np.zeros(len(X_normal_test)), np.ones(len(X_anomaly_test))], axis=0
    )

    X_train_supervised, y_train_supervised = shuffle(
        X_train_supervised, y_train_supervised, random_state=random_seed
    )
    X_test_supervised, y_test_supervised = shuffle(
        X_test_supervised, y_test_supervised, random_state=random_seed
    )

    X_train_seq = reshape_for_sequence_nsl(X_train_supervised, timesteps=timesteps, features=features)
    X_test_seq = reshape_for_sequence_nsl(X_test_supervised, timesteps=timesteps, features=features)

    return X_train_seq, y_train_supervised, X_test_seq, y_test_supervised

# CIC 2018 데이터셋 전처리
def get_datasets_cic_multi_supervised(
    normal_csv="./CIC_ae_normal.csv",
    anomaly_pattern="./CIC_anomaly_ae_{}.csv",
    num_anomaly_files=14,
    random_seed=42,
    anomaly_ratio=0.2,
    timesteps=10,
    features=2
):
    """
    CIC-IDS2018 AE 버전 (다중 anomaly 파일)을 지도학습용으로 불러옴.
    - 정상 1개 파일, anomaly 여러 개 자동 로드 및 병합
    - 정상 데이터 절반 train/test split
    - anomaly 일부만 학습 포함 (비율 anomaly_ratio)
    - 학습에 사용된 anomaly는 테스트셋에서 제거
    - CNN-LSTM 입력을 위해 reshape (예: 10×2)
    """

    np.random.seed(random_seed)
    random.seed(random_seed)

    # ----------------------------
    # 주요 피처 (20개)
    # ----------------------------
    SELECTED_FEATURES = [
        "Tot Fwd Pkts", "Tot Bwd Pkts", "TotLen Fwd Pkts", "TotLen Bwd Pkts",
        "Fwd Pkt Len Max", "Fwd Pkt Len Mean", "Bwd Pkt Len Max", "Bwd Pkt Len Mean",
        "Flow Byts/s", "Flow Pkts/s", "Flow IAT Mean", "Flow IAT Std",
        "Fwd IAT Mean", "Fwd IAT Std", "Bwd IAT Mean", "Bwd IAT Std",
        "Fwd Header Len", "ACK Flag Cnt", "Idle Mean", "Idle Std"
    ]

    # ----------------------------
    # 1. 정상 CSV 불러오기
    # ----------------------------
    df_normal = pd.read_csv(normal_csv)
    df_normal = df_normal[SELECTED_FEATURES].copy()

    # ----------------------------
    # 2. 여러 anomaly CSV 자동 병합
    # ----------------------------
    anomaly_dfs = []
    for i in range(1, num_anomaly_files + 1):
        path = anomaly_pattern.format(i)
        if os.path.exists(path):
            df_temp = pd.read_csv(path)
            df_temp = df_temp[SELECTED_FEATURES].copy()
            anomaly_dfs.append(df_temp)
        else:
            print(f"⚠️ Warning: {path} not found, skipping.")
    df_anomaly = pd.concat(anomaly_dfs, ignore_index=True)

    # ----------------------------
    # 3. NaN / inf 처리
    # ----------------------------
    for df in [df_normal, df_anomaly]:
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)

    # ----------------------------
    # 4. 정상 데이터 split
    # ----------------------------
    df_normal = shuffle(df_normal, random_state=random_seed)
    mid_idx = len(df_normal) // 2
    df_normal_train = df_normal.iloc[:mid_idx]
    df_normal_test = df_normal.iloc[mid_idx:]

    # ----------------------------
    # 5. 스케일링
    # ----------------------------
    scaler = MinMaxScaler()
    X_normal_train = scaler.fit_transform(df_normal_train)
    X_normal_test = scaler.transform(df_normal_test)
    X_anomaly_all = scaler.transform(df_anomaly)

    # ----------------------------
    # 6. anomaly 일부만 학습 포함
    # ----------------------------
    num_anomaly_to_add = int(len(X_anomaly_all) * anomaly_ratio)
    X_anomaly_train = X_anomaly_all[:num_anomaly_to_add]
    X_anomaly_test = X_anomaly_all[num_anomaly_to_add:]

    # ----------------------------
    # 7. train / test 구성
    # ----------------------------
    X_train = np.concatenate([X_normal_train, X_anomaly_train], axis=0)
    y_train = np.concatenate([
        np.zeros(len(X_normal_train)), np.ones(len(X_anomaly_train))
    ])
    X_test = np.concatenate([X_normal_test, X_anomaly_test], axis=0)
    y_test = np.concatenate([
        np.zeros(len(X_normal_test)), np.ones(len(X_anomaly_test))
    ])

    # ----------------------------
    # 8. 셔플
    # ----------------------------
    X_train, y_train = shuffle(X_train, y_train, random_state=random_seed)
    X_test, y_test = shuffle(X_test, y_test, random_state=random_seed)

    # ----------------------------
    # 9. CNN-LSTM 입력 reshape (예: 20 → 10×2)
    # ----------------------------
    X_train_seq = reshape_for_sequence_nsl(X_train, timesteps=timesteps, features=features)
    X_test_seq = reshape_for_sequence_nsl(X_test, timesteps=timesteps, features=features)

    print(f"✅ Loaded CIC dataset: Normal {len(df_normal)}, Anomaly {len(df_anomaly)}")
    print(f"Train: {X_train_seq.shape}, Test: {X_test_seq.shape}")

    return X_train_seq, y_train, X_test_seq, y_test


# ----------------------------
# 메인 실행 함수
# ----------------------------
def main():
    args = get_args()
    # X_train, y_train, X_test, y_test = get_datasets_nsl_supervised(
    #     random_seed=42, anomaly_ratio=0.2, timesteps=10, features=12
    # )

    X_train, y_train, X_test, y_test = get_datasets_cic_multi_supervised(
        normal_csv="./CIC2018/ae_datas/CIC_ae_normal.csv",
        anomaly_pattern="./CIC2018/ae_datas/CIC_anomaly_ae_{}.csv",
        num_anomaly_files=14,
        anomaly_ratio=0.2,
        timesteps=10,
        features=2
    )
    print("Train:", X_train.shape, y_train.shape)
    print("Test :", X_test.shape, y_test.shape)

    # ----------------------------
    # Global (Server) 모델 초기화
    # ----------------------------
    model = CNN_LSTM(timesteps=10, features=12)
    # _ = model(tf.zeros((1, 10, 12))) # NSL
    _ = model(tf.zeros((1, 10, 2))) # CIC
    model.compile(optimizer=Adam(0.0001), loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()

    # ----------------------------
    # 데이터 분할: 클라이언트별
    # ----------------------------
    client_data = np.array_split(X_train, args.client_nums)
    label_data = np.array_split(y_train, args.client_nums)

    # ----------------------------
    # 서버 평가 콜백
    # ----------------------------
    def server_evaluate(server_round: int, parameters, config):
        weights = parameters_to_ndarrays(parameters)
        model.set_weights(weights)

        y_prob = model.predict(X_test, verbose=0).reshape(-1)
        y_pred = (y_prob >= 0.5).astype(int)
        loss, acc = model.evaluate(X_test, y_test, verbose=0)

        report = classification_report(y_test, y_pred, target_names=["Normal", "Anomaly"], zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        try:
            auc = roc_auc_score(y_test, y_prob)
        except Exception:
            auc = 0.0

        with open(RESULT_PATH, "a") as f:
            f.write(f"\n[Round {server_round}] Server Evaluation Report\n")
            f.write(f"Loss: {loss:.4f}  Acc: {acc:.4f}  AUC: {auc:.4f}\n")
            f.write(report)

        try:
            plt.figure()
            RocCurveDisplay.from_predictions(y_test, y_prob)
            plt.savefig(ROC_PATH, bbox_inches="tight")
            plt.close()
        except Exception:
            pass

        return float(loss), {"acc": float(acc), "auc": float(auc)}

    # ----------------------------
    # 커스텀 FedAvg 전략
    # ----------------------------
    strategy = SaveEvaluationFedAvg(
        eval_server_args=None,
        fraction_fit=0.8,
        fraction_evaluate=0.8,
        min_fit_clients=args.client_nums,
        min_evaluate_clients=args.client_nums,
        min_available_clients=args.client_nums,
        evaluate_fn=server_evaluate,
    )

    # ----------------------------
    # 클라이언트 함수 정의
    # ----------------------------
    def client_fn(cid: str):
        cid_int = int(cid)
        client_model = CNN_LSTM(timesteps=10, features=12)
        # _ = client_model(tf.zeros((1, 10, 12))) # NSL
        _ = model(tf.zeros((1, 10, 2))) # CIC
        client_model.compile(optimizer=Adam(0.0005), loss="binary_crossentropy", metrics=["accuracy"])

        X_tr = client_data[cid_int]
        y_tr = label_data[cid_int]

        return FLClient(
            cid=cid_int,
            model=client_model,
            X_train=X_tr,
            y_train=y_tr,
            X_test=X_test,
            y_test=y_test,
            epochs=args.client_epochs,
            batch_size=args.batch_size,
        )

    # ----------------------------
    # 결과 파일 초기화
    # ----------------------------
    with open(RESULT_PATH, "w") as f:
        f.write("[Server Evaluation Report]\n")

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
    # 최종 글로벌 모델 weight 저장
    # ----------------------------
    if hasattr(strategy, "final_parameters") and strategy.final_parameters is not None:
        final_weights = parameters_to_ndarrays(strategy.final_parameters)
        model.set_weights(final_weights)

    # ----------------------------
    # 학습 결과 저장 및 시각화
    # ----------------------------
    save_and_plot_history(
        history,
        csv_path="./cnn_lstm/cnn_lstm_history",
        png_path="./cnn_lstm/cnn_lstm_history.png",
    )

    # ----------------------------
    # 최종 평가
    # ----------------------------
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n[Final] Loss: {loss:.4f}  Acc: {acc:.4f}")

    model.save_weights(WEIGHT_PATH)
    print(f"\n✅ Model weights saved to {WEIGHT_PATH}")


if __name__ == "__main__":
    main()
