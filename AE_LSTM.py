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

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

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

def get_datasets_cic_uns(random_seed=args.random_seed, timesteps=10, features=2):
    np.random.seed(random_seed)
    random.seed(random_seed)

    # ---------------------------
    # (0) 데이터 개수 설정
    # ---------------------------
    N_NORMAL_TRAIN = 2748235
    N_NORMAL_TEST = 2748235
    N_ANOMALY_TEST = 2748235

    # ---------------------------
    # (1) 선택 피처 (20개)
    # ---------------------------
    SELECTED_FEATURES = [
        "Tot Fwd Pkts", "Tot Bwd Pkts", "TotLen Fwd Pkts", "TotLen Bwd Pkts",
        "Fwd Pkt Len Max", "Fwd Pkt Len Mean", "Bwd Pkt Len Max", "Bwd Pkt Len Mean",
        "Flow Byts/s", "Flow Pkts/s", "Flow IAT Mean", "Flow IAT Std",
        "Fwd IAT Mean", "Fwd IAT Std", "Bwd IAT Mean", "Bwd IAT Std",
        "Fwd Header Len", "ACK Flag Cnt", "Idle Mean", "Idle Std"
    ]

    # ---------------------------
    # (2) 정상 데이터 로드 & 정리
    # ---------------------------
    df_normal = pd.read_csv("./CIC2018/ae_datas/CIC_ae_normal.csv")[SELECTED_FEATURES]
    df_normal = shuffle(df_normal, random_state=random_seed)
    df_normal = df_normal.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
    df_normal[df_normal < 0] = 0

    # ---------------------------
    # (3) 공격 데이터 로드 & 정리
    # ---------------------------
    anomaly_files = [f"./CIC2018/ae_datas/CIC_anomaly_ae_{i}.csv" for i in range(1, 15)]
    df_anomaly_list = [pd.read_csv(f)[SELECTED_FEATURES] for f in anomaly_files]
    df_anomaly = pd.concat(df_anomaly_list, ignore_index=True)
    df_anomaly = shuffle(df_anomaly, random_state=random_seed)
    df_anomaly = df_anomaly.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
    df_anomaly[df_anomaly < 0] = 0

    # ---------------------------
    # (4) 데이터 분할
    # ---------------------------
    total_normal_needed = N_NORMAL_TRAIN + N_NORMAL_TEST
    if len(df_normal) < total_normal_needed:
        raise ValueError(
            f"정상 데이터 개수({len(df_normal)})가 필요한 총 개수({total_normal_needed})보다 부족합니다."
        )

    df_normal_train = df_normal.iloc[:N_NORMAL_TRAIN]
    df_normal_test = df_normal.iloc[N_NORMAL_TRAIN:total_normal_needed]

    if len(df_anomaly) < N_ANOMALY_TEST:
        raise ValueError(
            f"이상 데이터 개수({len(df_anomaly)})가 테스트에 필요한 개수({N_ANOMALY_TEST})보다 부족합니다."
        )

    df_anomaly_test = df_anomaly.iloc[:N_ANOMALY_TEST]

    print(f"정상 훈련 데이터: {len(df_normal_train)}개")
    print(f"정상 테스트 데이터: {len(df_normal_test)}개")
    print(f"이상 테스트 데이터: {len(df_anomaly_test)}개")

    # ---------------------------
    # (5) MinMax 정규화
    # ---------------------------
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(df_normal_train.values)

    df_test = pd.concat([df_normal_test, df_anomaly_test], ignore_index=True)
    X_test = scaler.transform(df_test.values)
    y_test = np.concatenate([np.zeros(len(df_normal_test)), np.ones(len(df_anomaly_test))])
    X_test, y_test = shuffle(X_test, y_test, random_state=random_seed)

    # ---------------------------
    # (6) reshape for AE-LSTM
    # ---------------------------

    X_train_seq = reshape_for_sequence(X_train, timesteps, features)
    X_test_seq = reshape_for_sequence(X_test, timesteps, features)

    print(f"✅ Reshaped for AE-LSTM: Train {X_train_seq.shape}, Test {X_test_seq.shape}")

    return X_train_seq, X_test_seq, y_test


# ----------------------------
# 메인 함수
# ----------------------------
def main():
    args = get_args()

    # 데이터 불러오기
    X_train, X_test, y_test = get_datasets_kdd99()
    # X_train = reshape_for_sequence(X_train, timesteps=10, features=12)
    # X_test = reshape_for_sequence(X_test, timesteps=10, features=12)

    # ----------------------------
    # 서버 모델 정의 (비지도 AE-LSTM)
    # ----------------------------
    # model = AE_LSTM(timesteps=10, features=12) # NSL
    # _ = model(tf.zeros((1, 10, 12)))
    model = AE_LSTM(timesteps=10, features=2) # CIC
    _ = model(tf.zeros((1, 10, 2)))
    model.compile(optimizer=Adam(0.001), loss="mse")
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

        # Reconstruction
        recon = model.predict(X_test, verbose=0)
        mse = np.mean(np.square(X_test - recon), axis=(1, 2))
        threshold = np.percentile(mse, 95)  # 상위 5%를 이상치로 판단
        y_pred = (mse > threshold).astype(int)

        acc = np.mean(y_pred == y_test)
        try:
            auc = roc_auc_score(y_test, mse)
        except Exception:
            auc = 0.0

        report = classification_report(y_test, y_pred, target_names=["Normal", "Anomaly"], zero_division=0)

        print(f"\n[Server Round {server_round}] Acc={acc:.4f}, AUC={auc:.4f}, Th={threshold:.6f}")
        with open(RESULT_PATH, "a") as f:
            f.write(f"\n[Round {server_round}] Server Evaluation\n")
            f.write(f"Acc={acc:.4f}, AUC={auc:.4f}, Th={threshold:.6f}\n")
            f.write(report)

        try:
            plt.figure()
            RocCurveDisplay.from_predictions(y_test, mse)
            plt.savefig(ROC_PATH, bbox_inches="tight")
            plt.close()
        except Exception:
            pass

        return float(1 - acc), {"acc": acc, "auc": auc}

    # ----------------------------
    # 커스텀 FedAvg 전략
    # ----------------------------
    strategy = SaveEvaluationFedAvg(
        eval_server_args=None,
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
        # client_model = AE_LSTM(timesteps=10, features=12) # NSL
        # _ = client_model(tf.zeros((1, 10, 12)))
        client_model = AE_LSTM(timesteps=10, features=2) # CIC
        _ = client_model(tf.zeros((1, 10, 2)))
        client_model.compile(optimizer=Adam(0.001), loss="mse")

        X_tr = client_data[cid_int]

        return FLClient(
            cid=cid_int,
            model=client_model,
            X_train=X_tr,
            X_test=X_test,
            y_test=y_test,
            epochs=args.client_epochs,
            batch_size=args.batch_size,
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
        csv_path="./ae_lstm_nsl/ae_lstm_history",
        png_path="./ae_lstm_nsl/ae_lstm_history.png",
    )

    print(f"\n[Final] Accuracy={acc:.4f}, Threshold={threshold:.6f}")
    model.save_weights(WEIGHT_PATH)
    print(f"✅ Model weights saved to {WEIGHT_PATH}")


if __name__ == "__main__":
    main()
