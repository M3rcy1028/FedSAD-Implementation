import os
import numpy as np
import pandas as pd
import random
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler 
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from flwr.common import Scalar, parameters_to_ndarrays, ndarrays_to_parameters
import flwr as fl
from utils.arguments import get_args
from model_cnnlstm import SaveEvaluationFedAvg, CNN_LSTM, FLClient  # 커스텀 FedAvg, CNN-LSTM, Client

# ----------------------------
# path configuration
# ----------------------------
os.makedirs("./cnn_lstm", exist_ok=True)
WEIGHT_PATH = "./cnn_lstm/cnn_lstm_weights.h5"
MATRIX_PATH = "./cnn_lstm/cnn_lstm_cm.png"
RESULT_PATH = "./cnn_lstm/cnn_lstm_server.txt"
ROC_PATH = "./cnn_lstm/cnn_lstm_roc.png"

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


# ----------------------------
# reshape function for CNN input
# ----------------------------
def reshape_for_sequence_nsl(X, timesteps=10, features=12):
    n_samples, n_feats = X.shape
    if n_feats < timesteps * features:
        pad = np.zeros((n_samples, timesteps * features - n_feats))
        X = np.concatenate([X, pad], axis=1)
    X = X[:, :timesteps * features]
    return X.reshape(-1, timesteps, features)

# ----------------------------
# NSL-KDD supervised dataset loader
# ----------------------------
def get_datasets_nsl_supervised(random_seed=42, anomaly_ratio=0.2, timesteps=10, features=12):
    np.random.seed(random_seed)
    random.seed(random_seed)

    df_normal = pd.read_csv("./NSL-KDD/KDD_normal.csv")
    df_anomaly = pd.read_csv("./NSL-KDD/KDD_anomaly.csv")
    df_normal = shuffle(df_normal, random_state=random_seed)

    n_samples = 150_000
    df_normal = df_normal.sample(n=min(len(df_normal), n_samples * 2), random_state=random_seed)
    df_anomaly = df_anomaly.sample(n=min(len(df_anomaly), n_samples), random_state=random_seed)

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

# CIC 2018 dataset preprocessing
def get_datasets_cic_multi_supervised(
    normal_csv="./CIC2018/ae_datas_sampled/CIC_ae_normal.csv",
    anomaly_pattern="./CIC2018/v/CIC_anomaly_ae_{}.csv",
    num_anomaly_files=14,
    random_seed=123,
    anomaly_ratio=0.2,
    timesteps=10,
    features=8
):
    """
    Load CIC-IDS2018 AE version (multiple anomaly files) for supervised learning.
    - One normal file, multiple anomaly files auto-loaded and merged
    - Normal data split into half train/test
    - Only a portion of anomalies included in training (ratio: anomaly_ratio)
    - Anomalies used in training are excluded from test set
    - Reshaped for CNN-LSTM input (e.g., 10x2)
    """

    np.random.seed(random_seed)
    random.seed(random_seed)
    # ----------------------------
    # load normal CSV
    # ----------------------------
    df_normal = pd.read_csv(normal_csv)

    # ----------------------------
    # merge multiple anomaly CSV files automatically
    # ----------------------------
    anomlay_path = "./CIC2018/ae_datas_sampled/CIC_ae_anomaly.csv"
    df_anomaly = pd.read_csv(anomlay_path, low_memory=False)

    # ----------------------------
    # handle NaN / inf
    # ----------------------------
    for df in [df_normal, df_anomaly]:
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)

    # ----------------------------
    # split normal data
    # ----------------------------
    # 80% normal split
    df_normal = shuffle(df_normal, random_state=random_seed)
    split_point = int(len(df_normal) * 0.8)
    df_normal_train = df_normal.iloc[:split_point]
    df_normal_test = df_normal.iloc[split_point:]

    # ----------------------------
    # scaling
    # ----------------------------
    scaler = MinMaxScaler()
    X_normal_train = scaler.fit_transform(df_normal_train)
    X_normal_test = scaler.transform(df_normal_test)
    X_anomaly_all = scaler.transform(df_anomaly)

    # ----------------------------
    # include a portion of anomalies in training
    # ----------------------------
    num_anomaly_to_add = int(len(X_anomaly_all) * anomaly_ratio)
    X_anomaly_train = X_anomaly_all[:num_anomaly_to_add]
    X_anomaly_test = X_anomaly_all[num_anomaly_to_add:]

    # ----------------------------
    # build train / test sets
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
    # shuffle
    # ----------------------------
    X_train, y_train = shuffle(X_train, y_train, random_state=random_seed)
    X_test, y_test = shuffle(X_test, y_test, random_state=random_seed)

    # ----------------------------
    # CNN-LSTM input reshape (e.g., 20 -> 10x2)
    # ----------------------------
    X_train_seq = reshape_for_sequence_nsl(X_train, timesteps=timesteps, features=features)
    X_test_seq = reshape_for_sequence_nsl(X_test, timesteps=timesteps, features=features)

    print(f"Loaded CIC dataset: Normal {len(df_normal)}, Anomaly {len(df_anomaly)}")
    print(f"Train: {X_train_seq.shape}, Test: {X_test_seq.shape}")

    return X_train_seq, y_train, X_test_seq, y_test

# KDD99 dataset preprocessing
def get_datasets_kdd99_supervised(random_seed=42, anomaly_ratio=0.2, timesteps=10, features=12):
    np.random.seed(random_seed)
    random.seed(random_seed)

    df_normal = pd.read_csv("./KDD99/KDD99_normal.csv")
    df_anomaly = pd.read_csv("./KDD99/KDD99_anomaly.csv")
    df_normal = shuffle(df_normal, random_state=random_seed)

    scaler = MinMaxScaler()
    mid_idx = len(df_normal) // 2
    df_normal_train = df_normal.iloc[:mid_idx]
    df_normal_test = df_normal.iloc[mid_idx:]

    X_normal_train = scaler.fit_transform(df_normal_train)
    X_normal_test = scaler.transform(df_normal_test)
    X_anomaly_all = scaler.transform(df_anomaly)

    # include a portion of anomalies in training
    num_anomaly_train = int(len(X_anomaly_all) * anomaly_ratio)
    X_anomaly_train = X_anomaly_all[:num_anomaly_train]
    X_anomaly_test = X_anomaly_all[num_anomaly_train:]

    X_train = np.concatenate([X_normal_train, X_anomaly_train], axis=0)
    y_train = np.concatenate([
        np.zeros(len(X_normal_train)),
        np.ones(len(X_anomaly_train))
    ])
    X_test = np.concatenate([X_normal_test, X_anomaly_test], axis=0)
    y_test = np.concatenate([
        np.zeros(len(X_normal_test)),
        np.ones(len(X_anomaly_test))
    ])

    X_train, y_train = shuffle(X_train, y_train, random_state=random_seed)
    X_test, y_test = shuffle(X_test, y_test, random_state=random_seed)

    X_train_seq = reshape_for_sequence_nsl(X_train, timesteps=timesteps, features=features)
    X_test_seq = reshape_for_sequence_nsl(X_test, timesteps=timesteps, features=features)

    print(f"Loaded KDD99 (train {len(X_train_seq)}, test {len(X_test_seq)}) with {anomaly_ratio*100:.1f}% anomaly in training")
    return X_train_seq, y_train, X_test_seq, y_test

# InSDN dataset preprocessing
def get_datasets_insdn_supervised(
    normal_csv="./InSDN/ae_datas/InSDN_normal.csv",
    anomaly_csv="./InSDN/ae_datas/InSDN_anomaly.csv",
    random_seed=42,
    anomaly_ratio=0.4,
    timesteps=12,     
    features=7
):
    """
    InSDN 48-feature supervised dataset (for CNN-LSTM input).
    - Half of normal data for train, remaining half + all anomalies for test
    - Only a portion of anomalies (anomaly_ratio) included in train
    - Reshaped to (timesteps, features) = (8, 6) for CNN-LSTM input
    Returns: X_train_seq, y_train, X_test_seq, y_test
    """
    np.random.seed(random_seed)
    random.seed(random_seed)

    # ---------------------------
    # (1) Load & clean
    # ---------------------------
    df_normal = pd.read_csv(normal_csv, low_memory=False)
    df_anomaly = pd.read_csv(anomaly_csv, low_memory=False)

    def _clean(df):
        return df.apply(pd.to_numeric, errors="coerce") \
                 .replace([np.inf, -np.inf], np.nan) \
                 .fillna(0)
    df_normal = _clean(df_normal)
    df_anomaly = _clean(df_anomaly)

    # ---------------------------
    # (2) Split normal data
    # ---------------------------
    df_normal = shuffle(df_normal, random_state=random_seed)
    split_point = int(len(df_normal) * 0.8)
    df_normal_train = df_normal.iloc[:split_point] # for scaler training
    df_normal_test = df_normal.iloc[split_point:] # for actual testing

    # ---------------------------
    # (3) Scaling
    # ---------------------------
    scaler = MinMaxScaler()
    X_normal_train = scaler.fit_transform(df_normal_train.values)
    X_normal_test  = scaler.transform(df_normal_test.values)
    X_anomaly_all  = scaler.transform(df_anomaly.values)

    # ---------------------------
    # (4) Select anomalies for train/test
    # ---------------------------
    n_anom_train = int(len(X_anomaly_all) * anomaly_ratio)
    X_anomaly_train = X_anomaly_all[:n_anom_train]
    X_anomaly_test  = X_anomaly_all[n_anom_train:]

    # ---------------------------
    # (5) Combine & label
    # ---------------------------
    X_train = np.concatenate([X_normal_train, X_anomaly_train], axis=0)
    y_train = np.concatenate([
        np.zeros(len(X_normal_train), dtype=int),
        np.ones(len(X_anomaly_train), dtype=int)
    ])

    X_test = np.concatenate([X_normal_test, X_anomaly_test], axis=0)
    y_test = np.concatenate([
        np.zeros(len(X_normal_test), dtype=int),
        np.ones(len(X_anomaly_test), dtype=int)
    ])

    X_train, y_train = shuffle(X_train, y_train, random_state=random_seed)
    X_test,  y_test  = shuffle(X_test,  y_test,  random_state=random_seed)

    # ---------------------------
    # (6) Reshape for CNN-LSTM
    # ---------------------------
    def reshape_for_sequence_insdn(X, timesteps=12, features=7):
        n_samples, n_feats = X.shape
        if n_feats < timesteps * features:
            pad = np.zeros((n_samples, timesteps * features - n_feats))
            X = np.concatenate([X, pad], axis=1)
        X = X[:, :timesteps * features]
        return X.reshape(-1, timesteps, features)

    X_train_seq = reshape_for_sequence_insdn(X_train, timesteps, features)
    X_test_seq  = reshape_for_sequence_insdn(X_test,  timesteps, features)

    # ---------------------------
    # (7) Summary
    # ---------------------------
    print(f"[InSDN Supervised] Train: {X_train_seq.shape}, Test: {X_test_seq.shape}")
    print(f"y_train: {y_train.shape}, y_test: {y_test.shape}, anomaly_ratio(train)={anomaly_ratio}")

    return X_train_seq, y_train, X_test_seq, y_test

def get_datasets_unsw_supervised(
    normal_csv="./UNSW_NB15/ae_datas/UNSW_NB15_normal.csv",
    anomaly_csv="./UNSW_NB15/ae_datas/UNSW_NB15_anomaly.csv",
    random_seed=42,
    anomaly_ratio=0.2,
    timesteps=6,     
    features=7
):
    np.random.seed(random_seed)
    random.seed(random_seed)

    # ---------------------------
    # (1) Load & clean
    # ---------------------------
    df_normal = pd.read_csv(normal_csv, low_memory=False)
    df_anomaly = pd.read_csv(anomaly_csv, low_memory=False)

    def _clean(df):
        return df.apply(pd.to_numeric, errors="coerce") \
                 .replace([np.inf, -np.inf], np.nan) \
                 .fillna(0)
    df_normal = _clean(df_normal)
    df_anomaly = _clean(df_anomaly)

    # ---------------------------
    # (2) Split normal data
    # ---------------------------
    df_normal = shuffle(df_normal, random_state=random_seed)
    
    split_point = int(len(df_normal) * 0.8)
    df_normal_train = df_normal.iloc[:split_point] # for scaler training
    df_normal_test = df_normal.iloc[split_point:] # for actual testing

    # ---------------------------
    # (3) Scaling
    # ---------------------------
    scaler = MinMaxScaler()
    X_normal_train = scaler.fit_transform(df_normal_train.values)
    X_normal_test  = scaler.transform(df_normal_test.values)
    X_anomaly_all  = scaler.transform(df_anomaly.values)

    # ---------------------------
    # (4) Select anomalies for train/test
    # ---------------------------
    n_anom_train = int(len(X_anomaly_all) * anomaly_ratio)
    X_anomaly_train = X_anomaly_all[:n_anom_train]
    X_anomaly_test  = X_anomaly_all[n_anom_train:]

    # ---------------------------
    # (5) Combine & label
    # ---------------------------
    X_train = np.concatenate([X_normal_train, X_anomaly_train], axis=0)
    y_train = np.concatenate([
        np.zeros(len(X_normal_train), dtype=int),
        np.ones(len(X_anomaly_train), dtype=int)
    ])

    X_test = np.concatenate([X_normal_test, X_anomaly_test], axis=0)
    y_test = np.concatenate([
        np.zeros(len(X_normal_test), dtype=int),
        np.ones(len(X_anomaly_test), dtype=int)
    ])

    X_train, y_train = shuffle(X_train, y_train, random_state=random_seed)
    X_test,  y_test  = shuffle(X_test,  y_test,  random_state=random_seed)

    # ---------------------------
    # (6) Reshape for CNN-LSTM
    # ---------------------------
    def reshape_for_sequence_insdn(X, timesteps=12, features=7):
        n_samples, n_feats = X.shape
        if n_feats < timesteps * features:
            pad = np.zeros((n_samples, timesteps * features - n_feats))
            X = np.concatenate([X, pad], axis=1)
        X = X[:, :timesteps * features]
        return X.reshape(-1, timesteps, features)

    X_train_seq = reshape_for_sequence_insdn(X_train, timesteps, features)
    X_test_seq  = reshape_for_sequence_insdn(X_test,  timesteps, features)

    # ---------------------------
    # (7) Summary
    # ---------------------------
    print(f"[UNSW Supervised] Train: {X_train_seq.shape}, Test: {X_test_seq.shape}")
    print(f"y_train: {y_train.shape}, y_test: {y_test.shape}, anomaly_ratio(train)={anomaly_ratio}")

    return X_train_seq, y_train, X_test_seq, y_test

# ----------------------------
# main function
# ----------------------------
def main():
    args = get_args()
    # X_train, y_train, X_test, y_test = get_datasets_nsl_supervised(
    #     random_seed=42, anomaly_ratio=0.2, timesteps=10, features=12
    # )

    # X_train, y_train, X_test, y_test = get_datasets_cic_multi_supervised(
    #     normal_csv="./CIC2018/ae_datas_sampled/CIC_ae_normal.csv",
    #     anomaly_pattern="./CIC2018/ae_datas_sampled/CIC_anomaly_ae_{}.csv",
    #     num_anomaly_files=14,
    #     anomaly_ratio=0.5,
    #     timesteps=10,
    #     features=8
    # )

    # X_train, y_train, X_test, y_test = get_datasets_kdd99_supervised(
    #     random_seed=42, anomaly_ratio=0.2, timesteps=10, features=12
    # )

    X_train, y_train, X_test, y_test = get_datasets_insdn_supervised(timesteps=12, features=7)

    # UNSW_NB15
    # X_train, y_train, X_test, y_test = get_datasets_unsw_supervised(timesteps=6, features=7)

    print("Train:", X_train.shape, y_train.shape)
    print("Test :", X_test.shape, y_test.shape)

    # ----------------------------
    # Global (Server) model initialization
    # ----------------------------
    # model = CNN_LSTM(timesteps=10, features=12)
    # _ = model(tf.zeros((1, 10, 12))) # NSL
    # model = CNN_LSTM(timesteps=10, features=8)
    # _ = model(tf.zeros((1, 10, 8))) # CIC
    # model = CNN_LSTM(timesteps=10, features=12)
    # _ = model(tf.zeros((1, 10, 12))) # KDD99
    model = CNN_LSTM(timesteps=12, features=7) # 83
    _ = model(tf.zeros((1, 12, 7))) # InSDN
    # model = CNN_LSTM(timesteps=6, features=7) # 42
    # _ = model(tf.zeros((1, 6, 7))) # UNSW
    model.compile(optimizer=Adam(0.0001), loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()

    # ----------------------------
    # data split: per client
    # ----------------------------
    client_data = np.array_split(X_train, args.client_nums)
    label_data = np.array_split(y_train, args.client_nums)

    # ----------------------------
    # server evaluation callback
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
            RocCurveDisplay.from_predictions(y_test, y_prob) # TODO RocCurveDisplay 정의
            plt.savefig(ROC_PATH, bbox_inches="tight")
            plt.close()
        except Exception:
            pass

        return float(loss), {"acc": float(acc), "auc": float(auc)}

    # ----------------------------
    # custom FedAvg strategy
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
    # client function definition
    # ----------------------------
    def client_fn(cid: str):
        cid_int = int(cid)
        # client_model = CNN_LSTM(timesteps=10, features=12)
        # _ = client_model(tf.zeros((1, 10, 12))) # NSL
        # client_model = CNN_LSTM(timesteps=10, features=8)
        # _ = client_model(tf.zeros((1, 10, 8))) # CIC
        # client_model = CNN_LSTM(timesteps=10, features=12)
        # _ = client_model(tf.zeros((1, 10, 12))) # KDD99
        client_model = CNN_LSTM(timesteps=12, features=7)
        _ = client_model(tf.zeros((1, 12, 7))) # InSDN
        # client_model = CNN_LSTM(timesteps=6, features=7) # 42
        # _ = client_model(tf.zeros((1, 6, 7))) # UNSW
        client_model.compile(optimizer=Adam(0.0001), loss="binary_crossentropy", metrics=["accuracy"])

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
    # initialize results file
    # ----------------------------
    with open(RESULT_PATH, "w") as f:
        f.write("[Server Evaluation Report]\n")

    # ----------------------------
    # start federated learning
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
    # save final global model weights
    # ----------------------------
    if hasattr(strategy, "final_parameters") and strategy.final_parameters is not None:
        final_weights = parameters_to_ndarrays(strategy.final_parameters)
        model.set_weights(final_weights)

    # # ----------------------------
    # # save and visualize training results
    # # ----------------------------
    # save_and_plot_history(
    #     history,
    #     csv_path="./cnn_lstm/cnn_lstm_history",
    #     png_path="./cnn_lstm/cnn_lstm_history.png",
    # )

    # # ----------------------------
    # # final evaluation
    # # ----------------------------
    # loss, acc = model.evaluate(X_test, y_test, verbose=0)
    # print(f"\n[Final] Loss: {loss:.4f}  Acc: {acc:.4f}")

    model.save_weights(WEIGHT_PATH)
    print(f"\nModel weights saved to {WEIGHT_PATH}")

 
if __name__ == "__main__":
    main()
