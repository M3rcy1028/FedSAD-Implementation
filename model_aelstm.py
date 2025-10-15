'''
Unsupervised AE-LSTM model
 - Learns only normal data
 - Detects anomalies using reconstruction error
'''
from utils import *
from arguments import get_args
args = get_args()

BATCH_SIZE = 8
DROPOUT = 0.1
PERCENTILE = 85


# ---------------------- Server Aggregation ----------------------
class SaveEvaluationFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, eval_server_args=None, **kwargs):
        super().__init__(**kwargs)
        self.eval_server_args = eval_server_args
        self.final_parameters = None

    def aggregate_fit(self, rnd, results, failures):
        aggregated_parameters, metrics = super().aggregate_fit(rnd, results, failures)
        if aggregated_parameters is not None:
            self.final_parameters = aggregated_parameters
        return aggregated_parameters, metrics

    def evaluate(self, server_round, parameters):
        if self.eval_server_args is None:
            return None

        model = self.eval_server_args["model"]
        model.set_weights(fl.common.parameters_to_ndarrays(parameters))
        X_test_scaled = self.eval_server_args["X_test_scaled"]
        y_test = self.eval_server_args["y_test"]

        # Reconstruction & Error
        recon = model.predict(X_test_scaled, verbose=0)
        mse = np.mean(np.square(X_test_scaled - recon), axis=(1,2))
        threshold = np.percentile(mse, PERCENTILE)  # 상위 5% 이상을 이상치로 판단
        y_pred = (mse > threshold).astype(int)

        acc = np.mean(y_pred.flatten() == y_test)
        print(f"\n[Server Round {server_round}] Acc={acc:.4f}, Threshold={threshold:.6f}")

        report = classification_report(y_test, y_pred, target_names=["Normal", "Anomaly"], zero_division=0)
        with open(self.eval_server_args["result_path"], "a") as f:
            f.write(f"\n[Round {server_round}] Server Evaluation\nAcc={acc:.4f}\n")
            f.write(report)

        return 0.0, {"accuracy": acc}


# ---------------------- AE-LSTM (Unsupervised) ----------------------
@tf.keras.utils.register_keras_serializable(package="Custom")
class AE_LSTM(tf.keras.Model):
    """
    AutoEncoder + LSTM (Unsupervised)
    - Input = Output (Reconstruction)
    - Learns temporal dependencies of normal patterns
    """
    def __init__(self, timesteps=10, features=12, latent_dim=32, lstm_units=128, dropout_rate=DROPOUT):
        super().__init__()
        self.timesteps = timesteps
        self.features = features

        # Encoder (Feature compression)
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(timesteps, features)),
            layers.LSTM(lstm_units, return_sequences=False),
            layers.Dense(latent_dim, activation="relu"),
            layers.Dropout(dropout_rate)
        ])

        # Decoder (Reconstruction)
        self.decoder = tf.keras.Sequential([
            layers.Input(shape=(latent_dim,)),
            layers.RepeatVector(timesteps),
            layers.LSTM(lstm_units, return_sequences=True),
            layers.TimeDistributed(layers.Dense(features))
        ])

    def call(self, inputs, training=None):
        z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        return reconstructed


# ---------------------- Federated Client ----------------------
class FLClient(fl.client.NumPyClient):
    def __init__(self, cid, model, X_train, X_test, y_test, epochs=30, batch_size=BATCH_SIZE):
        self.cid = cid
        self.model = model
        self.X_train = X_train  # 정상 데이터만
        self.X_test = X_test
        self.y_test = y_test
        self.epochs = epochs
        self.batch_size = batch_size

        self.model(tf.zeros((1, X_train.shape[1], X_train.shape[2])))
        # self.model(tf.zeros((1, 10, 12))) # NSL-KDD
 
    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.compile(optimizer=Adam(0.001), loss="mse")
        self.model.fit(
            self.X_train, self.X_train,  # reconstruction
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=0,
            validation_split=0.1,
        )
        return self.model.get_weights(), len(self.X_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        recon = self.model.predict(self.X_test, verbose=0)
        mse = np.mean(np.square(self.X_test - recon), axis=(1, 2))
        threshold = np.percentile(mse, PERCENTILE)
        y_pred = (mse > threshold).astype(int)
        acc = np.mean(y_pred.flatten() == self.y_test)

        print(f"\n--- Client {self.cid} Evaluation ---")
        print(f"Threshold: {threshold:.6f} | Accuracy: {acc:.4f}")

        report = classification_report(self.y_test, y_pred, target_names=["Normal", "Anomaly"], zero_division=0)
        print(report)
        with open("client_aelstm_unsup.txt", "a") as f:
            f.write(f"\n--- Client {self.cid} Evaluation ---\n")
            f.write(f"Threshold: {threshold:.6f} | Acc: {acc:.4f}\n")
            f.write(report + "\n")

        return float(1 - acc), len(self.X_test), {}
