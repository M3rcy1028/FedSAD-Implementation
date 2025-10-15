'''
    Define Variational + Transformer AE model architecture
'''
from utils import *
from arguments import get_args
agrs = get_args()

BATCH_SIZE = args.batch_size
DROPOUT = args.dropout_rate
PERCENTILE = args.percentile

# Override Server Model
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

        preds = model.predict(X_test_scaled, verbose=0)
        y_pred = (preds > 0.5).astype(int)
        acc = np.mean(y_pred.flatten() == y_test)

        report = classification_report(y_test, y_pred, target_names=["Normal", "Anomaly"], zero_division=0)

        print(f"\n[Server Round {server_round}] Accuracy: {acc:.4f}")
        print(report)

        with open(self.eval_server_args["result_path"], "a") as f:
            f.write(f"\n[Round {server_round}] Server Evaluation Report\n")
            f.write(f"Accuracy: {acc:.4f}\n")
            f.write(report)

        return 0.0, {"accuracy": acc}

@tf.keras.utils.register_keras_serializable(package="Custom")
class CNN_LSTM(tf.keras.Model):
    def __init__(self, timesteps=10, features=12, cnn_filters=64, lstm_units=128, dropout_rate=0.1):
        super().__init__()
        self.timesteps = timesteps
        self.features = features

        # CNN Feature Extractor (시계열 각 step에 대한 로컬 패턴 학습)
        self.conv1 = layers.Conv1D(cnn_filters, 3, padding="same", activation="relu")
        self.conv2 = layers.Conv1D(cnn_filters, 3, padding="same", activation="relu")
        self.pool = layers.MaxPooling1D(pool_size=2)

        # LSTM Temporal modeling
        self.lstm = layers.LSTM(lstm_units, return_sequences=False)

        # Dense Classifier
        self.fc1 = layers.Dense(128, activation="relu")
        self.dropout = layers.Dropout(dropout_rate)
        self.fc2 = layers.Dense(64, activation="relu")

        # Output layer (binary classification)
        self.output_layer = layers.Dense(1, activation="sigmoid")

    def call(self, inputs, training=None):
        """
        inputs: (batch, timesteps=10, features=12)
        """
        x = self.conv1(inputs)       # (batch, 10, filters)
        x = self.conv2(x)
        x = self.pool(x)             # (batch, 5, filters)
        x = self.lstm(x)             # (batch, lstm_units)
        x = self.fc1(x)
        x = self.dropout(x, training=training)
        x = self.fc2(x)
        return self.output_layer(x)

class FLClient(fl.client.NumPyClient):
    def __init__(self, cid, model, X_train, y_train, X_test, y_test, epochs=10, batch_size=32):
        self.cid = cid
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.epochs = epochs
        self.batch_size = batch_size

        # 모델 빌드
        self.model(tf.zeros((1, X_train.shape[1], X_train.shape[2])))

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.compile(optimizer=Adam(0.001), loss="binary_crossentropy", metrics=["accuracy"])
        self.model.fit(
            self.X_train,
            self.y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=0,
            validation_split=0.1,
            class_weight={0:1, 1:2}
        )
        return self.model.get_weights(), len(self.X_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        preds = self.model.predict(self.X_test, verbose=0)
        y_pred = (preds > 0.5).astype(int)

        acc = np.mean(y_pred.flatten() == self.y_test)
        report = classification_report(self.y_test, y_pred, target_names=["Normal", "Anomaly"], zero_division=0)

        print(f"\n--- Client {self.cid} Evaluation ---")
        print(f"Accuracy: {acc:.4f}")
        print(report)

        with open("client_cnnlstm.txt", "a") as f:
            f.write(f"\n--- Client {self.cid} Evaluation ---\n")
            f.write(f"Accuracy: {acc:.4f}\n")
            f.write(report + "\n")

        return float(1 - acc), len(self.X_test), {}