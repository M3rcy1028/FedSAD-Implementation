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
PERCENTILE = 65


# ---------------------- Server Aggregation ----------------------
class SaveEvaluationFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, eval_server_args=None, **kwargs):
        super().__init__(**kwargs)
        self.eval_server_args = eval_server_args  # 반드시 {"model": model, ...} 포함
        self.final_parameters = None

    # 🔒 라운드 0 시작 전에 서버가 “빌드된” 글로벌 파라미터를 브로드캐스트
    def initialize_parameters(self, client_manager):
        # ...
        mdl = self.eval_server_args["model"]

        # ...
        if not mdl.weights:
            print("🧱 [Strategy] Building model for initial parameters...")
            # ⬇️⬇️⬇️ [FIX] 'mdl' 변수를 사용합니다. ⬇️⬇️⬇️
            try:
                # Keras 모델의 input shape에서 timesteps, features를 동적으로 가져오기
                input_shape = mdl.encoder.input_shape 
                timesteps, features = input_shape[1], input_shape[2]
                _ = mdl(tf.zeros((1, timesteps, features)))
                print(f"✅ Model built with shape (1, {timesteps}, {features})")
            except Exception as e:
                print(f"🚨 Model build failed in strategy: {e}")
                # fallback (하드코딩)
                _ = mdl(tf.zeros((1, 10, 4))) 


        nd = mdl.get_weights()
        return fl.common.ndarrays_to_parameters(nd)

    def aggregate_fit(self, rnd, results, failures):
        aggregated_parameters, metrics = super().aggregate_fit(rnd, results, failures)
        
        if aggregated_parameters is not None:
            self.final_parameters = aggregated_parameters
        else:
            # ⛑️ fallback: 
            print(f"⚠️ [Round {rnd}] Aggregation failed, using fallback weights.")
            mdl = self.eval_server_args["model"]
            current_weights_nd = mdl.get_weights()
            aggregated_parameters = fl.common.ndarrays_to_parameters(current_weights_nd)
            
            # ⬇️⬇️⬇️ [FIX] fallback 시에도 final_parameters를 업데이트합니다. ⬇️⬇️⬇️
            self.final_parameters = aggregated_parameters
            
        return aggregated_parameters, metrics

    def evaluate(self, server_round: int, parameters: fl.common.Parameters):
        """서버 검증: 파라미터가 None/빈 리스트면 현재 모델 가중치로 대체"""
        if self.eval_server_args is None:
            return None

        mdl = self.eval_server_args["model"]

        # 🛡️ 빈 파라미터 가드
        nd = []
        if parameters is not None:
            try:
                nd = fl.common.parameters_to_ndarrays(parameters)
            except Exception:
                nd = []
        if (parameters is None) or (not nd):
            # initialize_parameters에서 보장되지만, 혹시 모를 방어
            nd = mdl.get_weights()
        mdl.set_weights(nd)

        X_test = self.eval_server_args["X_test_scaled"]
        y_test = self.eval_server_args["y_test"]

        # multi-output 대응
        preds = mdl.predict(X_test, verbose=0)
        if isinstance(preds, dict):
            recon = preds["decoded"]
        else:
            recon = preds

        mse = np.mean(np.square(X_test - recon), axis=(1, 2))
        threshold = np.percentile(mse, 80)
        y_pred = (mse > threshold).astype(int)
        acc = float(np.mean(y_pred == y_test))

        with open(self.eval_server_args["result_path"], "a") as f:
            f.write(f"\n[Round {server_round}] Server Evaluation\nAcc={acc:.4f}\n")
            f.write(classification_report(y_test, y_pred,
                                          target_names=["Normal", "Anomaly"],
                                          zero_division=0))
        # Flower는 (loss, metrics) 반환
        return float(1 - acc), {"acc": acc}

# ---------------------- AE-LSTM (Unsupervised) ----------------------
@tf.keras.utils.register_keras_serializable(package="Custom")
class AE_LSTM(tf.keras.Model):
    """PSO-AE-LSTM inspired semi-supervised model"""
    def __init__(self, input_dim, timesteps, features,
                 latent_dim=64, dropout_rate=0.001):
        super().__init__()
        self.timesteps = timesteps
        self.features = features

        # ---------- Encoder (L1–L3) ----------
        self.encoder = tf.keras.Sequential([
            layers.LSTM(128, activation="relu", return_sequences=True,
                        input_shape=(timesteps, features)),
            layers.LSTM(64, activation="relu"),
            layers.Dropout(dropout_rate),
            layers.Dense(latent_dim, activation="relu", name="bottleneck"),
        ], name="Encoder")

        # ---------- Decoder (L4–L5) ----------
        self.decoder = tf.keras.Sequential([
            layers.Dense(latent_dim, activation="relu"),
            layers.RepeatVector(timesteps),
            layers.LSTM(64, activation="relu", return_sequences=True),
            layers.LSTM(128, activation="relu", return_sequences=True),
            layers.TimeDistributed(
                layers.Dense(features, activation="sigmoid")
            ),
        ], name="Decoder")

        # ---------- Classifier ----------
        self.classifier = tf.keras.Sequential([
            layers.Dense(64, activation="relu"),
            layers.Dropout(dropout_rate),
            layers.Dense(32, activation="relu"),
            layers.Dense(1, activation="sigmoid")
        ], name="Classifier")

    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        pred = self.classifier(encoded)
        return {"decoded": decoded, "pred": pred}

# ---------------------- Federated Client ----------------------
class FLClient(fl.client.NumPyClient):
    def __init__(self, cid, model, X_train, y_train_cls, X_test, y_test,
                 epochs=25, batch_size=64):
        self.cid = cid
        self.model = model
        self.X_train = X_train
        self.y_train_cls = y_train_cls
        self.X_test = X_test
        self.y_test = y_test
        self.epochs = epochs
        self.batch_size = batch_size

        # 🧩 모델 빌드 (중요!)
        _ = self.model(tf.zeros((1, X_train.shape[1], X_train.shape[2])))

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss={"decoded": "mse", "pred": "binary_crossentropy"},
            loss_weights={"decoded": 1.0, "pred": 0.1},
            metrics={"pred": ["accuracy"]}
        )
        self.model.fit(
            self.X_train,
            {"decoded": self.X_train, "pred": self.y_train_cls},
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=2,
            validation_split=0.2
        )
        return self.model.get_weights(), len(self.X_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        preds = self.model.predict(self.X_test, verbose=0)
        recon = preds["decoded"]
        cls_prob = preds["pred"].flatten()

        mse = np.mean(np.square(self.X_test - recon), axis=(1, 2))
        threshold = np.percentile(mse, 80)
        y_pred_recon = (mse > threshold).astype(int)
        y_pred_cls = (cls_prob > 0.5).astype(int)

        acc_recon = np.mean(y_pred_recon == self.y_test)
        acc_cls = np.mean(y_pred_cls == self.y_test)

        print(f"\n--- Client {self.cid} ---")
        print(f"Recon-Acc={acc_recon:.4f} | Cls-Acc={acc_cls:.4f} | Th={threshold:.4f}")
        
        # ---------- Confusion Matrices ----------
        cm_recon = confusion_matrix(self.y_test, y_pred_recon)
        cm_cls = confusion_matrix(self.y_test, y_pred_cls)

        print("\n📊 [Reconstruction-based Confusion Matrix]")
        print(cm_recon)
        print("\n📊 [Classifier-based Confusion Matrix]")
        print(cm_cls)

        # ---------- Reports ----------
        report_recon = classification_report(
            self.y_test, y_pred_recon,
            target_names=["Normal", "Anomaly"],
            zero_division=0
        )

        report_cls = classification_report(
            self.y_test, y_pred_cls,
            target_names=["Normal", "Anomaly"],
            zero_division=0
        )

        print("\n📊 [Reconstruction-based Report]")
        print(report_recon)
        print("📊 [Classifier-based Report]")
        print(report_cls)

        return float(1 - acc_recon), len(self.X_test), {"acc_cls": float(acc_cls)}

