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
    def __init__(self, eval_server_args, **kwargs):
        super().__init__(**kwargs)
        self.eval_server_args = eval_server_args
        self.round_history = []
        self.final_parameters = None  # 마지막 집계된 파라미터 저장

    # 서버 평가는 그대로 두되, aggregate_fit에서 마지막 weight 저장
    def aggregate_fit(self, rnd, results, failures):
        aggregated_parameters, metrics = super().aggregate_fit(rnd, results, failures)
        if aggregated_parameters is not None:
            self.final_parameters = aggregated_parameters  # 마지막 weight 저장
        return aggregated_parameters, metrics
        
    def evaluate(
        self,
        server_round: int,
        parameters: fl.common.Parameters,
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters on the server-side validation set."""
        if self.eval_server_args is None:
            return None

        # 1. Evaluate the global model
        model = self.eval_server_args['model']
        model.set_weights(fl.common.parameters_to_ndarrays(parameters))
        
        X_train_scaled = self.eval_server_args['X_train_scaled']
        X_test_scaled = self.eval_server_args['X_test_scaled']
        y_test = self.eval_server_args['y_test']
        result_path = self.eval_server_args['result_path']
        matrix_path = self.eval_server_args['matrix_path']

        # Perform evaluation directly in this method
        # 1. Calculate threshold using training data
        pred_train = model.predict(X_train_scaled, verbose=0)
        recon_errors_train = np.mean(np.square(X_train_scaled - pred_train), axis=1)
        # Assuming args.percentile is available
        threshold = np.percentile(recon_errors_train, 95)
        
        # 2. Evaluate on the common test dataset
        pred_test = model.predict(X_test_scaled, verbose=0)
        recon_errors_test = np.mean(np.square(X_test_scaled - pred_test), axis=1)

        # 3. Classify based on the calculated threshold
        y_pred = (recon_errors_test > threshold).astype(int)
        acc = np.mean(y_pred == y_test) 

        # 4. Save the report to file only
        report = classification_report(
            y_test, y_pred,
            target_names=["Normal", "Anomaly"],
            zero_division=0
        )
        
        with open(result_path, "a") as f:
            f.write(f"\n[Round {server_round}] Server Evaluation Report\n")
            f.write(f"Accuracy: {acc:.4f}\n")
            f.write(report)
        
        # 5. Return a loss and metrics to the Flower framework
        loss = mean_squared_error(X_test_scaled, pred_test)
        
        metrics = {"loss": loss}
        
        return loss, metrics

def transformer_block(embed_dim, num_heads, ff_dim, dropout_rate):
    inputs = layers.Input(shape=(None, embed_dim))
    attn_output = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=embed_dim // num_heads  # ✅ 수정: head당 차원 축소
    )(inputs, inputs)
    x = layers.LayerNormalization()(inputs + attn_output)
    x = layers.Dropout(dropout_rate)(x)

    ffn_output = layers.Dense(ff_dim, activation='relu')(x)
    ffn_output = layers.Dense(embed_dim)(ffn_output)
    x = layers.LayerNormalization()(x + ffn_output)
    x = layers.Dropout(dropout_rate)(x)

    return tf.keras.Model(inputs, x, name="TransformerBlock")

# Train model 
@register_keras_serializable(package="Custom")
class TransformerVAE(tf.keras.Model):
    def __init__(self, 
                input_dim, 
                latent_dim=32, 
                embed_dim=128, 
                num_heads=4, 
                ff_dim=128, 
                num_layers=3,
                dropout_rate=DROPOUT, 
                beta=0.001):

        super().__init__()
        self.latent_dim = latent_dim
        self.beta = beta

        # Encoder
        self.embedding = layers.Dense(embed_dim)
        self.expand = layers.Reshape((-1, embed_dim))  # sequence-aware
        self.encoder_blocks = [
            transformer_block(embed_dim, num_heads, ff_dim, dropout_rate)
            for _ in range(num_layers)
        ]
        self.flatten = layers.Flatten()
        self.dense_mu = layers.Dense(latent_dim)
        self.dense_log_var = layers.Dense(latent_dim)

        # Decoder
        self.dense_dec = layers.Dense(embed_dim)
        self.expand_dec = layers.Reshape((-1, embed_dim))
        self.decoder_blocks = [
            transformer_block(embed_dim, num_heads, ff_dim, dropout_rate)
            for _ in range(num_layers)
        ]
        self.output_layer = layers.Dense(input_dim)

    def sample(self, mu, log_var):
        eps = tf.random.normal(shape=tf.shape(mu))
        return mu + tf.exp(0.5 * log_var) * eps

    def call(self, inputs):
        # Encoder
        x = self.embedding(inputs)
        x = self.expand(x)
        for block in self.encoder_blocks:
            x = block(x)
        x_flat = self.flatten(x)

        mu = self.dense_mu(x_flat)
        log_var = self.dense_log_var(x_flat)
        z = self.sample(mu, log_var)

        # Decoder
        x_dec = self.dense_dec(z)
        x_dec = self.expand_dec(x_dec)
        for block in self.decoder_blocks:
            x_dec = block(x_dec)

        output = self.output_layer(tf.squeeze(x_dec, axis=1))

        # Losses
        recon_loss = tf.reduce_mean(tf.square(inputs - output))
        kl_loss = -0.5 * tf.reduce_mean(1 + log_var - tf.square(mu) - tf.exp(log_var))
        total_loss = recon_loss + self.beta * kl_loss
        self.add_loss(total_loss)

        return output

class FLClient(fl.client.NumPyClient):
    def __init__(self, cid, model, X_train, X_test, y_test, epochs=50):
        self.cid = cid
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_test = y_test
        self.epochs = epochs
        self.model(tf.zeros((1, X_train.shape[1])))   # 초기 weight로 초기화

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters) # 서버의 최신 가중치를 가져와 학습함
        self.model.compile(optimizer=Adam(0.001))
        self.model.fit(self.X_train, self.X_train,
                       epochs=self.epochs,
                       batch_size=BATCH_SIZE,
                       verbose=0,
                       validation_split=0.2)
        return self.model.get_weights(), len(self.X_train), {}

    def evaluate(self, parameters, config):
        # 로컬 결과 얻기
        self.model.set_weights(parameters)
        local_pred = self.model.predict(self.X_train, verbose=0)
        local_recon_errors = np.mean(np.square(self.X_train - local_pred), axis=1)
        threshold = np.percentile(local_recon_errors, PERCENTILE)
        
        # test 데이터셋으로 성능 평가하기
        pred_test = self.model.predict(self.X_test, verbose=0)
        recon_errors_test = np.mean(np.square(self.X_test - pred_test), axis=1)
        y_pred = (recon_errors_test > threshold).astype(int)

        # 평가 매트릭 생성
        acc = np.mean(y_pred == self.y_test)
        report = classification_report(
            self.y_test, y_pred,
            target_names=["Normal", "Anomaly"],
            zero_division=0
        )

        # ✅ 콘솔 출력
        print(f"\n--- Client {self.cid} Evaluation ---")
        print(f"Accuracy: {acc:.4f}")
        print(report)

        # ✅ 파일 저장
        with open("client.txt", "a") as f:
            f.write(f"\n--- Client {self.cid} Evaluation ---\n")
            f.write(f"Accuracy: {acc:.4f}\n")
            f.write(report + "\n")

        return float(1 - acc), len(self.X_test), {}