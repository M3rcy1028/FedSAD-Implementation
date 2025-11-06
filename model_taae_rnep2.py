'''
    Define Variational + Transformer AE model architecture
'''
from utils import *
from arguments import get_args
args = get_args()

BATCH_SIZE = args.batch_size
DROPOUT = args.dropout_rate
PERCENTILE = args.percentile

# 클라이언트 엔트로피 계산
def compute_client_entropy(X_local: np.ndarray, n_bins: int = 16, eps: float = 1e-12) -> float:
    """
    각 feature의 히스토그램 분포로 샤논 엔트로피를 계산하고 평균을 취함.
    """
    if X_local is None or X_local.shape[0] == 0:
        return 0.0
    feat_entropies = []
    for j in range(X_local.shape[1]):
        col = X_local[:, j]
        counts, _ = np.histogram(col, bins=n_bins)
        probs = counts / (counts.sum() + eps)
        probs = probs[probs > 0]
        if probs.size == 0:
            feat_entropies.append(0.0)
        else:
            H = -np.sum(probs * np.log(probs + eps))
            feat_entropies.append(float(H))
    return float(np.mean(feat_entropies))

def mix_weights_by_entropy(
    weights1: List[np.ndarray],
    weights2: List[np.ndarray],
    ent1: float,
    ent2: float,
) -> List[np.ndarray]:
    """
    RNEP 혼합: α = ent1/(ent1+ent2), (1-α) = ent2/(ent1+ent2).
    엔트로피가 높을수록 가중치를 더 많이 반영.
    """
    s = ent1 + ent2
    if s <= 0:
        a1, a2 = 0.5, 0.5
    else:
        a1, a2 = ent1 / s, ent2 / s
    return [a1 * w1 + a2 * w2 for w1, w2 in zip(weights1, weights2)]

# === 기존: SaveEvaluationFedAvg 를 RNEP 버전으로 변경 ===
class SaveEvaluationRNEP(fl.server.strategy.FedAvg):
    """
    FedAvg를 오버라이드하여 RNEP 방식(엔트로피 기반 페어 혼합)으로 집계.
    - 클라이언트 fit() metrics 에 "entropy" 가 반드시 포함되어야 함.
    - 결과 리스트를 무작위 페어링하여 각 페어의 가중치 혼합을 수행.
    - 모든 페어 혼합 결과를 단순 평균하여 최종 글로벌 파라미터로 설정.
    """
    def __init__(self, eval_server_args: Optional[Dict] = None, **kwargs):
        super().__init__(**kwargs)
        self.eval_server_args = eval_server_args
        self.round_history = []
        self.final_parameters = None

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[fl.common.Parameters], Dict[str, float]]:
        if not results:
            return None, {}

        # 1) 결과에서 파라미터와 엔트로피 수집
        entries = []  # [(ndarrays, entropy, num_examples)]
        for _, fit_res in results:
            params_nd = fl.common.parameters_to_ndarrays(fit_res.parameters)
            entropy = float(fit_res.metrics.get("entropy", 0.0))
            n = int(fit_res.num_examples)
            entries.append((params_nd, entropy, n))

        # 2) 무작위 셔플 후 페어링
        idxs = list(range(len(entries)))
        random.shuffle(idxs)

        mixed_param_lists = []  # 각 페어 혼합 결과 (ndarrays)
        i = 0
        while i < len(idxs):
            i1 = idxs[i]
            if i + 1 < len(idxs):
                i2 = idxs[i + 1]
                w1, e1, _ = entries[i1]
                w2, e2, _ = entries[i2]
                mixed = mix_weights_by_entropy(w1, w2, e1, e2)
                mixed_param_lists.append(mixed)
                i += 2
            else:
                # 짝이 없으면 그대로 통과 (odd client)
                w1, _, _ = entries[i1]
                mixed_param_lists.append([w.copy() for w in w1])
                i += 1

        # 3) 모든 혼합 결과를 단순 평균하여 최종 글로벌 파라미터 생성
        #    (리스트[텐서]들에 대해 element-wise 평균)
        num_buckets = len(mixed_param_lists)
        agg = []
        for layer_idx in range(len(mixed_param_lists[0])):
            stacked = np.stack([mixed_param_lists[b][layer_idx] for b in range(num_buckets)], axis=0)
            agg.append(stacked.mean(axis=0))

        aggregated_parameters = fl.common.ndarrays_to_parameters(agg)
        self.final_parameters = aggregated_parameters

        # 서버 메트릭(선택)
        metrics = {"rnp_pairs": len(mixed_param_lists)}

        return aggregated_parameters, metrics

    def evaluate(
        self,
        server_round: int,
        parameters: fl.common.Parameters,
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        """서버 측 공통 검증셋 평가 (기존 SaveEvaluationFedAvg와 동일 로직)."""
        if self.eval_server_args is None:
            return None

        model = self.eval_server_args['model']
        model.set_weights(fl.common.parameters_to_ndarrays(parameters))
        
        X_train_scaled = self.eval_server_args['X_train_scaled']
        X_test_scaled = self.eval_server_args['X_test_scaled']
        y_test = self.eval_server_args['y_test']
        result_path = self.eval_server_args['result_path']
        matrix_path = self.eval_server_args['matrix_path']  # 사용하지 않더라도 인자 유지

        # 1) train 기반 threshold 산출
        pred_train = model.predict(X_train_scaled, verbose=2)
        recon_train = np.mean(np.square(X_train_scaled - pred_train), axis=1)
        threshold = np.percentile(recon_train, PERCENTILE)

        # 2) test 평가
        pred_test = model.predict(X_test_scaled, verbose=2)
        recon_test = np.mean(np.square(X_test_scaled - pred_test), axis=1)
        y_pred = (recon_test > threshold).astype(int)
        acc = float(np.mean(y_pred == y_test))

        # 3) 리포트 파일 저장
        report = classification_report(
            y_test, y_pred,
            target_names=["Normal", "Anomaly"],
            zero_division=0
        )
        with open(result_path, "a") as f:
            f.write(f"\n[Round {server_round}] Server Evaluation Report\n")
            f.write(f"Accuracy: {acc:.4f}\n")
            f.write(report)

        # 4) Flower에 반환할 손실/메트릭
        loss = float(mean_squared_error(X_test_scaled, pred_test))
        return loss, {"loss": loss, "acc": acc}
        
@register_keras_serializable(package="Custom")
class GradientReversal(layers.Layer):
    def __init__(self, lambd=1.0, **kwargs):
        super().__init__(**kwargs)
        self.lambd = tf.Variable(lambd, trainable=False, dtype=tf.float32)

    def call(self, x):
        @tf.custom_gradient
        def _grl(x):
            def grad(dy):
                return -self.lambd * dy
            return x, grad
        return _grl(x)

    def set_lambda(self, value: float):
        self.lambd.assign(float(value))

# ---------------------------
# Transformer-based Adversarial AutoEncoder
# ---------------------------
@register_keras_serializable(package="Custom")
class TransformerAAE(tf.keras.Model):
    def __init__(self, input_dim, latent_dim=64, embed_dim=128, num_heads=4, ff_dim=128,
                 num_encoder_layers=3, num_decoder_layers=3,
                 dropout_rate=DROPOUT, beta=1.0, grl_lambda=1.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.beta = beta

        # -------------------
        # Encoder
        # -------------------
        self.embedding = layers.Dense(embed_dim)
        self.expand = layers.Reshape((1, embed_dim))

        self.encoder_blocks = []
        for _ in range(num_encoder_layers):
            attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
            norm1 = layers.LayerNormalization()
            drop1 = layers.Dropout(dropout_rate)
            ffn = tf.keras.Sequential([
                layers.Dense(ff_dim, activation="relu"),
                layers.Dense(embed_dim)
            ])
            norm2 = layers.LayerNormalization()
            drop2 = layers.Dropout(dropout_rate)
            self.encoder_blocks.append((attn, norm1, drop1, ffn, norm2, drop2))

        self.flatten = layers.Flatten()
        self.latent = layers.Dense(latent_dim)

        # -------------------
        # Decoder
        # -------------------
        self.dec_dense = layers.Dense(embed_dim)
        self.expand_dec = layers.Reshape((1, embed_dim))

        self.decoder_blocks = []
        for _ in range(num_decoder_layers):
            attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
            norm1 = layers.LayerNormalization()
            drop1 = layers.Dropout(dropout_rate)
            ffn = tf.keras.Sequential([
                layers.Dense(ff_dim, activation="relu"),
                layers.Dense(embed_dim)
            ])
            norm2 = layers.LayerNormalization()
            drop2 = layers.Dropout(dropout_rate)
            self.decoder_blocks.append((attn, norm1, drop1, ffn, norm2, drop2))

        self.output_layer = layers.Dense(input_dim)

        # -------------------
        # Adversarial Regularizer
        # -------------------
        self.grl = GradientReversal(lambd=grl_lambda)
        self.discriminator = tf.keras.Sequential([
            layers.Dense(128), layers.LeakyReLU(),
            layers.Dense(64), layers.LeakyReLU(),
            layers.Dense(1, activation="sigmoid")
        ])

        # Loss 함수
        self.mse = tf.keras.losses.MeanSquaredError()
        self.bce = tf.keras.losses.BinaryCrossentropy()

    def call(self, inputs, training=None, prior_labels=None):
        # -------------------
        # Encoder
        # -------------------
        x = self.embedding(inputs)
        x = self.expand(x)
        for attn, norm1, drop1, ffn, norm2, drop2 in self.encoder_blocks:
            attn_out = attn(x, x)
            x = drop1(norm1(x + attn_out), training=training)
            ffn_out = ffn(x)
            x = drop2(norm2(x + ffn_out), training=training)

        flat = self.flatten(x)
        z = self.latent(flat)

        # -------------------
        # Decoder
        # -------------------
        x_dec = self.dec_dense(z)
        x_dec = self.expand_dec(x_dec)
        for attn, norm1, drop1, ffn, norm2, drop2 in self.decoder_blocks:
            attn_out = attn(x_dec, x_dec)
            x_dec = drop1(norm1(x_dec + attn_out), training=training)
            ffn_out = ffn(x_dec)
            x_dec = drop2(norm2(x_dec + ffn_out), training=training)

        output = self.output_layer(tf.squeeze(x_dec, axis=1))

        # -------------------
        # Loss
        # -------------------
        recon_loss = self.mse(inputs, output)

        if prior_labels is not None:  # prior 분포와 맞추기 (예: 정규분포 샘플 vs 인코딩 z)
            z_grl = self.grl(z)
            d_pred = self.discriminator(z_grl, training=training)
            adv_loss = self.bce(prior_labels, d_pred)
            total_loss = recon_loss + self.beta * adv_loss
            self.add_loss(total_loss)
        else:
            self.add_loss(recon_loss)

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
        self.model.compile(optimizer=Adam(0.0001))
        self.model.fit(self.X_train, self.X_train,
                       epochs=self.epochs,
                       batch_size=BATCH_SIZE,
                       verbose=2,
                       validation_split=0.2)
        ent = compute_client_entropy(self.X_train)   # ★ 엔트로피 계산
        return self.model.get_weights(), len(self.X_train), {"entropy": float(ent)}

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
        with open("./rnep_frame_revised3/client.txt", "a") as f:
            f.write(f"\n--- Client {self.cid} Evaluation ---\n")
            f.write(f"Threshold: {threshold:.4f}\n")
            f.write(f"Accuracy: {acc:.4f}\n")
            f.write(report + "\n")

        return float(1 - acc), len(self.X_test), {}