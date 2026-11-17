'''
    Define Variational + Transformer AE model architecture
'''
import os
import numpy as np
import random
from typing import List, Tuple, Dict, Optional
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, roc_curve, roc_auc_score
from tensorflow.keras.saving import register_keras_serializable
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
import flwr as fl
from utils.arguments import get_taae_args

args = get_taae_args()

BATCH_SIZE = args.batch_size
DROPOUT = args.dropout_rate
PERCENTILE = args.percentile


def get_datasets_zenodo(
    base_dir=os.path.expanduser("~/IDS_Datasets/ITU_SDN_ULAK/raw"),
    random_seed=123,
    max_train_samples=150_000,
    balance_test=True,
    return_metadata=False,
):
    """Load ITU/ULAK original CSVs, preserving their published train/test split.

    Only BENIGN rows from Train_ULAK.csv train the autoencoder. Missing,
    infinite, and negative values use medians fitted on those training rows;
    MinMax scaling is also fitted there. Test balancing selects normal rows
    without replacement and retains the attack-class proportions. No log1p
    transform or label-based feature selection is applied to this dataset.
    """
    from pathlib import Path
    import pandas as pd
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler

    if max_train_samples is not None and max_train_samples < 1:
        raise ValueError("max_train_samples must be positive or None")
    base_dir = Path(base_dir)

    def read_file(name):
        path = base_dir / name
        try:
            frame = pd.read_csv(path, low_memory=False)
        except UnicodeDecodeError:
            frame = pd.read_csv(path, encoding="cp1252", low_memory=False)
        frame.columns = frame.columns.str.strip()
        if frame.columns.duplicated().any() or "Label" not in frame:
            raise ValueError(f"Invalid or duplicate column names in {path}")
        labels = frame.pop("Label").astype("string").str.strip()
        labels = labels.str.replace(r"\s+", " ", regex=True)
        if labels.isna().any() or labels.str.lower().isin(
                ["", "no label", "unknown", "nan"]).any():
            raise ValueError(f"Unannotated labels in {path}")
        values = frame.apply(pd.to_numeric, errors="coerce").astype(np.float64)
        invalid = ~np.isfinite(values.to_numpy()) | (values.to_numpy() < 0)
        count = int(invalid.sum())
        values = values.mask(invalid)
        return values, labels, count

    train, train_labels, invalid_train = read_file("Train_ULAK.csv")
    test, test_labels, invalid_test = read_file("Test_ULAK.csv")
    if train.columns.tolist() != test.columns.tolist():
        raise ValueError("Train/test feature names and order differ")
    normal_train_mask = train_labels.str.upper().eq("BENIGN").to_numpy(dtype=bool)
    if not normal_train_mask.any():
        raise ValueError("Train_ULAK.csv contains no BENIGN training rows")
    train_counts = train_labels.value_counts().to_dict()
    test_counts = test_labels.value_counts().to_dict()
    original_train_rows = len(train)
    original_test_rows = len(test)
    train = train.loc[normal_train_mask]
    # CICFlowMeter exports Fwd Header Length twice in this original release.
    # Identify the redundant export using training data only, retaining all
    # other columns (including features constant in normal training traffic).
    dropped = []
    for column in list(train.columns):
        if column.endswith(".1") and column[:-2] in train:
            if train[column].equals(train[column[:-2]]):
                dropped.append(column)
    train = train.drop(columns=dropped)
    test = test.drop(columns=dropped)
    train = train.sample(frac=1, random_state=random_seed)
    if max_train_samples is not None:
        train = train.iloc[:max_train_samples]

    normal_test = np.flatnonzero(test_labels.str.upper().eq("BENIGN"))
    attack_test = np.flatnonzero(~test_labels.str.upper().eq("BENIGN"))
    if not len(normal_test) or not len(attack_test):
        raise ValueError("Test_ULAK.csv must include normal and attack rows")
    if balance_test:
        count = min(len(normal_test), len(attack_test))
        rng = np.random.default_rng(random_seed)
        if len(normal_test) > count:
            normal_test = rng.choice(normal_test, count, replace=False)
        if len(attack_test) > count:
            attack_test, _ = train_test_split(
                attack_test, train_size=count, random_state=random_seed,
                stratify=test_labels.iloc[attack_test].to_numpy(),
            )
    selected = np.r_[normal_test, attack_test]
    selected = np.random.default_rng(random_seed).permutation(selected)
    test = test.iloc[selected]
    heldout_labels = test_labels.iloc[selected].to_numpy(dtype=str)
    y_test = np.asarray([0 if label.upper() == "BENIGN" else 1
                         for label in heldout_labels], dtype=np.int64)

    train_hashes = pd.util.hash_pandas_object(train, index=False).to_numpy()
    test_hashes = pd.util.hash_pandas_object(test, index=False).to_numpy()
    overlap = np.isin(test_hashes, np.unique(train_hashes))
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    train_values = imputer.fit_transform(train)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(train_values).astype(np.float32)
    X_test = scaler.transform(imputer.transform(test)).astype(np.float32)
    if not np.isfinite(X_train).all() or not np.isfinite(X_test).all():
        raise ValueError("Nonfinite values remain after preprocessing")
    report = {
        "dataset": "ITU_SDN_ULAK", "source_record": "https://zenodo.org/records/15303565",
        "train_file": "Train_ULAK.csv", "test_file": "Test_ULAK.csv",
        "original_train_rows": original_train_rows, "original_test_rows": original_test_rows,
        "original_train_class_counts": train_counts, "original_test_class_counts": test_counts,
        "normal_train_available": int(normal_train_mask.sum()), "training_rows": len(train),
        "normal_test_rows": int((y_test == 0).sum()), "attack_test_rows": int((y_test == 1).sum()),
        "test_class_counts": pd.Series(heldout_labels).value_counts().to_dict(),
        "split_seed": random_seed, "official_train_test_split_preserved": True,
        "test_balanced": balance_test, "max_train_samples": max_train_samples,
        "input_dim": X_train.shape[1], "feature_names": train.columns.tolist(),
        "dropped_redundant_columns": dropped,
        "invalid_cells_before_imputation": {"train_file": invalid_train, "test_file": invalid_test},
        "imputation": "negative/nonfinite -> normal-training median; all-missing training columns -> 0",
        "scaling": "MinMaxScaler fitted on sampled normal training rows only; test is not clipped",
        "log_transform": False,
        "training_constant_features": train.columns[np.ptp(train_values, axis=0) == 0].tolist(),
        "duplicate_training_feature_vectors": int(pd.Index(train_hashes).duplicated().sum()),
        "test_feature_vectors_also_present_in_normal_train": int(overlap.sum()),
        "feature_overlap_note": "Identical feature vectors are reported, not relabeled or removed; absent flow IDs/timestamps prevent proving identical source flows.",
    }
    print(f"[ITU/ULAK] Normal-only train: {X_train.shape}; test: {X_test.shape}; "
          f"Normal={report['normal_test_rows']}, Attack={report['attack_test_rows']}", flush=True)
    if return_metadata:
        return X_train, X_test, y_test, {
            "report": report, "imputer": imputer, "scaler": scaler,
            "train_source_rows": train.index.to_numpy(), "test_source_rows": test.index.to_numpy(),
            "test_attack_labels": heldout_labels,
        }
    return X_train, X_test, y_test


# Compute client entropy
def compute_client_entropy(X_local: np.ndarray, n_bins: int = 16, eps: float = 1e-12) -> float:
    """
    Computes Shannon entropy from the histogram distribution of each feature and averages across features.
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
    RNEP mixing: α = ent1/(ent1+ent2), (1-α) = ent2/(ent1+ent2).
    Higher entropy results in a larger contribution to the mixed weights.
    """
    s = ent1 + ent2
    if s <= 0:
        a1, a2 = 0.5, 0.5
    else:
        a1, a2 = ent1 / s, ent2 / s
    return [a1 * w1 + a2 * w2 for w1, w2 in zip(weights1, weights2)]

# === Original: SaveEvaluationFedAvg replaced with RNEP version ===
class SaveEvaluationRNEP(fl.server.strategy.FedAvg):
    """
    Overrides FedAvg to aggregate using the RNEP method (entropy-based pair mixing).
    - Client fit() metrics must include "entropy".
    - Results list is randomly paired and each pair's weights are mixed.
    - All pair-mixed results are simply averaged to produce the final global parameters.
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

        # 1) Collect parameters and entropy from the client results
        entries = []  # [(ndarrays, entropy, num_examples)]
        for _, fit_res in results:
            params_nd = fl.common.parameters_to_ndarrays(fit_res.parameters)
            entropy = float(fit_res.metrics.get("entropy", 0.0))
            n = int(fit_res.num_examples)
            entries.append((params_nd, entropy, n))

        # 2) Shuffle and pair the clients
        idxs = list(range(len(entries)))
        random.shuffle(idxs)

        mixed_param_lists = []  # mixed result per pair (ndarrays)
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
                # pass through as-is if unpaired (odd client)
                w1, _, _ = entries[i1]
                mixed_param_lists.append([w.copy() for w in w1])
                i += 1

        # 3) Average all mixed results into the new global parameters
        #    (element-wise mean over the lists of tensors)
        num_buckets = len(mixed_param_lists)
        agg = []
        for layer_idx in range(len(mixed_param_lists[0])):
            stacked = np.stack([mixed_param_lists[b][layer_idx] for b in range(num_buckets)], axis=0)
            agg.append(stacked.mean(axis=0))

        aggregated_parameters = fl.common.ndarrays_to_parameters(agg)
        self.final_parameters = aggregated_parameters

        # server metrics (optional)
        metrics = {"rnp_pairs": len(mixed_param_lists)}

        return aggregated_parameters, metrics

    def evaluate(
        self,
        server_round: int,
        parameters: fl.common.Parameters,
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        """Server-side common validation set evaluation (same logic as the original SaveEvaluationFedAvg)."""
        if self.eval_server_args is None:
            return None

        model = self.eval_server_args['model']
        model.set_weights(fl.common.parameters_to_ndarrays(parameters))
        
        X_train_scaled = self.eval_server_args['X_train_scaled']
        X_test_scaled = self.eval_server_args['X_test_scaled']
        y_test = self.eval_server_args['y_test']
        result_path = self.eval_server_args['result_path']
        matrix_path = self.eval_server_args['matrix_path']  # keep the argument even if unused

        # 1) Threshold from training reconstruction errors
        prediction_batch_size = self.eval_server_args.get("prediction_batch_size")
        pred_train = model.predict(X_train_scaled, batch_size=prediction_batch_size, verbose=2)
        recon_train = np.mean(np.square(X_train_scaled - pred_train), axis=1)
        threshold = np.percentile(recon_train, self.eval_server_args.get("percentile", PERCENTILE))

        # 2) Evaluate on the test set
        pred_test = model.predict(X_test_scaled, batch_size=prediction_batch_size, verbose=2)
        recon_test = np.mean(np.square(X_test_scaled - pred_test), axis=1)
        y_pred = (recon_test > threshold).astype(int)
        acc = float(np.mean(y_pred == y_test))

        # 3) Save the report
        report = classification_report(
            y_test, y_pred,
            target_names=["Normal", "Anomaly"],
            zero_division=0
        )
        with open(result_path, "a") as f:
            f.write(f"\n[Round {server_round}] Server Evaluation Report\n")
            f.write(f"Accuracy: {acc:.4f}\n")
            f.write(report)

        # 4) Loss / metrics returned to Flower
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
    def __init__(self, input_dim, latent_dim=32, embed_dim=128, num_heads=4, ff_dim=128,
                 num_encoder_layers=3, num_decoder_layers=3,
                 dropout_rate=DROPOUT, beta=0.3, grl_lambda=1.0): # adjusted to 0.3 for CIC
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

        # Loss functions
        self.mse = tf.keras.losses.MeanSquaredError()
        self.bce = tf.keras.losses.BinaryCrossentropy()
        # Build D before Flower first calls get_weights()/set_weights().
        self.discriminator.build((None, self.latent_dim))
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.recon_tracker = tf.keras.metrics.Mean(name="recon_loss")
        self.adv_tracker = tf.keras.metrics.Mean(name="adv_loss")

    def call(self, inputs, training=None, prior_labels=None, return_latent=False):
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
        # Optional latent output
        # -------------------
        # Keep prior_labels only for existing initialization-call compatibility.
        # Training losses are computed in train_step(), not in call().
        if return_latent:
            return output, z
        return output

    @property
    def metrics(self):
        return [self.loss_tracker, self.recon_tracker, self.adv_tracker]

    def _loss_terms(self, x, training):
        reconstruction, z = self(x, training=training, return_latent=True)
        target = tf.cast(x, reconstruction.dtype)
        recon_loss = tf.reduce_mean(tf.square(target - reconstruction))

        # Prior samples are positive; encoded samples are negative for D.
        z_prior = tf.random.normal(tf.shape(z), dtype=z.dtype)
        d_prior = self.discriminator(z_prior, training=training)
        d_encoded = self.discriminator(self.grl(z), training=training)
        adv_loss = 0.5 * (
            self.bce(tf.ones_like(d_prior), d_prior)
            + self.bce(tf.zeros_like(d_encoded), d_encoded)
        )
        total_loss = recon_loss + tf.cast(self.beta, recon_loss.dtype) * adv_loss
        return total_loss, recon_loss, adv_loss

    def _update_loss_metrics(self, losses, batch_size):
        for tracker, value in zip(self.metrics, losses):
            tracker.update_state(value, sample_weight=tf.cast(batch_size, tf.float32))
        return {tracker.name: tracker.result() for tracker in self.metrics}

    def train_step(self, data):
        # Autoencoding uses x as its target; no attack labels are required.
        x, _, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        if sample_weight is not None:
            raise ValueError("This training loop does not support sample weights.")
        with tf.GradientTape() as tape:
            losses = self._loss_terms(x, training=True)
        variables = self.trainable_variables
        gradients = tape.gradient(losses[0], variables)
        disconnected = [v.name for g, v in zip(gradients, variables) if g is None]
        if disconnected:
            raise RuntimeError("Disconnected training variables: " + ", ".join(disconnected))
        self.optimizer.apply_gradients(zip(gradients, variables))
        return self._update_loss_metrics(losses, tf.shape(x)[0])

    def test_step(self, data):
        x, _, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        if sample_weight is not None:
            raise ValueError("This training loop does not support sample weights.")
        losses = self._loss_terms(x, training=False)
        return self._update_loss_metrics(losses, tf.shape(x)[0])

class FLClient(fl.client.NumPyClient):
    def __init__(self, cid, model, X_train, X_test, y_test, epochs=50, learning_rate=0.00001,
                 batch_size=None, result_path=None, prediction_batch_size=None, percentile=None):
        self.cid = cid
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_test = y_test
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = BATCH_SIZE if batch_size is None else batch_size
        self.result_path = "./FedSAD_Results/client.txt" if result_path is None else result_path
        self.prediction_batch_size = prediction_batch_size
        self.percentile = PERCENTILE if percentile is None else percentile
        self.model(tf.zeros((1, X_train.shape[1])))   # initialize with initial weights

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters) # load the latest server weights for training
        self.model.compile(optimizer=Adam(self.learning_rate))
        print(f"[Client {self.cid}] learning_rate={self.learning_rate}, "
              f"latent_dim={getattr(self.model, 'latent_dim', 'unknown')}", flush=True)
        self.model.fit(self.X_train, self.X_train,
                       epochs=self.epochs,
                       batch_size=self.batch_size,
                       verbose=2,
                       validation_split=0.2)
        ent = compute_client_entropy(self.X_train)   # ★ compute entropy
        return self.model.get_weights(), len(self.X_train), {"entropy": float(ent)}

    def evaluate(self, parameters, config):
        # get local results
        self.model.set_weights(parameters)
        local_pred = self.model.predict(self.X_train, batch_size=self.prediction_batch_size, verbose=0)
        local_recon_errors = np.mean(np.square(self.X_train - local_pred), axis=1)
        threshold = np.percentile(local_recon_errors, self.percentile)
        
        # evaluate performance on test dataset
        pred_test = self.model.predict(self.X_test, batch_size=self.prediction_batch_size, verbose=0)
        recon_errors_test = np.mean(np.square(self.X_test - pred_test), axis=1)
        y_pred = (recon_errors_test > threshold).astype(int)

        # generate evaluation metrics
        acc = np.mean(y_pred == self.y_test)
        report = classification_report(
            self.y_test, y_pred,
            target_names=["Normal", "Anomaly"],
            zero_division=0
        )

        # ✅ console output
        print(f"\n--- Client {self.cid} Evaluation ---")
        print(f"Accuracy: {acc:.4f}")
        print(report)

        # ✅ save to file
        with open(self.result_path, "a") as f:
            f.write(f"\n--- Client {self.cid} Evaluation ---\n")
            f.write(f"Accuracy: {acc:.4f}\n")
            f.write(report + "\n")

        return float(1 - acc), len(self.X_test), {}
