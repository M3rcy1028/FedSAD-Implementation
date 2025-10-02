from arguments import get_args
from utils import *
from model_aae import TransformerAAE, FLClient  # FLClient class can be used if imported

WEIGHT_PATH = "./rnep/rnep_vae_transformer_weights.h5"
MATRIX_PATH = "./rnep/rnep_cm.png"
RESULT_PATH = "./rnep/rnep_server.txt"
CSV_PATH = "./rnep/rnep_history", 
PNG_PATH = "./rnep/rnep_history.png"
ROC_PATH = "./rnep/rnep_roc.png"

args = get_args()
client_nums = args.client_nums
client_epochs = args.client_epochs
global_rounds = args.server_rounds
batch_size = args.batch_size
percentile = args.percentile
client_models = []

os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

# 각 서버의 엔트로피 계산
def compute_client_entropy(X_local, n_bins=16, eps=1e-12):
    """
    X_local: (n_samples, n_features) numpy array
    방식: 각 feature 별로 히스토그램(빈 n_bins) -> 확률 분포 p -> H = -sum p log p
          각 feature entropy의 평균을 클라이언트 엔트로피로 사용.
    """
    if X_local.shape[0] == 0:
        return 0.0
    feat_entropies = []
    for j in range(X_local.shape[1]):
        col = X_local[:, j]
        # 히스토그램 (밀도가 아닌 counts -> 확률화)
        counts, _ = np.histogram(col, bins=n_bins)
        probs = counts / (counts.sum() + eps)
        probs = probs[probs > 0]
        if probs.size == 0:
            feat_entropies.append(0.0)
        else:
            H = -np.sum(probs * np.log(probs + eps))
            feat_entropies.append(H)
    return float(np.mean(feat_entropies))

# RNEP 로컬 학습
def evaluate_model(model, X_train, X_test, y_test, percentile):
    """모델 평가: train 기반 threshold → test 예측 → accuracy 반환"""
    # Reconstruction error (train)
    X_train_pred = model.predict(X_train, verbose=0)
    train_errors = np.mean(np.square(X_train - X_train_pred), axis=1)
    threshold = np.percentile(train_errors, percentile)

    # Reconstruction error (test)
    X_test_pred = model.predict(X_test, verbose=0)
    recon_errors_test = np.mean(np.square(X_test - X_test_pred), axis=1)
    y_pred = (recon_errors_test > threshold).astype(int)

    acc = np.mean(y_pred == y_test)
    return acc, threshold, y_pred


def local_train(rnd, X_local, X_test_scaled, y_test, client_idx, epochs=50, batch_size=16, verbose=1):
    """클라이언트 로컬 학습 및 평가"""
    model = client_models[client_idx]
    if X_local is None or len(X_local) == 0:
        return 0

    model.compile(optimizer=Adam(0.0001))
    model.fit(X_local, X_local,
              epochs=epochs,
              batch_size=batch_size,
              verbose=verbose,
              validation_split=0.1)

    # 평가
    acc, _, _ = evaluate_model(model, X_local, X_test_scaled, y_test, percentile)

    # 로그 저장
    with open("./rnep/rnep_server_report.txt", "a") as f:
        f.write(f"\n--- Client {client_idx} Round {rnd+1} ---\n")
        f.write(f"Accuracy: {acc:.4f}\n")

    return acc

# RNEP 가중치 교환 함수 (엔트로피 기반)
def mix_weights_by_entropy(weights1, weights2, ent1, ent2):
    s = ent1 + ent2
    if s == 0:
        a1, a2 = 0.5, 0.5
    else:
        a1, a2 = ent1 / s, ent2 / s
    mixed = [a1 * w1 + a2 * w2 for w1, w2 in zip(weights1, weights2)]
    return mixed

def main():
    X_train_scaled, X_test_scaled, y_test = get_datasets_cic()
    client_data = np.array_split(X_train_scaled, client_nums)
    input_dim = X_train_scaled.shape[1]

    # 중앙 모델 초기화
    central_model = TransformerAAE(input_dim)
    _ = central_model(tf.zeros((1, input_dim)), prior_labels=tf.zeros((1,1)))

    # 클라이언트 초기화
    for i in range(client_nums):
        m = TransformerAAE(input_dim)
        _ = m(tf.zeros((1, input_dim)), prior_labels=tf.zeros((1,1)))
        client_models.append(m)

    # 결과 파일 초기화
    open("./rnep/rnep_server_report.txt", "w").write("Server-level Client Reports\n")
    open("./rnep/rnep_central_report.txt", "w").write("Central Report\n")

    print("Start P2P RNEP simulation...")

    best_cid = None
    best_acc = -1

    for rnd in range(global_rounds):
        print(f"\n=== Global Round {rnd+1}/{global_rounds} ===")

        # (1) 클라이언트 로컬 학습
        round_accs = []
        for cid in range(client_nums):
            acc = local_train(rnd, client_data[cid], X_test_scaled, y_test,
                              cid, epochs=client_epochs, batch_size=batch_size, verbose=1)
            round_accs.append(acc)

            # 중앙 모델 기준: 가장 높은 accuracy 가진 client를 추적
            if acc > best_acc:
                best_acc = acc
                best_cid = cid

        avg_acc = np.mean(round_accs)
        print(f"[Round {rnd+1}] Average Client Accuracy: {avg_acc:.4f}")

        with open("./rnep/rnep_central_report.txt", "a") as f:
            f.write(f"[Round {rnd+1}] Average Client Accuracy: {avg_acc:.4f}\n")

    # (2) 가장 잘한 클라이언트 weight로 중앙 모델 업데이트
    central_model.set_weights(client_models[best_cid].get_weights())
    central_model.save_weights(WEIGHT_PATH)
    print(f"✅ Central model weights copied from Client {best_cid} (acc={best_acc:.4f})")

    # (3) 최종 평가
    eval_server(
        central_model,
        X_train_scaled,
        X_test_scaled,
        y_test,
        result_path=RESULT_PATH,
        matrix_path=MATRIX_PATH,
        roc_path=ROC_PATH
    )
    print("Server-level RNEP simulation completed.")

if __name__ == "__main__":
    main()