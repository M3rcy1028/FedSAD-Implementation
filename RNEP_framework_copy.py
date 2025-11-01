from arguments import get_args
from utils import *
from model_taae_rnep_copy import SaveEvaluationRNEP, TransformerAAE, FLClient  # FLClient class can be used if imported

os.makedirs("./rnep_frame_revised2", exist_ok=True)
WEIGHT_PATH = "./rnep_frame_revised2/rnep_frame_aae_transformer_weights.h5"
MATRIX_PATH = "./rnep_frame_revised2/rnep_frame_cm.png"
RESULT_PATH = "./rnep_frame_revised2/rnep_frame_server.txt"
ROC_PATH = "./rnep_frame_revised2/rnep_frame_roc.png"
CSV_PATH = "./rnep_frame_revised2/rnep_frame_history"
PNG_PATH = "./rnep_frame_revised2/rnep_frame_history.png"

os.environ["CUDA_VISIBLE_DEVICES"] = "2" 

def main():
    args = get_args()
    
    # 🔽 [수정] get_datasets_cic가 5개 값을 반환
    X_train_scaled, X_val_scaled, y_val, X_test_scaled, y_test = get_datasets_cic_val()
    
    client_data = np.array_split(X_train_scaled, args.client_nums)
    
    # 서버 평가를 위한 모델/데이터 준비
    input_dim = X_train_scaled.shape[1]
    print(input_dim)
    central_model = TransformerAAE(input_dim)
    _ = central_model(tf.zeros((1, input_dim)), prior_labels=tf.zeros((1,1)))
    central_model.compile(optimizer=Adam(0.0001), loss="mse")

    # # 🔹 서버 모델만 pretrain weight 로드
    # PRETRAIN_PATH = "rnep_frame_251021/rnep_frame_aae_transformer_weights.h5"
    # central_model.load_weights(PRETRAIN_PATH)
    # print(f"[Server] Loaded pre-trained weights from {PRETRAIN_PATH}")

    eval_server_args = {
        "model": central_model,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled, # 🔽 [수정] utils에서 분할된 (50%) 테스트셋
        "y_test": y_test,               # 🔽 [수정] utils에서 분할된 (50%) 테스트 레이블
        "result_path": RESULT_PATH,
        "matrix_path": MATRIX_PATH,
    }

    strategy = SaveEvaluationRNEP(
        eval_server_args=eval_server_args,
        fraction_fit=0.8,
        fraction_evaluate=0.8,
        min_fit_clients=args.client_nums,
        min_evaluate_clients=args.client_nums,
        min_available_clients=args.client_nums,
        evaluate_fn=None
    )

    # client_fn as closure: captures client_data, X_test_scaled, y_test, input_dim
    def client_fn(cid: str):
        cid_int = int(cid)
        client_model = TransformerAAE(input_dim)
        _ = client_model(tf.zeros((1, input_dim)), prior_labels=tf.zeros((1,1)))
        
        # PRETRAIN_PATH = "rnep_frame_251021/rnep_frame_aae_transformer_weights.h5"
        # try:
        #     client_model.load_weights(PRETRAIN_PATH)
        #     print(f"[Client {cid_int}] Loaded pre-trained weights from {PRETRAIN_PATH}")
        # except Exception as e:
        #     print(f"[Client {cid_int}] Warning: failed to load weights from {PRETRAIN_PATH} — {e}")
        
        # 🔽 [수정] FLClient 생성자에 X_val_scaled, y_val 추가
        return FLClient(
            cid_int,
            client_model,
            client_data[cid_int],
            X_val_scaled,       # 🔽 [수정]
            y_val,              # 🔽 [수정]
            X_test_scaled,
            y_test,
            epochs=args.client_epochs
        )

    # Clear previous results file
    with open(RESULT_PATH, "w") as f:
        f.write("[Server Evaluation Report]\n")

    # start_simulation with safe settings
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=args.client_nums,
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=args.server_rounds),
        client_resources={"num_cpus": 1},
        # num_parallel_clients=args.num_parallel_clients,   # 권장 1~2
        ray_init_args={"include_dashboard": False, "ignore_reinit_error": True},   
        # client_fn_eval=client_fn 
    )

    # 최종 학습된 weight를 central_model에 적용
    if strategy.final_parameters is not None:
        from flwr.common import parameters_to_ndarrays
        final_weights = parameters_to_ndarrays(strategy.final_parameters)
        central_model.set_weights(final_weights)
    
    save_and_plot_history(
        history, 
        csv_path=CSV_PATH, 
        png_path=PNG_PATH
    )

    eval_server(
        central_model,
        X_train_scaled,
        X_test_scaled, # 🔽 [수정] 분할된 (50%) 테스트셋으로 최종 평가
        y_test,        # 🔽 [수정] 분할된 (50%) 테스트 레이블로 최종 평가
        result_path=RESULT_PATH,
        matrix_path=MATRIX_PATH,
        roc_path=ROC_PATH
    )
    
    # 5. Save the trained model weights
    central_model.save_weights(WEIGHT_PATH)
    print(f"\nModel weights saved to {WEIGHT_PATH}")

if __name__ == "__main__":
    main()
