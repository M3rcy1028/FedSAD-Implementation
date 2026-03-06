from arguments import get_args
from utils import *
from model_aae import SaveEvaluationFedAvg, TransformerAAE, FLClient  # FLClient class can be used if imported

WEIGHT_PATH = "./avg/avg_vae_transformer_weights.h5"
MATRIX_PATH = "./avg/avg_cm.png"
RESULT_PATH = "./avg/avg_server.txt"
ROC_PATH = "./avg/avg_roc.png"

os.environ["CUDA_VISIBLE_DEVICES"] = "1" 

def main():
    args = get_args()
    X_train_scaled, X_test_scaled, y_test = get_datasets_cic()
    client_data = np.array_split(X_train_scaled, args.client_nums)
    input_dim = X_train_scaled.shape[1]

    # Global model (only for initial weights / final evaluation)
    global_model = TransformerAAE(input_dim)
    _ = global_model(tf.zeros((1, input_dim)), prior_labels=tf.zeros((1,1)))
    global_model.compile(optimizer=Adam(0.0001), loss="mse")
    # try:
    #     global_model.load_weights("./initial_weight.h5")
    #     print(f"✅ Loaded weights.")
    # except Exception as e:
    #     print(f"❌ Failed to load weights: {e}")
    #     global_model.save_weights("./initial_weight_3.h5")
    #     print("\nInitial Model saved.\n")

    # eval_server에 필요한 인자들을 묶어서 strategy에 전달
    eval_server_args = {
        'model': global_model,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'y_test': y_test,
        'result_path': RESULT_PATH,
        'matrix_path': MATRIX_PATH
    }

    # Custom strategy with round-by-round evaluation
    strategy = SaveEvaluationFedAvg(
        eval_server_args=eval_server_args,
        fraction_fit=0.8,  # training
        fraction_evaluate=0.8,  # evaluation
        min_fit_clients=11,
        min_evaluate_clients=11,
        min_available_clients=11,
        evaluate_fn=None # Let the custom strategy handle evaluation
    )

    # client_fn as closure: captures client_data, X_test_scaled, y_test, input_dim
    def client_fn(cid: str):
        cid_int = int(cid)
        client_model = TransformerAAE(input_dim)
        _ = client_model(tf.zeros((1, input_dim)), prior_labels=tf.zeros((1,1)))
        return FLClient(
            cid_int,
            client_model,
            client_data[cid_int],
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
    )

    # 최종 학습된 weight를 global_model에 적용
    if strategy.final_parameters is not None:
        from flwr.common import parameters_to_ndarrays
        final_weights = parameters_to_ndarrays(strategy.final_parameters)
        global_model.set_weights(final_weights)
    
    save_and_plot_history(
        history, 
        csv_path="./avg/avg_history", 
        png_path="./avg/avg_history.png"
    )

    eval_server(
        global_model,
        X_train_scaled,
        X_test_scaled,
        y_test,
        result_path=RESULT_PATH,
        matrix_path=MATRIX_PATH,
        roc_path=ROC_PATH
    )
    
    # 5. Save the trained model weights
    global_model.save_weights(WEIGHT_PATH)
    print(f"\nModel weights saved to {WEIGHT_PATH}")

if __name__ == "__main__":
    main()