import argparse

def get_args():
    '''
        Declare hyperparameters
    '''
    parser = argparse.ArgumentParser(add_help=False)

    # --- Evaluation Data ---
    parser.add_argument("--eval_dataset", type=str, default="cic", 
                        choices=["cic", "unsw"], help="Select dataset: cic or unsw")
    parser.add_argument("--base_data_path", type=str, default="/data/SDP_Dataset/Unified_model")
    parser.add_argument("--n_samples", type=int, default=150_000)
    

    # --- Train Parameters ---
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    parser.add_argument('--percentile', type=int, default=82)
    parser.add_argument('--random_seed', type=int, default=123)

    # --- Federated Learning Parameters ---
    # Clients (Local Servers)
    parser.add_argument('--client_nums', type=int, default=15)   # 서버 수가 50까지 될 수도 있음
    parser.add_argument('--client_epochs', type=int, default=30)  # == local epochs

    # Server (Center Aggregator for evaluation)
    parser.add_argument('--set_verbose', type=int, default=2)
    parser.add_argument('--server_rounds', type=int, default=10)  # server aggregation

    # --- Model Paths & Evaluation ---
    parser.add_argument("--threshold_std", type=float, default=0.25)

    # parse_args()를 사용하여 모든 인자를 확정합니다.
    args = parser.parse_args()
    return args