import argparse

def get_args():
    '''
        Declare hyperparameters
    '''
    parser = argparse.ArgumentParser()

    # --- Data Directories ---
    parser.add_argument("--taae_data_dir", type=str,
                        default="/data/SDP_Dataset/Unified_model/cic_rnep")
    parser.add_argument("--gsad_data_dir", type=str,
                        default="/data/SDP_Dataset/Unified_model/cic_graph")

    parser.add_argument("--n_samples", type=int, default=150_000)

    # --- Train Parameters ---
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    parser.add_argument('--percentile', type=int, default=85)
    parser.add_argument('--random_seed', type=int, default=123)

    # --- Federated Learning Parameters ---
    '''
        FL 관점에서 client-server이고, 
        server간 연합학습이므로,   client는 각 server가 되고
                                server는 각 server를 집계하는 center임
    '''
    # Clients (Local Servers)
    parser.add_argument('--client_nums', type=int, default=15)   # 서버 수가 50까지 될 수도 있음
    parser.add_argument('--client_epochs', type=int, default=30)  # == local epochs

    # Server (Center Aggregator)
    parser.add_argument('--set_verbose', type=int, default=2)
    parser.add_argument('--server_rounds', type=int, default=10)  # server aggregation

    # --- Model Paths & Evaluation ---
    parser.add_argument("--taae_model_path", type=str,
                        default="fedsad_weights/fedsad_cic_weights.h5")
    parser.add_argument("--gsad_model_path", type=str,
                        default="fedsad_weights/normal_stats.pkl")
    parser.add_argument("--threshold_std", type=float, default=0.25)
    parser.add_argument("--result_path", type=str,
                        default="fedsad_results/fedsad_cic")

    # parse_args()를 사용하여 모든 인자를 확정합니다.
    args = parser.parse_args()
    return args

# 하이퍼파라미터 불러오기
args = get_args()