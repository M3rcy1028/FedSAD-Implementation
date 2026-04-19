import argparse

# dataset_configs = {
#         "cic": {
#             "taae_dir": f"{args.base_data_path}/cic_rnep",
#             "gsad_dir": f"{args.base_data_path}/cic_graph",
#             "taae_model": "fedsad_weights/fedsad_cic_weights.h5",
#             "gsad_model": "fedsad_weights/normal_stats.pkl",
#             "result_path": "fedsad_results/fedsad_cic",
#             "normal_file": "CIC_ae_normal.csv", # 정상 파일 이름
#             "anomaly_prefix": "CIC_anomaly_ae_" # 이상 파일 접두사
#         },
#         "unsw": {
#             "taae_dir": f"{args.base_data_path}/unsw_rnep",
#             "gsad_dir": f"{args.base_data_path}/unsw_graph",
#             "taae_model": "fedsad_weights/fedsad_unsw_weights.h5",
#             "gsad_model": "fedsad_weights/normal_stats.pkl",
#             "result_path": "fedsad_results/fedsad_unsw",
#             "normal_file": "UNSW_NB15_normal.csv", # 정상 파일 이름 (예시)
#             "anomaly_prefix": "UNSW_NB15_anomaly_"  # 이상 파일 접두사
#         }
#     }

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

def get_fedsad_args():
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

    args = parser.parse_args()

    dataset_configs = {
        "cic": {
            "taae_dir": f"{args.base_data_path}/cic_rnep",
            "gsad_dir": f"{args.base_data_path}/cic_graph",
            "taae_model": "fedsad_weights/fedsad_cic_weights.h5",
            "gsad_model": "fedsad_weights/normal_stats.pkl",
            "result_path": "fedsad_results/fedsad_cic",
            "normal_file": "CIC_ae_normal.csv", # 정상 파일 이름
            "anomaly_prefix": "CIC_anomaly_ae_" # 이상 파일 접두사
        },
        "unsw": {
            "taae_dir": f"{args.base_data_path}/unsw_rnep",
            "gsad_dir": f"{args.base_data_path}/unsw_graph",
            "taae_model": "fedsad_weights/fedsad_unsw_weights.h5",
            "gsad_model": "fedsad_weights/normal_stats.pkl",
            "result_path": "fedsad_results/fedsad_unsw",
            "normal_file": "UNSW_NB15_normal.csv", 
            "anomaly_prefix": "UNSW_NB15_anomaly_"
        }
    }

    return args, dataset_configs

def get_gsad_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--time_window", type=str, default="1T")
    parser.add_argument("--threshold_std", type=float, default=3)

    parser.add_argument("--ae_data_dir", type=str,
                        default="/data/SDP_Dataset/Unified_model/cic_graph")
    parser.add_argument("--normal_file", type=str, default=None)
    parser.add_argument("--result_path", type=str, default="results/CSE-CIC-IDS2018/gsad")
    
    args = parser.parse_args()
    
    return args

def get_taae_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    parser.add_argument('--percentile', type=int, default=82)

    args = parser.parse_args()
    
    return args