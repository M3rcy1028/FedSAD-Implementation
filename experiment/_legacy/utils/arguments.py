import argparse
from pathlib import Path

def get_args(argv=None):
    '''
        Declare hyperparameters
    '''
    parser = argparse.ArgumentParser()

    # --- Evaluation Data ---
    parser.add_argument("--eval_dataset", type=str, default="cic", 
                        choices=["cic", "unsw", "ciciot", "zenodo", "genis", "gotham"],
                        help="Select dataset; zenodo uses the published ITU/ULAK train/test CSVs")
    parser.add_argument("--base_data_path", type=str, default="/data/SDP_Dataset/Unified_model")
    parser.add_argument("--n_samples", "--max_train_samples", type=int, default=150_000)
    

    # --- Train Parameters ---
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    parser.add_argument('--percentile', type=float, default=82)
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

    zenodo = parser.add_argument_group("Zenodo ITU/ULAK RNEP options")
    zenodo.add_argument("--data_dir", type=Path,
                        default=Path("/home/tako/naryun/IDS_Datasets/ITU_SDN_ULAK/raw"))
    zenodo.add_argument("--output_dir", type=Path,
                        help="Detailed run directory; dataset-specific default when omitted")
    zenodo.add_argument("--learning_rate", type=float, default=3e-5)
    zenodo.add_argument("--latent_dim", type=int, default=32)
    zenodo.add_argument("--gpu_per_client", type=float, default=0.2)
    zenodo.add_argument("--prepare_only", action="store_true",
                        help="Save preprocessed Zenodo/GeNIS data without training")

    genis = parser.add_argument_group("GeNIS matched source-model comparison")
    genis.add_argument("--genis_dir", type=Path,
                       default=Path("/home/tako/naryun/IDS_Datasets/GeNIS"))
    genis.add_argument("--flow_interval", type=int, choices=[5, 10, 30, 60], default=10)
    genis.add_argument("--taae_version", choices=["both", "previous", "current"], default="previous",
                       help="GeNIS RNEP model: previous (default), current, or both")
    genis.add_argument("--genis_prepared", type=Path, help="Shared prepared split for a single-model worker")
    genis.add_argument("--genis_latent_dim", type=int,
                       help="Optional common latent dimension; default preserves each source model's dimension")
    genis.add_argument("--result_path", type=Path,
                       help="Published GeNIS directory; defaults to results/GeNIS/<selected model>")
    genis.add_argument("--genis_attack_fraction", type=float, default=0.5,
                       help="Attack fraction in AE-LSTM/CNN-LSTM training data")
    genis.add_argument("--genis_sequence_split", choices=["shared", "new"], default="shared",
                       help="shared preserves RNEP validation/test rows for sequence models")
    genis.add_argument("--genis_worker", action="store_true", help=argparse.SUPPRESS)

    gotham = parser.add_argument_group("Gotham packet CSV / previous TAAE RNEP")
    gotham.add_argument("--gotham_dir", type=Path, default=Path("/data/SDP_Dataset/Gotham"))
    gotham.add_argument("--gotham_prepared", type=Path)
    gotham.add_argument("--gotham_eval_samples", type=int, default=25000,
                        help="Normal rows per evaluation partition; equal attack count")
    gotham.add_argument("--gotham_client_partition", choices=["devices", "iid"], default="iid",
                        help="iid (default): shuffle normal training rows across clients; devices: keep device groups")

    # parse_args()를 사용하여 모든 인자를 확정합니다.
    args = parser.parse_args(argv)
    if args.eval_dataset in {"zenodo", "genis", "gotham"}:
        if min(args.client_nums, args.server_rounds, args.client_epochs,
               args.batch_size, args.n_samples, args.latent_dim) < 1:
            parser.error("Counts and dimensions must be positive")
        if not 0 <= args.percentile <= 100 or args.learning_rate <= 0 or args.gpu_per_client < 0:
            parser.error("Invalid percentile, learning rate, or GPU resource request")
        if args.genis_latent_dim is not None and args.genis_latent_dim < 1:
            parser.error("GeNIS latent dimension must be positive")
        if not 0 <= args.genis_attack_fraction < 1:
            parser.error("GeNIS attack fraction must be in [0, 1)")
        if args.gotham_eval_samples < 1:
            parser.error("Gotham evaluation sample count must be positive")
        if args.eval_dataset == "gotham" and args.taae_version != "previous":
            parser.error("Gotham uses taae_previous.py; keep --taae_version previous")
    return args

def get_fedsad_args():
    '''
        Declare hyperparameters
    '''
    parser = argparse.ArgumentParser(add_help=False)

    # --- Evaluation Data ---
    parser.add_argument("--eval_dataset", type=str, default="cic", 
                        choices=["cic", "unsw", "ciciot"], help="Select dataset: cic, unsw or ciciot")
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
            "taae_model": "results/CSE-CIC-IDS2018/taae/taae_cic_weights.h5",
            "gsad_model": "results/CSE-CIC-IDS2018/gsad/normal_stats.pkl",
            "result_path": "results/CSE-CIC-IDS2018/fedsad",
            "normal_file": "CIC_ae_normal.csv", 
            "anomaly_prefix": "CIC_anomaly_ae_" 
        },
        "ciciot": {
            "taae_dir": f"{args.base_data_path}/ciciot_rnep",
            "gsad_dir": f"{args.base_data_path}/ciciot_graph",
            "taae_model": "results/CICIoT2023/taae/taae_ciciot_weights.h5",
            "gsad_model": "results/CICIoT2023/gsad/normal_stats.pkl",
            "result_path": "results/CICIoT2023/fedsad",
            "normal_file": "CICIoT_ae_normal.csv",
            "anomaly_prefix": "CICIoT_anomaly_ae_"
        },
        "unsw": {
            "taae_dir": f"{args.base_data_path}/unsw_rnep",
            "gsad_dir": f"{args.base_data_path}/unsw_graph",
            "taae_model": "results/UNSW_NB15/taae/taae_unsw_weights.h5",
            "gsad_model": "results/UNSW_NB15/gsad/normal_stats.pkl",
            "result_path": "results/UNSW_NB15/fedsad",
            "normal_file": "UNSW_NB15_normal.csv", 
            "anomaly_prefix": "UNSW_NB15_anomaly_"
        }
    }

    return args, dataset_configs

def get_gsad_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--time_window", type=str, default="1T")
    parser.add_argument("--threshold_std", type=float, default=0.25) #unsw 5

    parser.add_argument("--ae_data_dir", type=str,
                        default="/data/SDP_Dataset/Unified_model/cic_graph")
    parser.add_argument("--normal_file", type=str, default=None)
    parser.add_argument("--result_path", type=str, default="results/CSE-CIC-IDS2018/gsad")
    parser.add_argument("--eval_dataset", type=str, default="cic", 
                        choices=["cic", "unsw", "ciciot", "genis", "genis_frame"],
                        help="Select dataset: cic, unsw, ciciot, genis or genis_frame")
    args = parser.parse_args()
    
    return args

def get_taae_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    parser.add_argument('--percentile', type=float, default=82)

    # taae.py is imported by rnep.py/fedsad.py, so ignore their flags (e.g. --eval_dataset)
    args, _ = parser.parse_known_args()
    
    return args
