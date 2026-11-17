import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from gsad import run_anomaly_detection
from utils.metrics import save_report, plt_confusion_matrix
import argparse
import os

if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--time_window", type=str, default="1T")
    parser.add_argument("--threshold_std", type=float, default=0.25)

    parser.add_argument("--ae_data_dir", type=str,
                        default="/data/SDP_Dataset/Unified_model/cic_graph")
    parser.add_argument("--normal_file", type=str, default=None)
    parser.add_argument("--result_path", type=str, default="results/CSE-CIC-IDS2018/gsad")
    parser.add_argument("--eval_dataset", type=str, default="cic", 
                        choices=["cic", "unsw"], help="Select dataset: cic or unsw")
    args = parser.parse_args()

    os.makedirs(args.result_path,exist_ok=True)
    model_path=os.path.join(args.result_path, "normal_stats.pkl")
    report_path=os.path.join(args.result_path,"gsad_server.txt")
    matrix_path=os.path.join(args.result_path,"gsad_cm.png")

    dataset_file_configs = {
    "cic": {
        "normal_file": "CIC_ae_normal.csv",
        "anomaly_prefix": "CIC_anomaly_ae_"
    },
        "unsw": {
            "normal_file": "UNSW_NB15_normal.csv",
            "anomaly_prefix": "UNSW_NB15_anomaly_"
        }
    }

    file_config = dataset_file_configs.get(args.eval_dataset.lower())
    if not file_config:
        print(f"[ERROR] Dataset '{args.eval_dataset}' is not supported.")
        exit(1)

    normal_file = args.normal_file or os.path.join(args.ae_data_dir, file_config["normal_file"])

    anomaly_files = [
        os.path.join(args.ae_data_dir, f)
        for f in os.listdir(args.ae_data_dir)
        if f.startswith(file_config["anomaly_prefix"]) and f.endswith(".csv")
    ]

    y_true, y_pred = run_anomaly_detection(args, normal_file, anomaly_files, model_path)
    
    save_report(y_true,y_pred,report_path)
    plt_confusion_matrix(y_true,y_pred,matrix_path)

