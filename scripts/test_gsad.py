import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from gsad import run_anomaly_detection
from utils import save_report, plt_confusion_matrix
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
    
    args = parser.parse_args()

    os.makedirs(args.result_path,exist_ok=True)
    model_path=os.path.join(args.result_path, "normal_stats.pkl")
    report_path=os.path.join(args.result_path,"gsad_server.txt")
    matrix_path=os.path.join(args.result_path,"gsad_cm.png")

    normal_file = args.normal_file or os.path.join(args.ae_data_dir, "CIC_ae_normal.csv")

    anomaly_files = [
        os.path.join(args.ae_data_dir, f)
        for f in os.listdir(args.ae_data_dir)
        if f.startswith("CIC_anomaly_ae_") and f.endswith(".csv")
    ]

    y_true, y_pred = run_anomaly_detection(args, normal_file, anomaly_files)
    
    save_report(y_true,y_pred,report_path)
    plt_confusion_matrix(y_true,y_pred,matrix_path)

