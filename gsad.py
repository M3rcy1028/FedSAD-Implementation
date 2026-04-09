import pandas as pd
import os
import networkx as nx
import numpy as np
import pickle
from itertools import combinations
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from tqdm import tqdm
import argparse
from utils import *

# ===============================
#   GSAD-based Feature Extraction
# ===============================
def create_gsad_from_window(df_window):
    if df_window.empty:
        return None

    nodes = df_window['Dst Port'].unique()
    G = nx.Graph()
    G.add_nodes_from(nodes)

    if len(nodes) > 1:
        for node_pair in combinations(nodes, 2):
            G.add_edge(*node_pair)

    return G


def extract_gsad_features(G):
    if G is None or G.number_of_nodes() == 0:
        return {'num_nodes': 0, 'num_edges': 0, 'density': 0, 'avg_degree': 0}

    density = nx.density(G)
    degrees = [val for (_, val) in G.degree()]
    avg_degree = np.mean(degrees) if len(degrees) else 0

    return {
        'num_nodes': G.number_of_nodes(),
        'num_edges': G.number_of_edges(),
        'density': density,
        'avg_degree': avg_degree
    }


# ===============================
#   Statistics Computation
# ===============================
def compute_stats_from_df(df, time_window):
    if df.empty:
        empty = pd.Series({'num_nodes': 0, 'num_edges': 0,
                           'density': 0, 'avg_degree': 0}, dtype=float)
        return {"count": 0, "mean": empty, "var": empty}

    df = df.copy()

    if "Timestamp" not in df.columns:
        raise KeyError("❌ 'Timestamp' column is required.")

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], dayfirst=True, errors="coerce")
    df.set_index("Timestamp", inplace=True)
    df.sort_index(inplace=True)

    features = []

    for _, df_window in df.groupby(pd.Grouper(freq=time_window)):
        G = create_gsad_from_window(df_window)
        if G:
            features.append(extract_gsad_features(G))

    if not features:
        empty = pd.Series({'num_nodes': 0, 'num_edges': 0,
                           'density': 0, 'avg_degree': 0}, dtype=float)
        return {"count": 0, "mean": empty, "var": empty}

    df_feat = pd.DataFrame(features)

    return {
        "count": len(df_feat),
        "mean": df_feat.mean(numeric_only=True),
        "var": df_feat.var(numeric_only=True, ddof=0)
    }


# ===============================
#   Training
# ===============================
def train_single_model(args, normal_file,savefile_path):
    if not os.path.exists(normal_file):
        raise FileNotFoundError(f"❌ {normal_file} not found")

    df_normal = pd.read_csv(normal_file)
    df_train = df_normal.sample(frac=0.8, random_state=123).copy()

    stats = compute_stats_from_df(df_train, args.time_window)

    mean = stats["mean"]
    std = np.sqrt(stats["var"].clip(lower=0))

    normal_stats = pd.DataFrame([mean, std], index=["mean", "std"])
    normal_stats = normal_stats[["num_nodes", "num_edges", "density", "avg_degree"]]


    with open(savefile_path, "wb") as f:
        pickle.dump(normal_stats, f)

    print("\n=== Train ===")
    print(f"Saved to: {savefile_path}")
    print(normal_stats)

    return normal_stats


# ===============================
#   Evaluation
# ===============================
def run_anomaly_detection(args, normal_file, anomaly_files):

    with open(STAT_FILE_PATH, "rb") as f:
        normal_stats = pickle.load(f)

    df_normal = pd.read_csv(normal_file)
    df_test = df_normal.drop(df_normal.sample(frac=0.8, random_state=123).index).copy()

    df_test["Timestamp"] = pd.to_datetime(df_test["Timestamp"], dayfirst=True, errors="coerce")
    df_test.set_index("Timestamp", inplace=True)
    df_test.sort_index(inplace=True)

    y_true, y_pred = [], []

    def check(feat):
        for name, val in feat.items():
            if name not in normal_stats.columns:
                continue

            mean = float(normal_stats.loc["mean", name])
            std = float(normal_stats.loc["std", name])

            lower = mean - args.threshold_std * std
            upper = mean + args.threshold_std * std

            if not (lower <= val <= upper):
                return 1
        return 0

    # normal
    for _, df_window in tqdm(df_test.groupby(pd.Grouper(freq=args.time_window))):
        G = create_gsad_from_window(df_window)
        if not G:
            continue
        y_pred.append(check(extract_gsad_features(G)))
        y_true.append(0)

    # anomaly
    df_anomaly = pd.concat([pd.read_csv(f) for f in anomaly_files], ignore_index=True)

    df_anomaly["Timestamp"] = pd.to_datetime(df_anomaly["Timestamp"], dayfirst=True, errors="coerce")
    df_anomaly.set_index("Timestamp", inplace=True)
    df_anomaly.sort_index(inplace=True)

    for _, df_window in tqdm(df_anomaly.groupby(pd.Grouper(freq=args.time_window))):
        G = create_gsad_from_window(df_window)
        if not G:
            continue
        y_pred.append(check(extract_gsad_features(G)))
        y_true.append(1)

    return y_true, y_pred


# ===============================
#   Main
# ===============================
if __name__ == "__main__":
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

    train_single_model(args, normal_file,model_path)
    y_true,y_pred=run_anomaly_detection(args, normal_file, anomaly_files)

    
    save_report(y_true,y_pred,report_path)
    plt_confusion_matrix(y_true,y_pred,matrix_path)