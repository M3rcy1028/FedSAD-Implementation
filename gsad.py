import pandas as pd
import os
import networkx as nx
import numpy as np
import pickle
from itertools import combinations
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from tqdm import tqdm
from utils.arguments import get_gsad_args
from utils.metrics import save_report, plt_confusion_matrix

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

    df["Timestamp"] = pd.to_datetime(df["Timestamp"],
               dayfirst = True,
               errors="coerce")
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
def train_single_model(args, normal_file, savefile_path):

    df = pd.read_csv(normal_file)
    # 시간 정렬
    df["Timestamp"] = pd.to_datetime(df["Timestamp"],
               errors="coerce")
    df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp")

    n = len(df)
    train_end = int(n * 0.6)
    df_train_full = df.iloc[:train_end]

    df_train = df_train_full[df_train_full["Label"] == "Benign"]
    stats = compute_stats_from_df(df_train, args.time_window)

    mean = stats["mean"]
    std = np.sqrt(stats["var"].clip(lower=0))

    normal_stats = pd.DataFrame([mean, std], index=["mean", "std"])
    normal_stats = normal_stats[["num_nodes", "num_edges", "density", "avg_degree"]]

    with open(savefile_path, "wb") as f:
        pickle.dump(normal_stats, f)

    print("\n=== Train (Time-based 60%) ===")
    return normal_stats

# ===============================
#   Evaluation
# ===============================
def run_anomaly_detection(args, data_file , model_path):

    with open(model_path, "rb") as f:
        normal_stats = pickle.load(f)

    df_all = pd.read_csv(data_file)

    df_all["Timestamp"] = pd.to_datetime(df_all["Timestamp"], errors="coerce")
    df_all = df_all.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)

    n = len(df_all)
    train_end = int(n * 0.6)

    df_future = df_all.iloc[train_end:].copy()


    # -------------------------------
    # 3. sliding window split
    # -------------------------------
    df_future.set_index("Timestamp", inplace=True)

    windows = [
        w for _, w in df_future.groupby(pd.Grouper(freq=args.time_window))
        if not w.empty
    ]

    valid_windows = []
    test_windows = []

    split = int(len(windows) * 0.5)

    valid_windows = windows[:split]
    test_windows  = windows[split:]

    df_valid = pd.concat(valid_windows).sort_index()
    df_test  = pd.concat(test_windows).sort_index()

    # -------------------------------
    # detection 함수
    # -------------------------------
    def check(feat):
        for name, val in feat.items():
            if name not in normal_stats.columns:
                continue

            mean = float(normal_stats.loc["mean", name])
            std = float(normal_stats.loc["std", name])

            if not (mean - args.threshold_std * std <= val <= mean + args.threshold_std * std):
                return 1
        return 0

    # -------------------------------
    # VALID
    # -------------------------------
    y_true_valid, y_pred_valid = [], []

    for _, df_window in tqdm(df_valid.groupby(pd.Grouper(freq=args.time_window)), desc="VALID"):
        G = create_gsad_from_window(df_window)
        if not G:
            continue

        feat=extract_gsad_features(G)
        pred = check(feat)
        ratio = (df_window["Label"] != "Benign").mean()
        label = 1 if ratio > 0.3 else 0
        # label = 1 if (df_window["Label"] != "Benign").any() else 0

        y_pred_valid.append(pred)
        y_true_valid.append(label)

        if pred == 0:
            normal_stats = update_stats(normal_stats, [feat])

    # -------------------------------
    # TEST
    # -------------------------------
    y_true_test, y_pred_test = [], []

    for _, df_window in tqdm(df_test.groupby(pd.Grouper(freq=args.time_window)), desc="TEST"):
        G = create_gsad_from_window(df_window)
        if not G:
            continue
        
        feat = extract_gsad_features(G)
        ratio = (df_window["Label"] != "Benign").mean()
        label = 1 if ratio > 0.3 else 0
        # label = 1 if (df_window["Label"] != "Benign").any() else 0

        pred = check(feat)

        y_pred_test.append(pred)
        y_true_test.append(label)


    return y_true_valid, y_pred_valid, y_true_test, y_pred_test

def update_stats(normal_stats, feat_list, alpha=0.2):

    df_feat = pd.DataFrame(feat_list)

    old_mean = normal_stats.loc["mean"]
    old_std  = normal_stats.loc["std"]
    old_var  = old_std ** 2

    new_mean = df_feat.mean()
    new_std  = df_feat.std(ddof=0)
    new_var  = new_std ** 2

    updated_mean = (1 - alpha) * old_mean + alpha * new_mean

    updated_var = (
        (1 - alpha) * (old_var + (old_mean - updated_mean) ** 2)
        + alpha * (new_var + (new_mean - updated_mean) ** 2)
    )

    updated_std = np.sqrt(updated_var)

    return pd.DataFrame(
        [updated_mean, updated_std],
        index=["mean", "std"]
    )

# ===============================
#   Main
# ===============================
if __name__ == "__main__":

    args = get_gsad_args()

    os.makedirs(args.result_path,exist_ok=True)
    model_path=os.path.join(args.result_path, "normal_stats.pkl")
    report_path=os.path.join(args.result_path,"gsad_server.txt")
    matrix_path=os.path.join(args.result_path,"gsad_cm.png")

    # normal_file = args.normal_file or os.path.join(args.ae_data_dir, "CIC_ae_normal.csv")
    normal_file = args.normal_file or os.path.join(args.ae_data_dir, "combined_dataset.csv")

    # anomaly_files = [
    #     os.path.join(args.ae_data_dir, f)
    #     for f in os.listdir(args.ae_data_dir)
    #     if f.startswith("CIC_anomaly_ae_") and f.endswith(".csv")
    # ]

    train_single_model(args, normal_file, model_path)

    y_true_valid, y_pred_valid, y_true_test, y_pred_test = run_anomaly_detection(
        args, normal_file, model_path
    )

    # validation 결과
    save_report(y_true_valid, y_pred_valid, report_path.replace(".txt", "_valid.txt"))

    # test 결과
    save_report(y_true_test, y_pred_test, report_path)
    plt_confusion_matrix(y_true_test, y_pred_test, matrix_path)
    # y_true,y_pred=run_anomaly_detection(args, normal_file, anomaly_files, model_path)

    
    # save_report(y_true,y_pred,report_path)
    # plt_confusion_matrix(y_true,y_pred,matrix_path)