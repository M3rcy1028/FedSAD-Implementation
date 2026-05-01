import pandas as pd
import os
import networkx as nx
import numpy as np
import pickle
from collections import Counter, defaultdict
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

    G = nx.Graph()

    # port 등장 빈도
    port_counts = Counter(df_window['Dst Port'])

    ports = list(port_counts.keys())


    # pairwise edge 생성
    for u, v in combinations(ports, 2):
        weight = port_counts[u] + port_counts[v]
        G.add_edge(u, v, weight=weight)

    # -------------------------
    # self-loop fallback (핵심 추가)
    # -------------------------
    if G.number_of_edges() == 0:
        for port in ports:
            G.add_node(port)
            G.add_edge(port, port, weight=port_counts[port])

    return G


import numpy as np
from scipy.stats import entropy

def extract_gsad_features(G):
    if G is None or G.number_of_nodes() == 0:
        return {
            'num_nodes': 0,
            # 'num_edges': 0,
            # 'total_weight': 0,
            'avg_weight': 0,
            # 'max_weight': 0,
            # 'weight_std': 0,
            'weight_entropy': 0
            # 'max_strength': 0,
            # 'std_strength': 0
            # 'top1_ratio': 0
        }

    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()

    # -------------------------
    # Edge weight 처리
    # -------------------------
    weights = []

    for _, _, data in G.edges(data=True):
        w = data.get("weight", 1.0)  # weight 없으면 1
        weights.append(w)

    weights = np.array(weights) if len(weights) > 0 else np.array([0.0])

    total_weight = weights.sum()
    avg_weight = weights.mean()
    max_weight = weights.max()
    weight_std = weights.std()

    # -------------------------
    # Entropy (중요)
    # -------------------------
    if total_weight > 0:
        prob = weights / total_weight
        weight_entropy = entropy(prob)
    else:
        weight_entropy = 0

    # -------------------------
    # Node strength (weighted degree)
    # -------------------------
    strengths = {}
    for node in G.nodes():
        total = 0
        for neighbor, data in G[node].items():
            w = data.get("weight", 1.0)

            if neighbor == node:
                total += w
            else:
                total += w

        strengths[node] = total

    if strengths:
        strength_values = np.array(list(strengths.values()))
        max_strength = strength_values.max()
        std_strength = strength_values.std()
    else:
        max_strength = 0
        std_strength = 0

    # -------------------------
    # 집중도 (top-k)
    # -------------------------
    # if total_weight > 0:
    #     weights_sorted = np.sort(weights)[::-1]
    #     top1_ratio = weights_sorted[0] / total_weight
    # else:
    #     top1_ratio = 0

    return {
        'num_nodes': num_nodes,
        # 'num_edges': num_edges,
        # 'total_weight': total_weight,
        'avg_weight': avg_weight,
        # 'max_weight': max_weight,
        # 'weight_std': weight_std,
        'weight_entropy': weight_entropy
        # 'max_strength': max_strength,
        # 'std_strength': std_strength
        # 'top1_ratio': top1_ratio
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
def train_single_model(args, normal_file, anomaly_files, savefile_path):

    df_normal = pd.read_csv(normal_file)
    df_normal["Label"] = "Benign"
    # 시간 정렬
    df_normal["Timestamp"] = pd.to_datetime(df_normal["Timestamp"],
                dayfirst=True,
               errors="coerce")
    df_normal = df_normal.dropna(subset=["Timestamp"]).sort_values("Timestamp")
    # anomaly
    df_anomaly_list = []

    for f in anomaly_files:
        df_tmp = pd.read_csv(f)

        # 파일 이름에서 라벨 추출
        label_name = os.path.basename(f).replace(".csv", "")

        df_tmp["Label"] = label_name 
        df_anomaly_list.append(df_tmp)

    df_anomaly = pd.concat(df_anomaly_list, ignore_index=True)

    # anomaly Timestamp 처리 먼저
    df_anomaly["Timestamp"] = pd.to_datetime(
        df_anomaly["Timestamp"],
        dayfirst=True,
        errors="coerce"
    )
    df_anomaly = df_anomaly.dropna(subset=["Timestamp"]).sort_values("Timestamp")

    # anomaly의 가장 이른 시점
    anomaly_start_time = df_anomaly["Timestamp"].min()

    # normal에서도 같은 기준 적용
    df_normal_filter = df_normal[df_normal["Timestamp"] < anomaly_start_time].copy()

    # 다시 정렬 (안전)
    df_normal_filter = df_normal_filter.sort_values("Timestamp").reset_index(drop=True)
    print(f"df_normal_filter: {len(df_normal_filter)}")

    stats = compute_stats_from_df(df_normal_filter, args.time_window)

    mean = stats["mean"]
    var  = stats["var"].clip(lower=1e-6) 
    std  = np.sqrt(var)

    selected_features = [
        "num_nodes",
        # "num_edges",
        # "total_weight",
        "avg_weight",
        # "max_weight",
        # "weight_std",
        "weight_entropy"
        # "max_strength",
        # "std_strength"
        # "top1_ratio"
    ]

    normal_stats = pd.DataFrame(
        [mean[selected_features], std[selected_features]],
        index=["mean", "std"]
    )

    normal_stats.to_csv("normal_stats.csv")
    with open(savefile_path, "wb") as f:
        pickle.dump(normal_stats, f)

    print("\n=== Train (Time-based 60%) ===")
    return normal_stats

# ===============================
#   Evaluation
# ===============================
def run_anomaly_detection(args, normal_file, anomaly_files , model_path):

    k_dict = parse_feature_thresholds(args.feature_thresholds)

    with open(model_path, "rb") as f:
        normal_stats = pickle.load(f)

    df_normal = pd.read_csv(normal_file)
    df_normal["Label"] = "Benign"  

    # anomaly
    df_anomaly_list = []

    for f in anomaly_files:
        df_tmp = pd.read_csv(f)

        # 파일 이름에서 라벨 추출
        label_name = os.path.basename(f).replace(".csv", "")

        df_tmp["Label"] = label_name 
        df_anomaly_list.append(df_tmp)

    df_anomaly = pd.concat(df_anomaly_list, ignore_index=True)


    df_normal["Timestamp"] = pd.to_datetime(df_normal["Timestamp"], dayfirst=True, errors="coerce")
    df_normal = df_normal.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)

    df_anomaly["Timestamp"] = pd.to_datetime(df_anomaly["Timestamp"], dayfirst=True, errors="coerce")
    df_anomaly = df_anomaly.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)

    # -----------------------------
    # 1. anomaly 시작 시점 기준
    # -----------------------------
    anomaly_start_time = df_anomaly["Timestamp"].min()

    # -----------------------------
    # 2. anomaly 이전 normal 제거 (train에 이미 사용된 것으로 간주)
    # -----------------------------
    df_normal_after = df_normal[df_normal["Timestamp"] >= anomaly_start_time].copy()

    # -----------------------------
    # 3. random sampling (test normal)
    # -----------------------------
    sample_size = 200000

    df_test_normal = df_normal_after.sample(
        n=sample_size,
        random_state=42
    )

    # index 설정 (기존 흐름 유지)
    df_test_normal.set_index("Timestamp", inplace=True)
    df_anomaly.set_index("Timestamp", inplace=True)

    normal_windows = [
        w for _, w in df_test_normal.groupby(pd.Grouper(freq=args.time_window))
        if not w.empty
    ]

    anomaly_windows = [
        w for _, w in df_anomaly.groupby(pd.Grouper(freq=args.time_window))
        if not w.empty
    ]


    normal_test = normal_windows
    anomaly_test = anomaly_windows

    

    def check(feat):
        outlier_count = 0

        for name, val in feat.items():
            if name not in normal_stats.columns:
                continue

            mean = float(normal_stats.loc["mean", name])
            std  = float(normal_stats.loc["std", name])

            k = k_dict[name]

            if not (mean - k * std <= val <= mean + k * std):
                outlier_count += 1
            # if not (val <= mean + k * std):
            #     outlier_count += 1

        return 1 if outlier_count >= 1 else 0
  
    # -------------------------------
    # TEST
    # -------------------------------
    y_true_test, y_pred_test = [], []

    # normal
    for w in tqdm(normal_test, desc="TEST-NORMAL"):
        G = create_gsad_from_window(w)
        if not G:
            continue

        feat = extract_gsad_features(G)
        pred = check(feat)

        y_true_test.append(0)
        y_pred_test.append(pred)

    # anomaly
    for w in tqdm(anomaly_test, desc="TEST-ANOMALY"):
        G = create_gsad_from_window(w)
        if not G:
            continue

        feat = extract_gsad_features(G)
        pred = check(feat)

        y_true_test.append(1)
        y_pred_test.append(pred)


    return y_true_test, y_pred_test

def parse_feature_thresholds(threshold_str):
    if threshold_str is None:
        return None

    result = {}
    pairs = threshold_str.split(",")

    for p in pairs:
        key, val = p.split(":")
        result[key.strip()] = float(val)

    return result

# ===============================
#   Main
# ===============================
if __name__ == "__main__":

    args = get_gsad_args()

    os.makedirs(args.result_path,exist_ok=True)
    model_path=os.path.join(args.result_path, "normal_stats.pkl")
    report_path=os.path.join(args.result_path,"gsad_server.txt")
    matrix_path=os.path.join(args.result_path,"gsad_cm.png")

    normal_file = args.normal_file or os.path.join(args.ae_data_dir, "UNSW_NB15_normal.csv")
    # normal_file = args.normal_file or os.path.join(args.ae_data_dir, "CIC_ae_normal.csv")

    

    # anomaly_files = [
    #     os.path.join(args.ae_data_dir, f)
    #     for f in os.listdir(args.ae_data_dir)
    #     if f.startswith("UNSW_NB15_anomaly") and f.endswith(".csv")
    # ]
    anomaly_files = [
        os.path.join(args.ae_data_dir, f)
        for f in os.listdir(args.ae_data_dir)
        if f.startswith("CIC_anomaly_ae_") and f.endswith(".csv")
    ]


    train_single_model(args, normal_file, anomaly_files, model_path)

    y_true_test, y_pred_test = run_anomaly_detection(
        args, normal_file, anomaly_files, model_path
    )

    # test 결과
    save_report(y_true_test, y_pred_test, report_path)
    plt_confusion_matrix(y_true_test, y_pred_test, matrix_path)
    # y_true,y_pred=run_anomaly_detection(args, normal_file, anomaly_files, model_path)

    
    # save_report(y_true,y_pred,report_path)
    # plt_confusion_matrix(y_true,y_pred,matrix_path)