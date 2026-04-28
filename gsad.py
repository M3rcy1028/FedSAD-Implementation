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

    for u, v in combinations(ports, 2):
        weight = port_counts[u] + port_counts[v]   # weight 정의
        G.add_edge(u, v, weight=weight)

    return G


import numpy as np
from scipy.stats import entropy

def extract_gsad_features(G):
    if G is None or G.number_of_nodes() == 0:
        return {
            'num_nodes': 0,
            'num_edges': 0,
            'total_weight': 0,
            'avg_weight': 0,
            'max_weight': 0,
            'weight_std': 0,
            'weight_entropy': 0,
            'max_strength': 0,
            'std_strength': 0,
            'top1_ratio': 0
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
    strengths = dict(G.degree(weight='weight'))

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
    if total_weight > 0:
        weights_sorted = np.sort(weights)[::-1]
        top1_ratio = weights_sorted[0] / total_weight
    else:
        top1_ratio = 0

    return {
        'num_nodes': num_nodes,
        'num_edges': num_edges,
        'total_weight': total_weight,
        'avg_weight': avg_weight,
        'max_weight': max_weight,
        'weight_std': weight_std,
        'weight_entropy': weight_entropy,
        'max_strength': max_strength,
        'std_strength': std_strength,
        'top1_ratio': top1_ratio
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
    df["Label"] = "Benign"
    # 시간 정렬
    df["Timestamp"] = pd.to_datetime(df["Timestamp"],
                dayfirst=True,
               errors="coerce")
    df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp")

    # 정상만

    # 60% train (과거)
    n = len(df)
    train_end = int(n * 0.6)
    df_train_full = df.iloc[:train_end]

    df_train = df_train_full[df_train_full["Label"] == "Benign"]
    stats = compute_stats_from_df(df_train, args.time_window)

    stats = compute_stats_from_df(df_train, args.time_window)

    mean = stats["mean"]
    var  = stats["var"].clip(lower=1e-6) 
    std  = np.sqrt(var)

    selected_features = [
        "num_nodes",
        "num_edges",
        "total_weight",
        "avg_weight",
        "max_weight",
        "weight_std",
        "weight_entropy",
        "max_strength",
        "std_strength",
        "top1_ratio"
    ]
    normal_stats = pd.DataFrame(
        [mean[selected_features], std[selected_features]],
        index=["mean", "std"]
    )

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

    n_normal = len(df_normal)
    train_end = int(n_normal * 0.6)

    df_all_normal = df_normal.iloc[train_end:].copy()
    df_all_normal.set_index("Timestamp", inplace=True)
    df_anomaly.set_index("Timestamp", inplace=True)
    # df_future = pd.concat([df_all_normal, df_anomaly], ignore_index=True)

    # df_future = df_future.sort_values("Timestamp").reset_index(drop=True)

    # -------------------------------
    # 3. sliding window split
    # -------------------------------
    # df_future.set_index("Timestamp", inplace=True)

    # windows = [
    #     w for _, w in df_future.groupby(pd.Grouper(freq=args.time_window))
    #     if not w.empty
    # ]

    normal_windows = [
        w for _, w in df_all_normal.groupby(pd.Grouper(freq=args.time_window))
        if not w.empty
    ]

    anomaly_windows = [
        w for _, w in df_anomaly.groupby(pd.Grouper(freq=args.time_window))
        if not w.empty
    ]

    def split_windows(windows):
        split = int(len(windows) * 0.5)
        return windows[:split], windows[split:]

    normal_valid, normal_test = split_windows(normal_windows)
    anomaly_valid, anomaly_test = split_windows(anomaly_windows)

    

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

        return 1 if outlier_count >= 1 else 0
  

    # -------------------------------
    # VALID
    # -------------------------------
    y_true_valid, y_pred_valid = [], []
    feature_margins = defaultdict(list)
    # normal → label 0
    for w in tqdm(normal_valid, desc="VALID-NORMAL"):
        G = create_gsad_from_window(w)
        if not G:
            continue

        feat = extract_gsad_features(G)
        pred = check(feat)

        z_dict = compute_margin_per_feature(feat, normal_stats)

        if pred == 1:
            for f, z in z_dict.items():
                feature_margins[f].append(z)

        y_true_valid.append(0)
        y_pred_valid.append(pred)

        # drift adaptation (normal만 업데이트)
        if pred == 0:
            normal_stats = update_stats(normal_stats, [feat])


    # anomaly → label 1
    for w in tqdm(anomaly_valid, desc="VALID-ANOMALY"):
        G = create_gsad_from_window(w)
        if not G:
            continue

        feat = extract_gsad_features(G)
        pred = check(feat)

        y_true_valid.append(1)
        y_pred_valid.append(pred)


    print("\n=== Feature-wise Margin Statistics ===")

    for f, values in feature_margins.items():
        arr = np.array(values)
        if len(arr) == 0:
            continue

        k = k_dict[f]

        print(f"\n[{f}]")
        print(f"count: {len(arr)}")
        print(f"mean : {arr.mean():.4f}")
        print(f"std  : {arr.std():.4f}")
        print(f"min  : {arr.min():.4f}")
        print(f"max  : {arr.max():.4f}")
        print(f"threshold k: {k}")
        print(f"> k  : {(arr > k).mean():.2%}")
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

def compute_margin_per_feature(feat, normal_stats):
    z_dict = {}

    for name, val in feat.items():
        if name not in normal_stats.columns:
            continue

        mean = float(normal_stats.loc["mean", name])
        std  = float(normal_stats.loc["std", name])

        if std < 1e-3:
            continue

        z = abs(val - mean) / std
        z_dict[name] = z

    return z_dict

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

    # normal_file = args.normal_file or os.path.join(args.ae_data_dir, "UNSW_NB15_normal.csv")
    normal_file = args.normal_file or os.path.join(args.ae_data_dir, "CIC_ae_normal.csv")
    # normal_file = args.normal_file or os.path.join(args.ae_data_dir, "all_processed.csv")

    # anomaly_files = [
    #     os.path.join(args.ae_data_dir, f)
    #     for f in os.listdir(args.ae_data_dir)
    #     if f.startswith("UNSW_NB15_anomaly") and f.endswith(".csv")
    # ]
    anomaly_files = [
        os.path.join(args.ae_data_dir, f)
        for f in os.listdir(args.ae_data_dir)
        if f.startswith("CIC_anomaly") and f.endswith(".csv")
    ]

    train_single_model(args, normal_file, model_path)

    y_true_valid, y_pred_valid, y_true_test, y_pred_test = run_anomaly_detection(
        args, normal_file, anomaly_files, model_path
    )

    # validation 결과
    save_report(y_true_valid, y_pred_valid, report_path.replace(".txt", "_valid.txt"))

    # test 결과
    save_report(y_true_test, y_pred_test, report_path)
    plt_confusion_matrix(y_true_test, y_pred_test, matrix_path)
    # y_true,y_pred=run_anomaly_detection(args, normal_file, anomaly_files, model_path)

    
    # save_report(y_true,y_pred,report_path)
    # plt_confusion_matrix(y_true,y_pred,matrix_path)