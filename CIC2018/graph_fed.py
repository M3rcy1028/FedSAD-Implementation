import pandas as pd
import os
import networkx as nx
import numpy as np
import pickle
from itertools import combinations
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# --- 설정값 ---
SAVE_STAT = "normal_stats.pkl"
TIME_WINDOW = '1T'       # 데이터를 몇 분 단위로 묶을지
THRESHOLD_STD = 3        # 정상 범위를 판단할 표준편차 배수
NUM_CLIENTS = 5          # 연합학습 클라이언트 수
NUM_SAMPLES = 10736459   # (선택) 샘플링 개수

# ===============================
#   데이터 분할 (client 파일 생성)
# ===============================
def split_dataset_for_clients(input_file="./graph_datas/CIC_normal.csv", num_clients=5):
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"❌ {input_file} 없음")

    df = pd.read_csv(input_file)
    df = df.sample(frac=1, random_state=5).reset_index(drop=True)

    # (선택) 너무 크면 샘플링
    if NUM_SAMPLES and len(df) > NUM_SAMPLES:
        df = df.iloc[:NUM_SAMPLES].copy()

    splits = np.array_split(df, num_clients)

    client_paths = []
    for i, split in enumerate(splits, start=1):
        out_file = f"./client/CIC_normal_client{i}.csv"
        split.to_csv(out_file, index=False)
        client_paths.append(out_file)
        print(f"✅ {out_file} 저장 완료 (rows={len(split)})")

    return client_paths

# ===============================
#   그래프 기반 feature 추출
# ===============================
def create_graph_from_window(df_window):
    if df_window.empty:
        return None
    nodes = df_window['Dst Port'].unique()
    G = nx.Graph()
    G.add_nodes_from(nodes)
    if len(nodes) > 1:
        for node_pair in combinations(nodes, 2):
            G.add_edge(*node_pair)
    return G

def extract_graph_features(G):
    if G is None or G.number_of_nodes() == 0:
        return {'num_nodes':0,'num_edges':0,'density':0,'avg_degree':0}
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
#   클라이언트 로컬 통계 계산
# ===============================
def local_compute_stats(client_data_path):
    df = pd.read_csv(client_data_path)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df.set_index("Timestamp", inplace=True)

    features = []
    for _, df_window in df.groupby(pd.Grouper(freq=TIME_WINDOW)):
        G = create_graph_from_window(df_window)
        if G:
            features.append(extract_graph_features(G))

    if not features:
        empty = pd.Series({'num_nodes':0,'num_edges':0,'density':0,'avg_degree':0}, dtype=float)
        return {"count": 0, "mean": empty, "var": empty}

    df_feat = pd.DataFrame(features)
    count = len(df_feat)
    mean = df_feat.mean(numeric_only=True)
    var = df_feat.var(numeric_only=True, ddof=0)
    return {"count": count, "mean": mean, "var": var}

# ===============================
#   서버 집계 (Federated Aggregation)
# ===============================
def federated_aggregate(pieces):
    pieces = [p for p in pieces if p["count"] > 0]
    if not pieces:
        raise ValueError("❌ 유효한 클라이언트 통계 없음")

    common_cols = set(pieces[0]["mean"].index)
    for p in pieces[1:]:
        common_cols &= set(p["mean"].index)
    common_cols = list(common_cols)

    N = sum(p["count"] for p in pieces)
    num = sum((p["count"] * p["mean"][common_cols] for p in pieces))
    global_mean = num / N

    second_mom_sum = None
    for p in pieces:
        term = p["count"] * (p["var"][common_cols] + p["mean"][common_cols]**2)
        second_mom_sum = term if second_mom_sum is None else (second_mom_sum + term)
    global_var = (second_mom_sum / N) - (global_mean**2)
    global_var = global_var.clip(lower=0)
    global_std = np.sqrt(global_var)

    normal_stats = pd.DataFrame([global_mean, global_std], index=["mean","std"])
    return normal_stats[["num_nodes","num_edges","density","avg_degree"]]

def run_federated_learning(client_paths):
    print("=== [연합학습] 정상 통계 학습 ===")
    pieces = []
    for cp in client_paths:
        part = local_compute_stats(cp)
        print(f"  • {cp}: windows={part['count']}")
        pieces.append(part)

    normal_stats = federated_aggregate(pieces)
    with open(SAVE_STAT, "wb") as f:
        pickle.dump(normal_stats, f)
    print("✅ 연합학습 전역 정상 통계 저장 완료 →", SAVE_STAT)
    print(normal_stats)
    return normal_stats

# ===============================
#   평가 (기존 코드 활용)
# ===============================
def run_anomaly_detection():
    if not os.path.exists(SAVE_STAT):
        raise FileNotFoundError(f"❌ {SAVE_STAT} 없음")

    with open(SAVE_STAT, "rb") as f:
        normal_stats = pickle.load(f)

    print("\n--- 평가 데이터 준비 ---")
    normal_file = "./graph_datas/CIC_normal.csv"
    anomaly_file = "./graph_datas/CIC_anomaly.csv"

    if not os.path.exists(normal_file) or not os.path.exists(anomaly_file):
        print("❌ 평가용 CSV 없음")
        return

    # ✅ Normal 데이터: NUM_SAMPLES 이후 부분만 평가용으로 사용
    df_normal = pd.read_csv(normal_file)
    if NUM_SAMPLES and len(df_normal) > NUM_SAMPLES:
        df_normal_eval = df_normal.iloc[NUM_SAMPLES:].copy()
    else:
        df_normal_eval = pd.DataFrame()  # 없으면 빈값

    df_anomaly = pd.read_csv(anomaly_file)

    # Timestamp 파싱
    if not df_normal_eval.empty:
        df_normal_eval["Timestamp"] = pd.to_datetime(df_normal_eval["Timestamp"])
        df_normal_eval.set_index("Timestamp", inplace=True)
    df_anomaly["Timestamp"] = pd.to_datetime(df_anomaly["Timestamp"])
    df_anomaly.set_index("Timestamp", inplace=True)

    y_true, y_pred = [], []

    # ✅ Normal 평가
    for _, df_window in df_normal_eval.groupby(pd.Grouper(freq=TIME_WINDOW)):
        G = create_graph_from_window(df_window)
        if not G: 
            continue
        feat = extract_graph_features(G)
        is_anom = 0
        for name, val in feat.items():
            if name not in normal_stats.columns: 
                continue
            mean, std = normal_stats.loc["mean", name], normal_stats.loc["std", name]
            if not (mean - THRESHOLD_STD*std <= val <= mean + THRESHOLD_STD*std):
                is_anom = 1; break
        y_pred.append(is_anom)
        y_true.append(0)  # 정상

    # ✅ Anomaly 평가
    for _, df_window in df_anomaly.groupby(pd.Grouper(freq=TIME_WINDOW)):
        G = create_graph_from_window(df_window)
        if not G: 
            continue
        feat = extract_graph_features(G)
        is_anom = 0
        for name, val in feat.items():
            if name not in normal_stats.columns: 
                continue
            mean, std = normal_stats.loc["mean", name], normal_stats.loc["std", name]
            if not (mean - THRESHOLD_STD*std <= val <= mean + THRESHOLD_STD*std):
                is_anom = 1; break
        y_pred.append(is_anom)
        y_true.append(1)  # 이상

    print("\n--- Classification Report ---")
    print(classification_report(y_true, y_pred, target_names=["Normal","Anomaly"]))
    print("\n--- Confusion Matrix ---")
    print(confusion_matrix(y_true, y_pred))

def federated_rounds(client_paths, num_rounds=5, alpha=1.0):
    """
    Federated Learning with multiple rounds
    - num_rounds: 라운드 수
    - alpha: local update 비율 (0~1)
    """
    # 초기 전역 통계 (라운드0: 그냥 전체 클라 평균)
    print(f"=== [연합학습] {num_rounds} Rounds 시작 ===")
    pieces = [local_compute_stats(cp) for cp in client_paths if os.path.exists(cp)]
    global_stats = federated_aggregate(pieces)

    for rnd in range(1, num_rounds + 1):
        updated_pieces = []
        for cp in client_paths:
            local = local_compute_stats(cp)

            # 클라이언트 로컬 업데이트 (기존 global과 local을 혼합)
            new_mean = (1 - alpha) * global_stats.loc["mean"] + alpha * local["mean"]
            new_var  = (1 - alpha) * (global_stats.loc["std"]**2) + alpha * local["var"]
            new_std  = np.sqrt(new_var.clip(lower=0))

            updated_pieces.append({
                "count": local["count"],
                "mean": new_mean,
                "var": new_var
            })

        # 서버 집계
        global_stats = federated_aggregate(updated_pieces)
        print(f"  🔁 Round {rnd} 완료")

    # 저장
    with open(SAVE_STAT, "wb") as f:
        pickle.dump(global_stats, f)
    print("✅ 최종 전역 통계 저장 완료 →", SAVE_STAT)
    print(global_stats)
    return global_stats


# ===============================
#   실행
# ===============================
if __name__ == "__main__":
    # 1) 데이터 분할
    client_files = split_dataset_for_clients("./graph_datas/CIC_normal.csv", num_clients=NUM_CLIENTS)

    # 2) 라운드 연합학습
    global_stats = federated_rounds(client_files, num_rounds=10, alpha=0.5)

    # 3) 평가
    run_anomaly_detection()
