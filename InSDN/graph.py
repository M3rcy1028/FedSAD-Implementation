import pandas as pd
import os
import networkx as nx
import numpy as np
import pickle
from itertools import combinations
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# --- 설정값 ---
SAVE_STAT = "normal_stats.pkl"
TIME_WINDOW = '1T'       # 데이터를 몇 분 단위로 묶을지
THRESHOLD_STD = 3        # 정상 범위를 판단할 표준편차 배수
NUM_SAMPLES = 68424 // 2      # 샘플링 개수

def create_graph_from_window(df_window):
    """시간 창(window) 데이터를 기반으로 그래프 생성"""
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
    """그래프에서 특징(feature) 추출"""
    if G is None or G.number_of_nodes() == 0:
        return {'num_nodes':0,'num_edges':0,'density':0,'avg_degree':0}
    density = nx.density(G)
    degrees = [val for (node, val) in G.degree()]
    avg_degree = np.mean(degrees)
    return {
        'num_nodes': G.number_of_nodes(),
        'num_edges': G.number_of_edges(),
        'density': density,
        'avg_degree': avg_degree
    }

def run_anomaly_detection():
    # --- 1. 정상 데이터 학습 ---
    print(f"--- 1. 정상 데이터 학습 {NUM_SAMPLES} ---")
    try:
        df_normal_all = pd.read_csv('./graph_datas/InSDN_normal.csv')
    except FileNotFoundError:
        print("❌ InSDN_normal.csv 파일을 찾을 수 없습니다.")
        return
    
    # ---- 데이터셋 나누기 ----
    df_normal_train = df_normal_all.iloc[:NUM_SAMPLES].copy()

    # 타임스탬프 변환 및 인덱스 설정
    df_normal_train['Timestamp'] = pd.to_datetime(
        df_normal_train['Timestamp'], 
        dayfirst=True, 
        format='mixed', 
        errors='coerce'
    )
    df_normal_train.set_index('Timestamp', inplace=True)

    # 정상 학습 (그래프 기반 feature 추출)
    normal_features = []
    for period, df_window in df_normal_train.groupby(pd.Grouper(freq=TIME_WINDOW)):
        graph = create_graph_from_window(df_window)
        if graph:
            normal_features.append(extract_graph_features(graph))

    df_normal_features = pd.DataFrame(normal_features)
    normal_stats = df_normal_features.describe().loc[['mean','std']]

    # 결과 저장
    with open('normal_stats.pkl', 'wb') as f:
        pickle.dump(normal_stats, f)
    print("✅ 정상 데이터 통계 저장 완료: normal_stats.pkl")

    print("\n🎯 정상 통계 학습 완료! 이제 experiment_all_attacks_simple() 실행하세요.")


# 공격 유형별 평가
def evaluate_attack_counts(file_path, normal_stats, attack_name):
    try:
        df_attack = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"❌ 파일 없음: {file_path}")
        return

    df_attack['Timestamp'] = pd.to_datetime(
        df_attack['Timestamp'], 
        dayfirst=True, 
        format='mixed', 
        errors='coerce'
    )
    df_attack.set_index("Timestamp", inplace=True)

    pred_normal = 0
    pred_anomaly = 0

    for _, df_window in df_attack.groupby(pd.Grouper(freq=TIME_WINDOW)):
        graph = create_graph_from_window(df_window)
        if not graph:
            continue
        features = extract_graph_features(graph)

        is_anomaly = 0
        for feature_name in features:
            if feature_name not in normal_stats.columns:
                continue
            mean = normal_stats.loc["mean", feature_name]
            std = normal_stats.loc["std", feature_name]
            lower, upper = mean - THRESHOLD_STD*std, mean + THRESHOLD_STD*std
            if not (lower <= features[feature_name] <= upper):
                is_anomaly = 1
                break

        if is_anomaly:
            pred_anomaly += 1
        else:
            pred_normal += 1

    print(f"{attack_name}: pred_normal={pred_normal}, pred_anomaly={pred_anomaly}")


def experiment_all_attacks_simple():
    with open("normal_stats.pkl", "rb") as f:
        normal_stats = pickle.load(f)

    label_mapping = {
        0: "BFA",
        1: "DDoS",
        2: "DoS",
        3: "Probe",
        4: "U2R"
    }

    for num, attack_name in label_mapping.items():
        file_path = f"./graph_datas/InSDN_anomaly_{num}.csv"
        evaluate_attack_counts(file_path, normal_stats, attack_name)

# --- 실행 ---
if __name__ == "__main__":
    run_anomaly_detection()
    experiment_all_attacks_simple()
