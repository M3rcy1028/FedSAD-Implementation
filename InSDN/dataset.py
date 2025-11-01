import kagglehub
import pandas as pd
import glob
import requests
import zipfile
import io
import os

def download():
    # KaggleHub로 데이터셋 다운로드
    # path = kagglehub.dataset_download("muhammadumarjavaid/insdn-dataset-2020")
    path = kagglehub.dataset_download("badcodebuilder/insdn-dataset")

    print("Path to dataset files:", path)

# 웹사이트에서 다운로드
def download_insdn_dataset(save_dir="./data"):
    url = "https://aseados.ucd.ie/datasets/SDN/InSDN_DatasetCSV.zip"  # ← 실제 zip 파일 URL로 교체
    os.makedirs(save_dir, exist_ok=True)

    print("Downloading InSDN dataset from:", url)
    response = requests.get(url)
    response.raise_for_status()

    # 압축 해제
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        z.extractall(save_dir)

    print(f"✅ Dataset downloaded and extracted to: {save_dir}")

# Preprocessing
def split_dataset():
    # Dst port, timestamps
    # 경로 설정
    DATA_PATH = "./InSDN_Combined.csv"  # 원본 CSV 파일 경로
    SAVE_DIR = "./raw_datas"
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 데이터 불러오기
    df = pd.read_csv(DATA_PATH)

    # 'Label' 열이 존재하는지 확인
    if "Label" not in df.columns:
        raise ValueError("⚠️ 'Label' 컬럼이 존재하지 않습니다. CSV 구조를 확인하세요.")

    # Label 목록 출력
    labels = df["Label"].unique()
    print("📋 라벨 목록:")
    for i, lbl in enumerate(labels):
        print(f"{i}: {lbl}")

    # Normal / Anomaly 분리
    df_normal = df[df["Label"].str.lower() == "normal"]
    df_anomaly = df[df["Label"].str.lower() != "normal"]

    # Normal 저장
    normal_path = os.path.join(SAVE_DIR, "InSDN_normal.csv")
    df_normal.to_csv(normal_path, index=False)
    print(f"\n✅ Normal 저장 완료: {len(df_normal)}개 → {normal_path}")

    # Anomaly 저장
    anomaly_path = os.path.join(SAVE_DIR, "InSDN_anomaly.csv")
    df_anomaly.to_csv(anomaly_path, index=False)
    print(f"\n✅ Normal 저장 완료: {len(df_anomaly)}개 → {anomaly_path}")

    # Anomaly 라벨별 인코딩
    label_map = {label: i for i, label in enumerate(sorted(df_anomaly["Label"].unique()))}
    print("\n🚨 비정상 라벨 인코딩 매핑:")
    for lbl, idx in label_map.items():
        print(f"{idx}: {lbl}")

    # 인코딩 후 저장
    for lbl, idx in label_map.items():
        subset = df_anomaly[df_anomaly["Label"] == lbl]
        file_name = f"InSDN_anomaly_{idx}.csv"
        subset.to_csv(os.path.join(SAVE_DIR, file_name), index=False)
        print(f"   ⤷ {lbl}: {len(subset)}개 저장 ({file_name})")

    print("\n🎯 전체 완료!")
    print(f" - Normal: {len(df_normal)}개")
    print(f" - Anomaly 총합: {len(df_anomaly)}개")
    print(f" - 저장 위치: {SAVE_DIR}")

# For Graph
def create_graph_dataset():
    SAVE_DIR = "./graph_datas"
    SRC_DIR = "./InSDN_split"   # InSDN_normal / anomaly_*.csv 있는 폴더
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 그래프용으로 사용할 최소 컬럼
    graph_cols = ["Timestamp", "Dst Port"]

    # --- 1. 정상 데이터 ---
    normal_path = os.path.join(SRC_DIR, "InSDN_normal.csv")
    if os.path.exists(normal_path):
        df_normal = pd.read_csv(normal_path)
        df_normal = df_normal[graph_cols].copy()
        df_normal.to_csv(os.path.join(SAVE_DIR, "InSDN_normal.csv"), index=False)
        print(f"✅ 정상 데이터 저장 완료: {len(df_normal)}개")
    else:
        print("❌ InSDN_normal.csv 파일이 없습니다.")

    # --- 2. 이상 데이터 ---
    anomaly_files = sorted(glob.glob(os.path.join(SRC_DIR, "InSDN_anomaly_*.csv")))
    if not anomaly_files:
        print("❌ 이상 데이터 파일이 없습니다.")
        return

    for file in anomaly_files:
        name = os.path.basename(file)
        df_anomaly = pd.read_csv(file)
        df_anomaly = df_anomaly[graph_cols].copy()
        save_path = os.path.join(SAVE_DIR, name)
        df_anomaly.to_csv(save_path, index=False)
        print(f"🚨 {name} 저장 완료 ({len(df_anomaly)}개)")

    print("\n🎯 모든 그래프용 CSV 변환 완료!")
    print(f"저장 위치: {SAVE_DIR}")

def feature_selection():
    # --- 디렉토리 경로 ---
    RAW_DIR = "./raw_datas"
    SAVE_DIR = "./ae_datas"
    os.makedirs(SAVE_DIR, exist_ok=True)

    # --- 피처 선택 ---
    SELECTED_FEATURES = [
        "Protocol", "Flow Duration",
        "Tot Fwd Pkts", "Tot Bwd Pkts",
        "TotLen Fwd Pkts", "TotLen Bwd Pkts",
        "Fwd Pkt Len Max", "Fwd Pkt Len Min",
        "Fwd Pkt Len Mean", "Fwd Pkt Len Std",
        "Bwd Pkt Len Max", "Bwd Pkt Len Min",
        "Bwd Pkt Len Mean", "Bwd Pkt Len Std",
        "Flow Byts/s", "Flow Pkts/s",
        "Flow IAT Mean", "Flow IAT Std",
        "Flow IAT Max", "Flow IAT Min",
        "Fwd IAT Tot", "Fwd IAT Mean",
        "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
        "Bwd IAT Tot", "Bwd IAT Mean",
        "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
        "Fwd Header Len", "Bwd Header Len",
        "Fwd Pkts/s", "Bwd Pkts/s",
        "Pkt Len Min", "Pkt Len Max",
        "Pkt Len Mean", "Pkt Len Std", "Pkt Len Var",
        "Pkt Size Avg",
        "Active Mean", "Active Std", "Active Max", "Active Min",
        "Idle Mean", "Idle Std", "Idle Max", "Idle Min"
    ]

    # --- 정상 / 비정상 데이터 로드 ---
    normal_path = os.path.join(RAW_DIR, "InSDN_normal.csv")
    anomaly_path = os.path.join(RAW_DIR, "InSDN_anomaly.csv")

    df_normal = pd.read_csv(normal_path, low_memory=False)
    df_anomaly = pd.read_csv(anomaly_path, low_memory=False)

    # --- 피처 필터링 ---
    df_normal_filtered = df_normal[SELECTED_FEATURES]
    df_anomaly_filtered = df_anomaly[SELECTED_FEATURES]

    # --- 저장 ---
    df_normal_filtered.to_csv(os.path.join(SAVE_DIR, "InSDN_normal_48.csv"), index=False)
    df_anomaly_filtered.to_csv(os.path.join(SAVE_DIR, "InSDN_anomaly_48.csv"), index=False)

    print(f"✅ 정상 데이터: {df_normal_filtered.shape}, 이상 데이터: {df_anomaly_filtered.shape}")
    print("📁 저장 완료 → ./ae_datas/")

# --- 피처 선택 (48개) ---
SELECTED_FEATURES = [
    "Protocol", "Flow Duration",
    "Tot Fwd Pkts", "Tot Bwd Pkts",
    "TotLen Fwd Pkts", "TotLen Bwd Pkts",
    "Fwd Pkt Len Max", "Fwd Pkt Len Min",
    "Fwd Pkt Len Mean", "Fwd Pkt Len Std",
    "Bwd Pkt Len Max", "Bwd Pkt Len Min",
    "Bwd Pkt Len Mean", "Bwd Pkt Len Std",
    "Flow Byts/s", "Flow Pkts/s",
    "Flow IAT Mean", "Flow IAT Std",
    "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Tot", "Fwd IAT Mean",
    "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
    "Bwd IAT Tot", "Bwd IAT Mean",
    "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
    "Fwd Header Len", "Bwd Header Len",
    "Fwd Pkts/s", "Bwd Pkts/s",
    "Pkt Len Min", "Pkt Len Max",
    "Pkt Len Mean", "Pkt Len Std", "Pkt Len Var",
    "Pkt Size Avg",
    "Active Mean", "Active Std", "Active Max", "Active Min",
    "Idle Mean", "Idle Std", "Idle Max", "Idle Min"
]

def feature_selection_multi():
    # --- 디렉토리 경로 ---
    RAW_DIR = "./raw_datas"
    SAVE_DIR = "./ae_datas"
    os.makedirs(SAVE_DIR, exist_ok=True)

    # --- 정상 데이터 처리 ---
    normal_path = os.path.join(RAW_DIR, "InSDN_normal.csv")
    if os.path.exists(normal_path):
        df_normal = pd.read_csv(normal_path, low_memory=False)
        # df_normal_filtered = df_normal[SELECTED_FEATURES]
        df_normal_filtered = df_normal
        save_path = os.path.join(SAVE_DIR, "InSDN_normal_48.csv")
        df_normal_filtered.to_csv(save_path, index=False)
        print(f"✅ 정상 데이터 저장 완료: {df_normal_filtered.shape} → {save_path}")
    else:
        print("❌ 정상 데이터 파일이 없습니다.")

    # --- 비정상 데이터 처리 (anomaly_0~4) ---
    for i in range(5):
        anomaly_file = os.path.join(RAW_DIR, f"InSDN_anomaly_{i}.csv")
        if not os.path.exists(anomaly_file):
            print(f"⚠️ {anomaly_file} 없음, 건너뜀.")
            continue

        df_anomaly = pd.read_csv(anomaly_file, low_memory=False)

        # feature 존재 여부 확인
        missing_cols = [f for f in SELECTED_FEATURES if f not in df_anomaly.columns]
        if missing_cols:
            print(f"⚠️ {anomaly_file}에 누락된 컬럼 존재: {missing_cols[:5]} ...")

        df_anomaly_filtered = df_anomaly[[f for f in SELECTED_FEATURES if f in df_anomaly.columns]]
        save_path = os.path.join(SAVE_DIR, f"InSDN_anomaly_{i}_48.csv")
        df_anomaly_filtered.to_csv(save_path, index=False)
        print(f"🚨 이상 데이터 {i} 저장 완료: {df_anomaly_filtered.shape} → {save_path}")

    print("\n🎯 Feature Selection 전체 완료!")
    print(f"📁 결과 저장 폴더: {SAVE_DIR}")

def split_and_save_labelwise():
    # 🔹 폴더 내 모든 CSV 읽기
    DATA_DIR = "./InSDN_DatasetCSV"
    SAVE_DIR = "./ae_datas"
    os.makedirs(SAVE_DIR, exist_ok=True)

    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    if not csv_files:
        raise ValueError(f"❌ {DATA_DIR} 폴더에 CSV 파일이 없습니다.")

    print(f"📂 총 {len(csv_files)}개 CSV 파일을 병합합니다...")

    # 파일 통합
    df_list = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            print(f"  → {os.path.basename(file)}: {len(df)} rows")
            df_list.append(df)
        except Exception as e:
            print(f"⚠️ {file} 읽기 실패: {e}")

    df = pd.concat(df_list, ignore_index=True)
    print(f"\n📊 통합 데이터 크기: {len(df)} rows, 컬럼 {len(df.columns)}개")

    # Label 컬럼 확인
    if "Label" not in df.columns:
        raise ValueError("⚠️ 'Label' 컬럼이 없습니다. CSV 구조를 확인하세요.")

    # ✅ 라벨 문자열 정규화 추가
    df["Label"] = df["Label"].astype(str).str.strip().str.upper()

    # 선택된 피처 필터링
    # features_to_keep = [col for col in SELECTED_FEATURES if col in df.columns]
    # missing = set(SELECTED_FEATURES) - set(features_to_keep)
    # if missing:
    #     print(f"⚠️ 누락된 피처 {len(missing)}개: {missing}")

    # df = df[features_to_keep + ["Label"]]

    # Normal / Anomaly 분리
    df_normal = df[df["Label"].str.lower() == "normal"]
    df_anomaly = df[df["Label"].str.lower() != "normal"]

    # 저장
    normal_path = os.path.join(SAVE_DIR, "InSDN_normal.csv")
    anomaly_path = os.path.join(SAVE_DIR, "InSDN_anomaly.csv")

    (df_normal.drop(columns=["Label"])).to_csv(normal_path, index=False)
    (df_anomaly.drop(columns=["Label"])).to_csv(anomaly_path, index=False)

    print(f"\n✅ Normal 저장 완료: {len(df_normal)}개 → {normal_path}")
    print(f"✅ Anomaly 저장 완료: {len(df_anomaly)}개 → {anomaly_path}")

    # 라벨별 인코딩 및 저장
    label_map = {label: i for i, label in enumerate(sorted(df_anomaly["Label"].unique()))}
    print("\n🚨 비정상 라벨 인코딩 매핑:")
    for lbl, idx in label_map.items():
        print(f"{idx}: {lbl}")

    for lbl, idx in label_map.items():
        subset = df_anomaly[df_anomaly["Label"] == lbl]
        subset = subset.drop(columns=["Label"])
        file_name = f"InSDN_anomaly_{idx}.csv"
        subset.to_csv(os.path.join(SAVE_DIR, file_name), index=False)
        print(f"   ⤷ {lbl}: {len(subset)}개 저장 ({file_name})")

    print("\n🎯 전체 완료!")
    print(f" - Normal: {len(df_normal)}개")
    print(f" - Anomaly 총합: {len(df_anomaly)}개")
    print(f" - 라벨 수: {len(label_map)}개")
    print(f" - 저장 위치: {SAVE_DIR}")
    print(df_normal.shape)
    print(df_anomaly.shape)


if __name__ == "__main__":
    split_and_save_labelwise()