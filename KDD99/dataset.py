import kagglehub
import pandas as pd
import os
from kagglehub import KaggleDatasetAdapter


def download():
    # ✅ 다운로드할 파일명 지정
    file_path = "kddcup99_csv.csv"

    # ✅ 최신 KaggleHub 방식으로 호출 (Deprecation Warning 해결)
    dataset = kagglehub.dataset_load(
        KaggleDatasetAdapter.HUGGING_FACE,  # 어댑터 지정
        "toobajamal/kdd99-dataset",         # 데이터셋 핸들
        file_path,                          # 내부 파일 경로
        pandas_kwargs={"low_memory": False} # 옵션 (pandas read_csv 인자)
    )

    print("✅ Dataset successfully loaded via KaggleHub.")
    print(f"📂 File path: {file_path}")
    print(f"Data shape: {dataset.shape}")

    # ✅ 로컬로 저장도 가능
    save_path = "./KDD99_data"
    os.makedirs(save_path, exist_ok=True)
    dataset.to_csv(os.path.join(save_path, "kddcup99_csv.csv"), index=False)
    print(f"💾 Saved CSV to {save_path}/kddcup99_csv.csv")

def EncodeDataset():
    df = pd.read_csv("./raw_data/kddcup99_csv.csv")

    df_encoded = pd.get_dummies(df, columns=['service', 'flag'], dtype=int)
    df_encoded.to_csv("KDD99_encoded.csv", index=False)


def SplitDataset():
    # 파일명
    encode_csv = "KDD99_encoded.csv"
    save_dir = "./KDD99_split"
    os.makedirs(save_dir, exist_ok=True)

    # 데이터 불러오기
    df = pd.read_csv(encode_csv)

    # ✅ TCP 기반만 사용
    df_tcp = df[df["protocol_type"] == "tcp"]
    print(f"전체 데이터 크기: {df_tcp.shape}")
    
    # ✅ 정상 / 이상 분리
    df_normal = df_tcp[df_tcp["label"] == "normal"].drop(columns=["protocol_type", "label"], errors="ignore")
    df_anomaly = df_tcp[df_tcp["label"] != "normal"].drop(columns=["protocol_type"], errors="ignore")

    print(f"✅ Normal: {len(df_normal)} rows")
    print(f"✅ Anomaly: {len(df_anomaly)} rows")

    # ✅ 기본 normal/anomaly 저장
    df_normal.to_csv(os.path.join(save_dir, "KDD99_normal.csv"), index=False)
    df_anomaly.to_csv(os.path.join(save_dir, "KDD99_anomaly.csv"), index=False)
    print("💾 Saved KDD99_normal.csv / KDD99_anomaly.csv")

    # ---------------------------------------------------
    # 🔹 공격 유형별 분리 (자동 인덱스 부여)
    # ---------------------------------------------------
    unique_labels = sorted(df_tcp[df_tcp["label"] != "normal"]["label"].unique())
    label_map = {}
    summary = []

    for idx, label in enumerate(unique_labels):
        subset = df_tcp[df_tcp["label"] == label].drop(columns=["protocol_type", "label"], errors="ignore")
        file_name = f"KDD99_anomaly_{idx}.csv"
        save_path = os.path.join(save_dir, file_name)
        subset.to_csv(save_path, index=False)

        label_map[idx] = label
        summary.append({"인덱스": idx, "공격 유형": label, "샘플 수": len(subset)})
        print(f"   ⤷ {idx}: {label} ({len(subset)} rows) → {file_name}")

    # ✅ 매핑 테이블 저장
    df_summary = pd.DataFrame(summary)
    map_path = os.path.join(save_dir, "KDD99_label_map.csv")
    df_summary.to_csv(map_path, index=False, encoding="utf-8-sig")
    print(f"\n📁 저장 완료 → {map_path}")
    print(df_summary)

EncodeDataset()
SplitDataset()