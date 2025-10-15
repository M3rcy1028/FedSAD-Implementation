import kagglehub
import pandas as pd
import os

def download():
    # KaggleHub로 데이터셋 다운로드
    path = kagglehub.dataset_download("toobajamal/kdd99-dataset")
    print(os.listdir(path))   # ['kddcup99_csv.csv'] 확인됨

    # ✅ CSV 파일 경로 지정해서 읽기
    df = pd.read_csv(os.path.join(path, "kddcup99_csv.csv"))

    print(df.shape)
    print(df.head())

def EncodeDataset():
    df = pd.read_csv("kddcup99.csv")

    df_encoded = pd.get_dummies(df, columns=['service', 'flag'], dtype=int)
    df_encoded.to_csv("KDD99_encoded.csv", index=False)

def SplitDataset():
    # 파일명
    encode_csv = "KDD99_encoded.csv"

    # 데이터 불러오기
    df = pd.read_csv(encode_csv)
    # 조건: label protocol_type
    normal_df = df[(df['label'] == 'normal') & (df['protocol_type'] == 'tcp')]
    anomaly_df = df[(df['label'] != 'normal') & (df['protocol_type'] == 'tcp')]

    normal_df = normal_df.drop(columns='protocol_type', axis=1)
    anomaly_df = anomaly_df.drop(columns='protocol_type', axis=1)
    normal_df = normal_df.drop(columns='label', axis=1)
    anomaly_df = anomaly_df.drop(columns='label', axis=1)

    print(f"normal_df: {len(normal_df)} rows")
    print(f"anomaly_df: {len(anomaly_df)} rows")

    # 저장
    normal_df.to_csv("KDD99_normal.csv", index=False)
    anomaly_df.to_csv("KDD99_anomaly.csv", index=False)

EncodeDataset()
SplitDataset()