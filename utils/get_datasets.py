import os
import pandas as pd
import numpy as np
import random
from typing import List, Optional

from sklearn.preprocessing import MinMaxScaler 
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
from utils.metrics import plot_roc_curve, plt_confusion_matrix

RANDOM_SEED = 123


def rnep_normal_split(n_rows: int, train_ratio: float = 0.8, seed: int = RANDOM_SEED):
    """Positional train/test row indices exactly as the RNEP loaders split normal data
    (sklearn shuffle with RANDOM_SEED, first int(n * train_ratio) rows = train).
    Apply it to the CSV in file order, before any sorting, so GSAD/FedSAD use the same rows as RNEP."""
    perm = shuffle(np.arange(n_rows), random_state=seed)
    split_point = int(n_rows * train_ratio)
    return np.sort(perm[:split_point]), np.sort(perm[split_point:])


def normal_split_indices(df: pd.DataFrame):
    """Normal train/test positional indices. Uses an explicit `split` column (train/test)
    when the CSV carries one (e.g. GeNIS frame data built from the RNEP training pipeline),
    otherwise the RNEP random split (rnep_normal_split)."""
    if "split" in df.columns:
        split = df["split"].astype(str).values
        return np.flatnonzero(split == "train"), np.flatnonzero(split == "test")
    return rnep_normal_split(len(df))

# For datasets
def get_datasets_nsl():
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    df_normal = pd.read_csv("./NSL-KDD/KDD_normal.csv")
    df_anomaly = pd.read_csv("./NSL-KDD/KDD_anomaly.csv")

    df_normal = shuffle(df_normal, random_state=RANDOM_SEED)

    scaler = MinMaxScaler()
    mid_idx = len(df_normal) // 2
    df_normal_train = df_normal[:mid_idx]
    df_normal_test = df_normal[mid_idx:]

    X_train_scaled = scaler.fit_transform(df_normal_train)
    df_test = pd.concat([df_normal_test, df_anomaly], ignore_index=True)
    y_test = np.concatenate([np.zeros(len(df_normal_test)), np.ones(len(df_anomaly))])
    X_test_scaled, y_test = shuffle(scaler.transform(df_test), y_test, random_state=RANDOM_SEED)

    return X_train_scaled, X_test_scaled, y_test

def get_datasets_kdd99():
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    df_normal = pd.read_csv("./KDD99/KDD99_normal.csv")
    df_anomaly = pd.read_csv("./KDD99/KDD99_anomaly.csv")

    df_normal = shuffle(df_normal, random_state=RANDOM_SEED)

    scaler = MinMaxScaler()
    mid_idx = len(df_normal) // 2
    df_normal_train = df_normal[:mid_idx]
    df_normal_test = df_normal[mid_idx:]

    X_train_scaled = scaler.fit_transform(df_normal_train)
    df_test = pd.concat([df_normal_test, df_anomaly], ignore_index=True)
    y_test = np.concatenate([np.zeros(len(df_normal_test)), np.ones(len(df_anomaly))])
    X_test_scaled, y_test = shuffle(scaler.transform(df_test), y_test, random_state=random_seed) #TODO random_seed 정의 필요

    return X_train_scaled, X_test_scaled, y_test

# CIC
# def get_datasets_cic(random_seed=args.random_seed):
#     np.random.seed(random_seed)
#     random.seed(random_seed)

#     normal_path = "./CIC2018/ae_datas_all_features/CIC_ae_normal.csv"
#     df_normal = pd.read_csv(normal_path, low_memory=False)

#     # 클린업
#     df_normal = df_normal.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
#     df_normal[df_normal < 0] = 0
#     df_normal = shuffle(df_normal, random_state=random_seed)

#     anomaly_files = [
#         f"./CIC2018/ae_datas_all_features/CIC_anomaly_ae_{i}.csv" for i in range(1, 15)
#     ]
#     anomaly_dfs = []
#     for path in anomaly_files:
#         if os.path.exists(path):
#             df_temp = pd.read_csv(path, low_memory=False)
#             anomaly_dfs.append(df_temp)
#         else:
#             print(f"⚠️ Warning: {path} not found, skipping.")

#     df_anomaly = pd.concat(anomaly_dfs, ignore_index=True)
#     df_anomaly = shuffle(df_anomaly, random_state=random_seed)
#     df_anomaly = df_anomaly.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
#     df_anomaly[df_anomaly < 0] = 0

#     # ---------------------------
#     # (3) 개수 제한 (각 150,000개)
#     # ---------------------------
#     N_SAMPLES = args.n_samples
#     df_normal = df_normal.sample(n=min(len(df_normal), N_SAMPLES * 2), random_state=random_seed)
#     df_anomaly = df_anomaly.sample(n=min(len(df_anomaly), N_SAMPLES), random_state=random_seed)

#     mid_idx = len(df_normal) // 2
#     df_normal_train = df_normal.iloc[:N_SAMPLES].copy() if len(df_normal) >= N_SAMPLES else df_normal.iloc[:mid_idx]
#     df_normal_test = df_normal.iloc[-N_SAMPLES:].copy() if len(df_normal) >= N_SAMPLES * 2 else df_normal.iloc[mid_idx:]

#     print(f"정상 데이터 총 {len(df_normal)}개 → Train {len(df_normal_train)}, Test {len(df_normal_test)}")
#     print(f"이상 데이터 총 {len(df_anomaly)}개 (모두 테스트에 사용)")

#     # ---------------------------
#     # (4) MinMax 정규화
#     # ---------------------------
#     scaler = MinMaxScaler()
#     X_train = scaler.fit_transform(df_normal_train.values)

#     df_test = pd.concat([df_normal_test, df_anomaly], ignore_index=True)
#     X_test = scaler.transform(df_test.values)
#     y_test = np.concatenate([
#         np.zeros(len(df_normal_test)),
#         np.ones(len(df_anomaly))
#     ])

#     # ---------------------------
#     # (5) 셔플 & 리턴
#     # ---------------------------
#     X_test, y_test = shuffle(X_test, y_test, random_state=random_seed)

#     print(f"최종 Train shape: {X_train.shape}, Test shape: {X_test.shape}")
#     print(f"y_test: Normal={np.sum(y_test==0)}, Anomaly={np.sum(y_test==1)}")

#     return X_train, X_test, y_test

def get_datasets_cic_sam():
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    # ---------------------------
    # (1) 정상 데이터 로드 & 정리
    # ---------------------------
    normal_path = "./CIC2018/sampled_2/CIC_ae_normal.csv"
    df_normal = pd.read_csv(normal_path, low_memory=False)

    # 공통 피처만 사용 (파일에 존재하는 컬럼만 선택)

    # 클린업
    df_normal = df_normal.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
    df_normal[df_normal < 0] = 0
    df_normal = shuffle(df_normal, random_state=RANDOM_SEED)

    # ---------------------------
    # (2) 이상 데이터 로드 & 정리
    # ---------------------------
    # anomaly_files = [
    #     f"./CIC2018/ae_datas_sampled/CIC_anomaly_ae_{i}.csv" for i in range(1, 15)
    # ]
    # anomaly_dfs = []
    # for path in anomaly_files:
    #     if os.path.exists(path):
    #         df_temp = pd.read_csv(path, low_memory=False)
    #         # use_cols = [c for c in existing_cols if c in df_temp.columns]
    #         # df_temp = df_temp[use_cols].copy()
    #         anomaly_dfs.append(df_temp)
    #     else:
    #         print(f"⚠️ Warning: {path} not found, skipping.")

    # df_anomaly = pd.concat(anomaly_dfs, ignore_index=True)
    anomlay_path = "./CIC2018/sampled_2/CIC_ae_anomaly.csv"
    df_anomaly = pd.read_csv(anomlay_path, low_memory=False)

    df_anomaly = shuffle(df_anomaly, random_state=RANDOM_SEED)
    df_anomaly = df_anomaly.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
    df_anomaly[df_anomaly < 0] = 0

    split_point = int(len(df_normal) * 0.8)
    df_normal_train = df_normal.iloc[:split_point]
    df_normal_test = df_normal.iloc[split_point:]

    print(f"정상 데이터 총 {len(df_normal)}개 → Train {len(df_normal_train)}, Test {len(df_normal_test)}")
    print(f"이상 데이터 총 {len(df_anomaly)}개 (모두 테스트에 사용)")

    # ---------------------------
    # (4) MinMax 정규화
    # ---------------------------
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(df_normal_train.values)

    df_test = pd.concat([df_normal_test, df_anomaly], ignore_index=True)
    X_test = scaler.transform(df_test.values)
    y_test = np.concatenate([
        np.zeros(len(df_normal_test)),
        np.ones(len(df_anomaly))
    ])

    # ---------------------------
    # (5) 셔플 & 리턴
    # ---------------------------
    X_test, y_test = shuffle(X_test, y_test, random_state=RANDOM_SEED)

    print(f"최종 Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"y_test: Normal={np.sum(y_test==0)}, Anomaly={np.sum(y_test==1)}")

    return X_train, X_test, y_test

# InSDN
def get_datasets_insdn():
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    # ---------------------------
    # (1) Load dataset
    # ---------------------------
    normal_path = "./InSDN/raw_datas/InSDN_normal.csv"
    anomaly_path = "./InSDN/raw_datas/InSDN_anomaly.csv"

    df_normal = pd.read_csv(normal_path)
    df_anomaly = pd.read_csv(anomaly_path)

    print(f"✅ Loaded InSDN data → Normal: {df_normal.shape}, Anomaly: {df_anomaly.shape}")

    # ---------------------------
    # (2) Shuffle normal data
    # ---------------------------
    df_normal = shuffle(df_normal, random_state=RANDOM_SEED)

    # ---------------------------
    # (3) Split train/test for normal
    # ---------------------------
    split_point = int(len(df_normal) * 0.8)
    df_normal_train = df_normal.iloc[:split_point]
    df_normal_test = df_normal.iloc[split_point:]

    # ---------------------------
    # (4) Preprocessing: numeric cleanup
    # ---------------------------
    df_normal_train = df_normal_train.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
    df_normal_test = df_normal_test.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
    df_anomaly = df_anomaly.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)

    def _clean(df):
        df = df.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
        # ✅ Drop last column
        return df.iloc[:, :-1]

    df_normal_train = _clean(df_normal_train)
    df_normal_test = _clean(df_normal_test)
    df_anomaly = _clean(df_anomaly)
    
    # ---------------------------
    # (5) Scaling (MinMax)
    # ---------------------------
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(df_normal_train.values)

    df_test = pd.concat([df_normal_test, df_anomaly], ignore_index=True)
    X_test_scaled = scaler.transform(df_test.values)

    # ---------------------------
    # (6) Labeling (0=Normal, 1=Anomaly)
    # ---------------------------
    y_test = np.concatenate([np.zeros(len(df_normal_test)), np.ones(len(df_anomaly))])

    # ---------------------------
    # (7) Shuffle test set
    # ---------------------------
    X_test_scaled, y_test = shuffle(X_test_scaled, y_test, random_state=RANDOM_SEED)

    # ---------------------------
    # (8) Print summary
    # ---------------------------
    print(f"Train: {X_train_scaled.shape}, Test: {X_test_scaled.shape}, y_test: {y_test.shape}")
    print(f"Normal train: {len(df_normal_train)}, Normal test: {len(df_normal_test)}, Anomaly: {len(df_anomaly)}")

    return X_train_scaled, X_test_scaled, y_test

# UNSW_NB15
def get_datasets_unsw():
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    # ---------------------------
    # (1) Load dataset
    # ---------------------------
    normal_path = "./UNSW_NB15/ae_datas/UNSW_NB15_normal.csv"
    anomaly_path = "./UNSW_NB15/ae_datas/UNSW_NB15_anomaly.csv"

    df_normal = pd.read_csv(normal_path)
    df_anomaly = pd.read_csv(anomaly_path)

    print(f"✅ Loaded UNSW_NB15 data → Normal: {df_normal.shape}, Anomaly: {df_anomaly.shape}")

    # ---------------------------
    # (2) Shuffle normal data
    # ---------------------------
    df_normal = shuffle(df_normal, random_state=RANDOM_SEED)

    # ---------------------------
    # (3) Split train/test for normal
    # ---------------------------
    split_point = int(len(df_normal) * 0.8)
    df_normal_train = df_normal.iloc[:split_point]
    df_normal_test = df_normal.iloc[split_point:]

    # ---------------------------
    # (4) Preprocessing: numeric cleanup
    # ---------------------------
    df_normal_train = df_normal_train.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
    df_normal_test = df_normal_test.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
    df_anomaly = df_anomaly.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # ---------------------------
    # (5) Scaling (MinMax)
    # ---------------------------
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(df_normal_train.values)

    df_test = pd.concat([df_normal_test, df_anomaly], ignore_index=True)
    X_test_scaled = scaler.transform(df_test.values)

    # ---------------------------
    # (6) Labeling (0=Normal, 1=Anomaly)
    # ---------------------------
    y_test = np.concatenate([np.zeros(len(df_normal_test)), np.ones(len(df_anomaly))])

    # ---------------------------
    # (7) Shuffle test set
    # ---------------------------
    X_test_scaled, y_test = shuffle(X_test_scaled, y_test, random_state=RANDOM_SEED)

    # ---------------------------
    # (8) Print summary
    # ---------------------------
    print(f"Train: {X_train_scaled.shape}, Test: {X_test_scaled.shape}, y_test: {y_test.shape}")
    print(f"Normal train: {len(df_normal_train)}, Normal test: {len(df_normal_test)}, Anomaly: {len(df_anomaly)}")

    return X_train_scaled, X_test_scaled, y_test

def eval_server(model, X_train_scaled, X_test_scaled, y_test, result_path, matrix_path, roc_path):
    from utils.arguments import get_args
    args = get_args()
    
    X_test_pred = model.predict(X_test_scaled, verbose=0)
    recon_errors = np.mean(np.square(X_test_scaled - X_test_pred), axis=1)

    X_train_pred = model.predict(X_train_scaled, verbose=0)
    train_errors = np.mean(np.square(X_train_scaled - X_train_pred), axis=1)

    threshold = np.percentile(train_errors, args.percentile)
    y_pred = (recon_errors > threshold).astype(int)
    print(f"[Threshold: {threshold}%]")

    server_report = classification_report(
        y_test, y_pred,
        target_names=["Normal", "Anomaly"],
        zero_division=0
    )
    #TODO modify save_report function
    print("\n📊 [Server Classification Report]\n")
    print(server_report)

    with open(result_path, "a") as f:
        f.write("\n📊 [Server Classification Report]\n")
        f.write(server_report + "\n")

    plt_confusion_matrix(y_test, y_pred, matrix_path)

    auc_score = plot_roc_curve(
                    y_test, recon_errors, 
                    roc_path=roc_path, 
                    title="Server ROC Curve"
                )
    
    return server_report


def sanitize_numeric(df: pd.DataFrame, exclude: Optional[List[str]] = None) -> pd.DataFrame:
    clean = df.copy()
    exclude = exclude or []
    
    # 문자열 컬럼들을 숫자로 매핑
    if 'proto' in clean.columns:
        proto_mapping = {
            'tcp': 6, 'udp': 17, 'icmp': 1, 'igmp': 2, 'sctp': 132,
            'ospf': 89, 'unas': 0, 'others': 255, 'pipe': 99
        }
        clean['proto'] = clean['proto'].map(proto_mapping).fillna(0).astype(int)
    
    if 'state' in clean.columns:
        state_mapping = {
            'FIN': 1, 'CON': 2, 'ECO': 3, 'REQ': 4, 'RST': 5, 
            'PAR': 6, 'URN': 7, 'no': 0, 'CLO': 8, 'TXD': 9,
            'ACC': 10, 'INT': 11
        }
        clean['state'] = clean['state'].map(state_mapping).fillna(0).astype(int)
    
    target_cols = [c for c in clean.columns if c not in exclude]
    if target_cols:
        numeric = (
            clean[target_cols]
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
        )
        numeric[numeric < 0] = 0
        clean[target_cols] = numeric
    return clean

def _pair_files(taae_files, gsad_files):
    """Pair RNEP/GSAD anomaly CSVs by file name (os.listdir order differs between dirs)."""
    taae_by_name = {os.path.basename(f): f for f in taae_files}
    gsad_by_name = {os.path.basename(f): f for f in gsad_files}
    if set(taae_by_name) != set(gsad_by_name):
        raise ValueError(f"RNEP/GSAD anomaly files differ: "
                         f"{sorted(set(taae_by_name) ^ set(gsad_by_name))}")
    return [(taae_by_name[n], gsad_by_name[n]) for n in sorted(taae_by_name)]


def _gsad_by_time(df_gsad):
    """Parse Timestamp, drop unparsable rows, index by time (row_id kept as a column)."""
    df = df_gsad.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], dayfirst=True, errors="coerce")
    df = df[df["Timestamp"].notna()]
    df = sanitize_numeric(df, exclude=["Timestamp", "row_id"])
    df.set_index("Timestamp", inplace=True)
    df.sort_index(inplace=True, kind="stable")
    return df


def fedsad_data_preprocessing(taae_normal_file, taae_anomaly_file,
                              gsad_normal_file, gsad_anomaly_file, SCALER, prescaled=False):
    """RNEP and GSAD CSVs are row-aligned (same flow on the same line). Every row gets a
    `row_id`; the normal split is taken in file order exactly like the RNEP loaders
    (rnep_normal_split), and GSAD windows later fetch their RNEP rows by row_id.

    Returns:
        df_gsad_normal_test / df_gsad_anomaly_test: Timestamp index, columns incl. `row_id`
        df_taae_normal_test / df_taae_anomaly_test: indexed by `row_id`, feature columns only
        train_normal_scaled: scaled RNEP train rows (for the RNEP threshold)
    """
    # ---- Normal ----
    df_taae_normal_raw = pd.read_csv(taae_normal_file)
    df_gsad_normal_raw = pd.read_csv(gsad_normal_file)
    if len(df_taae_normal_raw) != len(df_gsad_normal_raw):
        raise ValueError(f"RNEP/GSAD normal CSVs are not row-aligned: "
                         f"{len(df_taae_normal_raw)} vs {len(df_gsad_normal_raw)} rows")
    print(f"[INFO] Normal data loaded: {len(df_taae_normal_raw)} rows (RNEP/GSAD row-aligned)")

    df_taae_normal_raw.index.name = "row_id"
    df_gsad_normal_raw["row_id"] = np.arange(len(df_gsad_normal_raw))

    train_idx, test_idx = normal_split_indices(df_gsad_normal_raw)
    df_gsad_normal_raw = df_gsad_normal_raw.drop(columns=["split"], errors="ignore")

    df_taae_normal_train_raw = df_taae_normal_raw.iloc[train_idx]
    df_taae_normal_test = df_taae_normal_raw.iloc[test_idx].copy()
    df_gsad_normal_test = _gsad_by_time(df_gsad_normal_raw.iloc[test_idx])

    # RNEP scaler (fit on RNEP train rows only)
    # prescaled: vectors are already imputed/scaled/encoded (keep negatives, no clipping)
    taae_train_features = df_taae_normal_train_raw if prescaled else sanitize_numeric(df_taae_normal_train_raw)
    SCALER.fit(taae_train_features.values)
    train_normal_scaled = SCALER.transform(taae_train_features.values)
    print(f"[INFO] TAAE scaler fitted on {train_normal_scaled.shape[1]} features "
          f"({len(train_idx)} train / {len(test_idx)} test normal rows)")

    # ---- Anomaly (all used for test) ----
    taae_list, gsad_list, offset = [], [], 0
    for taae_path, gsad_path in _pair_files(taae_anomaly_file, gsad_anomaly_file):
        df_t = pd.read_csv(taae_path)
        df_g = pd.read_csv(gsad_path)
        if len(df_t) != len(df_g):
            raise ValueError(f"{os.path.basename(taae_path)}: RNEP/GSAD not row-aligned "
                             f"({len(df_t)} vs {len(df_g)} rows)")
        ids = np.arange(offset, offset + len(df_t))
        df_t.index = pd.Index(ids, name="row_id")
        df_g["row_id"] = ids
        offset += len(df_t)
        taae_list.append(df_t)
        gsad_list.append(df_g)

    df_taae_anomaly_test = pd.concat(taae_list)
    df_gsad_anomaly_test = _gsad_by_time(pd.concat(gsad_list, ignore_index=True))

    print(f"[INFO] GSAD Normal test rows: {len(df_gsad_normal_test)} | GSAD Anomaly test rows: {len(df_gsad_anomaly_test)}")
    print(f"[INFO] TAAE Normal test rows: {len(df_taae_normal_test)} | TAAE Anomaly test rows: {len(df_taae_anomaly_test)}")
    return df_gsad_normal_test, df_gsad_anomaly_test, df_taae_normal_test, df_taae_anomaly_test, train_normal_scaled

