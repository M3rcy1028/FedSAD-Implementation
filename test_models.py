import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import tensorflow as tf
from model_taae_rnep import TransformerAAE
import matplotlib.pyplot as plt
import seaborn as sns

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# --------------------------------------------------
# KDD99 Dataset Evaluation Function
# --------------------------------------------------
def evaluate_taae_kdd99(model, base_dir="./KDD99/KDD99_split", features=117, percentile=95):
    normal_path = os.path.join(base_dir, "KDD99_normal.csv")

    # anomaly_x.csv 리스트 추출
    anomaly_files = sorted([
        f for f in os.listdir(base_dir)
        if f.startswith("KDD99_anomaly_") and f.endswith(".csv")
    ])

    # ✅ 전체 anomaly 합본 파일도 함께 평가
    anomaly_all_path = os.path.join(base_dir, "KDD99_anomaly.csv")
    if os.path.exists(anomaly_all_path):
        anomaly_files = ["KDD99_anomaly.csv"] + anomaly_files

    print(f"\n📊 Found {len(anomaly_files)} anomaly datasets for KDD99 evaluation")

    results = []

    # ✅ 정상 데이터 로드 및 절반 분리 (Train / Test)
    df_normal = pd.read_csv(normal_path)
    df_normal = df_normal.sample(frac=1, random_state=123).reset_index(drop=True)
    split_point = int(len(df_normal) * 0.8)
    df_normal_train = df_normal.iloc[:split_point]
    df_normal_test = df_normal.iloc[split_point:]

    # 정규화 fit은 Train(normal)으로만 수행
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(df_normal_train)
    X_normal_test = scaler.transform(df_normal_test)

    # ✅ Train(normal) reconstruction error → threshold 계산
    preds_train = model.predict(X_train, verbose=0)
    train_errors = np.mean(np.square(X_train - preds_train), axis=1)
    threshold = np.percentile(train_errors, percentile)
    print(f"\n📏 Threshold ({percentile}th percentile): {threshold:.6f}")

    # 시각화를 위해 합본(merged) anomaly 데이터를 저장할 변수
    df_merged_anomaly_for_plot = None 

    # --------------------------------------------------
    # 각 anomaly_i.csv 에 대해 평가
    # --------------------------------------------------
    # for file in anomaly_files:
    #     anomaly_path = os.path.join(base_dir, file)
    #     df_anomaly = pd.read_csv(anomaly_path)
    #     df_anomaly = df_anomaly.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)

    #     # ✅ label 컬럼이 있으면 제거
    #     if "label" in df_anomaly.columns:
    #         df_anomaly = df_anomaly.drop(columns=["label"])

    #     # KDD99_anomaly.csv (합본) 데이터를 시각화용으로 저장
    #     if file == "KDD99_anomaly.csv" and df_merged_anomaly_for_plot is None:
    #          df_merged_anomaly_for_plot = df_anomaly.copy()

    #     X_anomaly = scaler.transform(df_anomaly)
    #     X_test = np.concatenate([X_normal_test, X_anomaly])
    #     y_test = np.concatenate([np.zeros(len(X_normal_test)), np.ones(len(X_anomaly))])

    #     preds_test = model.predict(X_test, verbose=0)
    #     test_errors = np.mean(np.square(X_test - preds_test), axis=1)
    #     y_pred = (test_errors > threshold).astype(int)

    #     # Metrics 계산
    #     acc = accuracy_score(y_test, y_pred)
    #     prec = precision_score(y_test, y_pred, zero_division=0)
    #     rec = recall_score(y_test, y_pred, zero_division=0)
    #     f1 = f1_score(y_test, y_pred, zero_division=0)

    #     pred_normal = int(np.sum(y_pred == 0))
    #     pred_anomaly = int(np.sum(y_pred == 1))

    #     print(f"\n🚨 {file}")
    #     print(f"Samples: {len(X_test)} | Accuracy={acc:.6f}, Precision={prec:.6f}, Recall={rec:.6f}, F1={f1:.6f}")

    #     results.append({
    #         "File": file,
    #         "Samples": len(X_test),
    #         "Accuracy": acc,
    #         "Precision": prec,
    #         "Recall": rec,
    #         "F1": f1,
    #         "Pred_Normal": pred_normal,
    #         "Pred_Anomaly": pred_anomaly
    #     })

    # --------------------------------------------------
    # 결과 저장
    # --------------------------------------------------
    df = pd.DataFrame(results)
    save_path = os.path.join(base_dir, f"evaluation_mixed_summary_p{percentile}.csv")
    df.to_csv(save_path, index=False)
    print(f"\n📁 Saved summary → {save_path}")
    print(df.round(6))

    # --------------------------------------------------
    # 🔥 공격 유형별 Reconstruction Error 히스토그램 (Merged 제외)
    # --------------------------------------------------
    # (1) Normal error 계산
    preds_normal = model.predict(X_normal_test, verbose=0)
    errors_normal = np.mean(np.square(X_normal_test - preds_normal), axis=1)

    # (2) 공격 유형별 reconstruction error 저장 (번호 -> 이름 매핑, merged 제외)
    attack_name_map = {
        0: "back",
        1: "buffer_overflow",
        2: "ftp_write",
        3: "guess_passwd",
        4: "imap",
        5: "ipsweep",
        6: "land",
        7: "loadmodule",
        8: "multihop",
        9: "neptune",
        10: "nmap",
        11: "perl",
        12: "phf",
        13: "portsweep",
        14: "rootkit",
        15: "satan",
        16: "spy",
        17: "warezclient",
        18: "warezmaster"
    }

    error_by_attack = {}
    numeric_keys = []

    for file in anomaly_files:
        if file == "KDD99_anomaly.csv":  # skip merged file entirely
            continue

        # 파일명에서 숫자 추출 (예: KDD99_anomaly_3.csv -> 3)
        try:
            num_str = file.replace("KDD99_anomaly_", "").replace(".csv", "")
            attack_num = int(num_str)
        except Exception:
            # 파일명 포맷이 다르면 건너뜀
            continue

        path = os.path.join(base_dir, file)
        df_anomaly = pd.read_csv(path)
        df_anomaly = df_anomaly.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)

        X_anomaly = scaler.transform(df_anomaly.values)
        preds_anomaly = model.predict(X_anomaly, verbose=0)
        errors_anomaly = np.mean(np.square(X_anomaly - preds_anomaly), axis=1)

        # use mapped name if available, else fallback to numeric string
        attack_label = attack_name_map.get(attack_num, f"attack_{attack_num}")
        error_by_attack[attack_num] = (attack_label, errors_anomaly)
        numeric_keys.append(attack_num)

    # 정렬: 숫자 오름차순, 하지만 요청에 따라 0(back)이 있으면 맨 뒤로 보낸다
    numeric_keys = sorted(set(numeric_keys))

    # (3) 히스토그램 플롯 (Normal + 공격별)
    plt.figure(figsize=(12,7))

    # Normal 먼저
    plt.hist(errors_normal, bins=200, alpha=0.6, label="Normal", color="green", density=True)

    # 팔레트 (항목 수에 맞춤)
    n_attacks = len(numeric_keys)
    palette = plt.cm.tab20(np.linspace(0, 1, max(3, n_attacks)))  # 충분한 색 확보

    for i, atk_num in enumerate(numeric_keys):
        atk_name, errs = error_by_attack[atk_num]
        plt.hist(errs, bins=200, alpha=0.5, label=f"{atk_name}", color=palette[i], density=True)

    # Threshold
    plt.axvline(threshold, color="blue", linestyle="--", label=f"Threshold ({threshold:.6f})")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Reconstruction Error (log scale)")
    plt.ylabel("Density (log scale)")
    plt.title("Reconstruction Error per Attack Type (KDD99)")
    plt.legend(fontsize=8, loc="upper right", ncol=1)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig("./KDD99_distribution.png", dpi=300)
    plt.close()

    print("📊 Saved → ./KDD99_distribution.png")

def evaluate_taae_indsn(model, base_dir="./InSDN/ae_datas", features=117, percentile=95):
    normal_path = os.path.join(base_dir, "InSDN_normal.csv")

    # anomaly_x.csv 리스트 추출
    anomaly_files = sorted([
        f for f in os.listdir(base_dir)
        if f.startswith("InSDN_anomaly_") and f.endswith(".csv")
    ])

    # ✅ 전체 anomaly 합본 파일도 함께 평가
    anomaly_all_path = os.path.join(base_dir, "InSDN_anomaly.csv")
    if os.path.exists(anomaly_all_path):
        anomaly_files = ["InSDN_anomaly.csv"] + anomaly_files

    print(f"\n📊 Found {len(anomaly_files)} anomaly datasets for InSDN evaluation")

    results = []

    # ✅ 정상 데이터 로드 및 절반 분리 (Train / Test)
    df_normal = pd.read_csv(normal_path)
    df_normal = df_normal.sample(frac=1, random_state=123).reset_index(drop=True)
    split_point = int(len(df_normal) * 0.8)
    df_normal_train = df_normal.iloc[:split_point]
    df_normal_test = df_normal.iloc[split_point:]

    def _clean(df):
        df = df.apply(pd.to_numeric, errors="coerce")
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
        # 너무 큰 값 잘라내기 (예: 1e6 이상)
        df = np.clip(df, -1e6, 1e6)
        return df

    df_normal_train = _clean(df_normal_train)
    df_normal_test = _clean(df_normal_test)

    # 정규화 fit은 Train(normal)으로만 수행
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(df_normal_train)
    X_normal_test = scaler.transform(df_normal_test)

    # ✅ Train(normal) reconstruction error → threshold 계산
    preds_train = model.predict(X_train, verbose=0)
    train_errors = np.mean(np.square(X_train - preds_train), axis=1)
    threshold = np.percentile(train_errors, percentile)
    print(f"\n📏 Threshold ({percentile}th percentile): {threshold:.6f}")

    # 시각화를 위해 합본(merged) anomaly 데이터를 저장할 변수
    df_merged_anomaly_for_plot = None 

    # --------------------------------------------------
    # 각 anomaly_i.csv 에 대해 평가
    # --------------------------------------------------
    for file in anomaly_files:
        anomaly_path = os.path.join(base_dir, file)
        df_anomaly = pd.read_csv(anomaly_path)
        df_anomaly = df_anomaly.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)

        # ✅ label 컬럼이 있으면 제거
        if "label" in df_anomaly.columns:
            df_anomaly = df_anomaly.drop(columns=["label"])

        # InSDN_anomaly.csv (합본) 데이터를 시각화용으로 저장
        if file == "InSDN_anomaly.csv" and df_merged_anomaly_for_plot is None:
             df_merged_anomaly_for_plot = df_anomaly.copy()

        X_anomaly = scaler.transform(df_anomaly)
        X_test = np.concatenate([X_normal_test, X_anomaly])
        y_test = np.concatenate([np.zeros(len(X_normal_test)), np.ones(len(X_anomaly))])

        preds_test = model.predict(X_test, verbose=0)
        test_errors = np.mean(np.square(X_test - preds_test), axis=1)
        y_pred = (test_errors > threshold).astype(int)

        # Metrics 계산
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        pred_normal = int(np.sum(y_pred == 0))
        pred_anomaly = int(np.sum(y_pred == 1))

        print(f"\n🚨 {file}")
        print(f"Samples: {len(X_test)} | Accuracy={acc:.6f}, Precision={prec:.6f}, Recall={rec:.6f}, F1={f1:.6f}")

        results.append({
            "File": file,
            "Samples": len(X_test),
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1": f1,
            "Pred_Normal": pred_normal,
            "Pred_Anomaly": pred_anomaly
        })

    # --------------------------------------------------
    # 결과 저장
    # --------------------------------------------------
    df = pd.DataFrame(results)
    save_path = os.path.join(base_dir, f"evaluation_mixed_summary_p{percentile}.csv")
    df.to_csv(save_path, index=False)
    print(f"\n📁 Saved summary → {save_path}")
    print(df.round(6))

    # --------------------------------------------------
    # 🔥 공격 유형별 Reconstruction Error 히스토그램 (Merged 제외)
    # --------------------------------------------------
    # (1) Normal error 계산
    preds_normal = model.predict(X_normal_test, verbose=0)
    errors_normal = np.mean(np.square(X_normal_test - preds_normal), axis=1)

    # (2) 공격 유형별 reconstruction error 저장 (번호 -> 이름 매핑, merged 제외)
    attack_name_map = {
        0: "back",
        1: "buffer_overflow",
        2: "ftp_write",
        3: "guess_passwd",
        4: "imap",
        5: "ipsweep",
        6: "land",
        7: "loadmodule",
        8: "multihop",
        9: "neptune",
        10: "nmap",
        11: "perl",
        12: "phf",
        13: "portsweep",
        14: "rootkit",
        15: "satan",
        16: "spy",
        17: "warezclient",
        18: "warezmaster"
    }

    error_by_attack = {}
    numeric_keys = []

    for file in anomaly_files:
        if file == "InSDN_anomaly.csv":  # skip merged file entirely
            continue

        # 파일명에서 숫자 추출 (예: InSDN_anomaly_3.csv -> 3)
        try:
            num_str = file.replace("InSDN_anomaly_", "").replace(".csv", "")
            attack_num = int(num_str)
        except Exception:
            # 파일명 포맷이 다르면 건너뜀
            continue

        path = os.path.join(base_dir, file)
        df_anomaly = pd.read_csv(path)
        df_anomaly = df_anomaly.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)

        X_anomaly = scaler.transform(df_anomaly.values)
        preds_anomaly = model.predict(X_anomaly, verbose=0)
        errors_anomaly = np.mean(np.square(X_anomaly - preds_anomaly), axis=1)

        # use mapped name if available, else fallback to numeric string
        attack_label = attack_name_map.get(attack_num, f"attack_{attack_num}")
        error_by_attack[attack_num] = (attack_label, errors_anomaly)
        numeric_keys.append(attack_num)

    # 정렬: 숫자 오름차순, 하지만 요청에 따라 0(back)이 있으면 맨 뒤로 보낸다
    numeric_keys = sorted(set(numeric_keys))

    # (3) 히스토그램 플롯 (Normal + 공격별)
    plt.figure(figsize=(12,7))

    # Normal 먼저
    plt.hist(errors_normal, bins=200, alpha=0.6, label="Normal", color="green", density=True)

    # 팔레트 (항목 수에 맞춤)
    n_attacks = len(numeric_keys)
    palette = plt.cm.tab20(np.linspace(0, 1, max(3, n_attacks)))  # 충분한 색 확보

    for i, atk_num in enumerate(numeric_keys):
        atk_name, errs = error_by_attack[atk_num]
        plt.hist(errs, bins=200, alpha=0.5, label=f"{atk_name}", color=palette[i], density=True)

    # Threshold
    plt.axvline(threshold, color="blue", linestyle="--", label=f"Threshold ({threshold:.6f})")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Reconstruction Error (log scale)")
    plt.ylabel("Density (log scale)")
    plt.title("Reconstruction Error per Attack Type (InSDN)")
    plt.legend(fontsize=8, loc="upper right", ncol=1)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig("./InSDN_distribution.png", dpi=300)
    plt.close()

    print("📊 Saved → ./InSDN_distribution.png")

def evaluate_taae_cic(model, base_dir="./CIC2018/ae_datas_all_features", features=117, percentile=95):
    normal_path = os.path.join(base_dir, "CIC_ae_normal.csv")

    # anomaly_x.csv 리스트 추출
    anomaly_files = sorted([
        f for f in os.listdir(base_dir)
        if f.startswith("CIC_anomaly_ae_") and f.endswith(".csv")
    ])

    # ✅ 전체 anomaly 합본 파일도 함께 평가
    anomaly_all_path = os.path.join(base_dir, "CIC_anomaly.csv")
    if os.path.exists(anomaly_all_path):
        anomaly_files = ["CIC_anomaly.csv"] + anomaly_files

    print(f"\n📊 Found {len(anomaly_files)} anomaly datasets for CIC evaluation")

    results = []

    # ✅ 정상 데이터 로드 및 절반 분리 (Train / Test)
    df_normal = pd.read_csv(normal_path)
    df_normal = df_normal.sample(frac=1, random_state=123).reset_index(drop=True)
    split_point = int(len(df_normal) * 0.8)
    df_normal_train = df_normal.iloc[:split_point]
    df_normal_test = df_normal.iloc[split_point:]

    def _clean(df):
        df = df.apply(pd.to_numeric, errors="coerce")
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
        # 너무 큰 값 잘라내기 (예: 1e6 이상)
        df = np.clip(df, -1e6, 1e6)
        return df

    df_normal_train = _clean(df_normal_train)
    df_normal_test = _clean(df_normal_test)

    # 정규화 fit은 Train(normal)으로만 수행
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(df_normal_train)
    X_normal_test = scaler.transform(df_normal_test)

    # ✅ Train(normal) reconstruction error → threshold 계산
    preds_train = model.predict(X_train, verbose=0)
    train_errors = np.mean(np.square(X_train - preds_train), axis=1)
    threshold = np.percentile(train_errors, percentile)
    print(f"\n📏 Threshold ({percentile}th percentile): {threshold:.6f}")

    # 시각화를 위해 합본(merged) anomaly 데이터를 저장할 변수
    df_merged_anomaly_for_plot = None 

    # --------------------------------------------------
    # 각 anomaly_i.csv 에 대해 평가
    # --------------------------------------------------
    for file in anomaly_files:
        anomaly_path = os.path.join(base_dir, file)
        df_anomaly = pd.read_csv(anomaly_path)
        df_anomaly = df_anomaly.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)

        # ✅ label 컬럼이 있으면 제거
        if "label" in df_anomaly.columns:
            df_anomaly = df_anomaly.drop(columns=["label"])

        # CIC_anomaly.csv (합본) 데이터를 시각화용으로 저장
        if file == "CIC_anomaly.csv" and df_merged_anomaly_for_plot is None:
             df_merged_anomaly_for_plot = df_anomaly.copy()

        X_anomaly = scaler.transform(df_anomaly)
        X_test = np.concatenate([X_normal_test, X_anomaly])
        y_test = np.concatenate([np.zeros(len(X_normal_test)), np.ones(len(X_anomaly))])

        preds_test = model.predict(X_test, verbose=0)
        test_errors = np.mean(np.square(X_test - preds_test), axis=1)
        y_pred = (test_errors > threshold).astype(int)

        # Metrics 계산
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        pred_normal = int(np.sum(y_pred == 0))
        pred_anomaly = int(np.sum(y_pred == 1))

        print(f"\n🚨 {file}")
        print(f"Samples: {len(X_test)} | Accuracy={acc:.6f}, Precision={prec:.6f}, Recall={rec:.6f}, F1={f1:.6f}")

        results.append({
            "File": file,
            "Samples": len(X_test),
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1": f1,
            "Pred_Normal": pred_normal,
            "Pred_Anomaly": pred_anomaly
        })

    # --------------------------------------------------
    # 결과 저장
    # --------------------------------------------------
    df = pd.DataFrame(results)
    save_path = os.path.join(base_dir, f"evaluation_mixed_summary_p{percentile}.csv")
    df.to_csv(save_path, index=False)
    print(f"\n📁 Saved summary → {save_path}")
    print(df.round(6))

    # --------------------------------------------------
    # 🔥 공격 유형별 Reconstruction Error 히스토그램 (Merged 제외)
    # --------------------------------------------------
    # (1) Normal error 계산
    preds_normal = model.predict(X_normal_test, verbose=0)
    errors_normal = np.mean(np.square(X_normal_test - preds_normal), axis=1)

    # (2) 공격 유형별 reconstruction error 저장 (번호 -> 이름 매핑, merged 제외)
    attack_name_map = {
        1: "DDOS attack-HOIC",
        2: "DDoS attacks-LOIC-HTTP",
        3: "DoS attacks-Hulk",
        4: "Bot",
        5: "FTP-BruteForce",
        6: "SSH-Bruteforce",
        7: "Infiltration",
        8: "DoS attacks-SlowHTTPTest",
        9: "DoS attacks-GoldenEye",
        10: "DoS attacks-Slowloris",
        11: "DDOS attack-LOIC-UDP",
        12: "Brute Force -Web",
        13: "Brute Force -XSS",
        14: "SQL Injection"
    }

    error_by_attack = {}
    numeric_keys = []

    for file in anomaly_files:
        if file == "CIC_anomaly.csv":  # skip merged file entirely
            continue

        # 파일명에서 숫자 추출 (예: CIC_anomaly_3.csv -> 3)
        try:
            num_str = file.replace("CIC_anomaly_", "").replace(".csv", "")
            attack_num = int(num_str)
        except Exception:
            # 파일명 포맷이 다르면 건너뜀
            continue

        path = os.path.join(base_dir, file)
        df_anomaly = pd.read_csv(path)
        df_anomaly = df_anomaly.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)

        X_anomaly = scaler.transform(df_anomaly.values)
        preds_anomaly = model.predict(X_anomaly, verbose=0)
        errors_anomaly = np.mean(np.square(X_anomaly - preds_anomaly), axis=1)

        # use mapped name if available, else fallback to numeric string
        attack_label = attack_name_map.get(attack_num, f"attack_{attack_num}")
        error_by_attack[attack_num] = (attack_label, errors_anomaly)
        numeric_keys.append(attack_num)

    # 정렬: 숫자 오름차순, 하지만 요청에 따라 0(back)이 있으면 맨 뒤로 보낸다
    numeric_keys = sorted(set(numeric_keys))

    # (3) 히스토그램 플롯 (Normal + 공격별)
    plt.figure(figsize=(12,7))

    # Normal 먼저
    plt.hist(errors_normal, bins=200, alpha=0.6, label="Normal", color="green", density=True)

    # 팔레트 (항목 수에 맞춤)
    n_attacks = len(numeric_keys)
    palette = plt.cm.tab20(np.linspace(0, 1, max(3, n_attacks)))  # 충분한 색 확보

    for i, atk_num in enumerate(numeric_keys):
        atk_name, errs = error_by_attack[atk_num]
        plt.hist(errs, bins=200, alpha=0.5, label=f"{atk_name}", color=palette[i], density=True)

    # Threshold
    plt.axvline(threshold, color="blue", linestyle="--", label=f"Threshold ({threshold:.6f})")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Reconstruction Error (log scale)")
    plt.ylabel("Density (log scale)")
    plt.title("Reconstruction Error per Attack Type (CIC)")
    plt.legend(fontsize=8, loc="upper right", ncol=1)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig("./CIC_distribution.png", dpi=300)
    plt.close()

    print("📊 Saved → ./CIC_distribution.png")


# --------------------------------------------------
# 모델 로드 및 평가 실행
# --------------------------------------------------
if __name__ == "__main__":
    # input_dim = 115  # ✅ KDD99 feature 수
    # model = TransformerAAE(input_dim=input_dim)
    # _ = model(tf.zeros((1, input_dim)), prior_labels=tf.zeros((1, 1)))  # build
    # PRETRAIN_PATH = "Results/KDD99/rnep/rnep_frame_aae_transformer_weights.h5"
    
    # input_dim = 83  # ✅ InSDN feature 수
    # model = TransformerAAE(input_dim=input_dim)
    # _ = model(tf.zeros((1, input_dim)), prior_labels=tf.zeros((1, 1)))  # build
    # PRETRAIN_PATH = "Results/InSDN/rnep/rnep_frame_aae_transformer_weights.h5"
    
    input_dim = 78  # ✅ CIC2018 feature 수
    model = TransformerAAE(input_dim=input_dim)
    _ = model(tf.zeros((1, input_dim)), prior_labels=tf.zeros((1, 1)))  # build
    PRETRAIN_PATH = "rnep_cic2018/rnep_frame_aae_transformer_weights.h5"

    model.load_weights(PRETRAIN_PATH)
    print(f"✅ Loaded pre-trained weights from {PRETRAIN_PATH}")

    PERCENTILE = 90
    # evaluate_taae_kdd99(
    #     model=model,
    #     base_dir="./KDD99/KDD99_split",
    #     features=input_dim,
    #     percentile=PERCENTILE
    # )
    # evaluate_taae_indsn(
    #     model=model,
    #     base_dir="./InSDN/ae_datas",
    #     features=input_dim,
    #     percentile=PERCENTILE
    # )
    evaluate_taae_cic(
        model=model,
        base_dir="./CIC2018/ae_datas_all_features",
        features=input_dim,
        percentile=PERCENTILE
    )
    print("PERCENTILE =", PERCENTILE)