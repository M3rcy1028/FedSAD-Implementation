import argparse
import torch
import numpy as np
from copy import deepcopy
import pandas as pd

from models.MVIDS.load_data import load_dataset, make_loaders, build_model
from models.MVIDS.train import test_loop


def load_checkpoint(checkpoint_path, device):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    print("Checkpoint loaded.")
    print(f"Saved view_lengths: {checkpoint['view_lengths']}")

    return checkpoint


def run_test(args):
    device = torch.device(args.device)

    # ---------------------------
    # 체크포인트 로드
    # ---------------------------
    checkpoint = load_checkpoint(args.model_path, device)

    saved_args = checkpoint["args"]
    preprocessor = checkpoint["preprocessor"]
    view_lengths = checkpoint["view_lengths"]

    # ---------------------------
    # 데이터셋 로드
    # ---------------------------
    features, labels, _ = load_dataset(
        saved_args["dataset"],
        args.data_path
    )

    # ---------------------------
    # 공격 유형 로드
    # ---------------------------
    dataset_key = saved_args["dataset"].lower()

    attack_col_names = {
        "cic": "Attack",
        "unsw": "attack_cat"
    }

    attack_types = None

    original_df = pd.read_csv(args.data_path)

    attack_col = attack_col_names.get(dataset_key)

    if attack_col and attack_col in original_df.columns:
        attack_types = original_df[attack_col].values
        print(f"Found attack type column: {attack_col}")

    # ---------------------------
    # DataLoader 생성
    # ---------------------------
    dl_train, _, dl_test, _, _, _, _ = make_loaders(
        features,
        labels,
        preprocessor,
        argparse.Namespace(**saved_args),
        attack_types=attack_types
    )

    # ---------------------------
    # 모델 생성
    # ---------------------------
    model = build_model(view_lengths)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    print("Model weights loaded successfully.")

    # ---------------------------
    # 테스트
    # ---------------------------
    metrics = test_loop(
        model,
        dl_test,
        argparse.Namespace(**saved_args)
    )

    print("\n===== TEST RESULT =====")
    for key, value in metrics.items():
        if isinstance(value, (float, np.floating)):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_path",
        type=str,
        default="results/UNSW_NB15/mvids/mvids_weights.pt"
    )

    parser.add_argument(
        "--data_path",
        type=str,
        default="/data/SDP_Dataset/UNSW_NB15/UNSW_NB15_preprocessed.csv"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0"
    )

    args = parser.parse_args()

    run_test(args)