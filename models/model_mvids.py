from copy import deepcopy
from dataclasses import replace
from typing import List

import numpy as np
import torch
import os
import argparse
from MVIDS.kg_builder import make_kg_provider
from MVIDS.load_data import load_dataset
from MVIDS.train import test_loop, train_loop
from MVIDS.load_data import build_model, make_loaders
import pandas as pd

def run(args):
    dataset_key = args.dataset.lower()
    supported_datasets = {"unsw", "cic"}
    if dataset_key not in supported_datasets:
        raise NotImplementedError(
            f"지원하지 않는 데이터셋입니다: '{args.dataset}'. "
            f"현재는 {sorted(supported_datasets)} 만 사용할 수 있습니다."
        )

    features, labels, preprocessor_template = load_dataset(args.dataset, args.data_path)

    # ---------------------------
    # 공격 유형 로드
    # ---------------------------
    attack_types = None
    original_df = pd.read_csv(args.data_path)

    attack_col_names = {
        'cic': 'Attack',
        'unsw': 'attack_cat'
    }

    attack_col = attack_col_names.get(dataset_key)

    if attack_col and attack_col in original_df.columns:
        attack_types = original_df[attack_col].values
        print(f"Found attack type column: {attack_col}")
        print(f"Dataset size: {len(features):,} samples")

    schema = deepcopy(preprocessor_template.schema)

    f1_scores: List[float] = []

    for run_idx in range(args.repeats):
        run_seed = args.seed + run_idx
        np.random.seed(run_seed)
        torch.manual_seed(run_seed)

        preprocessor = preprocessor_template.__class__(deepcopy(schema))

        dl_train, _, dl_test, view_lengths, train_views, feature_names, y_train = make_loaders(
            features,
            labels,
            preprocessor,
            args,   
            attack_types=attack_types
        )

        # KG 생성 (model6 기준 항상 사용)
        args.y_train = y_train
        _, kg_metadata = make_kg_provider(train_views, feature_names, args)

    # ---------------------------
    # 모델 생성
    # ---------------------------
    model = build_model(view_lengths)

    os.makedirs(args.result_path, exist_ok=True )

    save_path=os.path.join(args.result_path,"mvids_weights.pt")
    # ---------------------------
    # 학습
    # ---------------------------
    model = train_loop(model, dl_train, None, args, save_path=save_path)

    # ---------------------------
    # 저장
    # ---------------------------
    save_dict = {
        'model_state_dict': model.state_dict(),
        'view_lengths': view_lengths,
        'args': vars(args),  
        'preprocessor': preprocessor,
        'feature_names': feature_names
    }

    print(f"Saving checkpoint with view_lengths: {view_lengths}")
    torch.save(save_dict, save_path)

    # ---------------------------
    # 평가
    # ---------------------------
    metrics = test_loop(model, dl_test, args)
    f1_scores.append(metrics["f1"])

    mean_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0
    std_f1 = float(np.std(f1_scores)) if f1_scores else 0.0

    print(f"Mean F1 over {args.repeats} runs: {mean_f1:.4f} ± {std_f1:.4f}")
    return f1_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='unsw')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--views', type=int, default=3)
    parser.add_argument('--kg_topk', type=int, default=50)
    parser.add_argument('--kg_thresh', type=float, default=0.3)
    parser.add_argument('--kg_binary', type=float, default=0.3)
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data_path', type=str, default="/data/SDP_Dataset/UNSW_NB15/UNSW_NB15_preprocessed.csv")
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--result_path',type=str, default="results/UNSW-NB15/mvids")
    args = parser.parse_args()

    run(args)
