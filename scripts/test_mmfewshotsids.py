import os
import time
import argparse
import numpy as np
import torch

from models.MMFEWSHOTSIDS.load_data import (
    MultimodalDataset,
    build_graph_A,
    build_basic_loader,
    fixed_count_split,
    attack_balanced_split
)

from models.MMFEWSHOTSIDS.multimodal_ids import MultiModalIDS
from models.MMFEWSHOTSIDS.train import evaluate
from models.MMFEWSHOTSIDS.traffic_feature_graph import build_packet_sequence_tensor


def prepare_loader(dataset, args):

    use_attack_balanced = (
        getattr(args, "attack_balanced_split", False)
        and getattr(dataset, "attack_col", None)
    )

    if use_attack_balanced:
        train_subset, test_subset = attack_balanced_split(
            dataset,
            args.k_train,
            args.k_test,
            seed=args.seed,
            train_per_attack=getattr(args, "attack_train_per_type", None),
            test_per_attack=getattr(args, "attack_test_per_type", None),
        )
    else:
        train_subset, test_subset = fixed_count_split(
            dataset,
            args.k_train,
            args.k_test,
            seed=args.seed,
            n_way=args.n_way
        )

    test_loader = build_basic_loader(
        test_subset,
        args.batch_size,
        shuffle=False
    )

    dataset.fit_transforms(train_subset.indices)
    return test_loader


def measure_latency(model, loader, device, warmup=10):

    model.eval()

    latencies = []
    total_samples = 0

    with torch.no_grad():

        # Warmup
        for idx, batch in enumerate(loader):

            if idx >= warmup:
                break

            batch = [
                x.to(device) if torch.is_tensor(x) else x
                for x in batch
            ]

            _ = model(*batch[:-1])

        # Real measurement
        for batch in loader:

            batch_size = batch[0].shape[0]
            total_samples += batch_size

            batch = [
                x.to(device) if torch.is_tensor(x) else x
                for x in batch
            ]

            if device.type == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()

            _ = model(*batch[:-1])

            if device.type == "cuda":
                torch.cuda.synchronize()

            end = time.perf_counter()

            latency_ms = (end - start) * 1000
            latencies.append(latency_ms)

    avg_latency = np.mean(latencies)
    std_latency = np.std(latencies)

    throughput = total_samples / (sum(latencies) / 1000)

    print("\n========== Latency ==========")
    print(f"Average latency : {avg_latency:.3f} ms")
    print(f"Std latency     : {std_latency:.3f} ms")
    print(f"Min latency     : {np.min(latencies):.3f} ms")
    print(f"Max latency     : {np.max(latencies):.3f} ms")
    print(f"Throughput      : {throughput:.2f} samples/sec")
    print("=============================\n")


def main(args):

    device = torch.device(args.device)

    print(f"[INFO] Loading tensor from {args.tensor_path}")

    tensor = np.load(args.tensor_path)

    if args.dataset == "cicids2018":
        label_col = "Label"
        attack_col = "Attack"
    else:
        label_col = "label"
        attack_col = "attack_cat"

    dataset = MultimodalDataset(
        csv_path=args.data_path,
        tensor_data=tensor,
        label_col=label_col,
        attack_col=attack_col,
        dataset_type=args.dataset,
    )

    print(
        "[INFO] Target dataset | "
        f"S-cont={dataset.num_cont} | "
        f"S-disc={dataset.num_disc} | "
        f"G-features={len(getattr(dataset, 'g_cols', []))}"
    )


    A_tensor = build_graph_A(
        dataset.df,
        dataset.g_cols,
        topk=args.kg_topk,
        thresh=args.kg_thresh
    ).to(device)

    test_loader = prepare_loader(dataset, args)

    model = MultiModalIDS(
        num_cont=dataset.num_cont,
        disc_cardinalities=dataset.cardinalities,
        num_classes=len(dataset.classes()),
        fusion_dim=args.fusion_dim,
        fusion_type=args.fusion_type,
        transformer_layers=args.s_depth,
        transformer_heads=args.s_heads,
        g_dropout=args.g_dropout,
        s_dropout=args.s_dropout,
        fusion_dropout=args.fusion_dropout,
        g_in_dim=A_tensor.shape[0],
        A_tensor=A_tensor,
    ).to(device)

    checkpoint = torch.load(args.model_path, map_location=device)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    print(f"[INFO] Loaded model from {args.model_path}")

    metrics = evaluate(
        model,
        test_loader,
        device,
        num_classes=len(dataset.classes()),
        class_labels=dataset.classes(),
        save_confusion_path=os.path.join(
            args.result_path,
            "confusion_matrix.png"
        ),
    )

    print(
        "[TEST] "
        f"ACC={metrics['ACC']:.4f} | "
        f"F1={metrics['F1']:.4f} | "
        f"PREC={metrics['Precision']:.4f} | "
        f"REC={metrics['Recall']:.4f}"
    )

    if args.measure_latency:
        measure_latency(
            model,
            test_loader,
            device,
            warmup=args.warmup
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='cicids2018')
    parser.add_argument('--device', type=str, default='cuda:0')

    parser.add_argument(
        '--data_path',
        type=str,
        default="/data/SDP_Dataset/CICIDS2018/NF-CICIDS2018-v3-balanced.csv"
    )

    parser.add_argument(
        '--tensor_path',
        type=str,
        default="results/CSE-CIC-IDS2018/mmfewshotsids/traffic_tensor.npy"
    )

    parser.add_argument(
        '--result_path',
        type=str,
        default="results/CSE-CIC-IDS2018/mmfewshotsids"
    )

    parser.add_argument(
        '--model_path',
        type=str,
        default="results/CSE-CIC-IDS2018/mmfewshotsids/mmfewshotsids_cicids2018.pt"
    )

    parser.add_argument('--batch_size', type=int, default=32)

    parser.add_argument('--fusion_dim', type=int, default=128)
    parser.add_argument('--fusion_type', type=str, default='hif')

    parser.add_argument('--s_heads', type=int, default=4)
    parser.add_argument('--s_depth', type=int, default=2)

    parser.add_argument('--g_dropout', type=float, default=0.1)
    parser.add_argument('--s_dropout', type=float, default=0.1)
    parser.add_argument('--fusion_dropout', type=float, default=0.1)

    parser.add_argument('--kg_topk', type=int, default=50)
    parser.add_argument('--kg_thresh', type=float, default=0.3)

    parser.add_argument('--k_train', type=int, default=15)
    parser.add_argument('--k_test', type=int, default=30)

    parser.add_argument('--attack_balanced_split', default=True)

    parser.add_argument('--n_way', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)

    # latency option
    parser.add_argument('--measure_latency', action='store_true')
    parser.add_argument('--warmup', type=int, default=10)

    args = parser.parse_args()

    main(args)