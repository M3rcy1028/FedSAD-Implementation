import copy
import os
import torch
from MMFEWSHOTSIDS.load_data import MultimodalDataset, build_graph_A
from MMFEWSHOTSIDS.multimodal_ids import MultiModalIDS
from MMFEWSHOTSIDS.train import evaluate, train_self
from MMFEWSHOTSIDS.load_data import attack_balanced_split, build_basic_loader, fixed_count_split
from MMFEWSHOTSIDS.traffic_feature_graph import build_packet_sequence_tensor
import argparse


def save_subset_csv(dataset, subset, filepath):
    if subset is None or not filepath:
        return
    indices = getattr(subset, "indices", None)
    if indices is None:
        raise ValueError("Subset does not expose indices; unable to save CSV split.")
    dirpath = os.path.dirname(filepath)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    df_to_save = dataset.df.iloc[list(indices)]
    df_to_save.to_csv(filepath, index=False)
    print(f"[INFO] Saved split to {filepath} (rows={len(df_to_save)})")


def prepare_loaders(dataset, args):
    use_attack_balanced = getattr(args, "attack_balanced_split", False) and getattr(dataset, "attack_col", None)
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
            dataset, args.k_train, args.k_test, seed=args.seed, n_way=args.n_way
        )
    dataset.fit_transforms(train_subset.indices)
    train_loader = build_basic_loader(train_subset, args.batch_size, shuffle=True)
    test_loader = build_basic_loader(test_subset, args.batch_size, shuffle=False)
    return train_subset, test_subset, train_loader, test_loader



def main(args):
    torch.manual_seed(args.seed)
    os.makedirs(args.result_path, exist_ok=True)

    model_path = os.path.join(args.result_path, f"mmfewshotsids_{args.dataset}.pt")

    tensor=build_packet_sequence_tensor(
        csv_path=args.data_path,
        output_path=args.tensor_path
    )

    if args.dataset == "cicids2018":
        label_col = "Label"
        attack_col = "Attack"
    else:
        label_col = "label"
        attack_col = "attack_cat"

    # Dataset
    target_dataset = MultimodalDataset(
        csv_path=args.data_path,
        tensor_data=tensor,
        label_col=label_col,
        attack_col=attack_col,
        dataset_type=args.dataset,
    )

    print(
        "[INFO] Target dataset | "
        f"S-cont={target_dataset.num_cont} | "
        f"S-disc={target_dataset.num_disc} | "
        f"G-features={len(getattr(target_dataset, 'g_cols', []))}"
    )

    # Graph
    A_tensor = build_graph_A(target_dataset.df, target_dataset.g_cols,
                             topk=args.kg_topk, thresh=args.kg_thresh)
    device = torch.device(args.device)
    A_tensor = A_tensor.to(device)

    # Loader
    train_subset, test_subset, train_loader, test_loader = prepare_loaders(
        target_dataset,
        args=args
    )

    save_subset_csv(target_dataset, train_subset,
                     os.path.join(args.result_path, "train_split.csv"))
    save_subset_csv(target_dataset, test_subset,
                     os.path.join(args.result_path, "test_split.csv"))

    # Model
    model = MultiModalIDS(
        num_cont=target_dataset.num_cont,
        disc_cardinalities=target_dataset.cardinalities,
        num_classes=len(target_dataset.classes()),
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

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_finetune)

    # Train
    train_self(args, model, train_loader, optimizer)

    # Evaluate
    metrics = evaluate(
        model,
        test_loader,
        device,
        num_classes=len(target_dataset.classes()),
        class_labels=target_dataset.classes(),
        save_confusion_path=os.path.join(args.result_path, "confusion_matrix.png"),
    )

    print(
        "[TEST] "
        f"ACC={metrics['ACC']:.4f} | F1={metrics['F1']:.4f} | "
        f"PREC={metrics['Precision']:.4f} | REC={metrics['Recall']:.4f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cicids2018')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr_finetune', type=float, default=5e-4)
    parser.add_argument('--views', type=int, default=3)
    parser.add_argument('--kg_topk', type=int, default=50)
    parser.add_argument('--kg_thresh', type=float, default=0.3)
    parser.add_argument('--kg_binary', type=float, default=0.3)
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data_path', type=str, default="/data/SDP_Dataset/CICIDS2018/NF-CICIDS2018-v3-balanced.csv")
    parser.add_argument('--result_path',type=str, default="results/CSE-CIC-IDS2018/mmfewshotsids")
    parser.add_argument('--tensor_path', type=str, default="results/CSE-CIC-IDS2018/mmfewshotsids/traffic_tensor.npy")
    parser.add_argument('--device', type=str, default="cuda:0")
    
    
    parser.add_argument('--model_variant', type=str, default='self')

    parser.add_argument('--fusion_dim', type=int, default=128)
    parser.add_argument('--fusion_type', type=str, default='hif')
    parser.add_argument('--s_heads', type=int, default=4)
    parser.add_argument('--s_depth', type=int, default=2)

    parser.add_argument('--g_dropout', type=float, default=0.1)
    parser.add_argument('--s_dropout', type=float, default=0.1)
    parser.add_argument('--fusion_dropout', type=float, default=0.1)

    parser.add_argument('--k_train', type=int, default=15)
    parser.add_argument('--k_test', type=int, default=30)

    parser.add_argument('--attack_balanced_split', default=True)
    parser.add_argument('--attack_train_per_type', type=int, default=None)
    parser.add_argument('--attack_test_per_type', type=int, default=None)

    parser.add_argument('--n_way', type=int, default=2)
    args = parser.parse_args()
    main(args)
