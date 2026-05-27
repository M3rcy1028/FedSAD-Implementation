from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch
from MVIDS.preprocess import MultiViewPreprocessor
from MVIDS.multiview import MultiViewKnowledgeNet
from MVIDS.preprocess import (
    CICIDS2018_VIEW_SCHEMA,
    MultiViewPreprocessor,
    UNSW_VIEW_SCHEMA,
)
from tqdm.auto import tqdm

VIEW_SCHEMAS = {
    "unsw": UNSW_VIEW_SCHEMA,
    "cic": CICIDS2018_VIEW_SCHEMA,
}


class MultiViewDataset(Dataset):
    """Dataset returning per-sample multi-view tensors matching the paper's design."""

    def __init__(self, views: Sequence[np.ndarray], labels: np.ndarray):
        if not views:
            raise ValueError("`views` must contain at least one view array.")
        self.views = [np.asarray(v, dtype=np.float32) for v in views]
        self.labels = labels.astype(np.float32)
        lengths = {v.shape[0] for v in self.views}
        if len(lengths) != 1:
            raise ValueError("All view arrays must share the same sample dimension.")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        sample_views: List[torch.Tensor] = []
        for view in self.views:
            sample = torch.from_numpy(view[idx]).unsqueeze(0)  # [1, F]
            sample_views.append(sample)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return sample_views, label


def ensure_binary(labels: pd.Series) -> np.ndarray:
    if labels.dtype.kind in {"i", "u"}:
        return labels.to_numpy(dtype=np.int64)
    # treat "normal" and "benign" (case-insensitive) as normal traffic
    labels_str = labels.astype(str)
    is_normal = labels_str.str.lower().isin({"normal", "benign"})
    binary = (~is_normal).astype(int)
    return binary.to_numpy(dtype=np.int64)


def load_dataset(
    dataset: str ,data_path: str
) -> Tuple[pd.DataFrame, np.ndarray, MultiViewPreprocessor]:
    """Load a dataset and return raw features, labels, and an unfitted preprocessor."""

    stage_bar = tqdm(total=4, desc=f"Load {dataset.upper()}", leave=False)

    try:
        stage_bar.set_postfix_str("read train csv")
        train_df = pd.read_csv(data_path)
        stage_bar.update()

        stage_bar.set_postfix_str("encode labels")
        possible_labels = ["label", "Label"]

        label_col=None
        for col in possible_labels:
            if col in train_df.columns:
                label_col=col
                break

        labels = ensure_binary(train_df[label_col])
        stage_bar.update()

        stage_bar.set_postfix_str("merge features")
        feature_df = train_df.drop(columns=[label_col], errors="ignore")

        if "attack_cat" in train_df.columns:
            feature_df = feature_df.drop(columns=["attack_cat"], errors="ignore")
        stage_bar.update()
    finally:
        stage_bar.close()

    schema = VIEW_SCHEMAS[dataset]
    preprocessor = MultiViewPreprocessor(schema)
    return feature_df, labels.astype(np.float32), preprocessor


def make_loaders(
    features: pd.DataFrame,
    labels: np.ndarray,
    preprocessor: MultiViewPreprocessor,
    args,
    attack_types: np.ndarray = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[int], List[np.ndarray], Dict[str, List[str]]]:
    """Split data into train/val/test, fit preprocessing on train only, and create DataLoaders."""
    idx = np.arange(len(labels))
    
    # stratify by attack type (if available), otherwise by binary label
    if attack_types is not None:
        stratify = attack_types
        print(f"Stratifying by attack types: {len(np.unique(attack_types))} types")
    else:
        stratify = labels if len(np.unique(labels)) > 1 else None
        print("Stratifying by binary labels (Normal vs Attack)")

    idx_train, idx_test, y_train, y_test = train_test_split(
        idx,
        labels,
        test_size=0.2,
        random_state=args.seed,
        stratify=stratify,
    )

    feature_train = features.iloc[idx_train]
    feature_test = features.iloc[idx_test]

    train_views = preprocessor.fit_transform(feature_train)
    test_views = preprocessor.transform(feature_test)

    ds_train = MultiViewDataset(train_views, y_train)
    ds_test = MultiViewDataset(test_views, y_test)

    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, drop_last=False)
    dl_test = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False)

    view_lengths = [v.shape[1] for v in train_views]
    feature_names = preprocessor.get_feature_names()
    # return without validation
    return dl_train, None, dl_test, view_lengths, train_views, feature_names, y_train


def build_model(view_lengths: Sequence[int], kg_provider=None):
    """Instantiate the deep multi-view model described in the paper."""
    use_kg = kg_provider is not None
    kg_dim = getattr(kg_provider, "kg_dim", 0) if use_kg else 0
    model = MultiViewKnowledgeNet(
        view_lengths=view_lengths,
        cnn_channels=32,
        fusion_hidden=500,
        kg_dim=kg_dim,
        dropout=0.3,
    )
    return model

#TODO refactor code structure
def load_unsw() -> Tuple[pd.DataFrame, np.ndarray, MultiViewPreprocessor]:
    """Backward-compatible helper for code paths expecting the UNSW loader."""
    return load_dataset("unsw")
