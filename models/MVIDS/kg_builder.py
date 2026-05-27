from __future__ import annotations

import itertools
from typing import Dict, List, Sequence, Tuple

import networkx as nx
import numpy as np
import torch
from tqdm.auto import tqdm
from sklearn.feature_selection import f_classif


def _normalize_for_binary(arr: np.ndarray) -> np.ndarray:
    """Min-max normalise per feature before binarisation."""
    if arr.size == 0:
        return arr.astype(np.float32)
    arr = np.asarray(arr, dtype=np.float32)
    minv = np.nanmin(arr, axis=0, keepdims=True)
    maxv = np.nanmax(arr, axis=0, keepdims=True)
    span = np.where((maxv - minv) < 1e-6, 1.0, maxv - minv)
    norm = (arr - minv) / span
    return np.nan_to_num(norm, nan=0.0, posinf=0.0, neginf=0.0)


def _binarise(arr: np.ndarray, threshold: float) -> np.ndarray:
    """Convert continuous features to binary indicators via thresholding."""
    if arr.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    norm = _normalize_for_binary(arr)
    return (norm > threshold).astype(np.float32)


def _co_occurrence_matrix(
    bin_a: np.ndarray,
    bin_b: np.ndarray | None = None,
) -> np.ndarray:
    """Compute co-occurrence frequency matrix between two binary matrices."""
    if bin_a.size == 0:
        if bin_b is None:
            return np.zeros((0, 0), dtype=np.float32)
        return np.zeros((bin_a.shape[1], bin_b.shape[1]), dtype=np.float32)
    if bin_b is None:
        bin_b = bin_a
    if bin_a.shape[0] == 0 or bin_a.shape[1] == 0 or bin_b.shape[1] == 0:
        return np.zeros((bin_a.shape[1], bin_b.shape[1]), dtype=np.float32)
    co_occ = bin_a.T @ bin_b
    denom = float(max(bin_a.shape[0], 1))
    co_occ = co_occ / denom
    co_occ = np.nan_to_num(co_occ, nan=0.0, posinf=0.0, neginf=0.0)
    return co_occ.astype(np.float32)

def build_kg_features(
    view_arrays: Sequence[np.ndarray],
    feature_names: Dict[str, List[str]],
    args,
) -> Tuple[nx.Graph, List[Dict[str, float]]]:

    """Construct KG aligned with paper (single-view + MI + co-occurrence + centrality)."""

    G = nx.Graph()

    # =========================
    # [CHANGE 1] merge multi-view -> single view
    # =========================
    X = np.concatenate(view_arrays, axis=1)
    all_features = sum(feature_names.values(), [])

    # =========================
    # [CHANGE 2] top-k feature selection based on Mutual Information
    # =========================

    y = args.y_train
    scores, _ = f_classif(X, y)

    # prevent NaN (important)
    scores = np.nan_to_num(scores, nan=0.0)

    topk = min(args.kg_topk, X.shape[1])
    topk_idx = np.argsort(scores)[-topk:]

    X_selected = X[:, topk_idx]
    selected_features = [all_features[i] for i in topk_idx]
    # =========================
    # keep existing binarization
    # =========================
    bin_X = _binarise(X_selected, args.kg_thresh)

    # =========================
    # keep existing co-occurrence
    # =========================
    co_occ = _co_occurrence_matrix(bin_X)
    np.fill_diagonal(co_occ, 0.0)

    # =========================
    # [CHANGE 3] simplify nodes
    # =========================
    for i, feat in enumerate(selected_features):
        G.add_node(feat, index=i)

    # =========================
    # [CHANGE 4] intra-view only (remove inter-view)
    # =========================
    for i in range(len(selected_features)):
        for j in range(i + 1, len(selected_features)):
            weight = float(co_occ[i, j])
            if weight >= args.kg_thresh:
                G.add_edge(selected_features[i], selected_features[j], weight=weight)

    if G.number_of_nodes() == 0:
        return G, []

    # =========================
    # keep centrality (paper)
    # =========================
    try:
        centrality = nx.eigenvector_centrality_numpy(G, weight="weight")
    except nx.NetworkXException:
        centrality = {node: 0.0 for node in G.nodes}

    sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)

    selected: List[Dict[str, float]] = []
    for node, score in sorted_nodes[:topk]:
        node_data = G.nodes[node]
        selected.append(
            {
                "node": node,
                "feature_index": int(node_data["index"]),
                "score": float(score),
            }
        )

    return G, selected


def make_kg_provider(
    view_arrays: Sequence[np.ndarray],
    feature_names: Dict[str, List[str]],
    args,
) -> Tuple[None, Dict[str, object]]:

    """Return KG metadata only (paper-aligned)."""

    G, selected = build_kg_features(view_arrays, feature_names, args)

    centrality_scores = [item["score"] for item in selected]

    metadata = {
        "graph": G,
        "selected": selected,
        "centrality": centrality_scores,
    }

    return None, metadata