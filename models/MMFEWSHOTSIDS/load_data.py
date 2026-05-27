import torch
from torch.utils.data import DataLoader,Dataset, Subset
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import numpy as np
from typing import Sequence, Union, Optional, Dict, Iterable, Tuple

# G-branch candidates (CICIDS2018)
G_COL_CANDIDATES_CICIDS = [
    "IN_BYTES", "OUT_BYTES", "IN_PKTS", "OUT_PKTS",
    "FLOW_DURATION_MILLISECONDS", "DURATION_IN", "DURATION_OUT",
    "SRC_TO_DST_IAT_MIN", "SRC_TO_DST_IAT_MAX", "SRC_TO_DST_IAT_AVG", "SRC_TO_DST_IAT_STDDEV",
    "DST_TO_SRC_IAT_MIN", "DST_TO_SRC_IAT_MAX", "DST_TO_SRC_IAT_AVG", "DST_TO_SRC_IAT_STDDEV",
    "SRC_TO_DST_AVG_THROUGHPUT", "DST_TO_SRC_AVG_THROUGHPUT",
    "MIN_TTL", "MAX_TTL",
    "LONGEST_FLOW_PKT", "SHORTEST_FLOW_PKT",
    "MIN_IP_PKT_LEN", "MAX_IP_PKT_LEN",
    "NUM_PKTS_UP_TO_128_BYTES", "NUM_PKTS_128_TO_256_BYTES",
    "NUM_PKTS_256_TO_512_BYTES", "NUM_PKTS_512_TO_1024_BYTES", "NUM_PKTS_1024_TO_1514_BYTES",
    "RETRANSMITTED_IN_BYTES", "RETRANSMITTED_IN_PKTS",
    "RETRANSMITTED_OUT_BYTES", "RETRANSMITTED_OUT_PKTS",
    "SRC_TO_DST_SECOND_BYTES", "DST_TO_SRC_SECOND_BYTES",
    "TCP_WIN_MAX_IN", "TCP_WIN_MAX_OUT",
]

# S-branch discrete candidates (CICIDS2018)
S_DISC_CANDIDATES_CICIDS = [
    "L4_SRC_PORT", "L4_DST_PORT", "PROTOCOL", "L7_PROTO",
    "TCP_FLAGS", "CLIENT_TCP_FLAGS", "SERVER_TCP_FLAGS",
    "ICMP_TYPE", "ICMP_IPV4_TYPE", "DNS_QUERY_ID", "DNS_QUERY_TYPE", "DNS_TTL_ANSWER",
    "FTP_COMMAND_RET_CODE",
]

S_CONT_CANDIDATES_CICIDS = [
    'IN_BYTES', 'IN_PKTS',
    'OUT_BYTES', 'OUT_PKTS',
    'FLOW_DURATION_MILLISECONDS', 'DURATION_IN', 'DURATION_OUT',
    'MIN_TTL', 'MAX_TTL',
    'LONGEST_FLOW_PKT', 'SHORTEST_FLOW_PKT',
    'MIN_IP_PKT_LEN', 'MAX_IP_PKT_LEN',
    'SRC_TO_DST_SECOND_BYTES', 'DST_TO_SRC_SECOND_BYTES',
    'RETRANSMITTED_IN_BYTES', 'RETRANSMITTED_IN_PKTS',
    'RETRANSMITTED_OUT_BYTES', 'RETRANSMITTED_OUT_PKTS',
    'SRC_TO_DST_AVG_THROUGHPUT', 'DST_TO_SRC_AVG_THROUGHPUT',
    'NUM_PKTS_UP_TO_128_BYTES', 'NUM_PKTS_128_TO_256_BYTES',
    'NUM_PKTS_256_TO_512_BYTES', 'NUM_PKTS_512_TO_1024_BYTES',
    'NUM_PKTS_1024_TO_1514_BYTES',
    'TCP_WIN_MAX_IN', 'TCP_WIN_MAX_OUT',
    'SRC_TO_DST_IAT_MIN', 'SRC_TO_DST_IAT_MAX',
    'SRC_TO_DST_IAT_AVG', 'SRC_TO_DST_IAT_STDDEV',
    'DST_TO_SRC_IAT_MIN', 'DST_TO_SRC_IAT_MAX',
    'DST_TO_SRC_IAT_AVG', 'DST_TO_SRC_IAT_STDDEV'
]


# G-branch candidates (UNSW-NB15)
G_COL_CANDIDATES_UNSW = [
    "dur", "sbytes", "dbytes", "spkts", "dpkts", "sload", "dload",
    "sttl", "dttl", "sloss", "dloss", "swin", "dwin", "stcpb", "dtcpb",
    "smeansz", "dmeansz", "trans_depth", "res_bdy_len", "sjit", "djit",
    "stime", "ltime", "sintpkt", "dintpkt", "tcprtt", "synack", "ackdat",
    "ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm",
    "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
]

# S-branch discrete candidates (UNSW-NB15)
S_DISC_CANDIDATES_UNSW = [
    "sport", "dsport", "proto", "state", "service",
    "is_sm_ips_ports", "ct_state_ttl", "ct_ftp_cmd",
]

S_CONT_CANDIDATES_UNSW = [
    "dur", "sbytes", "dbytes", "sload", "dload",
    "spkts", "dpkts", "sttl", "dttl", "sloss", "dloss",
    "swin", "dwin", "stcpb", "dtcpb", "smeansz", "dmeansz",
    "trans_depth", "res_bdy_len", "sjit", "djit",
    "sintpkt", "dintpkt", "tcprtt", "synack", "ackdat",
    "ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm",
    "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm"
]

DATASET_PRESETS = {
    "cicids2018": {
        "g_candidates": G_COL_CANDIDATES_CICIDS,
        "s_disc_candidates": S_DISC_CANDIDATES_CICIDS,
        "s_cont_candidates": S_CONT_CANDIDATES_CICIDS,
        "label_col": "Label",
        "label_names": ["Benign", "Attack"],
    },
    "unsw": {
        "g_candidates": G_COL_CANDIDATES_UNSW,
        "s_disc_candidates": S_DISC_CANDIDATES_UNSW,
        "s_cont_candidates": S_CONT_CANDIDATES_UNSW,
        "label_col": "label",
        "label_names": ["Benign", "Attack"],
    },

}


class MultimodalDataset(Dataset):
    """
    Updated version

    - Modality 1: traffic tensor (N,1,16,16,16)
    - Modality 2: network feature set
    """

    def __init__(
        self,
        csv_path: str,
        tensor_data: Optional[np.ndarray] = None,  
        cont_cols: Optional[Sequence[str]] = None,
        disc_cols: Optional[Sequence[str]] = None,
        g_cols: Optional[Sequence[str]] = None,
        label_col: str = "Label",
        attack_col: Optional[str] = "Attack",
        id_col: Optional[str] = "id",
        encoding: str = "utf-8",
        dataset_type: str = "cicids2018",
    ):
        preset_key = dataset_type.lower()
        if preset_key not in DATASET_PRESETS:
            raise ValueError(f"Unsupported dataset_type '{dataset_type}'")

        preset = DATASET_PRESETS[preset_key]

        self.cont_cols = list(cont_cols or preset["s_cont_candidates"])
        self.disc_cols = list(disc_cols or preset["s_disc_candidates"])
        self.g_cols = list(g_cols or preset["g_candidates"])

        self.label_col = label_col
        self.attack_col = attack_col

        print(f"[INFO] Loading CSV: {csv_path}")
        df = pd.read_csv(csv_path, encoding=encoding)

        # -----------------------------
        # column check
        # -----------------------------
        missing_cont = [c for c in self.cont_cols if c not in df.columns]
        missing_disc = [c for c in self.disc_cols if c not in df.columns]
        missing_g = [c for c in self.g_cols if c not in df.columns]

        if missing_cont:
            raise ValueError(f"Missing continuous columns: {missing_cont}")
        if missing_disc:
            raise ValueError(f"Missing discrete columns: {missing_disc}")
        if missing_g:
            raise ValueError(f"Missing graph columns: {missing_g}")
        if label_col not in df.columns:
            raise ValueError(f"Missing label column: {label_col}")

        # -----------------------------
        # dataframe processing
        # -----------------------------
        df = df.reset_index(drop=True)
        df["y"] = df[label_col].astype(int)

        self.df = df
        self.label_names = sorted(self.df["y"].unique().tolist())

        if self.attack_col and self.attack_col not in self.df.columns:
            self.attack_col = None

        # -----------------------------
        # tensor processing (core change)
        # -----------------------------
        self.graphs = None

        if tensor_data is not None:
            print("[INFO] Using in-memory tensor")
            self.graphs = self._reshape_to_3d(tensor_data)

            if len(self.graphs) != len(self.df):
                raise ValueError(
                    f"Tensor length ({len(self.graphs)}) != CSV length ({len(self.df)})"
                )

        # -----------------------------
        # internal variables
        # -----------------------------
        self._cont_stats: Dict[str, Tuple[float, float]] = {}
        self._disc_mappings: Dict[str, Dict[str, int]] = {}
        self.cardinalities: list[int] = []
        self.num_cont = len(self.cont_cols)
        self.num_disc = len(self.disc_cols)
        self._fitted = False

    # -----------------------------
    # tensor reshape
    # -----------------------------
    def _reshape_to_3d(self, arr):
        if arr.ndim == 5 and arr.shape[1:] == (1, 16, 16, 16):
            return arr
        if arr.ndim == 5 and arr.shape[1] == 1:
            return arr
        if arr.ndim == 4 and arr.shape[1:] == (16, 16, 16):
            return arr[:, None, :, :, :]
        if arr.ndim == 4 and arr.shape[1:] == (64, 64):
            N = arr.shape[0]
            return arr.reshape(N, 1, 16, 16, 16)
        if arr.ndim == 3 and arr.shape[1:] == (64, 64):
            N = arr.shape[0]
            return arr.reshape(N, 1, 16, 16, 16)
        raise ValueError("Invalid tensor shape")

    # -----------------------------
    # basic functions
    # -----------------------------
    def __len__(self):
        return len(self.df)

    def classes(self):
        return sorted(self.df["y"].unique().tolist())

    # -----------------------------
    # transform fitting
    # -----------------------------
    def fit_transforms(self, train_indices: Iterable[int]):
        indices = np.asarray(list(train_indices), dtype=int)
        if indices.size == 0:
            raise ValueError("Empty indices")

        # continuous normalize
        stats: Dict[str, Tuple[float, float]] = {}
        for c in self.cont_cols:
            vals = self.df.loc[indices, c].to_numpy(dtype=np.float32)
            mean = float(vals.mean())
            std = float(vals.std()) or 1.0
            stats[c] = (mean, std)
            self.df[c] = ((self.df[c] - mean) / std).astype("float32")

        self._cont_stats = stats

        # discrete encoding
        mappings: Dict[str, Dict[str, int]] = {}
        self.cardinalities = []

        for c in self.disc_cols:
            vals = self.df.loc[indices, c].astype(str)
            unique_vals = pd.Index(vals).unique()
            mapping = {val: i + 1 for i, val in enumerate(unique_vals)}
            mappings[c] = mapping

            self.df[c] = (
                self.df[c].astype(str)
                .map(mapping)
                .fillna(0)
                .astype("int64")
            )

            self.cardinalities.append(len(mapping) + 1)

        self._disc_mappings = mappings
        self._fitted = True

    # -----------------------------
    # data retrieval
    # -----------------------------
    def __getitem__(self, idx: int):
        if not self._fitted:
            raise RuntimeError("Call fit_transforms first")

        row = self.df.iloc[idx]

        # graph part
        if self.graphs is not None:
            g_tensor = torch.tensor(self.graphs[idx], dtype=torch.float32)
        else:
            # fallback (keep existing)
            g_tensor = torch.tensor(
                row[self.g_cols].values.astype("float32"),
                dtype=torch.float32
            )

        cont = torch.tensor(
            row[self.cont_cols].values.astype("float32"),
            dtype=torch.float32
        )

        disc = torch.tensor(
            row[self.disc_cols].values.astype("int64"),
            dtype=torch.long
        )

        y = torch.tensor(int(row["y"]), dtype=torch.long)

        return g_tensor, cont, disc, y
    
def build_graph_A(df, g_cols, topk=None, thresh=0.0):
    """
    Compute Pearson correlation matrix |corr| from df[g_cols] -> row-normalize to row-stochastic A.
    If topk is given, keep only topk per row. If thresh>0, values below are set to 0.
    """
    if not g_cols:
        return torch.zeros((256,256), dtype=torch.float32)  # dummy
    X = df[g_cols].to_numpy(dtype=np.float32)
    C = np.corrcoef(X, rowvar=False)   # (M,M)
    C = np.nan_to_num(np.abs(C), nan=0.0, posinf=0.0, neginf=0.0)

    M = C.shape[0]
    A = C
    if thresh > 0:
        A[A < thresh] = 0.0
    if topk is not None and topk < M:
        idx = np.argpartition(-A, kth=topk, axis=1)[:, :topk]
        mask = np.zeros_like(A, dtype=bool)
        rows = np.arange(M)[:, None]
        mask[rows, idx] = True
        A[~mask] = 0.0

    # row normalization
    row_sum = A.sum(axis=1, keepdims=True) + 1e-8
    A = A / row_sum

    # pad/trunc to 256x256 (for GGraphCNN 16x16 reshape)
    if M > 256:
        A = A[:256, :256]
    elif M < 256:
        pad0 = 256 - M
        A = np.pad(A, ((0,pad0),(0,pad0)), constant_values=0.0)

    return torch.tensor(A, dtype=torch.float32)


class FewShotEpisodeDataset(Dataset):
    """
    Episodic dataset that samples N-way, K-shot support and query sets from a base dataset or subset.
    """

    def __init__(
        self,
        base,
        n_way: int,
        k_shot: Union[int, Sequence[int]],
        k_query: int,
        episodes_per_epoch: int,
        seed: int = 42,
    ):
        if n_way <= 0 or k_query <= 0:
            raise ValueError("n_way and k_query must be > 0")

        self.n_way = n_way
        if isinstance(k_shot, (list, tuple, np.ndarray, set)):
            self.k_shot_choices = [int(x) for x in k_shot if int(x) > 0]
        else:
            self.k_shot_choices = [int(k_shot)]
        if not self.k_shot_choices or any(x <= 0 for x in self.k_shot_choices):
            raise ValueError("k_shot must contain positive integers")
        self.k_query = int(k_query)
        if self.k_query <= 0:
            raise ValueError("k_query must be > 0")
        self.episodes_per_epoch = episodes_per_epoch
        self.rng = np.random.default_rng(seed)

        if isinstance(base, Subset):
            self.source = base.dataset
            self.indices = list(base.indices)
        else:
            self.source = base
            self.indices = list(range(len(base)))

        self.class_to_indices = {}
        if hasattr(self.source, "df") and "y" in getattr(self.source, "df").columns:
            labels = self.source.df["y"].to_numpy()
            for idx in self.indices:
                cls = int(labels[idx])
                self.class_to_indices.setdefault(cls, []).append(idx)
        else:
            for idx in self.indices:
                _, _, _, y = self.source[idx]
                cls = int(y.item() if isinstance(y, torch.Tensor) else y)
                self.class_to_indices.setdefault(cls, []).append(idx)

        self.available_classes = sorted(cls for cls, idxs in self.class_to_indices.items() if idxs)
        if len(self.available_classes) < self.n_way:
            raise ValueError(
                f"Not enough classes ({len(self.available_classes)}) to sample {self.n_way}-way episodes."
            )

    def __len__(self):
        return self.episodes_per_epoch

    def _sample_indices_for_class(self, cls: int, k_shot: int, k_query: int):
        pool = self.class_to_indices[cls]
        needed = k_shot + k_query
        replace = len(pool) < needed
        chosen = self.rng.choice(pool, size=needed, replace=replace)
        support = chosen[:k_shot]
        query = chosen[k_shot:]
        return support.tolist(), query.tolist()

    def _gather(self, indices):
        g_list, cont_list, disc_list, y_list = [], [], [], []
        for idx in indices:
            g_vec, cont, disc, y = self.source[idx]
            g_list.append(g_vec)
            cont_list.append(cont)
            disc_list.append(disc)
            y_list.append(int(y))

        g = torch.stack(g_list)
        cont = torch.stack(cont_list)
        disc = torch.stack(disc_list)
        y = torch.tensor(y_list, dtype=torch.long)
        idx_tensor = torch.tensor(indices, dtype=torch.long)
        return g, cont, disc, y, idx_tensor

    def __getitem__(self, _):
        classes = self.rng.choice(self.available_classes, size=self.n_way, replace=False)
        k_shot = int(self.rng.choice(self.k_shot_choices))
        support_indices, query_indices = [], []
        for cls in classes:
            sup, que = self._sample_indices_for_class(int(cls), k_shot=k_shot, k_query=self.k_query)
            support_indices.extend(sup)
            query_indices.extend(que)

        g_sup, cont_sup, disc_sup, y_sup, idx_sup = self._gather(support_indices)
        g_que, cont_que, disc_que, y_que, idx_que = self._gather(query_indices)

        return {
            "support": {"g": g_sup, "cont": cont_sup, "disc": disc_sup, "y": y_sup, "indices": idx_sup},
            "query": {"g": g_que, "cont": cont_que, "disc": disc_que, "y": y_que, "indices": idx_que},
            "classes": torch.tensor(classes, dtype=torch.long),
            "meta": {"k_shot": k_shot, "k_query": self.k_query},
        }


def fixed_count_split(ds, train_per_class, test_per_class, seed=42, n_way=None):
    """
    Select exactly train/test samples per class (aligns with paper definition).
    """
    rng = np.random.default_rng(seed)
    labels = ds.df["y"].to_numpy()
    classes = sorted(np.unique(labels))
    if n_way is not None and n_way < len(classes):
        classes = classes[:n_way]

    train_indices, test_indices = [], []
    for cls in classes:
        cls_indices = np.where(labels == cls)[0]
        if len(cls_indices) < train_per_class + test_per_class:
            raise ValueError(
                f"Class {cls} has {len(cls_indices)} samples but "
                f"needs at least {train_per_class + test_per_class}."
            )
        rng.shuffle(cls_indices)
        train_indices.extend(cls_indices[:train_per_class].tolist())
        test_indices.extend(cls_indices[train_per_class:train_per_class + test_per_class].tolist())
    return Subset(ds, train_indices), Subset(ds, test_indices)


def build_basic_loader(subset, batch_size, shuffle):
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
    )


def attack_balanced_split(
    ds,
    train_per_class,
    test_per_class,
    seed=42,
    train_per_attack=None,
    test_per_attack=None,
):
    """
    Ensure each attack type contributes fixed counts to train/test.
    """
    attack_col = getattr(ds, "attack_col", None)
    if attack_col is None:
        raise ValueError("Dataset does not have attack_col; cannot build attack-balanced split.")

    rng = np.random.default_rng(seed)
    df = ds.df.reset_index(drop=True)
    labels = df["y"].to_numpy()

    train_attack = train_per_attack or train_per_class
    test_attack = test_per_attack or test_per_class

    attack_df = df[labels == 1]
    if attack_df.empty:
        raise ValueError("No attack samples found for attack-balanced split.")
    attack_counts = attack_df[attack_col].value_counts()
    attack_types = sorted(attack_counts.index.tolist())

    normal_target_train = train_per_class
    normal_target_test = test_per_class
    if attack_types:
        normal_target_train = train_attack * len(attack_types)
        normal_target_test = test_attack * len(attack_types)

    normal_indices = np.where(labels == 0)[0]
    if len(normal_indices) < normal_target_train + normal_target_test:
        raise ValueError(
            f"Normal class has {len(normal_indices)} samples but "
            f"needs {normal_target_train + normal_target_test}."
        )

    rng.shuffle(normal_indices)
    train_indices = normal_indices[:normal_target_train].tolist()
    test_indices = normal_indices[normal_target_train : normal_target_train + normal_target_test].tolist()
    for att in attack_types:
        att_indices = attack_df[attack_df[attack_col] == att].index.to_numpy()
        if len(att_indices) < train_attack + test_attack:
            print(
                f"[WARN] Attack type '{att}' has only {len(att_indices)} samples; "
                f"requires {train_attack + test_attack}. Skipping this type."
            )
            continue
        rng.shuffle(att_indices)
        train_indices.extend(att_indices[:train_attack].tolist())
        test_indices.extend(att_indices[train_attack : train_attack + test_attack].tolist())

    if not train_indices or not test_indices:
        raise ValueError("Attack-balanced split failed to gather any samples. Check counts and parameters.")

    expected_attacks = sum(
        1 for att in attack_types if attack_counts[att] >= (train_attack + test_attack)
    )
    actual_attacks = expected_attacks
    if actual_attacks != len(attack_types):
        skipped = len(attack_types) - actual_attacks
        print(
            f"[INFO] Attack-balanced split included {actual_attacks} attack types "
            f"and skipped {skipped} due to insufficient samples."
        )
    print(
        f"[INFO] attack-balanced split | "
        f"normal train/test={normal_target_train}/{normal_target_test} | "
        f"attack train/test per type={train_attack}/{test_attack}"
    )

    return Subset(ds, train_indices), Subset(ds, test_indices)