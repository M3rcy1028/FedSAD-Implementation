import numpy as np
import pandas as pd
from collections import OrderedDict
from typing import Dict, Iterable, List, Optional, Sequence

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tqdm.auto import tqdm

# -------------------------------
# Multi-view schema (Table 2)
# -------------------------------

# The schema mirrors Table 2 of “Multi-view intrusion detection framework using deep learning
# and knowledge graphs”. Each view encapsulates a semantically coherent set of UNSW-NB15
# features. Categorical attributes are handled with one-hot encoding, while numeric attributes
# are standardised.
UNSW_VIEW_SCHEMA: "OrderedDict[str, Dict[str, list[str]]]" = OrderedDict(
    [
        # ① FLOW VIEW
        (
            "flow",
            {
                "numeric": [
                    "dur",       # duration
                    "sbytes", "dbytes",
                    "spkts", "dpkts",
                    "sttl", "dttl",
                    "sloss", "dloss",
                ],
                "categorical": [],
            },
        ),

        # ② BASIC VIEW
        (
            "basic",
            {
                "numeric": [
                    "smeansz", "dmeansz",
                    "swin", "dwin",
                    "stcpb", "dtcpb",
                    "trans_depth", "res_bdy_len",
                ],
                "categorical": [
                    "proto", "state", "service",
                ],
            },
        ),

        # ③ CONTENT VIEW
        (
            "content",
            {
                "numeric": [
                    "sload", "dload",
                    "sjit", "djit",
                ],
                "categorical": [],
            },
        ),

        # ④ TIME VIEW
        (
            "time",
            {
                "numeric": [
                    "stime", "ltime",
                    "sintpkt", "dintpkt",
                    "tcprtt", "synack", "ackdat",
                ],
                "categorical": [],
            },
        ),

        # ⑤ ADDITIONAL VIEW
        (
            "additional",
            {
                "numeric": [],
                "categorical": [
                    "is_sm_ips_ports",
                    "ct_state_ttl",
                    "ct_ftp_cmd",
                    "ct_srv_src",
                    "ct_srv_dst",
                    "ct_dst_ltm",
                    "ct_src_ltm",
                    "ct_src_dport_ltm",
                    "ct_dst_sport_ltm",
                    "ct_dst_src_ltm",
                ],
            },
        ),
    ]
)


CICIDS2018_VIEW_SCHEMA: "OrderedDict[str, Dict[str, list[str]]]" = OrderedDict(
    [
        (
            "network_traffic",
            {
                "numeric": [
                    "IN_BYTES",
                    "OUT_BYTES",
                    "IN_PKTS",
                    "OUT_PKTS",
                    "SRC_TO_DST_SECOND_BYTES",
                    "DST_TO_SRC_SECOND_BYTES",
                    "SRC_TO_DST_AVG_THROUGHPUT",
                    "DST_TO_SRC_AVG_THROUGHPUT",
                ],
                "categorical": [],
            },
        ),
        (
            "temporal",
            {
                "numeric": [
                    "FLOW_DURATION_MILLISECONDS",
                    "SRC_TO_DST_IAT_MIN",
                    "SRC_TO_DST_IAT_MAX",
                    "SRC_TO_DST_IAT_AVG",
                    "SRC_TO_DST_IAT_STDDEV",
                    "DST_TO_SRC_IAT_MIN",
                    "DST_TO_SRC_IAT_MAX",
                    "DST_TO_SRC_IAT_AVG",
                    "DST_TO_SRC_IAT_STDDEV",
                    "DURATION_IN",
                    "DURATION_OUT",
                ],
                "categorical": [],
            },
        ),
        (
            "protocol_service",
            {
                "numeric": [],
                "categorical": [
                    "PROTOCOL",
                    "L7_PROTO",
                    "DNS_QUERY_TYPE",
                    "DNS_TTL_ANSWER",
                    "FTP_COMMAND_RET_CODE",
                    "ICMP_TYPE",
                    "ICMP_IPV4_TYPE",
                    "TCP_FLAGS",
                    "CLIENT_TCP_FLAGS",
                    "SERVER_TCP_FLAGS",
                ],
            },
        ),
        (
            "security",
            {
                "numeric": [
                    "MIN_TTL",
                    "MAX_TTL",
                    "RETRANSMITTED_IN_BYTES",
                    "RETRANSMITTED_IN_PKTS",
                    "RETRANSMITTED_OUT_BYTES",
                    "RETRANSMITTED_OUT_PKTS",
                    "TCP_WIN_MAX_IN",
                    "TCP_WIN_MAX_OUT",
                ],
                "categorical": [],
            },
        ),
        (
            "connection",
            {
                "numeric": [
                    "L4_SRC_PORT",
                    "L4_DST_PORT",
                    "FLOW_END_MILLISECONDS",
                ],
                "categorical": [
                    "IPV4_SRC_ADDR",
                    "IPV4_DST_ADDR",
                ],
                "numeric_prefixes": [
                    "NUM_PKTS_",
                ],
            },
        ),
    ]
)


def _ensure_list(cols: Optional[Sequence[str]]) -> List[str]:
    if cols is None:
        return []
    if isinstance(cols, (list, tuple)):
        return list(cols)
    return [cols]


class MultiViewPreprocessor:
    """Scikit-learn based multi-view preprocessing helper."""

    def __init__(self, schema: "OrderedDict[str, Dict[str, Sequence[str]]]" = UNSW_VIEW_SCHEMA):
        self.schema = schema
        self.transformers_: Dict[str, Optional[ColumnTransformer]] = {}
        self.feature_names_: Dict[str, List[str]] = {}

    def fit(self, df: pd.DataFrame) -> "MultiViewPreprocessor":
        views_iter = list(self.schema.items())
        for view_name, view_cols in tqdm(
            views_iter,
            desc="Preprocess | fit views",
            leave=False,
        ):
            num_cols = self._resolve_columns(
                df.columns,
                _ensure_list(view_cols.get("numeric")),
                _ensure_list(view_cols.get("numeric_prefixes")),
            )
            cat_cols = self._resolve_columns(
                df.columns,
                _ensure_list(view_cols.get("categorical")),
                _ensure_list(view_cols.get("categorical_prefixes")),
            )

            transformers = []
            if num_cols:
                num_pipe = Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                )
                transformers.append(("num", num_pipe, num_cols))
            if cat_cols:
                cat_pipe = Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                )
                transformers.append(("cat", cat_pipe, cat_cols))

            if not transformers:
                self.transformers_[view_name] = None
                self.feature_names_[view_name] = []
                continue

            ct = ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.0)
            ct.fit(df)
            self.transformers_[view_name] = ct
            self.feature_names_[view_name] = self._collect_feature_names(ct)
        return self

    def fit_transform(self, df: pd.DataFrame) -> List[np.ndarray]:
        self.fit(df)
        return self.transform(df)

    def transform(self, df: pd.DataFrame) -> List[np.ndarray]:
        outputs: List[np.ndarray] = []
        view_names = list(self.schema.keys())
        for view_name in tqdm(
            view_names,
            desc="Preprocess | transform views",
            leave=False,
        ):
            ct = self.transformers_.get(view_name)
            if ct is None:
                outputs.append(np.zeros((len(df), 0), dtype=np.float32))
                continue
            arr = ct.transform(df)
            outputs.append(np.asarray(arr, dtype=np.float32))
        return outputs

    def _collect_feature_names(self, ct: ColumnTransformer) -> List[str]:
        names: List[str] = []
        for _, trans, cols in ct.transformers_:
            if trans == "drop":
                continue
            cols_list = _ensure_list(cols)
            last_step = trans
            if isinstance(trans, Pipeline):
                last_step = trans.steps[-1][1]
            if hasattr(last_step, "get_feature_names_out"):
                try:
                    out_names = last_step.get_feature_names_out(cols_list)
                except TypeError:
                    out_names = last_step.get_feature_names_out()
                names.extend([str(n) for n in out_names])
            else:
                names.extend([str(c) for c in cols_list])
        return names

    def _resolve_columns(
        self,
        df_columns: Iterable[str],
        explicit: Sequence[str],
        prefixes: Sequence[str],
    ) -> List[str]:
        resolved: List[str] = []
        df_cols_list = list(df_columns)
        for col in explicit:
            if col in df_cols_list:
                resolved.append(col)
        for prefix in prefixes:
            matches = [c for c in df_cols_list if c.startswith(prefix)]
            resolved.extend(matches)
        # Preserve original order while dropping duplicates
        seen = set()
        unique_cols = []
        for col in resolved:
            if col not in seen:
                unique_cols.append(col)
                seen.add(col)
        return unique_cols

    @property
    def view_names(self) -> List[str]:
        return list(self.schema.keys())

    def get_feature_names(self) -> Dict[str, List[str]]:
        return self.feature_names_
