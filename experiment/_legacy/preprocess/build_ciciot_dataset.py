"""
Build the CICIoT2023 files in the same layout as CSE-CIC-IDS2018 (cic_rnep / cic_graph).

Input : CICFlowMeter-4.0 flow CSVs  (CICIoT/FLOW/<class>/<pcap>_Flow.csv, from run_cicflowmeter.py)
Output: <out_root>/ciciot_rnep/   CICIoT_ae_normal.csv, CICIoT_anomaly_ae_{1..27}.csv, CICIoT_ae_anomaly.csv
        <out_root>/ciciot_graph/  same file names, columns [Dst Port, Timestamp]
        <out_root>/ciciot_attack_map.csv, <out_root>/ciciot_build_stats.csv

- TAAE files use the 78 CIC2018 feature columns (CIC2018 short names, same order).
- GSAD files hold the same rows in the same order as the TAAE files (row i == same flow).
- Timestamp is written as dd/mm/yyyy HH:MM:SS (24h), parsed by the GSAD code with dayfirst=True.
- Normal: random 200k benign flows. Anomaly: per-class cap so the total is 200k
  (rare classes kept as-is), see docs/CICIoT2023_integration_plan.md §4.3.

Usage:
  python preprocess/build_ciciot_dataset.py --flow_root CICIoT/FLOW --out_root CICIoT/processed
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

RANDOM_STATE = 123

# CICFlowMeter-4.0 column -> CSE-CIC-IDS2018 column (TAAE feature order of cic_rnep)
CFM_TO_CIC2018 = {
    "Dst Port": "Dst Port",
    "Protocol": "Protocol",
    "Flow Duration": "Flow Duration",
    "Total Fwd Packet": "Tot Fwd Pkts",
    "Total Bwd packets": "Tot Bwd Pkts",
    "Total Length of Fwd Packet": "TotLen Fwd Pkts",
    "Total Length of Bwd Packet": "TotLen Bwd Pkts",
    "Fwd Packet Length Max": "Fwd Pkt Len Max",
    "Fwd Packet Length Min": "Fwd Pkt Len Min",
    "Fwd Packet Length Mean": "Fwd Pkt Len Mean",
    "Fwd Packet Length Std": "Fwd Pkt Len Std",
    "Bwd Packet Length Max": "Bwd Pkt Len Max",
    "Bwd Packet Length Min": "Bwd Pkt Len Min",
    "Bwd Packet Length Mean": "Bwd Pkt Len Mean",
    "Bwd Packet Length Std": "Bwd Pkt Len Std",
    "Flow Bytes/s": "Flow Byts/s",
    "Flow Packets/s": "Flow Pkts/s",
    "Flow IAT Mean": "Flow IAT Mean",
    "Flow IAT Std": "Flow IAT Std",
    "Flow IAT Max": "Flow IAT Max",
    "Flow IAT Min": "Flow IAT Min",
    "Fwd IAT Total": "Fwd IAT Tot",
    "Fwd IAT Mean": "Fwd IAT Mean",
    "Fwd IAT Std": "Fwd IAT Std",
    "Fwd IAT Max": "Fwd IAT Max",
    "Fwd IAT Min": "Fwd IAT Min",
    "Bwd IAT Total": "Bwd IAT Tot",
    "Bwd IAT Mean": "Bwd IAT Mean",
    "Bwd IAT Std": "Bwd IAT Std",
    "Bwd IAT Max": "Bwd IAT Max",
    "Bwd IAT Min": "Bwd IAT Min",
    "Fwd PSH Flags": "Fwd PSH Flags",
    "Bwd PSH Flags": "Bwd PSH Flags",
    "Fwd URG Flags": "Fwd URG Flags",
    "Bwd URG Flags": "Bwd URG Flags",
    "Fwd Header Length": "Fwd Header Len",
    "Bwd Header Length": "Bwd Header Len",
    "Fwd Packets/s": "Fwd Pkts/s",
    "Bwd Packets/s": "Bwd Pkts/s",
    "Packet Length Min": "Pkt Len Min",
    "Packet Length Max": "Pkt Len Max",
    "Packet Length Mean": "Pkt Len Mean",
    "Packet Length Std": "Pkt Len Std",
    "Packet Length Variance": "Pkt Len Var",
    "FIN Flag Count": "FIN Flag Cnt",
    "SYN Flag Count": "SYN Flag Cnt",
    "RST Flag Count": "RST Flag Cnt",
    "PSH Flag Count": "PSH Flag Cnt",
    "ACK Flag Count": "ACK Flag Cnt",
    "URG Flag Count": "URG Flag Cnt",
    "CWR Flag Count": "CWE Flag Count",  # CIC2018 release spells it "CWE"
    "ECE Flag Count": "ECE Flag Cnt",
    "Down/Up Ratio": "Down/Up Ratio",
    "Average Packet Size": "Pkt Size Avg",
    "Fwd Segment Size Avg": "Fwd Seg Size Avg",
    "Bwd Segment Size Avg": "Bwd Seg Size Avg",
    "Fwd Bytes/Bulk Avg": "Fwd Byts/b Avg",
    "Fwd Packet/Bulk Avg": "Fwd Pkts/b Avg",
    "Fwd Bulk Rate Avg": "Fwd Blk Rate Avg",
    "Bwd Bytes/Bulk Avg": "Bwd Byts/b Avg",
    "Bwd Packet/Bulk Avg": "Bwd Pkts/b Avg",
    "Bwd Bulk Rate Avg": "Bwd Blk Rate Avg",
    "Subflow Fwd Packets": "Subflow Fwd Pkts",
    "Subflow Fwd Bytes": "Subflow Fwd Byts",
    "Subflow Bwd Packets": "Subflow Bwd Pkts",
    "Subflow Bwd Bytes": "Subflow Bwd Byts",
    "FWD Init Win Bytes": "Init Fwd Win Byts",
    "Bwd Init Win Bytes": "Init Bwd Win Byts",
    "Fwd Act Data Pkts": "Fwd Act Data Pkts",
    "Fwd Seg Size Min": "Fwd Seg Size Min",
    "Active Mean": "Active Mean",
    "Active Std": "Active Std",
    "Active Max": "Active Max",
    "Active Min": "Active Min",
    "Idle Mean": "Idle Mean",
    "Idle Std": "Idle Std",
    "Idle Max": "Idle Max",
    "Idle Min": "Idle Min",
}
FEATURES = list(CFM_TO_CIC2018.values())
assert len(FEATURES) == 78

NORMAL_CLASS = "Benign_Final"

# Attack numbering: descending by row count of the official CICIoT2023 CSV release
# (same convention as CIC2018, where 1 = largest class HOIC ... 14 = SQL Injection).
# PCAP folder names are used; the CSV release spells two of them differently.
ATTACKS = [
    ("DDoS-ICMP_Flood", "DDoS"),
    ("DDoS-UDP_Flood", "DDoS"),
    ("DDoS-TCP_Flood", "DDoS"),
    ("DDoS-PSHACK_Flood", "DDoS"),       # CSV: DDoS-PSHACK_FLOOD
    ("DDoS-SYN_Flood", "DDoS"),
    ("DDoS-RSTFINFlood", "DDoS"),        # CSV: DDoS-RSTFINFLOOD
    ("DDoS-SynonymousIP_Flood", "DDoS"),
    ("DoS-UDP_Flood", "DoS"),
    ("DoS-TCP_Flood", "DoS"),
    ("DoS-SYN_Flood", "DoS"),
    ("Mirai-greeth_flood", "Mirai"),
    ("Mirai-udpplain", "Mirai"),
    ("Mirai-greip_flood", "Mirai"),
    ("DDoS-ICMP_Fragmentation", "DDoS"),
    ("VulnerabilityScan", "Recon"),
    ("MITM-ArpSpoofing", "Spoofing"),
    ("DDoS-UDP_Fragmentation", "DDoS"),
    ("DDoS-ACK_Fragmentation", "DDoS"),
    ("DNS_Spoofing", "Spoofing"),
    ("Recon-HostDiscovery", "Recon"),
    ("Recon-OSScan", "Recon"),
    ("Recon-PortScan", "Recon"),
    ("DoS-HTTP_Flood", "DoS"),
    ("DDoS-HTTP_Flood", "DDoS"),
    ("DDoS-SlowLoris", "DDoS"),
    ("DictionaryBruteForce", "BruteForce"),
    ("BrowserHijacking", "Web"),
    ("CommandInjection", "Web"),
    ("SqlInjection", "Web"),
    ("XSS", "Web"),
    ("Backdoor_Malware", "Web"),
    ("Recon-PingSweep", "Recon"),
    ("Uploading_Attack", "Web"),
]

# Classes CICFlowMeter cannot represent as attack flows (docs/CICIoT2023_integration_plan.md §11.2):
# packets without a TCP/UDP header get protocol 0 / port 0 and collapse into a few hundred huge
# flows, so >95% of the sampled flows would be background benign traffic labelled as attack.
EXCLUDED = {
    "DDoS-ICMP_Flood": "ICMP, no L4 ports",
    "DDoS-ICMP_Fragmentation": "IP fragments without L4 header",
    "Recon-PingSweep": "ICMP, no L4 ports",
    "Mirai-greeth_flood": "GRE encapsulation, inner packet not parsed",
    "Mirai-greip_flood": "GRE encapsulation, inner packet not parsed",
    "MITM-ArpSpoofing": "ARP is not IP; relayed traffic looks benign at IP level",
}
ATTACKS = [(c, cat) for c, cat in ATTACKS if c not in EXCLUDED]
assert len(ATTACKS) == 27


def load_class(flow_root, cls):
    files = sorted(glob.glob(os.path.join(flow_root, cls, "*_Flow.csv")))
    if not files:
        raise FileNotFoundError(f"no flow csv for {cls}")
    df = pd.concat([pd.read_csv(f, low_memory=False) for f in files], ignore_index=True)
    n_raw = len(df)

    missing = [c for c in CFM_TO_CIC2018 if c not in df.columns]
    if missing:
        raise KeyError(f"{cls}: missing columns {missing}")

    feats = df[list(CFM_TO_CIC2018)].rename(columns=CFM_TO_CIC2018)
    feats = feats.apply(pd.to_numeric, errors="coerce")

    # CICFlowMeter prints 12h time with AM/PM; store 24h so dayfirst parsing is unambiguous
    ts = pd.to_datetime(df["Timestamp"], format="%d/%m/%Y %I:%M:%S %p", errors="coerce")

    bad = feats.replace([np.inf, -np.inf], np.nan).isna().any(axis=1) | ts.isna()
    feats, ts = feats[~bad], ts[~bad]
    feats = feats.astype({"Dst Port": "int64", "Protocol": "int64"})
    feats["Timestamp"] = ts.dt.strftime("%d/%m/%Y %H:%M:%S")
    return feats.reset_index(drop=True), n_raw, int(bad.sum())


def water_fill_caps(counts, budget):
    """Per-class cap c so that sum(min(n_i, c)) == budget (classes below c kept whole)."""
    if sum(counts.values()) <= budget:
        return dict(counts)
    lo, hi = 0, max(counts.values())
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if sum(min(n, mid) for n in counts.values()) <= budget:
            lo = mid
        else:
            hi = mid - 1
    take = {k: min(n, lo) for k, n in counts.items()}
    # hand out the remainder one by one to classes that still have rows
    rest = budget - sum(take.values())
    for k in sorted(counts, key=lambda k: -counts[k]):
        if rest == 0:
            break
        if counts[k] > take[k]:
            take[k] += 1
            rest -= 1
    return take


def write_pair(df, rnep_path, graph_path):
    df[FEATURES].to_csv(rnep_path, index=False)
    df[["Dst Port", "Timestamp"]].to_csv(graph_path, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flow_root", default="CICIoT/FLOW")
    ap.add_argument("--out_root", default="CICIoT/processed")
    ap.add_argument("--n_normal", type=int, default=200_000)
    ap.add_argument("--n_anomaly", type=int, default=200_000)
    args = ap.parse_args()

    rnep_dir = os.path.join(args.out_root, "ciciot_rnep")
    graph_dir = os.path.join(args.out_root, "ciciot_graph")
    os.makedirs(rnep_dir, exist_ok=True)
    os.makedirs(graph_dir, exist_ok=True)
    # drop per-class files from a previous build so stale class ids are not picked up by prefix
    for f in glob.glob(os.path.join(rnep_dir, "CICIoT_anomaly_ae_*.csv")) + \
             glob.glob(os.path.join(graph_dir, "CICIoT_anomaly_ae_*.csv")):
        os.remove(f)

    stats = []

    # ---- normal
    df_n, n_raw, n_bad = load_class(args.flow_root, NORMAL_CLASS)
    df_n = df_n.sample(n=min(args.n_normal, len(df_n)), random_state=RANDOM_STATE).reset_index(drop=True)
    write_pair(df_n, os.path.join(rnep_dir, "CICIoT_ae_normal.csv"),
               os.path.join(graph_dir, "CICIoT_ae_normal.csv"))
    stats.append({"id": 0, "class": NORMAL_CLASS, "category": "Benign",
                  "flows_raw": n_raw, "dropped_nan_inf": n_bad, "flows_clean": n_raw - n_bad,
                  "sampled": len(df_n)})
    print(f"[normal] raw={n_raw} dropped={n_bad} sampled={len(df_n)}", flush=True)

    # ---- anomaly
    pools = {}
    for i, (cls, cat) in enumerate(ATTACKS, 1):
        df_a, n_raw, n_bad = load_class(args.flow_root, cls)
        pools[cls] = df_a
        stats.append({"id": i, "class": cls, "category": cat, "flows_raw": n_raw,
                      "dropped_nan_inf": n_bad, "flows_clean": len(df_a)})
        print(f"[{i:2d}] {cls}: raw={n_raw} dropped={n_bad}", flush=True)

    take = water_fill_caps({c: len(d) for c, d in pools.items()}, args.n_anomaly)

    merged = []
    for i, (cls, _) in enumerate(ATTACKS, 1):
        df_s = pools[cls].sample(n=take[cls], random_state=RANDOM_STATE).reset_index(drop=True)
        write_pair(df_s, os.path.join(rnep_dir, f"CICIoT_anomaly_ae_{i}.csv"),
                   os.path.join(graph_dir, f"CICIoT_anomaly_ae_{i}.csv"))
        merged.append(df_s)
        stats[i]["sampled"] = len(df_s)

    # merged file name must not start with the per-class prefix (CICIoT_anomaly_ae_)
    pd.concat(merged, ignore_index=True)[FEATURES].to_csv(
        os.path.join(rnep_dir, "CICIoT_ae_anomaly.csv"), index=False)

    st = pd.DataFrame(stats)
    st.to_csv(os.path.join(args.out_root, "ciciot_build_stats.csv"), index=False)
    st[st["id"] > 0][["id", "class", "category"]].to_csv(
        os.path.join(args.out_root, "ciciot_attack_map.csv"), index=False)
    print(st.to_string(index=False))
    print(f"[done] normal={len(df_n)} anomaly={int(st['sampled'][1:].sum())} -> {args.out_root}")


if __name__ == "__main__":
    main()
