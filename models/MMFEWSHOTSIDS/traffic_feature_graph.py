import numpy as np
import pandas as pd


def safe_get(row, col, default=0.0):
    return float(row[col]) if col in row and not pd.isna(row[col]) else default


def build_packet_sequence_tensor(
    csv_path,
    output_path,
    max_packets=16,
    feature_dim=16,
):
    print(f"[INFO] Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    N = len(df)
    tensor = np.zeros((N, 1, 16, 16, 16), dtype=np.float32)

    for i, row in df.iterrows():
        # -----------------------------
        # 1. packet 개수 추정
        # -----------------------------
        in_pkts = safe_get(row, "IN_PKTS")
        out_pkts = safe_get(row, "OUT_PKTS")

        total_pkts = int(in_pkts + out_pkts)
        total_pkts = max(1, min(total_pkts, max_packets))

        # -----------------------------
        # 2. packet size 분포 기반 생성
        # -----------------------------
        packet_sizes = []

        def add_packets(col, value):
            cnt = int(safe_get(row, col))
            packet_sizes.extend([value] * cnt)

        add_packets("NUM_PKTS_UP_TO_128_BYTES", 64)
        add_packets("NUM_PKTS_128_TO_256_BYTES", 192)
        add_packets("NUM_PKTS_256_TO_512_BYTES", 384)
        add_packets("NUM_PKTS_512_TO_1024_BYTES", 768)
        add_packets("NUM_PKTS_1024_TO_1514_BYTES", 1200)

        # fallback
        if len(packet_sizes) == 0:
            avg_size = safe_get(row, "IN_BYTES") / (in_pkts + 1e-6)
            packet_sizes = [avg_size] * total_pkts

        # 길이 맞추기
        packet_sizes = packet_sizes[:max_packets]
        if len(packet_sizes) < max_packets:
            packet_sizes += [packet_sizes[-1]] * (max_packets - len(packet_sizes))

        # -----------------------------
        # 3. IAT 생성 (temporal 정보)
        # -----------------------------
        iat_mean = safe_get(row, "SRC_TO_DST_IAT_AVG", 1.0)
        iat_std = safe_get(row, "SRC_TO_DST_IAT_STDDEV", 0.1)

        iats = np.random.normal(iat_mean, iat_std + 1e-6, max_packets)
        iats = np.clip(iats, 0, None)

        # -----------------------------
        # 4. direction (in/out)
        # -----------------------------
        directions = np.zeros(max_packets)
        split = int(max_packets * (in_pkts / (in_pkts + out_pkts + 1e-6)))
        directions[:split] = 1  # inbound = 1, outbound = 0

        # -----------------------------
        # 5. packet feature 구성
        # -----------------------------
        packets = []

        for j in range(max_packets):
            pkt = [
                packet_sizes[j],
                iats[j],
                directions[j],
                safe_get(row, "MIN_TTL"),
                safe_get(row, "MAX_TTL"),
                safe_get(row, "TCP_WIN_MAX_IN"),
                safe_get(row, "TCP_WIN_MAX_OUT"),
                safe_get(row, "SRC_TO_DST_AVG_THROUGHPUT"),
                safe_get(row, "DST_TO_SRC_AVG_THROUGHPUT"),
                safe_get(row, "RETRANSMITTED_IN_PKTS"),
                safe_get(row, "RETRANSMITTED_OUT_PKTS"),
                safe_get(row, "LONGEST_FLOW_PKT"),
                safe_get(row, "SHORTEST_FLOW_PKT"),
                safe_get(row, "MIN_IP_PKT_LEN"),
                safe_get(row, "MAX_IP_PKT_LEN"),
                j / max_packets,  # position encoding 느낌
            ]
            packets.append(pkt)

        packets = np.array(packets, dtype=np.float32)  # (16,16)

        # -----------------------------
        # 6. normalization (per sample)
        # -----------------------------
        mean = packets.mean(axis=0, keepdims=True)
        std = packets.std(axis=0, keepdims=True) + 1e-6
        packets = (packets - mean) / std

        # -----------------------------
        # 7. 2D → 3D tensor 변환
        # -----------------------------
        grid = packets.reshape(16, 16)

        tensor[i, 0] = np.repeat(grid[:, :, None], 16, axis=2)

    print(f"[INFO] Final tensor shape: {tensor.shape}")

    np.save(output_path, tensor)
    print(f"[INFO] Saved tensor to: {output_path}")
    return tensor


if __name__ == "__main__":
    build_packet_sequence_tensor(
        csv_path="/data/SDP_Dataset/CICIDS2018/NF-CICIDS2018-v3-balanced.csv",
        output_path="traffic_tensor.npy"
    )