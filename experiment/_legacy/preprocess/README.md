# CICIoT2023 Preprocessing

Builds CICIoT2023 files in the same layout and with the same 78 CICFlowMeter features as the
CSE-CIC-IDS2018 data (`cic_rnep` / `cic_graph`). See `docs/CICIoT2023_integration_plan.md` for the rationale.

## 1. Build the CICFlowMeter image (once)

```bash
docker build -t cicflowmeter:4.0 preprocess/cicflowmeter
```

The image builds the official Java CICFlowMeter-4.0 (`ahlashkari/CICFlowMeter`) with its bundled jnetpcap.

## 2. PCAP → flow CSV

```bash
python preprocess/run_cicflowmeter.py --pcap_root CICIoT/PCAP --out_root CICIoT/FLOW --workers 6
```

- Benign: every pcap. Other classes: one pcap (the smallest) — enough for the per-class sample size.
  Use `--all_files` to process everything.
- Flows are written inside the container first and copied out at the end
  (writing straight to a Windows bind mount is ~12x slower).
- Already extracted pcaps are skipped, so the script can be re-run after a failure.

## 3. Flow CSV → experiment files

```bash
python preprocess/build_ciciot_dataset.py --flow_root CICIoT/FLOW --out_root CICIoT/processed
```

Output:

```text
CICIoT/processed/
├── ciciot_rnep/    CICIoT_ae_normal.csv, CICIoT_anomaly_ae_{1..27}.csv, CICIoT_ae_anomaly.csv  (78 features)
├── ciciot_graph/   CICIoT_ae_normal.csv, CICIoT_anomaly_ae_{1..27}.csv                         (Dst Port, Timestamp)
├── ciciot_attack_map.csv
└── ciciot_build_stats.csv
```

- Normal: 200,000 random benign flows. Anomaly: 200,000 flows, per-class cap (rare classes kept whole).
- 27 of the 33 attack types are used. Six attacks without TCP/UDP headers (ICMP, GRE) or outside IP (ARP)
  are excluded (`EXCLUDED` in `build_ciciot_dataset.py`, reasons in `docs/CICIoT2023_integration_plan.md` §11.3).
- `ciciot_rnep` and `ciciot_graph` hold the same rows in the same order.
- Rows with NaN/Infinity features are dropped; negative values are kept (the loaders clip them).

## 4. Copy to the server

```text
CICIoT/processed/ciciot_rnep   → /data/SDP_Dataset/Unified_model/ciciot_rnep
CICIoT/processed/ciciot_graph  → /data/SDP_Dataset/Unified_model/ciciot_graph
```

## 5. Run on the server

All commands run from the repository root (`ciciot2023` branch).

```bash
# baselines and scripts/test_*.py read ./CICIoT2023/ciciot_rnep
ln -s /data/SDP_Dataset/Unified_model ./CICIoT2023
```

**(1) TAAE federated training (RNEP)** — writes to `./FedSAD_Results/`, then copy the weights to where `fedsad.py` looks

```bash
python rnep.py --eval_dataset ciciot
mkdir -p results/CICIoT2023/taae && cp FedSAD_Results/fedsad_weights.h5 results/CICIoT2023/taae/taae_ciciot_weights.h5
```

**(2) GSAD** — `--threshold_std` needs tuning for CICIoT2023 (CIC 0.25, UNSW 5)

```bash
python gsad.py --eval_dataset ciciot --ae_data_dir /data/SDP_Dataset/Unified_model/ciciot_graph --result_path results/CICIoT2023/gsad --threshold_std 0.25
```

**(3) FedSAD (TAAE + GSAD fusion)** — needs the outputs of (1) and (2)

```bash
python fedsad.py --eval_dataset ciciot --threshold_std 0.25
```

**(4) TAAE per-attack evaluation** — set `DATASET_TO_RUN = "CICIoT2023"` in `scripts/test_taae.py`

```bash
PYTHONPATH=. python scripts/test_taae.py
```

**(5) Baselines** — in `main()` comment out the current loader and uncomment the `# CICIoT2023` block
(`models/AE_LSTM.py`, `models/CNN_LSTM.py`). Results go to `./ae_lstm/`, `./cnn_lstm/`.

```bash
PYTHONPATH=.:models python models/AE_LSTM.py
PYTHONPATH=.:models python models/CNN_LSTM.py
```

Per-attack evaluation: set `DATASET_TO_RUN = "CICIoT2023"` in `scripts/test_aelstm.py` / `scripts/test_cnnlstm.py`.
Run them with `PYTHONPATH=.:models python scripts/test_aelstm.py` (they import `model_aelstm` / `model_cnnlstm`).
Note that both scripts currently call `exit()` right after `model.summary()`, so remove that line to run the evaluation.

Not wired for CICIoT2023 yet: MM-FewShot-IDS, MV-IDS (need NetFlow features), `models/FedAVG.py`, `models/CML.py`.
