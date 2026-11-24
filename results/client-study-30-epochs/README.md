# GeNIS client-count study: 30 local epochs

Client counts: 5, 10, 15, 20, 25, 30, 35, 40, 46 and 50. Each completes 10 server rounds with 30 local epochs per client per round. The previous TAAE model, source code, data split, seed 123, batch 16, learning rate 3e-5, dropout 0.1 and RNEP pairing recipe match the recorded 70-epoch study in `../client-study-70-epochs/`.

Training uses 31,123 normal rows; validation is 3,815 normal + 3,815 attack; test is 4,373 normal + 4,373 attack. Related source rows/groups do not overlap between partitions. Normal data are split into the same ordered client chunks. The final percentile is selected on validation macro F1, then applied once to the test set. A fixed-percentile-82 control is also saved. Round performance is validation only.

Resume or collect the recorded study from the project root in the `sdp_fl` environment (completed runs are reused):

```bash
python -u results/client-study-30-epochs/run_sweep.py

# Append additional client counts; completed runs are reused.
python -u results/client-study-30-epochs/run_sweep.py --add-clients 35 40 46 50
```

- `experiment_config.json`: actual configuration, data/source SHA256 values, partition sizes and comparison to the reference.
- `source_snapshot/`: exact model/training implementation copied from the 70-epoch study, hash checked before every run.
- `status.json`, `sweep.log`: overall progress.
- `clients_<N>/status.json`, `driver.log`: progress and log for each client count.
- `clients_<N>/rounds/round_<RR>/validation_metrics.json`: metrics for each completed round.
- `clients_<N>/round_metrics.csv`, `round_metrics.png`: validation curves.
- `clients_<N>/run/previous/`: training logs, full checkpoints, reconstruction scores, percentile analyses and completion verification.
- `clients_<N>/GeNIS_*.png`, `GeNIS_metrics.json`, `rnep_frame_genis_weights.h5`: final published figures, metrics and weights.
- `client_comparison.csv`, `client_comparison.png`, `final_comparison_table.md`: final client-count comparison, updated as each experiment finishes.
- `round_comparison.csv`, `round_comparison.png`: round-by-round validation comparison.

The separate `results/client-study-70-epochs/` archive retains the earlier measurements. All 10 runs in this directory completed with 30 local epochs and 10 server rounds.

The original six runs completed before extending to 35, 40, 46 and 50 clients. Their model/score hashes are recorded in `completed_original_six_sha256.json`; the original completion status and configuration are retained alongside it.

Operational paths were updated after renaming this directory. `experiment_config_as_recorded.json`, model/launch metadata and frozen source files retain their as-recorded content; `artifact_paths.json` maps historical study prefixes to the current directories.

To train a fresh copy with the same frozen sources and data without replacing the archive:

```bash
python -u results/client-study-30-epochs/run_sweep.py \
  --output-root outputs/genis/client-study-30-epochs \
  --add-clients 35 40 46 50
```
