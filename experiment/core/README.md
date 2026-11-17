# experiment/core

Model implementations and the original experiment pipelines. `run.py` loads these
modules through `experiment/backends.py`; the scripts here can also be run directly.

| Path | Contents |
|---|---|
| `fedsad.py`, `gsad.py` | FedSAD (GSAD + TAAE, Dempster-Shafer fusion) and the graph-based detector |
| `taae_previous.py`, `taae_current.py` | TAAE model and FL client. `previous` (latent 64) matches the archived weights; `current` is the newer variant |
| `train_*.py` | Federated / centralized training entry points (TAAE with RNEP, AE-LSTM, CNN-LSTM, FedAvg, CML, MV-IDS, MM-FewShot-IDS) |
| `models/` | AE-LSTM and CNN-LSTM definitions, MV-IDS and MM-FewShot-IDS packages |
| `scripts/evaluate_*.py` | Evaluate the archived weights in `results/` |
| `scripts/` (others) | GeNIS preprocessing, baselines, client sweep, window sweep and paper figures |
| `utils/` | Dataset loaders, metrics, GeNIS/Gotham helpers, window metrics, threshold search |
