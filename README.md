# FedSAD: Federated Learning-based Anomaly Detection in Software-Defined Perimeter Architecture for Distributed Authentication

Official implementation of the **FedSAD** framework, designed for secure and distributed anomaly detection within Software-Defined Perimeter (SDP) architectures.

## Overview

FedSAD is a novel framework that integrates **Federated Learning (FL)** with **Anomaly Detection** to enhance security in SDP-based environments. It focuses on distributed authentication and identifying malicious activities while preserving data privacy across multiple nodes.

FedSAD combines two detectors and fuses their evidence with Dempster-Shafer theory for every detection window:

- **TAAE** (Transformer Adversarial Autoencoder), trained with federated learning and the **RNEP** entropy-aware aggregation
- **GSAD** (graph-based anomaly detection), which compares each window's port-communication graph with normal statistics

This repository provides the FedSAD source code, the comparison baselines, evaluation scripts, the archived experimental results, dataset preprocessing details and the hyperparameter settings used in the paper.

## Experiment guide

Installation, source layout, data preparation, training and re-evaluation are documented in the [experiment and reproduction guide](docs/EXPERIMENTS.md).

- [Quick start](docs/EXPERIMENTS.md#quick-start) · [Directory structure](docs/EXPERIMENTS.md#directory-structure) · [Models and datasets](docs/EXPERIMENTS.md#models)
- [Dataset preparation](docs/EXPERIMENTS.md#datasets-and-paths) · [Run experiments](docs/EXPERIMENTS.md#running-experiments-with-runpy) · [Reproduce paper results](docs/EXPERIMENTS.md#reproducing-the-paper-results)
- [Original study settings](docs/EXPERIMENTS.md#original-study-details) · [Testing and limitations](docs/EXPERIMENTS.md#testing-and-known-limitations)

## Archived results

`results/` holds the paper figures, metrics, final weights and client-study records. See the [artifact layout](docs/EXPERIMENTS.md#archived-artifact-layout) for directory details and the [reproduction guide](docs/EXPERIMENTS.md#reproducing-the-paper-results) for evaluation commands.

### Reconstruction-error distributions

The five panels below present reconstruction-error distributions for KDD99, InSDN, CSE-CIC-IDS2018, UNSW-NB15 and GeNIS. The caption and accompanying analysis are provided here as supplementary material. The figures link to the archived artifacts in `results/`.

<table>
  <tbody>
    <tr>
      <td align="left"><strong>(a) KDD99</strong></td>
      <td align="left"><strong>(b) InSDN</strong></td>
    </tr>
    <tr>
      <td align="left"><img src="results/KDD99/taae/KDD99_distribution.png" alt="KDD99 reconstruction error"></td>
      <td align="left"><img src="results/InSDN/taae/InSDN_distribution.png" alt="InSDN reconstruction error"></td>
    </tr>
    <tr>
      <td align="left"><strong>(c) CSE-CIC-IDS2018</strong></td>
      <td align="left"><strong>(d) UNSW-NB15</strong></td>
    </tr>
    <tr>
      <td align="left"><img src="results/CSE-CIC-IDS2018/taae/CSE-CIC-IDS2018_distribution.png" alt="CSE-CIC-IDS2018 reconstruction error"></td>
      <td align="left"><img src="results/UNSW-NB15/taae/UNSW_NB15_distribution.png" alt="UNSW-NB15 reconstruction error"></td>
    </tr>
    <tr>
      <td align="left"><strong>(e) GeNIS</strong></td>
      <td></td>
    </tr>
    <tr>
      <td align="left"><img src="results/GeNIS/taae/GeNIS_distribution.png" alt="GeNIS reconstruction error"></td>
      <td></td>
    </tr>
  </tbody>
</table>

**Figure caption.** Reconstruction error distributions for normal (green) and anomalous traffic across five datasets: (a) KDD99, (b) InSDN, (c) CSE-CIC-IDS2018, (d) UNSW-NB15 and (e) GeNIS. The histograms display the density of reconstruction errors for each dataset, illustrating how normal samples generally cluster below the detection threshold, while anomalous samples tend toward higher values. The vertical blue dashed line denotes the detection threshold ($\tau$) derived from normal training reconstruction errors. For GeNIS, the operating percentile is selected on the validation set. Different colors distinguish the attack categories.

#### Relationship to recall and ROC performance

The attack-wise recall patterns and ROC characteristics correlate with these reconstruction-error distributions. Across datasets, normal traffic generally concentrates at low reconstruction errors below the threshold, while most attack types shift toward higher error values. Attack categories whose errors lie clearly above $\tau$ are detected reliably, supporting high recall. A clearer separation between normal and anomalous scores also supports stronger ROC performance.

Certain attack categories resemble normal behavior, leading to overlapping error distributions and reduced recall. An attack sample with a reconstruction error at or below $\tau$ is classified as normal and becomes a false negative. Thus, a category can have low recall even when the model detects most other attack categories successfully.

- **KDD99 — `back` attack (panel a).** The *back* distribution clusters near the normal region, with substantial density below the decision threshold. This overlap explains its lower recall compared with attack types whose reconstruction errors are concentrated above the threshold. The corresponding [KDD99 recall comparison](results/KDD99/KDD99_recall.png) shows this category-specific weakness alongside stronger detection of other attacks.
- **CSE-CIC-IDS2018 — `Infiltration` (panel c).** Many *Infiltration* samples have reconstruction errors in the benign region, consistent with feature patterns that closely resemble normal activities. Their substantial overlap with normal traffic places many samples below the threshold, reducing recall. This overlap is a major contributor to the lower AUC and reduced detection performance on CSE-CIC-IDS2018. Infiltration is also challenging across the approaches evaluated in the manuscript, as reflected in the [CSE-CIC-IDS2018 recall comparison](results/CSE-CIC-IDS2018/CSE-CIC-IDS2018_recall.png).
- **GeNIS — `recon-dns` (panel e).** The *recon-dns* errors lie in the low-error region occupied by normal traffic, below the selected threshold. All **10 test samples** have errors of approximately **0.00052–0.00110**, below $\tau \approx 0.00413$, and are classified as normal, giving **recall 0.0000**. The model reconstructs these DNS reconnaissance flows with small errors, consistent with a benign-like representation under the learned features. This is the same overlap-driven failure pattern observed for *back* and *Infiltration*. The other GeNIS attack categories are detected with high recall. Because *recon-dns* accounts for only 10 of the 4,373 attack samples, overall attack recall remains **0.9977** even though this category is missed entirely. The [GeNIS recall comparison](results/GeNIS/GeNIS_recall.png) and [per-attack summary](results/GeNIS/taae/GeNIS_summary.csv) show why aggregate scores should be interpreted alongside category-level results.

The two effects should be distinguished: recall depends on the selected threshold, whereas ROC-AUC summarizes score ranking across thresholds. Lowering $\tau$ may recover overlapping attack samples but also flags more normal traffic as anomalous; changing the threshold alone does not change the ROC-AUC. The overlapping distributions therefore explain both the missed attacks at the chosen operating point and the weaker score separation reflected in the ROC curve.

#### Combined ROC curves

The following figure compares TAAE score separation on KDD99, NSL-KDD, InSDN, CSE-CIC-IDS2018, UNSW-NB15 and GeNIS. Each curve is computed from the saved anomaly scores on that dataset's test split; higher scores indicate anomalous traffic. The diagonal denotes random ranking.

![ROC curves on six datasets](results/combined_roc_curves_6datasets.png)

| Dataset | ROC-AUC | Test normal | Test anomaly |
| --- | ---: | ---: | ---: |
| KDD99 | 0.9965 | 15,363 | 113,252 |
| NSL-KDD | 0.9855 | 12,289 | 60,127 |
| InSDN | 0.9866 | 13,685 | 275,465 |
| CSE-CIC-IDS2018 | 0.9217 | 40,000 | 200,000 |
| UNSW-NB15 | 0.9872 | 391,955 | 99,643 |
| GeNIS | 0.9904 | 4,373 | 4,373 |

The [ROC summary CSV](results/combined_roc_curves_6datasets.csv) links each curve to its saved score file. These are the archived test splits; their class counts differ from the current CLI defaults where the recorded protocol differs. ROC-AUC measures score ranking independently of the chosen decision threshold.

### Effect of client count on GeNIS

Measured configuration: **30 local epochs per client per round**, **10 server rounds**, previous TAAE with RNEP aggregation, latent dimension 64, batch size 16, learning rate `3e-5`, dropout 0.1 and seed 123. We evaluate **5, 10, 15, 20, 25, 30, 35, 40, 46 and 50 clients**.

The same ordered pool of **31,123 normal training rows** is partitioned using `numpy.array_split`. All runs share **7,630 validation rows** (3,815 normal and 3,815 attack) and **8,746 test rows** (4,373 normal and 4,373 attack), with identical frozen model/training sources and preprocessing. The [experiment configuration](results/client-study-30-epochs/experiment_config.json) records the source/data hashes; the [partition summary](results/client-study-30-epochs/partition_summary.csv) records the client sizes.

The table reports the final model after round 10. Each percentile is selected by **validation macro F1**, then applied to the held-out test set. **Precision, Recall and F1-score refer to Anomaly (1)**; Macro F1 averages the Normal and Anomaly F1 scores. Scores are rounded to four decimal places.

| Clients | Selected percentile | Accuracy | Precision | Recall | F1-score | Macro F1 | ROC-AUC |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 98.32 | 0.9835 | 0.9702 | 0.9977 | 0.9838 | 0.9835 | 0.9905 |
| 10 | 97.36 | 0.9752 | 0.9547 | 0.9977 | 0.9757 | 0.9752 | 0.9895 |
| 15 | 97.63 | 0.9790 | 0.9625 | 0.9968 | 0.9793 | 0.9790 | 0.9889 |
| 20 | 97.42 | 0.9802 | 0.9658 | 0.9957 | 0.9805 | 0.9802 | 0.9884 |
| 25 | 96.52 | 0.9761 | 0.9590 | 0.9947 | 0.9765 | 0.9761 | 0.9866 |
| 30 | 94.53 | 0.9559 | 0.9223 | 0.9957 | 0.9576 | 0.9558 | 0.9818 |
| 35 | 93.14 | 0.9455 | 0.9047 | 0.9959 | 0.9481 | 0.9453 | 0.9771 |
| 40 | 91.29 | 0.9152 | 0.8573 | 0.9961 | 0.9215 | 0.9146 | 0.9663 |
| 46 | 89.75 | 0.9078 | 0.8475 | 0.9947 | 0.9152 | 0.9071 | 0.9467 |
| 50 | 89.65 | 0.9036 | 0.8462 | 0.9865 | 0.9110 | 0.9029 | 0.9328 |

The [full comparison CSV](results/client-study-30-epochs/client_comparison.csv) includes the validation-selected results and the fixed-percentile-82 control. The figure compares both policies; the [round comparison](results/client-study-30-epochs/round_comparison.png) shows validation performance through round 10.

![GeNIS client-count comparison: 30 local epochs and 10 rounds](results/client-study-30-epochs/client_comparison.png)

#### Interpretation

- **5 clients performs best in this 30-epoch study.** Accuracy is **0.9835**, Macro F1 is **0.9835**, attack F1 is **0.9838** and ROC-AUC is **0.9905**. Its confusion matrix has **134 false positives and 10 false negatives**, with Normal recall **0.9694** and attack recall **0.9977**.
- **Larger client counts reduce score separation under this fixed training budget.** The 10–25-client accuracy ranges from **0.9752 to 0.9802**. At 30, 35, 40, 46 and 50 clients it falls to **0.9559, 0.9455, 0.9152, 0.9078 and 0.9036**, respectively. ROC-AUC falls to **0.9328** at 50 clients, so the deterioration involves score ranking as well as threshold calibration. Normal recall falls from **0.9694** at 5 clients to **0.8207** at 50, while attack recall remains **0.9865** at 50. False positives increase from **134 to 784**, and false negatives from **10 to 59**.
- **Client size and calibration both matter.** The training pool per client shrinks from **6,224–6,225 rows** at 5 clients to **622–623 rows** at 50. Each local epoch therefore has fewer optimizer steps per client, and RNEP mixes a different set of local models. These are possible contributors, not isolated causal findings. The validation-selected percentile shifts from **98.32 to 89.65**; even after this calibration, the highest client counts have lower performance. The fixed-percentile-82 control gives accuracy **0.8724–0.8938**, showing why the same threshold-selection policy must be used when comparing settings.
- **This is a single-seed result.** The measurements support 5 clients as the strongest setting among these counts for this split and 30-epoch/10-round budget. Repeated seeds and non-IID partitions are needed before generalizing that ordering.

Per-client weights, figures and completion checks are under `results/client-study-30-epochs/clients_<N>/`; the [study index](results/client-study-30-epochs/README.md) describes the artifacts and reproduction command. The separate [70-epoch archive](results/client-study-70-epochs/README.md) preserves the earlier measurements.

### FedSAD and GSAD confusion matrices

Confusion matrices of FedSAD and GSAD alone on 1-minute windows (Anomaly is the positive class). FedSAD uses `--threshold-std` 0.25 (CIC2018) and 5 (UNSW-NB15).

| Dataset | FedSAD (GSAD + TAAE) | GSAD |
| --- | --- | --- |
| CSE-CIC-IDS2018 | ![FedSAD CSE-CIC-IDS2018](results/CSE-CIC-IDS2018/fedsad/fedsad_cm.png) | ![GSAD CSE-CIC-IDS2018](results/CSE-CIC-IDS2018/gsad/gsad_cm.png) |
| UNSW-NB15 | ![FedSAD UNSW-NB15](results/UNSW-NB15/fedsad/fedsad_cm.png) | ![GSAD UNSW-NB15](results/UNSW-NB15/gsad/gsad_cm.png) |

### Pipeline latency

Per-window inference latency and throughput of the full FedSAD pipeline (1-minute detection windows).

<table>
  <thead>
    <tr><th rowspan="2">Metric</th><th colspan="2">CSE-CIC-IDS2018</th><th colspan="2">UNSW-NB15</th></tr>
    <tr><th>Normal</th><th>Anomaly</th><th>Normal</th><th>Anomaly</th></tr>
  </thead>
  <tbody>
    <tr><td>GSAD (ms)</td><td>0.201 ± 0.011</td><td>0.201 ± 0.011</td><td>0.214 ± 0.018</td><td>0.198 ± 0.014</td></tr>
    <tr><td>TAAE (ms)</td><td>103.580 ± 12.666</td><td>110.074 ± 15.748</td><td>115.166 ± 26.083</td><td>108.126 ± 9.924</td></tr>
    <tr><td>Throughput (win/sec)</td><td>9.59</td><td>8.99</td><td>7.88</td><td>9.14</td></tr>
  </tbody>
</table>

## Original study details

The following sections describe the paper's experiments and historical settings. New runs are identified by their own resolved configuration, data hashes and run ID.

### Experimental environment

All experiments were conducted on a server with the following specifications.

- CPU: Dual Intel Xeon Gold 5218R processors
- Memory: 503 GB RAM
- GPU: 3 × NVIDIA RTX A6000 GPUs (48 GB VRAM each)
- Operating System: Ubuntu 22.04.1 LTS
- CUDA Version: 12.2

For federated learning experiments, we implemented our training pipeline using the Flower framework (`flwr==1.19.0`) on Python 3.9.23. We extended and overrode the default aggregation strategy to integrate the RNEP-based entropy-aware pairing mechanism.

### Federated learning implementation

The proposed FedSAD framework is implemented using the Flower federated learning framework.

The default aggregation strategy was customized to support the proposed **RNEP-based entropy-aware pairing mechanism**. Instead of relying only on conventional centralized aggregation such as FedAvg, the proposed method considers entropy-based node information during federated aggregation.

The implementation includes:

- Federated client-side training
- RNEP-based entropy-aware node pairing
- Entropy-based aggregation logic
- Communication-efficient model synchronization
- Privacy-preserving distributed anomaly detection

### Datasets


We evaluate the proposed framework on six intrusion detection datasets: KDD99, NSL-KDD, InSDN, CSE-CIC-IDS2018, UNSW-NB15 and GeNIS. The sample pools below describe the prepared numeric benchmark CSVs and the selected TAAE pool across training, validation and testing. They are not final-test counts; the ROC table above reports the actual archived test sizes. Supervised models can use additional attack rows for training according to their loader protocol.

| Dataset | Normal Samples | Anomaly Samples | Purpose |
| :--- | :--- | :--- | :--- |
| KDD99 | 76,812 | 113,252 | Comparison with AE-LSTM-based anomaly detection models |
| NSL-KDD | 61,442 | 60,127 | Evaluation on an improved KDD-based IDS benchmark |
| InSDN | 68,424 | 275,465 | Benchmarking CNN-LSTM approaches in SDN environments |
| CSE-CIC-IDS2018 | 200,000 | 200,000 | Comparison with multimodal IDS frameworks |
| UNSW-NB15 | 1,959,772 | 99,643 | Evaluation on modern IDS traffic and comparison with multimodal frameworks |
| GeNIS | 39,311 | 8,188 | Evaluation on enterprise network scenarios with shared held-out flow groups |

#### Dataset descriptions

**KDD99** is a classic intrusion detection benchmark derived from the 1999 DARPA evaluation program. It contains connection-level flow records with attack categories such as DoS, Probe, Remote-to-Local (R2L), and User-to-Root (U2R). We use 76,812 normal samples and 113,252 anomaly samples.

**NSL-KDD** is an improved variant of KDD99 designed to alleviate the severe class imbalance and redundancy problems present in the original dataset. We use 61,442 normal samples and 60,127 anomaly samples.

**InSDN** is an intrusion detection dataset collected from an SDN environment. It contains both benign traffic and attack scenarios such as ARP spoofing and TCP SYN flooding. The prepared CSV pool contains 68,424 normal samples and 275,465 anomaly samples.

**CSE-CIC-IDS2018** is a large-scale dataset composed of realistic benign activities and diverse attack scenarios, including web attacks, brute force, botnets, and DDoS attacks. Given its large scale, we randomly sample 200,000 normal and 200,000 anomaly instances for our experiments.

**UNSW-NB15** is a modern IDS dataset generated using the IXIA PerfectStorm platform. It includes modern attack types such as Exploits, Reconnaissance, and Shellcode. We use 1,959,772 normal samples and 99,643 anomaly samples.

**GeNIS: GECAD Network Intrusion Scenarios** provides enterprise network traffic from simulated attack scenarios, including labeled flow CSVs at 5, 10, 30 and 60 second intervals. The official dataset is available from [Zenodo](https://zenodo.org/records/14919237); please also cite its [primary data article](https://doi.org/10.1016/j.dib.2025.111487).

We use the **10-second scenario CSVs**, with 39,311 normal rows across the training, validation and test partitions and 8,188 selected attack rows for validation and testing. Attack scenarios include denial of service, reconnaissance and brute-force activity.

### Dataset preprocessing

To bridge the gap between benchmark datasets and practical SDP deployment, packet-level datasets, including CSE-CIC-IDS2018 and UNSW-NB15, are processed through a data pipeline that extracts log-consistent features aligned with real SDP operational logs.

The preprocessing pipeline includes:

- Packet-level feature extraction
- Session-level feature construction
- Log-consistent representation generation
- Normalization and label transformation
- Train-test split according to the evaluation setting

Splits follow each dataset's preserved loader protocol rather than a single ratio for every benchmark:

| Dataset | Normal split in the current flow-model loaders | Attack usage and evaluation |
| --- | --- | --- |
| KDD99 | 50:50 train/test | TAAE trains on normal only; classifiers retain their model-specific attack-training fractions. |
| NSL-KDD | 50:50 train/test | TAAE trains on normal only; classifiers retain their model-specific attack-training fractions. |
| InSDN | 80:20 for TAAE/CNN-LSTM; 50:50 for AE-LSTM | Attack-training fractions depend on the model. |
| CSE-CIC-IDS2018 | 80:20 train/test | Normal and attack pools are sampled before applying the loader split. |
| UNSW-NB15 | 80:20 train/test | The model-specific feature layout and attack-training fractions are preserved. |
| GeNIS | Group-held-out partitions: 31,123 train, 3,815 validation and 4,373 test normal rows | Validation and test add 3,815 and 4,373 attack rows, respectively; all three flow models share those evaluation source rows. |

The archived figures use the split recorded with each saved model; changing a CLI default does not change an existing model's evaluation rows. New runs record split counts, source hashes and preprocessing details in their data report.

Numeric benchmark TAAE loaders fit MinMax scaling on normal training rows. Model-specific transforms and classifier training fractions are detailed in [Datasets and paths](docs/EXPERIMENTS.md#datasets-and-paths). For the GeNIS scenario flows, preparation removes records reused across files and groups related connections/windows before splitting, so groups do not overlap across train, validation and test. Signed measurements are preserved; identifiers, timestamps and exporter bookkeeping are excluded from model input. Median imputation, MinMax scaling and one-hot encoding are fitted on the training partition: normal training rows for TAAE, labeled training rows for AE-LSTM/CNN-LSTM.

The sequence classifiers pack each flow's features into a padded sequence rather than treating unrelated flows as a temporal sequence. Their shared held-out evaluation rows support comparison with TAAE while preserving their supervised training protocol. The [GeNIS preprocessing report](results/GeNIS/taae/preprocessing_report.json) records group-overlap checks, feature dimensions and attack types absent from training.

### CSE-CIC-IDS2018 sampling strategy

The original CSE-CIC-IDS2018 dataset contains 13,484,694 normal samples and 2,748,235 anomaly samples. Given its large scale, we randomly sample 200,000 normal and 200,000 anomaly instances for our experiments.

For anomaly samples, rare attack types are used as-is, while high-frequency attack types are proportionally downsampled to preserve attack diversity while preventing dominant classes from overwhelming the training process.

| Attack Type | Sampling Method | Sampled | Original |
| :--- | :--- | ---: | ---: |
| SQL Injection | Use As-Is | 87 | 87 |
| Brute Force-XSS |  | 230 | 230 |
| Brute Force-Web |  | 611 | 611 |
| DDoS Attack-LOIC-UDP |  | 1,730 | 1,730 |
| DoS Attack-Slowloris |  | 10,990 | 10,990 |
| DoS Attack-GoldenEye | | 41,508 | 41,508 |
| DoS Attack-SlowHTTPTest | Proportional Downsampling | 7,524 | 139,890 |
| Infiltration |  | 8,709 | 161,934 |
| Brute Force-SSH |  | 10,088 | 187,589 |
| Brute Force-FTP |  | 10,400 | 193,360 |
| Bot |  | 15,392 | 286,191 |
| DoS Attack-Hulk | | 24,844 | 461,912 |
| DDoS Attack-LOIC-HTTP |  | 30,990 | 576,191 |
| DDoS Attack-HOIC |  | 36,897 | 686,012 |

### Hyperparameter settings

The table below describes the original manuscript settings. New runs use the resolved values saved in their `config.json`.

| Parameters | CNN-LSTM | AE-LSTM | MM-FEWSHOTS-IDS | MV-IDS | Ours |
| :--- | :--- | :--- | :--- | :--- | :--- |
| No. of Classes | 2 | 2 | 2 | 2 | 2 |
| Batch Size | 8, 32 | 64, 128 | 32 | 128 | 8, 32 |
| Dropout Rate | 0.1, 0.25, 0.5 | 0.001 | 0.1 | 0.3 | 0.1 |
| Learning Rate | 0.0001 | 0.001 | 0.001 | 0.0001 | 0.0001 |
| Optimizer | Adam | Adam | Adam | Adam | Adam |
| Activation | ReLU, Softmax | ReLU, Sigmoid | ReLU, Softmax | ReLU, Softmax | LeakyReLU, ReLU, Sigmoid |
| Loss Function | BCE | MSE | CE | BCE | MSE, BCE |

### Reproducibility

Due to manuscript space limitations, detailed implementation and training procedures could not be fully included in the paper. This repository provides:

- Source code for the proposed FedSAD framework (GSAD, TAAE and the RNEP aggregation)
- The comparison baselines (AE-LSTM, CNN-LSTM, MV-IDS, MM-FewShot-IDS, FedAvg, centralized TAAE)
- Dataset preprocessing procedures and experimental configurations
- Hyperparameter settings and the server/software environment
- Evaluation scripts and the archived results, including the saved model weights

## License

This project is licensed under the [MIT License](LICENSE). Copyright (c) 2026 FedSAD contributors.

Third-party datasets and components retain their respective licenses and usage terms.

## Citation

If you use this repository in your research, please cite:

<!-- ```bibtex
@article{FedSAD,
  title={FedSAD: Federated Learning-based Anomaly Detection in Software-Defined Perimeter Architecture for Distributed Authentication},
  author={Anonymous},
  journal={Under Review},
  year={2026}
}
``` -->
