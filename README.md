# FedSAD: Federated Learning-based Anomaly Detection in Software-Defined Perimeter Architecture for Distributed Authentication

Official implementation of the **FedSAD** framework, designed for secure and distributed anomaly detection within Software-Defined Perimeter (SDP) architectures.

## Overview

FedSAD is a novel framework that integrates **Federated Learning (FL)** with **Anomaly Detection** to enhance security in SDP-based environments. It focuses on distributed authentication and identifying malicious activities while preserving data privacy across multiple nodes.

This repository provides the core logic for the FedSAD model, including comparison baselines, evaluation scripts, experimental results, dataset preprocessing details, and hyperparameter settings used in the paper.

## Project Structure

```text
.
├── models/                 # Baseline and comparison model definitions
├── scripts/                # Evaluation and result-saving scripts
├── results/                # Experimental results and performance metrics
├── utils/                  # Utility functions for data processing, training, and evaluation
├── fedsad.py               # Main implementation of the FedSAD framework
├── gsad.py                 # Graph-based Statistical Anomaly Detection module
├── rnep.py                 # RNEP-based federated learning aggregation module
├── taae.py                 # TAAE model implementation
├── requirements.txt        # Required Python packages
├── TAAE_ModelSummary.txt   # Detailed TAAE model architecture summary
└── taae_weights.tar.gz     # Saved TAAE model weights
```

## Environment and Dependencies

This project is developed and tested with **Python 3.9.23**. To ensure reproducibility, please use the specific versions of the core libraries listed below.

### Python Version

- **Python 3.9.23**

### Core Packages

| Package | Version | Description |
| :--- | :--- | :--- |
| **tensorflow** | 2.14.1 | Primary framework for TAAE model |
| **keras** | 2.14.0 | Neural network API used with TensorFlow |
| **flwr** | 1.19.0 | Federated Learning (Flower) orchestration |
| **numpy** | 1.25.2 | Numerical computing and array operations |
| **pandas** | 2.3.1 | Data manipulation and analysis |
| **scikit-learn** | 1.6.1 | Anomaly detection metrics and utilities |

Install dependencies using:

```bash
pip install -r requirements.txt
```

or manually:

```bash
pip install tensorflow==2.14.1 keras==2.14.0 flwr==1.19.0 scikit-learn==1.6.1 pandas==2.3.1 numpy==1.25.2
```

## Experimental Environment

All experiments were conducted on a server with the following specifications.

- CPU: Dual Intel Xeon Gold 5218R processors
- Memory: 503 GB RAM
- GPU: 3 × NVIDIA RTX A6000 GPUs (48 GB VRAM each)
- Operating System: Ubuntu 22.04.1 LTS
- CUDA Version: 12.2

For federated learning experiments, we implemented our training pipeline using the Flower framework (`flwr==1.19.0`) on Python 3.9.23. We extended and overrode the default aggregation strategy to integrate the RNEP-based entropy-aware pairing mechanism.

## Federated Learning Implementation

The proposed FedSAD framework is implemented using the Flower federated learning framework.

The default aggregation strategy was customized to support the proposed **RNEP-based entropy-aware pairing mechanism**. Instead of relying only on conventional centralized aggregation such as FedAvg, the proposed method considers entropy-based node information during federated aggregation.

The implementation includes:

- Federated client-side training
- RNEP-based entropy-aware node pairing
- Entropy-based aggregation logic
- Communication-efficient model synchronization
- Privacy-preserving distributed anomaly detection

## Datasets

To evaluate the proposed framework, we use five widely adopted intrusion detection datasets for fair comparison with existing IDS baselines.

| Dataset | Normal Samples | Anomaly Samples | Purpose |
| :--- | :--- | :--- | :--- |
| KDD99 | 76,812 | 113,252 | Comparison with AE-LSTM-based anomaly detection models |
| NSL-KDD | 61,442 | 90,848 | Evaluation on an improved KDD-based IDS benchmark |
| InSDN | 68,424 | 138,772 | Benchmarking CNN-LSTM approaches in SDN environments |
| CSE-CIC-IDS2018 | 200,000 | 200,000 | Comparison with multimodal IDS frameworks |
| UNSW-NB15 | 1,959,772 | 99,643 | Evaluation on modern IDS traffic and comparison with multimodal frameworks |

### Dataset Descriptions

**KDD99** is a classic intrusion detection benchmark derived from the 1999 DARPA evaluation program. It contains connection-level flow records with attack categories such as DoS, Probe, Remote-to-Local (R2L), and User-to-Root (U2R). We use 76,812 normal samples and 113,252 anomaly samples.

**NSL-KDD** is an improved variant of KDD99 designed to alleviate the severe class imbalance and redundancy problems present in the original dataset. We use 61,442 normal samples and 90,848 anomaly samples.

**InSDN** is an intrusion detection dataset collected from an SDN environment. It contains both benign traffic and attack scenarios such as ARP spoofing and TCP SYN flooding. We use 68,424 normal samples and 138,772 anomaly samples.

**CSE-CIC-IDS2018** is a large-scale dataset composed of realistic benign activities and diverse attack scenarios, including web attacks, brute force, botnets, and DDoS attacks. Given its large scale, we randomly sample 200,000 normal and 200,000 anomaly instances for our experiments.

**UNSW-NB15** is a modern IDS dataset generated using the IXIA PerfectStorm platform. It includes modern attack types such as Exploits, Reconnaissance, and Shellcode. We use 1,959,772 normal samples and 99,643 anomaly samples.

## Dataset Preprocessing

To bridge the gap between benchmark datasets and practical SDP deployment, packet-level datasets, including CSE-CIC-IDS2018 and UNSW-NB15, are processed through a data pipeline that extracts log-consistent features aligned with real SDP operational logs.

The preprocessing pipeline includes:

- Packet-level feature extraction
- Session-level feature construction
- Log-consistent representation generation
- Normalization and label transformation
- Train-test split according to the evaluation setting

Across all datasets, we apply different train-test split strategies depending on the evaluation purpose.

For the federated training comparison, including Centralized Baseline, FedAvg, and RNEP, we use a 50:50 split of normal samples for training and testing. In contrast, for the packet-level and unified model evaluation, we use an 80:20 train-test split.

## CSE-CIC-IDS2018 Sampling Strategy

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

## Hyperparameter Settings

The following hyperparameters were used for the compared models.

| Parameters | CNN-LSTM | AE-LSTM | MM-FEWSHOTS-IDS | MV-IDS | Ours |
| :--- | :--- | :--- | :--- | :--- | :--- |
| No. of Classes | 2 | 2 | 2 | 2 | 2 |
| Batch Size | 8, 32 | 64, 128 | 32 | 128 | 8, 32 |
| Dropout Rate | 0.1, 0.25, 0.5 | 0.001 | 0.1 | 0.3 | 0.1 |
| Learning Rate | 0.0001 | 0.001 | 0.001 | 0.0001 | 0.0001 |
| Optimizer | Adam | Adam | Adam | Adam | Adam |
| Activation | ReLU, Softmax | ReLU, Sigmoid | ReLU, Softmax | ReLU, Softmax | LeakyReLU, ReLU, Sigmoid |
| Loss Function | BCE | MSE | CE | BCE | MSE, BCE |

## Reproducibility

Due to manuscript space limitations, detailed implementation and training procedures could not be fully included in the paper. Therefore, this repository provides supplementary details to facilitate reproducibility.

This repository includes:

- Source code for the proposed FedSAD framework
- GSAD implementation
- TAAE implementation
- RNEP-based federated aggregation logic
- Dataset preprocessing procedures
- Experimental configurations
- Hyperparameter settings
- Server and software environment details
- Evaluation scripts and result-saving scripts
- Model architecture summary
- Saved TAAE model weights

The implementation corresponds to the experimental setup reported in the paper.

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
