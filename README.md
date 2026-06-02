# FedSAD: Federated Learning-based Anomaly Detection in Software-Defined Perimeter Architecture for Distributed Authentication

Official implementation of the **FedSAD** framework, designed for secure and distributed anomaly detection within Software-Defined Perimeter (SDP) architectures.

## 🚀 Overview
FedSAD is a novel framework that integrates **Federated Learning (FL)** with **Anomaly Detection** to enhance security in SDP-based environments. It focuses on distributed authentication and identifying malicious activities while preserving data privacy across multiple nodes.

This repository provides the core logic for the FedSAD model, including comparison baselines, evaluation scripts, and experimental results on various network security datasets.

## 📂 Project Structure
```
.
├── models/             # Baseline and comparison model definitions
├── scripts/            # Evaluation and result-saving scripts
├── results/            # Experimental results and performance metrics
├── fedsad.py           # Main implementation of the FedSAD framework
├── gsad.py             # Graph-based Statistical Anomaly Detection module
├── rnep.py             # Specific model architecture (TAAE-RNEP)
├── utils.py            # Utility functions for data processing and logging
└── FedSAD_ModelSummary.txt # Detailed model architecture summary
```

## Environment & Dependencies

This project is developed and tested with **Python 3.9.23**. To ensure reproducibility, please use the specific versions of the core libraries listed below.

### Python Version
- **Python 3.9.23**

### Core Packages
| Package | Version | Description |
| :--- | :--- | :--- |
| **tensorflow** | 2.14.1 | Primary framework for TAAE-RNEP model |
| **flwr** | 1.19.0 | Federated Learning (Flower) orchestration |
| **numpy** | 1.25.2 | Numerical computing and array operations |
| **pandas** | 2.3.1 | Data manipulation and analysis |
| **scikit-learn** | 1.6.1 | Anomaly detection metrics and utilities |

```
pip install tensorflow==2.14.1 flwr==1.19.0 scikit-learn==1.6.1 pandas==2.3.1 numpy==1.25.2
```
