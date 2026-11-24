GeNIS **5/10/15/20/25/30-client runs are complete.**

| Clients | Percentile | Accuracy | Precision | Recall | F1-score | Macro F1 | ROC-AUC |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 97.14 | 0.9601 | 0.9286 | 0.9968 | 0.9615 | 0.9600 | 0.9888 |
| 10 | 97.98 | 0.9792 | 0.9629 | 0.9968 | 0.9796 | 0.9792 | 0.9900 |
| 15 | 98.15 | 0.9820 | 0.9674 | 0.9977 | 0.9823 | 0.9820 | 0.9904 |
| 20 | 97.81 | 0.9803 | 0.9642 | 0.9977 | 0.9807 | 0.9803 | 0.9901 |
| 25 | 97.82 | 0.9793 | 0.9645 | 0.9952 | 0.9796 | 0.9793 | 0.9894 |
| 30 | 97.69 | 0.9790 | 0.9621 | 0.9973 | 0.9793 | 0.9790 | 0.9890 |

All runs use 70 local epochs, 10 server rounds, seed 123 and the same 8,746 test flows (4,373 normal / 4,373 attack). Thresholds are selected by validation macro F1; precision, recall and F1-score are for the attack (Anomaly) class.
