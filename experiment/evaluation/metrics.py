"""One score direction and explicit per-class metrics for every binary model."""
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


def report(labels, scores, threshold, inclusive=False):
    labels, scores = np.asarray(labels, dtype=int), np.asarray(scores).reshape(-1)
    if labels.shape != scores.shape or not np.isfinite(scores).all():
        raise ValueError("Scores and labels must be aligned finite vectors")
    predicted = ((scores >= threshold) if inclusive else (scores > threshold)).astype(int)
    result = {"threshold": float(threshold), "positive_class": "Anomaly (1)",
              "prediction_rule": "score >= threshold" if inclusive else "score > threshold",
              "classification_report": classification_report(labels, predicted, labels=[0, 1],
                    target_names=["Normal", "Anomaly"], output_dict=True, zero_division=0),
              "confusion_matrix": confusion_matrix(labels, predicted, labels=[0, 1]).tolist(),
              "roc_auc": float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else None}
    return result


def score(model, values, kind):
    result = model.predict(values, batch_size=512, verbose=0)
    if kind == "taae":
        errors = np.mean(np.square(values - result), axis=tuple(range(1, values.ndim)))
        return errors, errors
    if kind == "aelstm":
        return result["pred"].reshape(-1), np.mean(np.square(values - result["decoded"]), axis=(1, 2))
    return np.asarray(result).reshape(-1), None


def choose_threshold(config, train_errors, val_scores, y_val):
    percentile = config["percentile"]
    if config["model"] != "taae":
        return .5, {"policy": "classifier_probability", "threshold": .5}
    policy = config["threshold_policy"]
    independent = config.get("independent_validation", False) and len(np.unique(y_val)) == 2
    if policy == "validation" and not independent:
        raise ValueError("Validation threshold selection needs independent validation with both classes")
    if policy == "validation" or (policy == "auto" and independent):
        from experiment.backends import load
        sweep = load("scripts.sweep_rnep_percentiles")
        rows = sweep.threshold_metrics(train_errors, val_scores, y_val,
                                       np.round(np.linspace(0, 100, 10001), 2))
        best = sweep.best_row(rows, "macro_f1")
        return float(best["threshold"]), {"policy": "validation_macro_f1", "selection_set": "validation", **best}
    threshold = float(np.percentile(train_errors, percentile))
    return threshold, {"policy": "fixed_normal_training_percentile", "percentile": percentile,
                       "threshold": threshold, "selection_set": "normal training errors"}
