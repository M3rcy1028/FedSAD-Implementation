import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, roc_curve, roc_auc_score
import pandas as pd
import seaborn as sns
import time
import numpy as np
# For evaluation
def plt_confusion_matrix(y_test, y_pred, 
                    save_path, labels=['Normal', 'Anomaly'], 
                    title="Confusion Matrix"):
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    cm_table = pd.DataFrame(
        [[tn, fp], [fn, tp]],
        index=['Actual Normal', 'Actual Anomaly'],
        columns=['Predicted Normal', 'Predicted Anomaly']
    )
    print("\n🧾 [Confusion Matrix]")
    print(cm_table)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=[f'Predicted {l}' for l in labels],
                yticklabels=[f'Actual {l}' for l in labels])
    plt.title(title)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_roc_curve(y_true, scores, roc_path="./roc_curve.png", title="ROC Curve"):
    """
    y_true: true labels (0=Normal, 1=Anomaly)
    scores: continuous model scores (e.g. reconstruction error)
    roc_path: where to save the ROC curve
    """
    fpr, tpr, _ = roc_curve(y_true, scores)
    auc_score = roc_auc_score(y_true, scores)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(roc_path, dpi=300)
    plt.close()

    print(f"ROC curve saved to {roc_path}")
    return auc_score

# Save FL history
def save_and_plot_history(history, csv_path, png_path):
    """
    Save and plot Flower simulation history (centralized loss).
    """
    # Centralized loss history
    rounds, losses = zip(*history.losses_centralized)

    # Save CSV
    hist_df = pd.DataFrame({"round": rounds, "loss": losses})
    hist_df.to_csv(csv_path, index=False)
    print(f"\nTraining history saved to {csv_path}")

    # Save plot
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, losses, marker="o", label="Centralized Loss")
    plt.title("Centralized Loss Over Rounds")
    plt.xlabel("Federated Rounds")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(png_path)
    plt.close()
    print(f"Loss plot saved to {png_path}")

def save_report(y_test,y_pred, result_path,labels=['Normal', 'Anomaly'], 
                    title="Classification Report"):
    server_report = classification_report(
        y_test, y_pred,
        target_names=labels,
        zero_division=0,
        digits=4 
    )

    print("\n📊 [Server Classification Report]\n")
    print(server_report)

    with open(result_path, "a") as f:
        f.write(f"\n📊 [{title}]\n")
        f.write(server_report + "\n")

class TimeRegistry:
    def __init__(self):
        self.storage = {}

    def add(self, name, elapsed):
        if name not in self.storage:
            self.storage[name] = []
        self.storage[name].append(elapsed)

    def summary(self):
        result = {}
        for k, v in self.storage.items():
            result[k] = {
                "mean": np.mean(v) * 1000,
                "std": np.std(v) * 1000
            }
        return result

    def reset(self):
        self.storage.clear()


GLOBAL_TIMER = TimeRegistry()

def timed_step(name, registry=GLOBAL_TIMER):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            registry.add(name, elapsed)
            return result
        return wrapper
    return decorator

def print_latency(label, registry=GLOBAL_TIMER):
    stats = registry.summary()

    print(f"\n[{label}] Latency")
    for k, v in stats.items():
        print(f"{k}: {v['mean']:.4f} ± {v['std']:.4f} ms")

    total_mean = sum(v["mean"] for v in stats.values())
    print(f"Total: {total_mean:.4f} ms")