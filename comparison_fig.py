# Update chart: tilt annotation labels to prevent overlap (e.g., rotate text by 45°)
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

# KDD99
# data = """Attack_Type\tOurs\tCNN_LSTM\tAE_LSTM
# back\t0.233772\t0.99818\t0.9995
# buffer_overflow\t1\t0.7\t0.4333
# ftp_write\t1\t0\t0
# guess_passwd\t1\t0.98113\t0.9811
# imap\t1\t0.16667\t0.6667
# ipsweep\t1\t0.94681\t0.9574
# land\t1\t0.04762\t0.0476
# loadmodule\t1\t0.33333\t0.1111
# multihop\t1\t0.57143\t0.1429
# neptune\t1\t0.99828\t0.9999
# nmap\t1\t1\t1
# perl\t1\t0\t0
# phf\t1\t0\t0
# portsweep\t1\t0.99038\t0.9865
# rootkit\t1\t0\t0
# satan\t1\t0.99153\t0.9922
# spy\t1\t0\t0
# warezclient\t0.94902\t0.05196\t0.0049
# warezmaster\t1\t0.9\t0
# merged\t0.984636\t0.9889\t0.9898
# """

# InSDN
data = """Attack_Type\tOurs\tCNN_LSTM\tAE_LSTM
BFA\t0.9751\t0.1039\t0.2078
BOTNET\t0.5000\t0.4939\t0.6829
DDoS\t1.0000\t0.9954\t0.6060
DoS\t0.7742\t0.0563\t0.0000
Probe\t0.9691\t0.4122\t0.1780
U2R\t0.9412\t0.0588\t0.0588
Web-Attack\t0.9583\t0.0000\t0.0000
merged\t0.9446\t0.5993\t0.3331
"""

df = pd.read_csv(StringIO(data), sep="\t")

labels = df['Attack_Type'].tolist()
x = range(len(labels))
width = 0.25

fig, ax = plt.subplots(figsize=(18, 6))

bars1 = ax.bar([i - width for i in x], df['Ours'], width, label='Ours')
bars2 = ax.bar(x, df['CNN_LSTM'], width, label='CNN_LSTM')
bars3 = ax.bar([i + width for i in x], df['AE_LSTM'], width, label='AE_LSTM')

def fmt(val):
    s = f"{val:.4f}"
    return s.rstrip("0").rstrip(".")

# Place rotated annotation labels
for bars in (bars1, bars2, bars3):
    for bar in bars:
        height = bar.get_height()
        if abs(height - 1.0) > 1e-12:
            ax.annotate(fmt(height),
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8,
                        rotation=45)

ax.set_xticks(list(x))
ax.set_xticklabels(labels, rotation=30, ha='right')

ax.set_ylabel('Recall')
ax.set_title('Recall per Attack Type (InSDN)', pad=20)
ax.set_ylim(0, 1.15)
ax.legend()

plt.tight_layout(rect=[0, 0, 1, 0.95])
out_path = "./recall_grouped_bar_rotated_values.png"
plt.savefig(out_path, dpi=200)
plt.show()

out_path
