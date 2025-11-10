# Update chart: tilt annotation labels to prevent overlap (e.g., rotate text by 45°)
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

# KDD99
data = """Attack_Type\tOurs\tCNN_LSTM\tAE_LSTM
back\t0.233772\t0.99818\t0.9995
buffer_overflow\t1\t0.7\t0.4333
ftp_write\t1\t0\t0
guess_passwd\t1\t0.98113\t0.9811
imap\t1\t0.16667\t0.6667
ipsweep\t1\t0.94681\t0.9574
land\t1\t0.04762\t0.0476
loadmodule\t1\t0.33333\t0.1111
multihop\t1\t0.57143\t0.1429
neptune\t1\t0.99828\t0.9999
nmap\t1\t1\t1
perl\t1\t0\t0
phf\t1\t0\t0
portsweep\t1\t0.99038\t0.9865
rootkit\t1\t0\t0
satan\t1\t0.99153\t0.9922
spy\t1\t0\t0
warezclient\t0.94902\t0.05196\t0.0049
warezmaster\t1\t0.9\t0
"""

# InSDN
# data = """Attack_Type\tOurs\tCNN_LSTM\tAE_LSTM
# BFA\t0.9992\t0.1039\t0.2078
# BOTNET\t1.000\t0.4939\t0.6829
# DDoS\t1.0000\t0.9954\t0.6060
# DoS\t0.9997\t0.0563\t0.0000
# Probe\t1.000\t0.4122\t0.1780
# U2R\t0.9411\t0.0588\t0.0588
# Web-Attack\t1.000\t0.0000\t0.0000
# """

# UNSW-NB15
# data = """Attack_Type\tOurs\tCNN_LSTM\tAE_LSTM
# analysis\t0.99863\t0.98626\t0.84840
# backdoor\t0.99294\t0.99042\t0.85780
# dos\t0.99347\t0.98217\t0.90060
# exploits\t0.99696\t0.99159\t0.93550
# fuzzers\t0.99936\t0.82776\t0.37570
# generic\t0.99945\t0.76129\t0.31140
# reconnaissance\t0.99895\t0.97148\t0.91960
# shellcode\t1.00000\t0.99735\t0.96820
# worms\t1.00000\t1.00000\t0.98250
# """

# # CSE-CIC-IDS1028
# data = """Attack_Type\tOurs\tCNN_LSTM\tAE_LSTM
# DDOS attack-HOIC\t1.0000\t0.95287\t0.75012
# DDoS attacks-LOIC-HTTP\t0.4647\t0.44537\t0.96531
# DoS attacks-Hulk\t1.0000\t0.97895\t0.44334
# Bot\t1.0000\t0.99812\t0.99674
# FTP-BruteForce\t1.0000\t1.00000\t0.99844
# SSH-Bruteforce\t1.0000\t1.00000\t1.00000
# Infiltration\t0.3568\t0.19290\t0.49891
# DoS attacks-SlowHTTPTest\t1.0000\t1.00000\t0.03674
# DoS attacks-GoldenEye\t1.0000\t0.62116\t1.00000
# DoS attacks-Slowloris\t1.0000\t0.88963\t0.66072
# DDOS attack-LOIC-UDP\t1.0000\t0.98844\t0.90419
# Brute Force -Web\t0.6727\t0.02455\t0.02081
# Brute Force -XSS\t0.5000\t0.00000\t0.00491
# SQL Injection\t0.6092\t0.02299\t0.00000
# """

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
ax.set_title('Recall per Attack Type (KDD99)', pad=20)
ax.set_ylim(0, 1.15)
ax.legend(loc='center right')
# ax.legend()

plt.tight_layout(rect=[0, 0, 1, 0.95])
out_path = "./recall_grouped_bar_rotated_values.png"
plt.savefig(out_path, dpi=200)
plt.show()

out_path
