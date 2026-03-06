# Update chart: tilt annotation labels to prevent overlap (e.g., rotate text by 45°)
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

# # KDD99
# data = """Attack_Type\tCNN-LSTM\tAE-LSTM\tTAAE (Ours)
# back\t0.99818\t0.9995\t0.233772
# buffer_overflow\t0.7\t0.4333\t1
# ftp_write\t0\t0\t1
# guess_passwd\t0.98113\t0.9811\t1
# imap\t0.16667\t0.6667\t1
# ipsweep\t0.94681\t0.9574\t1
# land\t0.04762\t0.0476\t1
# loadmodule\t0.33333\t0.1111\t1
# multihop\t0.57143\t0.1429\t1
# neptune\t0.99828\t0.9999\t1
# nmap\t1\t1\t1
# perl\t0\t0\t1
# phf\t0\t0\t1
# portsweep\t0.99038\t0.9865\t1
# rootkit\t0\t0\t1
# satan\t0.99153\t0.9922\t1
# spy\t0\t0\t1
# warezclient2\t0.05196\t0.0049\t0.9490
# warezmaster\t0.9\t0\t1
# """

# InSDN
# data = """Attack_Type\tCNN-LSTM\tAE-LSTM\tTAAE (Ours)
# BFA\t0.1039\t0.2078\t0.9992
# BOTNET\t0.4939\t0.6829\t1.000
# DDoS\t0.9954\t0.6060\t1.0000
# DoS\t0.0563\t0.0000\t0.9997
# Probe\t0.4122\t0.1780\t1.000
# U2R\t0.0588\t0.0588\t0.9411
# Web-Attack\t0.0000\t0.0000\t1.000
# """

# # UNSW-NB15
# data = """Attack_Type\tCNN-LSTM\tAE-LSTM\tTAAE (Ours)
# analysis\t0.98626\t0.84840\t0.99863
# backdoor\t0.99042\t0.85780\t0.99294
# dos\t0.98217\t0.90060\t0.99347
# exploits\t0.99159\t0.93550\t0.99696
# fuzzers\t0.82776\t0.37570\t0.99936
# generic\t0.76129\t0.31140\t0.99945
# reconnaissance\t0.97148\t0.91960\t0.99895
# shellcode\t0.99735\t0.96820\t1.00000
# worms\t1.00000\t0.98250\t1.00000
# """

# CSE-CIC-IDS1028
data = """Attack_Type\tCNN-LSTM\tAE-LSTM\tTAAE (Ours)
DDOS attack-HOIC\t0.95287\t0.75012\t1.0000
DDoS attacks-LOIC-HTTP\t0.44537\t0.96531\t0.4647
DoS attacks-Hulk\t0.97895\t0.44334\t1.0000
Bot\t0.99812\t0.99674\t1.0000
FTP-BruteForce\t1.00000\t0.99844\t1.0000
SSH-Bruteforce\t1.0000\t1.00000\t1.00000
Infiltration\t0.19290\t0.49891\t0.3568
DoS attacks-SlowHTTPTest\t1.00000\t0.03674\t1.0000
DoS attacks-GoldenEye\t0.62116\t1.00000\t1.0000
DoS attacks-Slowloris\t0.88963\t0.66072\t1.0000
DDOS attack-LOIC-UDP\t0.98844\t0.90419\t1.0000
Brute Force -Web\t0.02455\t0.02081\t0.6727
Brute Force -XSS\t0.00000\t0.00491\t0.5000
SQL Injection\t0.02299\t0.00000\t0.6092
"""

# # CSE-CIC-IDS2018 (vs. Multimodals)
# data = """Attack_Type\tMM-FEWSHOTS-IDS\tMV-IDS\tFedSAD
# DDOS attack-HOIC\t1.0000\t1.0000\t1.0000
# DDoS attacks-LOIC-HTTP\t0.9333\t0.9997\t0.7435
# DoS attacks-Hulk\t0.4000\t1.0000\t1.0000
# Bot\t1.0000\t1.0000\t1.0000
# FTP-BruteForce\t1.0000\t1.00000\t1.0000
# SSH-Bruteforce\t1.00000\t0.9914\t1.0000
# Infiltration\t0.3000\t1.0000\t0.0104
# DoS attacks-SlowHTTPTest\t1.0000\t1.00000\t1.0000
# DoS attacks-GoldenEye\t0.8333\t1.00000\t1.0000
# DoS attacks-Slowloris\t0.9000\t1.0000\t1.0000
# DDOS attack-LOIC-UDP\t0.9667\t1.0000\t1.0000
# Brute Force -Web\t0.9667\t0.9262\t0.8120
# Brute Force -XSS\t1.0000\t0.9348\t0.9351
# SQL Injection\t1.0000\t0.5882\t0.6666
# """

# # UNSW-NB15 (vs. Multimodals)
# data = """Attack_Type\tMM-FEWSHOTS-IDS\tMV-IDS\tFedSAD
# analysis\t1.0000\t0.8421\t1.0000
# backdoor\t1.0000\t0.9924\t1.0000
# dos\t0.9667\t0.9841\t1.0000
# exploits\t1.0000\t0.9784\t1.0000
# fuzzers\t0.9667\t0.5639\t1.0000
# generic\t1.0000\t0.9927\t1.0000
# reconnaissance\t1.0000\t0.9674\t1.0000
# shellcode\t1.0000\t0.9801\t1.00000
# worms\t1.00000\t1.0000\t1.00000
# """

# # CSE-CIC-IDS2018 (GSAD vs TAAE vs FedSAD)
# data = """Attack_Type\tGSAD\tTAAE\tFedSAD
# DDOS attack-HOIC\t1.0000\t1.0000\t1.0000
# DDoS attacks-LOIC-HTTP\t1.0000\t0.4647\t0.7435
# DoS attacks-Hulk\t1.0000\t1.0000\t1.0000
# Bot\t0.9357\t1.0000\t1.0000
# FTP-BruteForce\t1.0000\t1.00000\t1.0000
# SSH-Bruteforce\t1.0000\t1.00000\t1.0000
# Infiltration\t0.0069\t0.3568\t0.0104
# DoS attacks-SlowHTTPTest\t1.0000\t1.00000\t1.0000
# DoS attacks-GoldenEye\t1.0000\t1.0000\t1.00000
# DoS attacks-Slowloris\t1.0000\t1.0000\t1.0000
# DDOS attack-LOIC-UDP\t1.0000\t1.0000\t1.0000
# # Brute Force -Web\t0.6015\t0.6727\t0.8120
# Brute Force -XSS\t0.9722\t0.3008\t0.9351
# SQL Injection\t1.0000\t0.6092\t0.6666
# """

# # UNSW-NB15 (GSAD vs TAAE vs FedSAD)
# data = """Attack_Type\tGSAD\tTAAE\tFedSAD
# analysis\t1.0000\t0.9986\t1.0000
# backdoor\t1.0000\t0.9929\t1.0000
# dos\t1.0000\t0.9934\t1.0000
# exploits\t1.0000\t0.9969\t1.0000
# fuzzers\t0.9667\t0.9993\t1.0000
# generic\t1.0000\t0.9994\t1.0000
# reconnaissance\t1.0000\t0.9989\t1.0000
# shellcode\t1.00000\t1.0000\t1.0000
# worms\t1.00000\t1.00000\t1.0000
# """



df = pd.read_csv(StringIO(data), sep="\t")

labels = df['Attack_Type'].tolist()
x = range(len(labels))
width = 0.25

fig, ax = plt.subplots(figsize=(18, 6))

# bars1 = ax.bar([i - width for i in x], df['GSAD'], width, label='GSAD')
# bars2 = ax.bar(x, df['TAAE'], width, label='TAAE')
# bars3 = ax.bar([i + width for i in x], df['FedSAD'], width, label='FedSAD')

# bars1 = ax.bar([i - width for i in x], df['MM-FEWSHOTS-IDS'], width, label='MM-FEWSHOTS-IDS')
# bars2 = ax.bar(x, df['MV-IDS'], width, label='MV-IDSS')
# bars3 = ax.bar([i + width for i in x], df['FedSAD'], width, label='FedSAD')

bars1 = ax.bar([i - width for i in x], df['CNN-LSTM'], width, label='CNN-LSTM')
bars2 = ax.bar(x, df['AE-LSTM'], width, label='AE-LSTM')
bars3 = ax.bar([i + width for i in x], df['TAAE (Ours)'], width, label='TAAE (Ours)')

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
ax.set_title('Recall per Attack Type (CSE-CIC-IDS2018)', pad=20)
# ax.set_title('Recall per Attack Type (UNSW-NB15)', pad=20)
ax.set_ylim(0, 1.15)
ax.legend(loc='upper right')
# ax.legend()

plt.tight_layout(rect=[0, 0, 1, 0.95])
out_path = "./recall_grouped_bar_rotated_values.png"
plt.savefig(out_path, dpi=200)
plt.show()

out_path
