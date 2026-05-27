# attack type label mapping
LABEL_MAPPING = {
	1: "DDOS attack-HOIC",
	2: "DDoS attacks-LOIC-HTTP",
	3: "DoS attacks-Hulk",
	4: "Bot",
	5: "FTP-BruteForce",
	6: "SSH-Bruteforce",
	7: "Infilteration",
	8: "DoS attacks-SlowHTTPTest",
	9: "DoS attacks-GoldenEye",
	10: "DoS attacks-Slowloris",
	11: "DDOS attack-LOIC-UDP",
	12: "Brute Force -Web",
	13: "Brute Force -XSS",
	14: "SQL Injection"
}


import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ensemble_dempster_shafer import (
	create_graph_from_window, extract_graph_features,
	graph_model_predict, rnep_model_predict, combine_dempster_shafer, _sanitize_numeric
)

# visualization options
USE_LOG_SCALE = True   # if True, use log scale on y-axis for better visibility of attack distribution
USE_DENSITY = False    # if True, compare using probability density instead of counts
TIME_WINDOW = "1T"

# extract normal window T, label from pkl
PKL_PATH = "run/ensemble_results_20251112_025740.pkl"  # replace with actual file name
with open(PKL_PATH, "rb") as f:
	results = pickle.load(f)
data = []
for r in results:
	m_a = r["m_comb"]["anomaly"]
	m_u = r["m_comb"]["uncertain"]
	T = m_a + 0.5 * m_u
	label = 0 if r["label"] == "Normal" else 1
	data.append({"T": T, "label": label})
df_normal = pd.DataFrame([d for d in data if d["label"] == 0])

# compute percentile from normal windows
normal_T = df_normal["T"]
q75 = np.percentile(normal_T, 95)
q85 = np.percentile(normal_T, 96)
q95 = np.percentile(normal_T, 97)

# test data file list by attack type
RNEP_DATA_DIR = "/data/SDP_Dataset/Unified_model/cic_rnep"
GRAPH_DATA_DIR = "/data/SDP_Dataset/Unified_model/cic_graph"
graph_files = [f for f in os.listdir(GRAPH_DATA_DIR) if f.startswith("CIC_anomaly_ae_") and f.endswith(".csv")]
rnep_files = [f for f in os.listdir(RNEP_DATA_DIR) if f.startswith("CIC_anomaly_ae_") and f.endswith(".csv")]

# Graph anomaly: add attack type tag per file, then concat, sort, and normalize
graph_dfs = []
for gf in graph_files:
	attack_num = int(gf.replace("CIC_anomaly_ae_", "").replace(".csv", ""))
	attack_name = LABEL_MAPPING.get(attack_num, f"Type {attack_num}")
	df_tmp = pd.read_csv(os.path.join(GRAPH_DATA_DIR, gf))
	df_tmp["Timestamp"] = pd.to_datetime(df_tmp["Timestamp"], dayfirst=True, errors="coerce")
	df_tmp = df_tmp[df_tmp["Timestamp"].notna()].copy()
	df_tmp["attack_type"] = attack_name
	graph_dfs.append(df_tmp)
df_graph_anomaly = pd.concat(graph_dfs, ignore_index=True)
df_graph_anomaly = _sanitize_numeric(df_graph_anomaly, exclude=["Timestamp", "attack_type"])
df_graph_anomaly.set_index("Timestamp", inplace=True)
df_graph_anomaly.sort_index(inplace=True)

# RNEP anomaly: concat into single sequence
df_rnep_anomaly_list = []
for rf in rnep_files:
	df_tmp = pd.read_csv(os.path.join(RNEP_DATA_DIR, rf))
	df_rnep_anomaly_list.append(df_tmp)
df_rnep_anomaly = pd.concat(df_rnep_anomaly_list, ignore_index=True)
df_rnep_anomaly.reset_index(drop=True, inplace=True)

# compute T by window matching all attack types at once (preserving attack_type tag)
anomaly_data = []
gb_graph = df_graph_anomaly.groupby(pd.Grouper(freq=TIME_WINDOW))
attack_types = df_graph_anomaly["attack_type"]
rnep_idx = 0
total_rnep_rows = len(df_rnep_anomaly)
for window_time, df_graph_window in gb_graph:
	if df_graph_window.empty:
		continue
	graph_window_size = len(df_graph_window)
	if rnep_idx + graph_window_size <= total_rnep_rows:
		df_rnep_window = df_rnep_anomaly.iloc[rnep_idx: rnep_idx + graph_window_size].copy()
		rnep_idx += graph_window_size
	else:
		remaining = (rnep_idx + graph_window_size) - total_rnep_rows
		df_rnep_part1 = df_rnep_anomaly.iloc[rnep_idx:].copy()
		df_rnep_part2 = df_rnep_anomaly.iloc[:remaining].copy()
		df_rnep_window = pd.concat([df_rnep_part1, df_rnep_part2], ignore_index=True)
		rnep_idx = remaining
	if df_rnep_window.empty:
		continue
	G = create_graph_from_window(df_graph_window)
	feat = extract_graph_features(G)
	m_graph = graph_model_predict(feat)
	m_rnep = rnep_model_predict(df_rnep_window)
	m_comb = combine_dempster_shafer(m_graph, m_rnep)
	T = m_comb["anomaly"] + 0.5 * m_comb["uncertain"]
	# use the most frequent attack_type within the window
	attack_label = df_graph_window["attack_type"].mode()[0] if "attack_type" in df_graph_window.columns else "Unknown"
	anomaly_data.append({"T": T, "label": 1, "attack_type": attack_label})

df_anomaly_all = pd.DataFrame(anomaly_data)
if df_anomaly_all.empty:
	print("No attack data to visualize.")
	exit()

# store distribution by attack type
attack_distributions = []
for attack_name, df_group in df_anomaly_all.groupby("attack_type"):
	df = pd.concat([df_normal, df_group[["T", "label"]]], ignore_index=True)
	attack_distributions.append({
		"name": attack_name,
		"anomaly_T": df_group["T"].to_numpy()
	})

if not attack_distributions:
	print("No attack data to visualize.")
	exit()

# overlay all attack types on a single graph
all_T_values = list(df_normal["T"])
for attack in attack_distributions:
	all_T_values.extend(attack["anomaly_T"])
bins = np.linspace(min(all_T_values), max(all_T_values), 41)

# compute normal histogram once and reuse
normal_T = df_normal["T"].to_numpy()
normal_counts, normal_bins = np.histogram(normal_T, bins=bins, density=USE_DENSITY)
normal_widths = np.diff(normal_bins)
normal_bin_lefts = normal_bins[:-1]

fig, ax = plt.subplots(figsize=(12, 7))
ax.bar(normal_bin_lefts, normal_counts, width=normal_widths, alpha=0.6, label="Normal", color="skyblue", align="edge")

attack_colors = plt.cm.tab20(np.linspace(0, 1, len(attack_distributions)))
for idx, attack in enumerate(attack_distributions):
	ax.hist(
		attack["anomaly_T"],
		bins=bins,
		alpha=0.35,
		label=attack["name"],
		color=attack_colors[idx],
		histtype="stepfilled",
		density=USE_DENSITY,
	)

ax.axvline(q75, color="blue", linestyle="--", label="Normal 95%")
ax.axvline(q85, color="green", linestyle="--", label="Normal 96%")
ax.axvline(q95, color="red", linestyle="--", label="Normal 97%")
ax.set_xlabel("Policy Threshold (T)",
			  fontsize="15")
ax.set_ylabel("Window Count",
			  fontsize="15")
ax.tick_params(axis='both', labelsize=15)
ax.set_title("Policy Threshold Distribution by Attack Type",
			 fontsize=20)
if USE_LOG_SCALE:
	ax.set_yscale("log")

# legend: Normal + all attack types + threshold baselines
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

legend_handles = [Patch(facecolor="skyblue", edgecolor="skyblue", alpha=0.6, label="Normal")]
def convert_attack_label(name):
	# keep as-is if 'attacks' is present; replace 'attack' with 'attacks' otherwise
	if "attacks" in name:
		return name
	return name.replace("attack", "attacks")

legend_handles.extend([
	Patch(
		facecolor=attack_colors[i],
		edgecolor=attack_colors[i],
		alpha=0.35,
		label=convert_attack_label(attack_distributions[i]["name"])
	)
	for i in range(len(attack_distributions))
])
legend_handles.extend([
	Line2D([0], [0], color="blue", linestyle="--", label="Normal 95%"),
	Line2D([0], [0], color="green", linestyle="--", label="Normal 96%"),
	Line2D([0], [0], color="red", linestyle="--", label="Normal 97%"),
])
ax.legend(handles=legend_handles, 
		  loc="upper right", 
		  bbox_to_anchor=(1.35, 1), 
		  fontsize=13)
fig.tight_layout()

img_path = "results/CSE-CIC-IDS2018/policy_threshold_distribution_all_attacks.png"
plt.savefig(img_path, bbox_inches="tight")
plt.close()
print(f"Image saved: {img_path}")
