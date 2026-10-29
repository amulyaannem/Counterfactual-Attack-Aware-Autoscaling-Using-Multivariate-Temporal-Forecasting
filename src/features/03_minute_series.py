"""Phase 3: minute-level aggregate series for Dataset A + overview plot."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"

path = PROC / "flows_clean.parquet"
df = pd.read_parquet(path) if path.exists() else pd.read_csv(PROC / "flows_clean.csv", parse_dates=["ts"])


def entropy(s):
    p = s.value_counts(normalize=True).to_numpy()
    return float(-(p * np.log2(p)).sum())


g = df.groupby("ts")
m = g.agg(
    x_flows=("row_id", "size"),
    x_pkts_fwd=("Total Fwd Packets", "sum"),
    x_pkts_bwd=("Total Backward Packets", "sum"),
    x_bytes_fwd=("Total Length of Fwd Packets", "sum"),
    x_bytes_bwd=("Total Length of Bwd Packets", "sum"),
    x_dur_median=("Flow Duration", "median"),
    x_syn=("SYN Flag Count", "sum"),
    x_ack=("ACK Flag Count", "sum"),
    x_rst=("RST Flag Count", "sum"),
    a_uniq_src=("Source IP", "nunique"),
    a_uniq_dst=("Destination IP", "nunique"),
    y_ddos_flows=("label_binary", "sum"),
)
m["a_src_entropy"] = g["Source IP"].apply(entropy)
m["y_benign_flows"] = m["x_flows"] - m["y_ddos_flows"]
m["y_attack_frac"] = m["y_ddos_flows"] / m["x_flows"]
vic = df[df["segment"] == "victim_service"]
m["a_victim_flows"] = vic.groupby("ts").size().reindex(m.index, fill_value=0)

m.to_csv(PROC / "minute_series_A.csv")
print(m.shape)
print(m.describe().T.to_string())

atk = m.index[m["y_ddos_flows"] > 0]
a0, a1 = atk.min(), atk.max()

panels = [
    ("Flows per minute", ["y_benign_flows", "y_ddos_flows"]),
    ("Flows to victim service (all labels)", ["a_victim_flows"]),
    ("Packets per minute (fwd / bwd)", ["x_pkts_fwd", "x_pkts_bwd"]),
    ("Median flow duration (us)", ["x_dur_median"]),
    ("Unique source IPs and source entropy", ["a_uniq_src", "a_src_entropy"]),
]
fig, axes = plt.subplots(len(panels), 1, figsize=(11, 13), sharex=True)
for ax, (title, cols) in zip(axes, panels):
    for c in cols:
        ax.plot(m.index, m[c], label=c)
    ax.axvspan(a0, a1, alpha=0.15, color="red")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
fig.tight_layout()
RES.mkdir(exist_ok=True)
fig.savefig(RES / "03_minute_series.png", dpi=130)
print(f"Saved {RES / '03_minute_series.png'}; attack minutes {a0} -> {a1}")