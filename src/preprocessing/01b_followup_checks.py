"""Phase 1b: read-only follow-up checks on Dataset A."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "data" / "raw" / "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"

df = pd.read_csv(CSV, low_memory=False)
df.columns = df.columns.str.strip()
df["ts"] = pd.to_datetime(df["Timestamp"], format="mixed")

print("Fwd Header Length identical to '.1' column:",
      df["Fwd Header Length"].equals(df["Fwd Header Length.1"]))

print("\n--- Distinct IPs / ports per label ---")
print(df.groupby("Label").agg(src_ips=("Source IP", "nunique"),
                              dst_ips=("Destination IP", "nunique"),
                              dst_ports=("Destination Port", "nunique")))

ddos = df[df.Label == "DDoS"]
print("\n--- Top source IPs (DDoS) ---")
print(ddos["Source IP"].value_counts().head(5))
print("\n--- Top destination IP:port (DDoS) ---")
print(ddos.groupby(["Destination IP", "Destination Port"]).size().nlargest(5))

print("\n--- Protocol by label ---")
print(pd.crosstab(df["Protocol"], df["Label"]))

# Phase boundaries come from the labels, used for analysis only
a0, a1 = ddos["ts"].min(), ddos["ts"].max()
df["phase"] = np.where(df.ts < a0, "pre", np.where(df.ts > a1, "post", "during"))
benign = df[df.Label == "BENIGN"]

print(f"\nAttack minutes: {a0} -> {a1}")
print("\n--- Benign flows per minute, by phase ---")
per_min = benign.groupby(["phase", "ts"]).size()
print(per_min.groupby(level="phase").describe())

print("\n--- Benign feature medians, by phase ---")
cols = ["Flow Duration", "Total Fwd Packets", "Total Backward Packets",
        "Average Packet Size", "SYN Flag Count", "ACK Flag Count"]
print(benign.groupby("phase")[cols].median())

print("\n--- Benign distinct src / dst IPs, by phase ---")
print(benign.groupby("phase").agg(src=("Source IP", "nunique"),
                                  dst=("Destination IP", "nunique")))