"""Phase 1c: demand on the attacked service (192.168.10.50:80) vs background traffic."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "data" / "raw" / "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
OUT = ROOT / "results" / "01c_minute_counts.csv"

VICTIM_IP, VICTIM_PORT = "192.168.10.50", 80

df = pd.read_csv(CSV, low_memory=False)
df.columns = df.columns.str.strip()
df["ts"] = pd.to_datetime(df["Timestamp"], format="mixed")

# Flow direction can be swapped in this dataset, so check both endpoints
to_service = (
    ((df["Destination IP"] == VICTIM_IP) & (df["Destination Port"] == VICTIM_PORT))
    | ((df["Source IP"] == VICTIM_IP) & (df["Source Port"] == VICTIM_PORT))
)
df["segment"] = np.where(to_service, "victim_service", "background")

a0, a1 = df.loc[df.Label == "DDoS", "ts"].agg(["min", "max"])
df["phase"] = np.where(df.ts < a0, "pre", np.where(df.ts > a1, "post", "during"))

print("Flows by segment and label:")
print(pd.crosstab(df["segment"], df["Label"]))

counts = df.groupby(["ts", "segment", "Label"]).size().unstack(["segment", "Label"], fill_value=0)
OUT.parent.mkdir(exist_ok=True)
counts.to_csv(OUT)
phase = np.where(counts.index < a0, "pre", np.where(counts.index > a1, "post", "during"))
print("\nMedian flows per minute by phase:")
print(counts.groupby(phase).median())

cols = ["Flow Duration", "Total Fwd Packets", "Total Backward Packets", "Average Packet Size"]
print("\nBenign feature medians by phase and segment:")
print(df[df.Label == "BENIGN"].groupby(["phase", "segment"])[cols].median())
print(f"\nSaved minute counts to {OUT}")