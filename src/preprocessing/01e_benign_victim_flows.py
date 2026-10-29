"""Phase 1e: who are the benign flows to the victim service?"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "data" / "raw" / "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
VICTIM_IP, VICTIM_PORT = "192.168.10.50", 80

df = pd.read_csv(CSV, low_memory=False)
df.columns = df.columns.str.strip()
df["ts"] = pd.to_datetime(df["Timestamp"], format="mixed")

to_service = (
    ((df["Destination IP"] == VICTIM_IP) & (df["Destination Port"] == VICTIM_PORT))
    | ((df["Source IP"] == VICTIM_IP) & (df["Source Port"] == VICTIM_PORT))
)
b = df[to_service & (df.Label == "BENIGN")].copy()
d = df[df.Label == "DDoS"]

b["client"] = b["Source IP"].where(b["Source IP"] != VICTIM_IP, b["Destination IP"])
print(f"Benign victim-service flows: {len(b):,}")
print(f"Time range: {b.ts.min()} -> {b.ts.max()}")
print(f"Distinct client IPs: {b.client.nunique()}")
print("\nTop client IPs:")
print(b.client.value_counts().head(10))
print(f"\nShare involving 172.16.0.1: "
      f"{((b['Source IP'] == '172.16.0.1') | (b['Destination IP'] == '172.16.0.1')).mean():.3f}")
print("\nProtocol:\n", b.Protocol.value_counts())

cols = ["Flow Duration", "Total Fwd Packets", "Total Backward Packets",
        "Total Length of Fwd Packets", "Total Length of Bwd Packets",
        "Average Packet Size", "SYN Flag Count", "ACK Flag Count", "PSH Flag Count"]
print("\nMedians, benign-to-victim vs DDoS:")
print(pd.concat({"benign_to_victim": b[cols].median(), "ddos": d[cols].median()}, axis=1))