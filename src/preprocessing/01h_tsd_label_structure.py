"""Phase 1h: where do the TSD labels come from? Flood bursts vs isolated windows; file segments."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
V6 = RAW / "tsd_mendeley_v6"
OUT = ROOT / "data" / "processed"
LABEL = "Traffic-Type"

DAYS = {
    "jan": ("Flood)-detection-train", "severity-detection-train",
            "jan-12-dataset-final-version-csv-ieeedataport.csv"),
    "mar": ("Flood)-detection-test", "severity-detection-test",
            "mar-11-dataset-final-version-csv-ieeedataport.csv"),
}


def find(substr):
    hits = [p for p in V6.rglob("*.csv") if substr in p.name]
    assert len(hits) == 1, (substr, hits)
    df = pd.read_csv(hits[0])
    df.columns = df.columns.str.strip()
    return df


OUT.mkdir(parents=True, exist_ok=True)
for day, (det_key, sev_key, old_name) in DAYS.items():
    det, sev = find(det_key), find(sev_key)
    old = pd.read_csv(RAW / old_name)
    fn = old.iloc[:, 0].to_numpy()
    tcp = det["#TCP-Packets"].to_numpy()
    att = (det[LABEL] == "DoS Attack").to_numpy()
    flood = tcp > 100_000
    idx = np.where(flood)[0]

    print("=" * 70)
    print(day.upper())
    print(f"flood windows (TCP > 1e5): {flood.sum()}, index range {idx.min()}-{idx.max()}, "
          f"contiguous: {idx.max() - idx.min() + 1 == flood.sum()}, about {flood.sum() * 5 / 60:.1f} min")
    print(f"attack windows inside flood: {int((att & flood).sum())}, outside: {int((att & ~flood).sum())}")
    print(f"flood windows NOT labelled attack: {int((flood & ~att).sum())}")
    print("\nSeverity label vs flood:")
    print(pd.crosstab(sev[LABEL], np.where(flood, "flood", "other")))

    q = [0.05, 0.25, 0.5, 0.75, 0.95, 1.0]
    print("\nTCP packets, attack windows outside flood:", det.loc[att & ~flood, "#TCP-Packets"].quantile(q).tolist())
    print("TCP packets, normal windows:              ", det.loc[~att, "#TCP-Packets"].quantile(q).tolist())
    quiet = tcp <= 10
    print(f"windows with TCP <= 10: {int(quiet.sum())}; labelled attack: {int((quiet & att).sum())} "
          f"({att[quiet].mean():.1%})")
    synack0 = det["#SYN-ACK-Packets"].to_numpy() == 0
    print(f"attack windows with SYN-ACK == 0: {att[att & synack0].size / max(att.sum(), 1):.1%}; "
          f"normal windows with SYN-ACK == 0: {(synack0 & ~att).sum() / (~att).sum():.1%}")

    seg = pd.Series(fn).value_counts()
    long_ = seg[seg >= 12]
    print(f"\nfile segments >= 12 windows (>= 1 min): {len(long_)}, covering {int(long_.sum())} of {len(fn)} windows")
    print("largest segments (file number: windows):", seg.head(5).to_dict())

    merged = det.copy()
    merged.insert(0, "file_number", fn)
    merged["severity"] = sev[LABEL].to_numpy()
    merged["in_flood"] = flood
    merged.to_csv(OUT / f"tsd_{day}.csv", index=False)
    print(f"saved {OUT / f'tsd_{day}.csv'}")