"""Phase 1g: validate TSD-DDoS v6, compare to DataPort copies, summarise attack structure, plot."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
V6 = RAW / "tsd_mendeley_v6"
RES = ROOT / "results"
SERIAL = "time period serial number"
COUNTS = ["#SYN-Packets", "#SYN-ACK-Packets", "#ACK-Packets", "#RST-Packets", "#TCP-Packets"]
LABEL = "Traffic-Type"


def find(substr):
    hits = [p for p in V6.rglob("*.csv") if substr in p.name]
    assert len(hits) == 1, (substr, hits)
    df = pd.read_csv(hits[0])
    df.columns = df.columns.str.strip()
    return df


jan, mar = find("jan-12-traffic"), find("mar-11-traffic")
det_tr, det_te = find("Flood)-detection-train"), find("Flood)-detection-test")
sev_tr, sev_te = find("severity-detection-train"), find("severity-detection-test")


def aligned(base, lab):
    return (np.array_equal(base[[SERIAL] + COUNTS].to_numpy(), lab[[SERIAL] + COUNTS].to_numpy()))


print("Jan base == detection-train features:", aligned(jan, det_tr))
print("Mar base == detection-test features:", aligned(mar, det_te))
print("Severity attack windows == detection attack windows (train):",
      bool(((sev_tr[LABEL] != "Normal") == (det_tr[LABEL] == "DoS Attack")).all()))
print("Severity attack windows == detection attack windows (test):",
      bool(((sev_te[LABEL] != "Normal") == (det_te[LABEL] == "DoS Attack")).all()))


def runs(is_attack):
    a = is_attack.to_numpy().astype(int)
    d = np.diff(np.r_[0, a, 0])
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
    return starts, ends - starts


ports = {
    "jan": ("jan-12-dataset-final-version-csv-ieeedataport.csv", jan, det_tr),
    "mar": ("mar-11-dataset-final-version-csv-ieeedataport.csv", mar, det_te),
}
for day, (name, base, lab) in ports.items():
    print("\n" + "=" * 70)
    print(day.upper())
    starts, lengths = runs(lab[LABEL] == "DoS Attack")
    print(f"attack runs: {len(starts)}; run length (windows): "
          f"min={lengths.min()} median={np.median(lengths)} max={lengths.max()}")
    print("first runs (start index, length):", list(zip(starts[:10].tolist(), lengths[:10].tolist())))
    for cls, grp in lab.groupby(LABEL):
        print(f"{cls:>10}: median SYN={grp['#SYN-Packets'].median():.0f} "
              f"SYN-ACK={grp['#SYN-ACK-Packets'].median():.0f} "
              f"ACK={grp['#ACK-Packets'].median():.0f} TCP={grp['#TCP-Packets'].median():.0f}")
    p = RAW / name
    if p.exists():
        old = pd.read_csv(p)
        old.columns = old.columns.str.strip()
        vals_old, vals_new = old.iloc[:, 1:].to_numpy(), base.to_numpy()
        print("DataPort counts identical to v6:",
              vals_old.shape == vals_new.shape and bool(np.array_equal(vals_old, vals_new)))
        fn = old.iloc[:, 0].to_numpy()
        sizes = pd.Series(fn).value_counts()
        print(f"pcap files: {len(sizes)}; windows per file: "
              f"median={sizes.median()} max={sizes.max()}; file boundaries: {int((np.diff(fn) != 0).sum())}")
        crossing = int((fn[starts] != fn[starts + lengths - 1]).sum())
        print(f"attack runs spanning a file boundary: {crossing} of {len(starts)}")
    else:
        print("DataPort copy not found; skipping file-boundary checks")

RES.mkdir(exist_ok=True)
fig, axes = plt.subplots(2, 1, figsize=(12, 8))
for ax, (title, df, lab) in zip(axes, [("Jan (train day)", jan, det_tr), ("Mar (test day)", mar, det_te)]):
    x = df[SERIAL]
    ax.semilogy(x, df["#SYN-Packets"] + 1, lw=0.5, label="SYN + 1")
    ax.semilogy(x, df["#TCP-Packets"], lw=0.5, label="TCP")
    att = (lab[LABEL] == "DoS Attack").to_numpy()
    ax.fill_between(x, 0, 1, where=att, transform=ax.get_xaxis_transform(), color="red", alpha=0.25)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig(RES / "01g_tsd_overview.png", dpi=130)
print(f"\nSaved {RES / '01g_tsd_overview.png'}")