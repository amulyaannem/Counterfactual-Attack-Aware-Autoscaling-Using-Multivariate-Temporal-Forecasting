import sys, re, argparse
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/home/ammu/Desktop/Documents/projects/ai_project")
ap = argparse.ArgumentParser()
ap.add_argument("files", nargs="+")
ap.add_argument("--tcp"); ap.add_argument("--syn"); ap.add_argument("--ack")
args = ap.parse_args()
THRESH = [3e4, 1e5, 3e5]
QS = [0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0]

def pick(cols, pat, avoid=None):
    m = [c for c in cols if re.search(pat, c, re.I) and not (avoid and re.search(avoid, c, re.I))]
    return m

def runs(mask):
    m = np.asarray(mask).astype(int)
    d = np.diff(np.r_[0, m, 0])
    return np.where(d == -1)[0] - np.where(d == 1)[0]

rows = []
for f in args.files:
    df = pd.read_csv(f)
    print(f"\n=== {f}  shape {df.shape}")
    print("columns:", list(df.columns))
    tcp = args.tcp or (pick(df.columns, r"tcp") or [None])
    syn = args.syn or (pick(df.columns, r"\bsyn\b", r"ack") or [None])
    ack = args.ack or (pick(df.columns, r"\back\b", r"syn") or [None])
    if isinstance(tcp, list): tcp = tcp[0] if len(tcp) == 1 else None
    if isinstance(syn, list): syn = syn[0] if len(syn) == 1 else None
    if isinstance(ack, list): ack = ack[0] if len(ack) == 1 else None
    if tcp is None:
        sys.exit("Could not pick a unique TCP column. Re-run with --tcp COLNAME (see columns above).")
    print(f"using tcp={tcp} syn={syn} ack={ack}")
    v = pd.to_numeric(df[tcp], errors="coerce")
    print(v.describe().to_string())
    for t in THRESH:
        fl = v > t
        flood, normal = v[fl], v[~fl]
        r = runs(fl)  # run lengths in ROW order; rows may not be contiguous in time
        rec = {"file": Path(f).name, "threshold": t, "n_flood": int(fl.sum()),
               "n_below": int((~fl).sum()), "n_runs": int(len(r)),
               "max_run": int(r.max()) if len(r) else 0}
        if len(flood):
            med = flood.median()
            for q in QS:
                rec[f"flood_rel_q{q}"] = flood.quantile(q) / med
            rec["flood_median"] = med
            rec["below_median"] = normal.median()
            rec["flood_median_over_below_median_UNCALIBRATED"] = med / max(normal.median(), 1)
        if syn and ack:
            H = pd.to_numeric(df[syn], errors="coerce") - pd.to_numeric(df[ack], errors="coerce")
            rec["flood_with_H_gt_0"] = int((H[fl] > 0).sum())
        rows.append(rec)
        print(f"\nthreshold {t:g}: flood {rec['n_flood']}, below {rec['n_below']}, "
              f"runs {rec['n_runs']}, longest run {rec['max_run']}")
        if len(flood):
            print("  flood magnitude relative to flood median:",
                  {q: round(rec[f'flood_rel_q{q}'], 3) for q in QS})

out = ROOT / "results"; out.mkdir(exist_ok=True)
pd.DataFrame(rows).to_csv(out / "tsd_flood_magnitudes.csv", index=False)
print("\nwrote", out / "tsd_flood_magnitudes.csv")