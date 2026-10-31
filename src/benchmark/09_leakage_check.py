"""Phase 6 gate: leakage checklist for the windowed benchmark. Exits non-zero if any check fails."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
CFG = yaml.safe_load(open(ROOT / "config.yaml"))
SYN = ROOT / CFG["paths"]["synthetic"]
STEP = CFG["step_s"]
FEATS = CFG["features"]
d = {s: pd.read_parquet(SYN / f"windows_{STEP}s_{s}.parquet") for s in CFG["splits"]}
results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))


# 1. Feature list contains only observed counters
check("features are x_ columns only", all(f.startswith("x_") for f in FEATS), str(FEATS))
y_cols = [c for c in d["train"].columns if c.startswith("y_")]

# 2. No feature equals a label over a whole split (label-derived feature)
bad = [(s, f, y) for s, df in d.items() for f in FEATS for y in y_cols if np.array_equal(df[f], df[y])]
check("no feature identical to a label", not bad, str(bad))

# 3. Features are reproducible from observed counters alone (recompute derived ones)
tr = d["train"]
ok = (np.array_equal(tr["x_syn_minus_ack"], tr["x_syn"] - tr["x_ack"])
      and np.allclose(tr["x_req_per_client"], tr["x_requests"] / np.maximum(tr["x_clients"], 1)))
check("derived features use only x_ inputs", ok)

# 4. No shuffling: every series is contiguous and time-ordered at the step size
gaps = {s: df.groupby("a_series", observed=True)["a_ts"].apply(lambda t: set(np.diff(t.to_numpy())))
        for s, df in d.items()}
ok = all(g == {STEP} for s in gaps for g in gaps[s])
check("series contiguous and ordered (diff == step)", ok)

# 5. No series spans splits; split days are chronological and separated by >= purge_bins
spans = {s: (df["a_ts"].min(), df["a_ts"].max()) for s, df in d.items()}
order = ["train", "val", "test"]
gap_ok = all(spans[a][1] + CFG["purge_bins"] * STEP < spans[b][0] for a, b in zip(order, order[1:]))
check("chronological train < val < test with purge gap", gap_ok,
      {s: [str(pd.to_datetime(v, unit="s")) for v in spans[s]] for s in order})
shared = set(d["train"]["a_series"]) & (set(d["val"]["a_series"]) | set(d["test"]["a_series"]))
check("no series id shared across splits", not shared)

# 6. Scaler fit on train only: recompute from train and from test, compare
sc = json.load(open(SYN / f"scaler_{STEP}s.json"))
lt = np.log1p(tr[FEATS].clip(lower=0))
ok = (sc["fit_split"] == "train" and sc["fit_rows"] == len(tr)
      and np.allclose(sc["mean"], lt.mean()) and np.allclose(sc["scale"], lt.std(ddof=0)))
te_mean = np.log1p(d["test"][FEATS].clip(lower=0)).mean()
check("scaler stats == train stats", ok)
check("scaler stats != test stats", not np.allclose(sc["mean"], te_mean))

# 7. Synthetic draws are independent across splits (different onsets for the same config)
on = {s: df.groupby("a_series", observed=True)["a_onset_bin"].first().to_numpy() for s, df in d.items()}
check("synthetic onsets differ across splits", not np.array_equal(on["train"], on["test"]))

# 8. Analysis-only columns are not features; real baseline burst excluded from evaluation
check("no a_ column in features", not any(f.startswith("a_") for f in FEATS))
va = d["val"]
check("TR1 real burst masked out of evaluation", (va.loc[va["a_baseline_burst"] == 1, "a_eval_mask"] == 0).all()
      and va["a_baseline_burst"].sum() > 0)

w = max(len(n) for n, _, _ in results)
for n, ok, det in results:
    print(f"[{'PASS' if ok else 'FAIL'}] {n:<{w}}  {det if det else ''}")
print(f"\n{sum(ok for _, ok, _ in results)}/{len(results)} checks passed")
sys.exit(0 if all(ok for _, ok, _ in results) else 1)
