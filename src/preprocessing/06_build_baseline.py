import sys, gzip
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
src = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data/raw/worldcup98/TR2.gz"
STEP = 5  # seconds
DT = np.dtype([("ts", ">u4"), ("client", ">u4"), ("obj", ">u4"), ("size", ">u4"),
               ("method", "u1"), ("status", "u1"), ("type", "u1"), ("server", "u1")])
a = np.frombuffer(gzip.open(src, "rb").read(), dtype=DT)
ts = a["ts"].astype(np.int64)
t0 = ts.min() - (ts.min() % STEP)
idx = (ts - t0) // STEP
req = np.bincount(idx)
# distinct clients per bin: needed later for entropy-style features
order = np.lexsort((a["client"], idx))
i, c = idx[order], a["client"][order]
new = np.r_[True, (i[1:] != i[:-1]) | (c[1:] != c[:-1])]
uniq = np.bincount(i[new], minlength=len(req))
out = ROOT / "data/processed" / f"{src.stem}_baseline_{STEP}s.csv"
out.parent.mkdir(parents=True, exist_ok=True)
np.savetxt(out, np.c_[np.arange(len(req)) * STEP + t0, req, uniq], fmt="%d",
           delimiter=",", header="ts_utc,requests,distinct_clients", comments="")
print(f"bins: {len(req):,}  requests/bin median {np.median(req):.0f}  max {req.max()}")
print(f"sum check: {req.sum():,} (should equal {len(a):,})")
print("wrote", out)