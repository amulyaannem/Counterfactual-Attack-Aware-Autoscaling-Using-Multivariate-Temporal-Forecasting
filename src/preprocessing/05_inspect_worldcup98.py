import sys, gzip, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data/raw/worldcup98/TR2.gz"
name = path.stem
DT = np.dtype([("ts", ">u4"), ("client", ">u4"), ("obj", ">u4"), ("size", ">u4"),
               ("method", "u1"), ("status", "u1"), ("type", "u1"), ("server", "u1")])
assert DT.itemsize == 20

raw = gzip.open(path, "rb").read()
print(f"decompressed bytes: {len(raw):,}  (mod 20 = {len(raw) % 20})")
if len(raw) % 20:
    sys.exit("Size is not a multiple of 20: format assumption is wrong. Stop and paste this output.")
a = np.frombuffer(raw, dtype=DT)
print(f"records: {len(a):,}")

ts = a["ts"].astype(np.int64)
t0, t1 = ts.min(), ts.max()
fmt = lambda t: np.datetime64(int(t), "s").astype(str)
print(f"UTC range: {fmt(t0)} -> {fmt(t1)}  ({(t1 - t0 + 1) / 3600:.2f} h)")
print("sorted by time:", bool(np.all(np.diff(ts) >= 0)))

per_s = np.bincount(ts - t0)
n = len(per_s)
print(f"seconds spanned: {n:,}; zero-request seconds: {(per_s == 0).sum():,}")
print(f"req/s: median {np.median(per_s):.0f}, p99 {np.percentile(per_s, 99):.0f}, max {per_s.max()}")

m = n // 60
per_min = per_s[: m * 60].reshape(m, 60).sum(axis=1)
print(f"req/min: median {np.median(per_min):.0f}, max {per_min.max()}, "
      f"max/median {per_min.max() / max(np.median(per_min), 1):.2f}")
k = n // 300
per_5 = per_s[: k * 300].reshape(k, 300).sum(axis=1)
print(f"5-min peak/median: {per_5.max() / max(np.median(per_5), 1):.2f}")

hours = (ts - t0) // 3600
print("requests per hour (from start):", np.bincount(hours).tolist())

cl = np.unique(a["client"], return_counts=True)[1]
print(f"unique clients: {len(cl):,}; top client share: {cl.max() / len(a):.4f}")
for f in ["method", "status", "type", "server"]:
    u, c = np.unique(a[f], return_counts=True)
    print(f"{f} raw values (top 8):", dict(zip(u[np.argsort(-c)][:8].tolist(), np.sort(c)[::-1][:8].tolist())))

out = ROOT / "data/processed"; out.mkdir(parents=True, exist_ok=True)
np.savetxt(out / f"worldcup98_{name}_per_minute.csv", per_min, fmt="%d",
           header=f"requests_per_minute start_utc={fmt(t0)}")
json.dump({"records": int(len(a)), "start_utc": fmt(t0), "end_utc": fmt(t1)},
          open(out / f"worldcup98_{name}_summary.json", "w"))