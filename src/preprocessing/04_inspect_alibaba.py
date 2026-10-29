"""Phase 4a: generic read-only inspection of the Alibaba microservices file."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if len(sys.argv) != 2:
    sys.exit("usage: python 04_inspect_alibaba.py <path-to-csv-or-parquet>")
p = Path(sys.argv[1])
df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p, low_memory=False)
df.columns = df.columns.astype(str).str.strip()

print("shape:", df.shape)
print(df.dtypes.to_string())
print(df.head(8).to_string())

na = df.isna().sum()
print("\nColumns with NaN:", na[na > 0].to_dict() if (na > 0).any() else "none")

num = df.select_dtypes(include=[np.number])
print("\nNumeric summary:")
print(num.describe(percentiles=[0.01, 0.5, 0.99]).T.to_string())
in01 = [c for c in num.columns if num[c].min() >= 0 and num[c].max() <= 1]
print("\nColumns entirely within [0, 1] (possible min-max scaling):", in01)

for c in df.columns:
    low = c.lower()
    if any(k in low for k in ("time", "stamp", "date")):
        s = df[c]
        print(f"\ntime-like column '{c}': dtype={s.dtype}, min={s.min()}, max={s.max()}, "
              f"monotonic_increasing={s.is_monotonic_increasing}")
        if pd.api.types.is_numeric_dtype(s):
            u = np.sort(s.dropna().unique())
            if len(u) > 1:
                d = np.diff(u)
                print(f"  gaps between distinct values: min={d.min()} median={np.median(d)} max={d.max()}")
    if not pd.api.types.is_numeric_dtype(df[c]) or any(k in low for k in ("id", "name")):
        print(f"id-like column '{c}': {df[c].nunique():,} distinct values")