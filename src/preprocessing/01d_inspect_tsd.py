"""Phase 1d: generic read-only inspection of the two TSD-DDoS files."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FILES = [
    "jan-12-dataset-final-version-csv-ieeedataport.csv",
    "mar-11-dataset-final-version-csv-ieeedataport.csv",
]

for name in FILES:
    path = ROOT / "data" / "raw" / name
    print("=" * 70)
    print(name)
    if not path.exists():
        print("MISSING")
        continue
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()
    print("shape:", df.shape)
    print("columns:", list(df.columns))
    print(df.dtypes.to_string())
    print("\nhead:\n", df.head(5).to_string())
    print("\ntail:\n", df.tail(3).to_string())
    print("\nNaN per column:", df.isna().sum().to_dict())
    print("duplicated rows:", int(df.duplicated().sum()))
    print("\nsummary:\n", df.describe(include="all").T.to_string())

    for c in df.columns:
        if df[c].nunique() <= 10:
            print(f"\nvalue_counts for '{c}':")
            print(df[c].value_counts(dropna=False).to_string())
        if any(k in c.lower() for k in ("time", "date")):
            ts = pd.to_datetime(df[c], errors="coerce", format="mixed")
            gaps = ts.dropna().diff().dropna().dt.total_seconds()
            print(f"\ntimestamp column '{c}': {ts.min()} -> {ts.max()}, unparsed={int(ts.isna().sum())}")
            if len(gaps):
                print(f"gap seconds: min={gaps.min()} median={gaps.median()} max={gaps.max()}")