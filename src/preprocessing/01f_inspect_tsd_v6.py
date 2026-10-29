"""Phase 1f: inspect the Mendeley v6 TSD-DDoS CSVs and compare base files with the DataPort copies."""
from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parents[2] / "data" / "raw"
V6 = RAW / "tsd_mendeley_v6"
BASE = ["jan-12-dataset-final-version-csv-ieeedataport.csv",
        "mar-11-dataset-final-version-csv-ieeedataport.csv"]

files = sorted(V6.rglob("*.csv"))
if not files:
    raise SystemExit(f"No CSV files found under {V6}")


def load(p):
    df = pd.read_csv(p, low_memory=False)
    df.columns = df.columns.str.strip()
    return df


for p in files:
    df = load(p)
    print("=" * 70)
    print(p.relative_to(V6))
    print("shape:", df.shape)
    print("columns:", list(df.columns))
    print(df.head(5).to_string())
    print("NaN total:", int(df.isna().sum().sum()))
    for c in df.columns:
        if df[c].nunique() <= 12:
            print(f"\nvalue_counts for '{c}':")
            print(df[c].value_counts(dropna=False).to_string())

print("\n" + "=" * 70)
print("Comparison with the DataPort copies")
for name in BASE:
    old = RAW / name
    new = next((q for q in files if q.name == name), None)
    if old.exists() and new is not None:
        a, b = load(old), load(new)
        print(f"{name}: DataPort {a.shape} vs v6 {b.shape}; "
              f"same columns: {list(a.columns) == list(b.columns)}; identical: {a.equals(b)}")
    else:
        print(f"{name}: no same-named file in v6 (see the file list above)")