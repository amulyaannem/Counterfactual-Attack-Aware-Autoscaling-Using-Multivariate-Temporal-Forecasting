"""Phase 1: read-only inspection of the Friday CSV. Writes results/01_inspect_report.txt"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "data" / "raw" / "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
OUT = ROOT / "results" / "01_inspect_report.txt"

lines = []


def log(msg=""):
    print(msg)
    lines.append(str(msg))


def section(title):
    log("\n" + "=" * 70)
    log(title)
    log("=" * 70)


def main():
    if not CSV.exists():
        sys.exit(f"File not found: {CSV}")
    OUT.parent.mkdir(exist_ok=True)

    try:
        df = pd.read_csv(CSV, low_memory=False)
        log("Encoding: utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(CSV, low_memory=False, encoding="latin-1")
        log("Encoding: latin-1 (utf-8 failed)")

    section("1. SHAPE")
    log(f"rows={df.shape[0]:,}  cols={df.shape[1]}")

    section("2. COLUMN NAMES (raw)")
    padded = [c for c in df.columns if c != c.strip()]
    log(f"Columns with leading/trailing whitespace: {len(padded)}")
    stripped = [c.strip() for c in df.columns]
    if len(set(stripped)) != len(stripped):
        log("WARNING: duplicate column names after stripping!")
    log(stripped)

    # Work on a renamed in-memory copy only; the file on disk is untouched.
    df.columns = stripped

    section("3. DTYPES")
    log(df.dtypes.value_counts().to_string())
    obj_cols = df.select_dtypes(include="object").columns.tolist()
    log(f"Object (non-numeric) columns: {obj_cols}")

    section("4. SAMPLE ROWS (first 3, transposed)")
    log(df.head(3).T.to_string())

    section("5. LABELS")
    label_col = next((c for c in df.columns if c.lower() == "label"), None)
    if label_col is None:
        log("No 'Label' column found!")
    else:
        vc = df[label_col].value_counts(dropna=False)
        log(vc.to_string())
        log((vc / len(df)).round(4).to_string())
        log(f"Distinct raw label values (repr): {[repr(v) for v in vc.index]}")

    section("6. MISSING / INFINITE / MALFORMED")
    na = df.isna().sum()
    log(f"Columns with NaN: {(na > 0).sum()}")
    log(na[na > 0].to_string() if (na > 0).any() else "No NaN values")

    id_like = {"Flow ID", "Source IP", "Destination IP", "Timestamp", label_col}
    num_cols = [c for c in df.columns if c not in id_like]
    bad_numeric = {}
    for c in num_cols:
        if df[c].dtype == "object":
            coerced = pd.to_numeric(df[c], errors="coerce")
            n_bad = int(coerced.isna().sum() - df[c].isna().sum())
            if n_bad:
                bad_numeric[c] = n_bad
            df[c] = coerced
    log(f"Object columns with non-numeric junk: {bad_numeric or 'none'}")

    num_df = df[num_cols].select_dtypes(include=[np.number])
    inf_counts = np.isinf(num_df).sum()
    log(f"Columns with +/-inf: {(inf_counts > 0).sum()}")
    if (inf_counts > 0).any():
        log(inf_counts[inf_counts > 0].to_string())
        rows_inf = np.isinf(num_df).any(axis=1)
        log(f"Rows with at least one inf: {int(rows_inf.sum()):,}")
        if label_col:
            log(df.loc[rows_inf, label_col].value_counts().to_string())

    section("7. DUPLICATES")
    log(f"Fully duplicated rows (excluding first occurrence): {int(df.duplicated().sum()):,}")
    feat_cols = [c for c in df.columns if c != label_col]
    h = pd.util.hash_pandas_object(df[feat_cols], index=False)
    log(f"Rows duplicated on all non-label columns: {int(h.duplicated().sum()):,}")
    if label_col:
        n_lab = df.groupby(h)[label_col].nunique()
        log(f"Identical-feature groups with CONFLICTING labels: {int((n_lab > 1).sum())}")

    section("8. CONSTANT COLUMNS")
    nun = num_df.nunique(dropna=True)
    log(f"Constant numeric columns: {nun[nun <= 1].index.tolist()}")

    section("9. NEGATIVE VALUES (numeric columns)")
    neg = (num_df < 0).sum()
    log(neg[neg > 0].to_string() if (neg > 0).any() else "None")

    section("10. NUMERIC SUMMARY (selected columns)")
    pick = [c for c in ["Flow Duration", "Total Fwd Packets", "Total Backward Packets",
                        "Flow Bytes/s", "Flow Packets/s"] if c in num_df.columns]
    if pick:
        log(num_df[pick].replace([np.inf, -np.inf], np.nan)
            .describe(percentiles=[.01, .5, .99]).T.to_string())

    section("11. TIMESTAMPS")
    if "Timestamp" not in df.columns:
        log("No Timestamp column!")
    else:
        raw = df["Timestamp"]
        log(f"Raw samples: {raw.head(5).tolist()}")
        log(f"Raw samples (tail): {raw.tail(3).tolist()}")
        for name, kw in [("month-first", dict(dayfirst=False)), ("day-first", dict(dayfirst=True))]:
            ts = pd.to_datetime(raw, errors="coerce", format="mixed", **kw)
            log(f"[{name}] unparsed={int(ts.isna().sum()):,}  min={ts.min()}  max={ts.max()}")
        ts = pd.to_datetime(raw, errors="coerce", format="mixed", dayfirst=False)
        log("\n(Check the raw samples above to decide which parsing is correct. Below uses month-first.)")
        log(f"Monotonic increasing in file order: {ts.is_monotonic_increasing}")
        log(f"Distinct timestamps: {ts.nunique():,}")
        log(f"Distinct seconds values (0-59): {ts.dt.second.nunique()}")
        uniq = ts.dropna().drop_duplicates().sort_values()
        gaps = uniq.diff().dropna().dt.total_seconds()
        if len(gaps):
            log(f"Gap between distinct timestamps (s): min={gaps.min()}  median={gaps.median()}  max={gaps.max()}")
        per_ts = ts.value_counts()
        log(f"Flows per distinct timestamp: min={per_ts.min()}  median={per_ts.median()}  max={per_ts.max()}")
        if label_col:
            tmp = pd.DataFrame({"ts": ts, "y": df[label_col]})
            tmp["m"] = tmp["ts"].dt.floor("min")
            by_min = tmp.groupby(["m", "y"]).size().unstack(fill_value=0)
            log("\nFlows per minute by label:")
            log(by_min.to_string())

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport saved to {OUT}")


if __name__ == "__main__":
    main()