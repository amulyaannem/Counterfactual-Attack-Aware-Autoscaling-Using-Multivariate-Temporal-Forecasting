"""Phase 2: reproducible, audit-preserving cleaning of Dataset A. The raw file is never modified."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
OUT_DIR = ROOT / "data" / "processed"
RES = ROOT / "results"

VICTIM_IP, VICTIM_PORT = "192.168.10.50", 80
RATE_COLS = ["Flow Bytes/s", "Flow Packets/s"]
ID_COLS = ["Flow ID", "Source IP", "Source Port", "Destination IP", "Destination Port", "Timestamp"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    df = pd.read_csv(RAW, low_memory=False)
    df.columns = df.columns.str.strip()
    df["Label"] = df["Label"].str.strip()
    df.insert(0, "row_id", np.arange(len(df)))  # original file order
    df["ts"] = pd.to_datetime(df["Timestamp"], format="mixed")

    # Duplicates are judged on the original columns, before any flags are added
    df["is_duplicate"] = df.drop(columns=["row_id", "ts"]).duplicated()

    num_cols = [c for c in df.columns
                if c not in ID_COLS + ["Label", "row_id", "ts", "is_duplicate"]]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")

    # Audit flags, computed before any value is changed
    df["had_inf"] = np.isinf(df[num_cols]).any(axis=1)
    df["had_nan"] = df[num_cols].isna().any(axis=1)
    df["neg_duration"] = df["Flow Duration"] < 0

    # inf -> NaN in the rate columns only; raw values kept alongside
    for c in RATE_COLS:
        df[c + "_raw"] = df[c]
        df[c] = df[c].replace([np.inf, -np.inf], np.nan)

    # Analysis-only variables (never model inputs)
    df["label_binary"] = (df["Label"] != "BENIGN").astype(int)
    to_service = (
        ((df["Destination IP"] == VICTIM_IP) & (df["Destination Port"] == VICTIM_PORT))
        | ((df["Source IP"] == VICTIM_IP) & (df["Source Port"] == VICTIM_PORT))
    )
    df["segment"] = np.where(to_service, "victim_service", "background")

    # Constant and duplicate columns: listed, not dropped
    nun = df[num_cols].nunique(dropna=True)
    constant = nun[nun <= 1].index.tolist()
    seen, duplicate_of = {}, {}
    for c in num_cols:
        sig = int(pd.util.hash_pandas_object(df[c], index=False).sum())
        if sig in seen and df[c].equals(df[seen[sig]]):
            duplicate_of[c] = seen[sig]
        else:
            seen.setdefault(sig, c)
    duplicate_of = {c: k for c, k in duplicate_of.items() if c not in constant}
    dup_cols = list(duplicate_of)

    analysis_only = ID_COLS + ["ts", "row_id", "segment", "Protocol"] + [c + "_raw" for c in RATE_COLS]
    flags = ["is_duplicate", "had_inf", "had_nan", "neg_duration"]
    excluded = sorted(set(constant) | set(dup_cols))
    features = [c for c in num_cols if c not in excluded and c not in analysis_only]
    flag_cols_unverified = [c for c in features if "Flag" in c]
    features = [c for c in features if c not in flag_cols_unverified]

    manifest = {
        "raw_file": RAW.name,
        "raw_sha256": sha256(RAW),
        "n_rows": int(len(df)),
        "labels": ["Label", "label_binary"],
        "analysis_only": analysis_only,
        "audit_flags": flags,
        "excluded_constant": constant,
        "excluded_duplicate": dup_cols,
        "duplicate_of": duplicate_of,
        "candidate_features": features,
        "excluded_unverified_flag_semantics": flag_cols_unverified,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RES.mkdir(exist_ok=True)
    try:
        df.to_parquet(OUT_DIR / "flows_clean.parquet", index=False)
        out_name = "flows_clean.parquet"
    except ImportError:
        df.to_csv(OUT_DIR / "flows_clean.csv", index=False)
        out_name = "flows_clean.csv"
    (OUT_DIR / "flows_manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"Saved {out_name} ({len(df):,} rows, {df.shape[1]} cols)")
    print("Flag counts:", {f: int(df[f].sum()) for f in flags})
    print("Segments:\n", df["segment"].value_counts().to_string())
    print("Label counts:\n", df["Label"].value_counts().to_string())
    print("Constant columns:", constant)
    print("Duplicate columns:", dup_cols)
    print("Duplicate pairs (dropped -> kept):", duplicate_of)
    print(f"Candidate features: {len(features)}")


if __name__ == "__main__":
    main()