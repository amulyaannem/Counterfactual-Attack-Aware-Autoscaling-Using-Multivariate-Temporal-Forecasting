"""Shared loading, features and evaluation for Phase 7+ models (same protocol for every model)."""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import precision_recall_curve

ROOT = Path(__file__).resolve().parents[2]
CFG = yaml.safe_load(open(ROOT / "config.yaml"))
SYN = ROOT / CFG["paths"]["synthetic"]
STEP, SEED = CFG["step_s"], CFG["seed"]
BASE = ["x_requests", "x_syn", "x_ack", "x_clients"]
RATIOS = ("syn_per_req", "ack_per_syn", "req_per_client")


def load(split):
    df = pd.read_parquet(SYN / f"windows_{STEP}s_{split}.parquet")
    n_s = df["a_series"].nunique()
    assert len(df) % n_s == 0 and (df.groupby("a_series", observed=True)["a_bin"].diff().dropna() == 1).all()
    return df, n_s


def wide(df, n_s, col):
    """(time x series) frame; every series in a split has the same length and is stored contiguously."""
    return pd.DataFrame(df[col].to_numpy(dtype=float).reshape(n_s, -1).T)


def flat(w):
    return w.to_numpy().T.ravel()


def features(df, n_s, h):
    """Trailing features from information available h steps before the target window (h = 0: current window)."""
    cols = {}
    srcs = {c: wide(df, n_s, c) for c in BASE}
    srcs["syn_per_req"] = srcs["x_syn"] / srcs["x_requests"].clip(lower=1)
    srcs["ack_per_syn"] = srcs["x_ack"] / srcs["x_syn"].clip(lower=1)
    srcs["req_per_client"] = srcs["x_requests"] / srcs["x_clients"].clip(lower=1)
    for name, w in srcs.items():
        cur = w.shift(h)
        past = w.shift(h + 1) if h == 0 else cur
        if h == 0:
            cols[name] = cur
        for lag in [1, 2, 12]:
            cols[f"{name}_lag{lag}"] = past.shift(lag - 1)
        for win in [12, 60, 720]:
            m = past.rolling(win, min_periods=win // 2).mean()
            cols[f"{name}_mean{win}"] = m
            cols[f"{name}_ratio{win}"] = (cur if h == 0 else past) / (m.abs() + 1)
        sd = past.rolling(60, min_periods=30).std()
        cols[f"{name}_std60"] = sd
        if h == 0:
            cols[f"{name}_z60"] = (cur - cols[f"{name}_mean60"]) / (sd + 1)
    return pd.DataFrame({k: flat(v) for k, v in cols.items()})


def level_free(cols):
    """Columns that do not carry the absolute traffic level (train/val/test days differ in level)."""
    return [c for c in cols if c.startswith(RATIOS) or "_ratio" in c or c.endswith("_z60")]


def best_f1_threshold(y, s):
    s = np.nan_to_num(s, nan=np.nanmin(s) - 1)  # no history yet -> lowest score, never flagged first
    p, r, t = precision_recall_curve(y, s)
    f = 2 * p * r / np.maximum(p + r, 1e-12)
    i = int(np.nanargmax(f[:-1]))
    return float(t[i]), float(f[i])


def fpr_threshold(y, s, fpr):
    """Smallest threshold whose false-positive rate on the given windows is at most `fpr`."""
    neg = np.sort(np.nan_to_num(s[y == 0], nan=-np.inf))[::-1]
    k = int(np.floor(fpr * len(neg)))
    return float(np.nextafter(neg[k], np.inf)) if k < len(neg) else float(neg[-1])


def detection_metrics(df, n_s, pred):
    m = df["a_eval_mask"].to_numpy() == 1
    y = df["y_attack_active"].to_numpy() == 1
    pred = pred.astype(bool)
    tp, fp = (pred & y & m).sum(), (pred & ~y & m).sum()
    fn, tn = (~pred & y & m).sum(), (~pred & ~y & m).sum()
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    out = {"precision": prec, "recall": rec, "f1": 2 * prec * rec / max(prec + rec, 1e-12), "fpr": fp / max(fp + tn, 1)}
    # same metrics without mimic series (legitimate look-alikes, unidentifiable by design)
    k = m & (df["a_family"].to_numpy() != "mimic")
    tp2, fp2, fn2 = (pred & y & k).sum(), (pred & ~y & k).sum(), (~pred & y & k).sum()
    p2, r2 = tp2 / max(tp2 + fp2, 1), tp2 / max(tp2 + fn2, 1)
    out.update({"precision_no_mimic": p2, "recall_no_mimic": r2, "f1_no_mimic": 2 * p2 * r2 / max(p2 + r2, 1e-12)})
    # event level: an event is a contiguous attack-active run inside one series
    yw, pw = y.reshape(n_s, -1), pred.reshape(n_s, -1)
    delays, missed = [], 0
    for ys, ps in zip(yw, pw):
        d = np.diff(np.r_[0, ys.astype(int), 0])
        for s, e in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
            hit = np.flatnonzero(ps[s:e])
            if len(hit):
                delays.append(hit[0] * STEP)
            else:
                missed += 1
    n_ev = len(delays) + missed
    out.update({"events": n_ev, "event_recall": len(delays) / max(n_ev, 1),
                "median_delay_s": float(np.median(delays)) if delays else np.nan})
    neg = ~y & m
    out["false_alarms_per_hour"] = (pred & neg).sum() / (neg.sum() * STEP / 3600)
    surge = neg & (df["y_surge_active"].to_numpy() == 1)
    out["fpr_during_legit_surge"] = (pred & surge).sum() / max(surge.sum(), 1)
    return out


def by_family(df, pred, model):
    rows = []
    for fam, g in df.assign(pred=pred.astype(bool)).groupby(["a_family", "a_attack_type"], observed=True):
        y = g["y_attack_active"] == 1
        rows.append({"model": model, "family": fam[0], "attack_type": fam[1],
                     "window_recall": (g["pred"] & y).sum() / max(y.sum(), 1) if y.any() else np.nan,
                     "fpr": (g["pred"] & ~y).sum() / max((~y).sum(), 1)})
    return rows
