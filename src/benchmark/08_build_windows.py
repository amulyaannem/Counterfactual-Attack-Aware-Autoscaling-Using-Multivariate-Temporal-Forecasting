"""Phase 6: windowed semi-synthetic benchmark.

Observed(t) = f(L(t), A(t)). L = real WorldCup98 baseline + SYNTHETIC surge, A = SYNTHETIC attack-like
traffic, so the legitimate component is known by construction (see docs/BENCHMARK_SPEC.md).
Column groups: x_ model inputs (observed counters only), y_ labels / ground truth, a_ analysis-only.
"""
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
CFG = yaml.safe_load(open(ROOT / "config.yaml"))
PROC = ROOT / CFG["paths"]["processed"]
SYN = ROOT / CFG["paths"]["synthetic"]
RES = ROOT / CFG["paths"]["results"]
SYN.mkdir(parents=True, exist_ok=True)
STEP = CFG["step_s"]
PER_MIN = 60 // STEP
LG, SG, AT = CFG["legit"], CFG["surge"], CFG["attack"]
K = LG["connections_per_request"]


def load_ack_ratios():
    """ACK/SYN per 5-s window from TSD non-flood windows (ACK counts every ACK-flagged packet, not handshakes)."""
    pool = []
    for f in LG["ack_ratio_files"]:
        t = pd.read_csv(ROOT / CFG["paths"]["raw"] / f)
        t.columns = [c.strip().replace("# ", "#") for c in t.columns]
        syn, ack, tcp = t["#SYN Packets"], t["#ACK Packets"], t["#TCP Packets"]
        keep = (tcp <= 1e5) & (syn - ack < 100) & (syn >= LG["ack_ratio_min_syn"])
        pool.append((ack[keep] / syn[keep]).to_numpy())
    return np.concatenate(pool)


ACK_RATIO = load_ack_ratios()


def surge_shape(n, onset, rise_min, plateau_min, hl_min):
    t = np.arange(n) - onset
    rise, plat, hl = rise_min * PER_MIN, plateau_min * PER_MIN, hl_min * PER_MIN
    s = np.where(t < 0, 0.0, np.minimum((t + 1) / rise, 1.0))
    s = np.where(t >= rise + plat, 0.5 ** ((t - rise - plat) / hl), s)
    return np.where(s < SG["tail_cutoff"], 0.0, s)


def plateau_shape(n, start, dur_min):
    s = np.zeros(n)
    s[start: start + dur_min * PER_MIN] = 1.0
    return s


def yoyo_shape(n, start):
    s = np.zeros(n)
    on, off = AT["yoyo_on_min"] * PER_MIN, AT["yoyo_off_min"] * PER_MIN
    for c in range(AT["yoyo_cycles"]):
        s[start + c * (on + off): start + c * (on + off) + on] = 1.0
    return s


def configs():
    """(family, attack_type, params) for every scenario in BENCHMARK_SPEC.md."""
    yield "none", "none", {}
    for p, r in itertools.product(SG["peak_ratio"], SG["rise_min"]):
        yield "surge_only", "none", {"peak_ratio": p, "rise_min": r}
    for t, rho, d in itertools.product(["syn", "app"], AT["rho"], AT["duration_min"]):
        yield "attack_only", t, {"rho": rho, "duration_min": d}
    for t, rho, r in itertools.product(["syn", "app"], AT["rho"], SG["rise_min"]):
        yield "surge_plus_attack", t, {"peak_ratio": AT["combined_peak_ratio"], "rise_min": r, "rho": rho,
                                       "duration_min": max(AT["duration_min"])}
    for t, rho in itertools.product(["syn", "app"], AT["rho"]):
        yield "yoyo", t, {"rho": rho}
    for rho, r in itertools.product(AT["rho"], SG["rise_min"]):
        yield "mimic", "app", {"rho": rho, "rise_min": r}
    for t, rho in itertools.product(["syn", "app"], AT["rho"]):
        yield "smokescreen", t, {"peak_ratio": AT["combined_peak_ratio"], "rise_min": min(SG["rise_min"]),
                                 "rho": rho, "duration_min": max(AT["duration_min"])}


def build_series(base, day, family, atype, p, rng):
    n = len(base)
    b_req = base["requests"].to_numpy()
    b_cl = base["distinct_clients"].to_numpy()
    rpc = np.median(b_req / np.maximum(b_cl, 1))  # requests per client on this day
    lo, hi = (h * 3600 // STEP for h in LG["onset_window_h"])
    onset = int(rng.integers(lo, hi))
    level = np.median(b_req[max(onset - LG["level_window_bins"], 0): onset])

    s_shape = np.zeros(n)
    if "peak_ratio" in p and family != "mimic":
        s_shape = surge_shape(n, onset, p["rise_min"], SG["plateau_min"], SG["decay_half_life_min"])
    extra = rng.poisson((p.get("peak_ratio", 1) - 1) * level * s_shape)
    l_req = b_req + extra
    l_cl = b_cl + np.rint(extra / rpc).astype(np.int64)

    a_shape = np.zeros(n)
    peak_bin = onset + p.get("rise_min", 0) * PER_MIN
    if family == "attack_only":
        a_shape = plateau_shape(n, onset, p["duration_min"])
    elif family == "surge_plus_attack":
        span = (p["rise_min"] + SG["plateau_min"] + SG["decay_half_life_min"]) * PER_MIN
        a_shape = plateau_shape(n, onset + int(rng.integers(0, span)), p["duration_min"])
    elif family == "smokescreen":
        a_shape = plateau_shape(n, peak_bin, p["duration_min"])
    elif family == "yoyo":
        a_shape = yoyo_shape(n, onset)
    elif family == "mimic":
        a_shape = surge_shape(n, onset, p["rise_min"], SG["plateau_min"], SG["decay_half_life_min"])
    active = a_shape > 0
    a = np.zeros(n, dtype=np.int64)
    if active.any():
        amp = p["rho"] * l_req[active].max()
        jitter = rng.uniform(*AT["plateau_jitter"], n) if family != "mimic" else 1.0
        a = rng.poisson(amp * a_shape * jitter)

    # Observed counters. SYN flood: SYNs only. App flood / mimic: handshakes complete, requests are served and
    # ACK packets follow the same per-window ACK/SYN ratio as legitimate traffic (conservative: packet-level look-alike).
    ratio = ACK_RATIO[rng.integers(0, len(ACK_RATIO), n)]
    l_syn = rng.poisson(K * l_req)
    l_ack = rng.poisson(l_syn * ratio)
    a_syn = a
    a_ack = rng.poisson(a * ratio) if atype == "app" else np.zeros(n, dtype=np.int64)
    a_req = a if atype == "app" else np.zeros(n, dtype=np.int64)
    if family == "mimic":
        a_cl = np.rint(a / rpc).astype(np.int64)
    elif atype == "app":
        a_cl = np.where(a > 0, np.minimum(AT["app_flood_clients"], a), 0)
    else:
        a_cl = np.zeros(n, dtype=np.int64)  # spoofed SYNs never produce a served request

    x_req = l_req + a_req
    x_cl = l_cl + a_cl
    df = pd.DataFrame({
        "x_requests": x_req,
        "x_syn": l_syn + a_syn,
        "x_ack": l_ack + a_ack,
        "x_clients": x_cl,
    })
    df["x_syn_minus_ack"] = df["x_syn"] - df["x_ack"]
    df["x_req_per_client"] = x_req / np.maximum(x_cl, 1)
    df["y_legit_requests"] = l_req
    df["y_legit_clients"] = l_cl
    df["y_attack_conn"] = a
    df["y_attack_active"] = active.astype(np.int8)
    df["y_surge_active"] = (s_shape > 0).astype(np.int8)
    df["y_attack_frac"] = a_syn / np.maximum(df["x_syn"], 1)
    df["a_ts"] = base["ts_utc"].to_numpy()
    df["a_bin"] = np.arange(n)
    df["a_baseline_requests"] = b_req
    burst = np.zeros(n, dtype=np.int8)
    for s, e in CFG["baseline_bursts"].get(day, []):
        burst[s: e + 1] = 1
    df["a_baseline_burst"] = burst
    df["a_eval_mask"] = (1 - burst).astype(np.int8)
    df["a_onset_bin"] = onset
    return df


# ---- build -------------------------------------------------------------------
cfg_list = list(configs())
index, frames = [], {}
for si, (split, day) in enumerate(CFG["splits"].items()):
    base = pd.read_csv(PROC / f"{day}_baseline_{STEP}s.csv")
    parts = []
    for ci, (family, atype, p) in enumerate(cfg_list):
        for rep in range(CFG["n_seeds"]):
            rng = np.random.default_rng(np.random.SeedSequence([CFG["seed"], si, ci, rep]))
            sid = f"{day}_{family}_{atype}_{ci:02d}_s{rep}"
            df = build_series(base, day, family, atype, p, rng)
            df.insert(0, "a_series", sid)
            df["a_split"], df["a_day"], df["a_family"], df["a_attack_type"], df["a_seed_rep"] = \
                split, day, family, atype, rep
            for key in ["peak_ratio", "rise_min", "rho", "duration_min"]:
                df[f"a_{key}"] = p.get(key, np.nan)
            parts.append(df)
            index.append({"series": sid, "split": split, "day": day, "family": family, "attack_type": atype,
                          "seed_rep": rep, "onset_bin": int(df["a_onset_bin"].iloc[0]), **p})
    out = pd.concat(parts, ignore_index=True)
    for c in ["a_series", "a_split", "a_day", "a_family", "a_attack_type"]:
        out[c] = out[c].astype("category")
    out.to_parquet(SYN / f"windows_{STEP}s_{split}.parquet", index=False)
    frames[split] = out
    print(f"{split} ({day}): {out['a_series'].nunique()} series, {len(out):,} rows, "
          f"attack-active rows {int(out['y_attack_active'].sum()):,}, surge-active {int(out['y_surge_active'].sum()):,}")
pd.DataFrame(index).to_csv(SYN / "series_index.csv", index=False)

# ---- scaler: log1p + standardise, fit on TRAIN only ----------------------------
feats = CFG["features"]
tr = frames["train"]
sc = StandardScaler().fit(np.log1p(tr[feats].clip(lower=0)))
json.dump({"fit_split": "train", "fit_day": CFG["splits"]["train"], "fit_rows": int(len(tr)),
           "transform": "log1p then (x - mean) / scale; negatives clipped to 0 before log1p",
           "features": feats, "mean": sc.mean_.tolist(), "scale": sc.scale_.tolist()},
          open(SYN / f"scaler_{STEP}s.json", "w"), indent=2)
print("scaler fit on train rows:", len(tr))

# ---- window-size sensitivity on the real baselines ---------------------------
rows = []
for day in CFG["splits"].values():
    r = pd.read_csv(PROC / f"{day}_baseline_{STEP}s.csv")["requests"].to_numpy()
    for step in [5, 10, 30, 60, 300]:
        k = step // STEP
        x = r[: len(r) // k * k].reshape(-1, k).sum(1)
        rows.append({"day": day, "step_s": step, "rows": len(x), "cv": x.std() / x.mean(),
                     "lag1_acf": np.corrcoef(x[:-1], x[1:])[0, 1],
                     "median_abs_diff_over_median": np.median(np.abs(np.diff(x))) / np.median(x),
                     "rows_per_5min_attack": 300 // step})
pd.DataFrame(rows).to_csv(RES / "08_window_sensitivity.csv", index=False)

# ---- figure: one example per family (train day, seed 0) -------------------------
fams = [f for f in dict.fromkeys(c[0] for c in cfg_list)]
fig, axes = plt.subplots(len(fams), 1, figsize=(11, 2.1 * len(fams)), sharex=False)
for ax, fam in zip(axes, fams):
    s = tr[(tr["a_family"] == fam) & (tr["a_seed_rep"] == 0)]
    s = s[s["a_series"] == s["a_series"].iloc[len(s) // 2]]
    o = int(s["a_onset_bin"].iloc[0])
    w = s.iloc[max(o - 360, 0): o + 1080]
    t = (w["a_bin"] - o) * STEP / 60
    ax.plot(t, w["x_syn"], lw=0.7, label="observed SYN (x_syn)")
    ax.plot(t, w["x_requests"], lw=0.7, label="observed requests (x_requests)")
    ax.plot(t, w["y_legit_requests"], lw=1.0, ls="--", c="k", label="true legitimate (y_legit_requests)")
    ax.set_title(f"{fam}  [{s['a_series'].iloc[0]}]  SYNTHETIC surge/attack on real WorldCup98 baseline",
                 fontsize=8)
    ax.set_ylabel(f"per {STEP}s", fontsize=7)
axes[0].legend(fontsize=7, loc="upper right")
axes[-1].set_xlabel("minutes from event onset")
fig.tight_layout()
fig.savefig(RES / "08_scenarios.png", dpi=130)
print("wrote", SYN, "and", RES / "08_scenarios.png")
