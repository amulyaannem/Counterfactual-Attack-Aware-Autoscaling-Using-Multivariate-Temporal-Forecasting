"""Phase 7: CPU-only baselines on the Phase 6 benchmark.

Detection task: flag attack-active 5-s windows (y_attack_active) from x_ counters and their past.
Forecast task: predict x_requests h steps ahead (h = 1 -> 5 s, h = 12 -> 1 min); also report the error against the
true legitimate demand (y_legit_requests), the counterfactual the project cares about.
Protocol: fit on train (TR2), choose thresholds / hyper-parameters on val (TR1), report on test (TR3).
Infeasible here: seasonal-naive (one day per series, no previous day) and ARIMA (EWMA = optimal ARIMA(0,1,1) forecast;
statsmodels not installed). XGBoost not installed: sklearn HistGradientBoosting is the boosted-tree baseline.
"""
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import precision_recall_curve

ROOT = Path(__file__).resolve().parents[2]
CFG = yaml.safe_load(open(ROOT / "config.yaml"))
SYN = ROOT / CFG["paths"]["synthetic"]
OUT = ROOT / CFG["paths"]["results"] / "phase7"
OUT.mkdir(parents=True, exist_ok=True)
STEP, SEED = CFG["step_s"], CFG["seed"]
BASE = ["x_requests", "x_syn", "x_ack", "x_clients"]
RUN_META = {
    "git_commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                                 text=True).stdout.strip(),
    "config_sha256": hashlib.sha256(open(ROOT / "config.yaml", "rb").read()).hexdigest()[:16],
    "seed": SEED,
}
LOG = open(OUT / "runs.jsonl", "w")


def log(rec):
    LOG.write(json.dumps({**RUN_META, **rec}, default=float) + "\n")
    LOG.flush()


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


def best_f1_threshold(y, s):
    s = np.nan_to_num(s, nan=np.nanmin(s) - 1)  # no history yet -> lowest score, never flagged first
    p, r, t = precision_recall_curve(y, s)
    f = 2 * p * r / np.maximum(p + r, 1e-12)
    i = int(np.nanargmax(f[:-1]))
    return float(t[i]), float(f[i])


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


data = {s: load(s) for s in ["train", "val", "test"]}
det_rows, fam_rows, fc_rows = [], [], []


def report(name, kind, params, s_val, s_test, fit_s, infer_s):
    yv = data["val"][0]["y_attack_active"].to_numpy()
    mv = data["val"][0]["a_eval_mask"].to_numpy() == 1
    thr, f1v = best_f1_threshold(yv[mv], s_val[mv])
    te, n_te = data["test"]
    met = detection_metrics(te, n_te, np.nan_to_num(s_test, nan=-np.inf) >= thr)
    rec = {"task": "detection", "model": name, "kind": kind, "params": params, "threshold": thr, "val_f1": f1v,
           "fit_s": fit_s, "infer_us_per_window": infer_s / len(te) * 1e6, **met}
    log(rec)
    det_rows.append({k: v for k, v in rec.items() if k != "params"} | {"params": json.dumps(params)})
    fam_rows.extend(by_family(te, np.nan_to_num(s_test, nan=-np.inf) >= thr, name))
    print(f"{name:<28} F1 {met['f1']:.3f}  F1-no-mimic {met['f1_no_mimic']:.3f}  P {met['precision']:.3f}  "
          f"R {met['recall']:.3f}  FPR {met['fpr']:.4f}  "
          f"event-recall {met['event_recall']:.3f}  delay {met['median_delay_s']}s  surge-FPR "
          f"{met['fpr_during_legit_surge']:.4f}")


# ---- 1. static thresholds on single counters --------------------------------------
print("== detection (threshold chosen on val by max F1, metrics on test)")
static = {
    "x_syn_minus_ack": lambda d: d["x_syn_minus_ack"].to_numpy(float),
    "x_syn": lambda d: d["x_syn"].to_numpy(float),
    "x_requests": lambda d: d["x_requests"].to_numpy(float),
    "syn_per_req": lambda d: (d["x_syn"] / d["x_requests"].clip(lower=1)).to_numpy(float),
}
for name, fn in static.items():
    t0 = time.perf_counter(); s_te = fn(data["test"][0]); dt = time.perf_counter() - t0
    report(f"static[{name}]", "static_threshold", {"feature": name}, fn(data["val"][0]), s_te, 0.0, dt)
te, n_te = data["test"]
met = detection_metrics(te, n_te, te["x_syn_minus_ack"].to_numpy() > 0)
log({"task": "detection", "model": "static[TSD rule SYN-ACK>0]", "kind": "fixed_rule", **met})
det_rows.append({"model": "static[TSD rule SYN-ACK>0]", "kind": "fixed_rule", "threshold": 0, **met})
print(f"{'static[TSD rule SYN-ACK>0]':<28} F1 {met['f1']:.3f}  R {met['recall']:.3f}  FPR {met['fpr']:.4f}")


# ---- 2. EWMA residual detectors (span chosen on val) ----------------------------
def ewma_z(df, n_s, col, span):
    w = wide(df, n_s, col)
    past = w.shift(1)
    m, sd = past.ewm(span=span, min_periods=12).mean(), past.ewm(span=span, min_periods=12).std()
    return flat((w - m) / (sd + 1))


for col in ["x_syn", "x_requests", "x_syn_minus_ack"]:
    best = None
    for span in [60, 360, 1440]:
        sv = ewma_z(*data["val"], col, span)
        mv = data["val"][0]["a_eval_mask"].to_numpy() == 1
        f1 = best_f1_threshold(data["val"][0]["y_attack_active"].to_numpy()[mv], sv[mv])[1]
        if best is None or f1 > best[1]:
            best = (span, f1, sv)
    t0 = time.perf_counter(); s_te = ewma_z(*data["test"], col, best[0]); dt = time.perf_counter() - t0
    report(f"ewma_z[{col}]", "ewma_residual", {"feature": col, "span_bins": best[0]}, best[2], s_te, 0.0, dt)

# ---- 3. tree models on lagged features ------------------------------------------
X = {s: features(*data[s], 0) for s in data}
y_tr = data["train"][0]["y_attack_active"].to_numpy()
print(f"lag features: {X['train'].shape[1]}")
models = {
    "hgb[lag features]": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, max_leaf_nodes=63,
                                                        early_stopping=True, random_state=SEED),
    "rf[lag features]": RandomForestClassifier(n_estimators=200, max_depth=14, min_samples_leaf=20, n_jobs=-1,
                                               random_state=SEED),
}
rng = np.random.default_rng(SEED)
for name, mdl in models.items():
    idx = np.arange(len(y_tr)) if name.startswith("hgb") else rng.choice(len(y_tr), 400_000, replace=False)
    t0 = time.perf_counter(); mdl.fit(X["train"].iloc[idx], y_tr[idx]); fit_s = time.perf_counter() - t0
    sv = mdl.predict_proba(X["val"])[:, 1]
    t0 = time.perf_counter(); s_te = mdl.predict_proba(X["test"])[:, 1]; dt = time.perf_counter() - t0
    params = {k: v for k, v in mdl.get_params().items() if k in
              ("max_iter", "learning_rate", "max_leaf_nodes", "n_estimators", "max_depth", "min_samples_leaf")}
    params["train_rows"] = int(len(idx))
    report(name, "tree_lagged", params, sv, s_te, fit_s, dt)

pd.DataFrame(det_rows).to_csv(OUT / "detection_test.csv", index=False)
pd.DataFrame(fam_rows).to_csv(OUT / "detection_by_family_test.csv", index=False)


# ---- 4. forecasting x_requests -------------------------------------------------
def fc_metrics(df, pred, h, name, params, fit_s, infer_s):
    m = (df["a_eval_mask"].to_numpy() == 1) & ~np.isnan(pred)
    obs, legit = df["x_requests"].to_numpy(float), df["y_legit_requests"].to_numpy(float)
    calm = (df["y_attack_active"].to_numpy() == 0) & (df["y_surge_active"].to_numpy() == 0)
    att = df["y_attack_active"].to_numpy() == 1
    e = pred - obs
    rec = {"task": "forecast", "model": name, "horizon_steps": h, "horizon_s": h * STEP, "params": params,
           "mae": np.abs(e[m]).mean(), "rmse": np.sqrt((e[m] ** 2).mean()),
           "mae_calm": np.abs(e[m & calm]).mean(), "mae_during_attack": np.abs(e[m & att]).mean(),
           "mae_vs_true_legit_during_attack": np.abs(pred - legit)[m & att].mean(),
           "fit_s": fit_s, "infer_us_per_window": infer_s / len(df) * 1e6}
    log(rec)
    fc_rows.append({k: v for k, v in rec.items() if k != "params"} | {"params": json.dumps(params)})
    print(f"h={h:<3} {name:<22} MAE {rec['mae']:.2f}  RMSE {rec['rmse']:.2f}  calm {rec['mae_calm']:.2f}  "
          f"attack(vs obs) {rec['mae_during_attack']:.2f}  attack(vs true legit) "
          f"{rec['mae_vs_true_legit_during_attack']:.2f}")


print("\n== forecast of x_requests (test)")
te, n_te = data["test"]
legit = te["y_legit_requests"].to_numpy(float)
att = (te["y_attack_active"].to_numpy() == 1) & (te["a_eval_mask"].to_numpy() == 1)
ref = np.abs(te["x_requests"].to_numpy(float) - legit)[att].mean()
log({"task": "forecast", "model": "reactive (L_hat = observed)", "mae_vs_true_legit_during_attack": ref})
fc_rows.append({"model": "reactive (L_hat = observed)", "horizon_steps": 0, "mae_vs_true_legit_during_attack": ref})
print(f"reactive reference: |observed - true legit| during attack = {ref:.2f}")
for h in [1, 12]:
    w = {s: wide(*data[s], "x_requests") for s in data}
    fc_metrics(te, flat(w["test"].shift(h)), h, "naive_last", {}, 0, 0)
    fc_metrics(te, flat(w["test"].shift(h).rolling(12, min_periods=6).mean()), h, "trailing_mean_12", {}, 0, 0)
    vb = data["val"][0]
    best = min([6, 12, 60], key=lambda sp: np.nanmean(np.abs(flat(w["val"].shift(h).ewm(span=sp).mean())
                                                             - vb["x_requests"].to_numpy(float))))
    fc_metrics(te, flat(w["test"].shift(h).ewm(span=best).mean()), h, "ewma", {"span_bins": best}, 0, 0)
    Xh = {s: features(*data[s], h) for s in ["train", "test"]}
    reg = HistGradientBoostingRegressor(max_iter=300, max_leaf_nodes=63, early_stopping=True, random_state=SEED)
    t0 = time.perf_counter(); reg.fit(Xh["train"], data["train"][0]["x_requests"].to_numpy(float))
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter(); p = reg.predict(Xh["test"]); dt = time.perf_counter() - t0
    fc_metrics(te, p, h, "hgb_regressor", {"max_iter": 300, "max_leaf_nodes": 63}, fit_s, dt)

pd.DataFrame(fc_rows).to_csv(OUT / "forecast_test.csv", index=False)
LOG.close()
print("\nwrote", OUT)
