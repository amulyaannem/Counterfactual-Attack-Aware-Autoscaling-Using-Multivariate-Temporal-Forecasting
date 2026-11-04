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

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier

from common import (CFG, ROOT, SEED, STEP, SYN, best_f1_threshold, by_family, detection_metrics, features, flat,
                    level_free, load, wide)

OUT = ROOT / CFG["paths"]["results"] / "phase7"
OUT.mkdir(parents=True, exist_ok=True)
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


data = {s: load(s) for s in ["train", "val", "test"]}
det_rows, fam_rows, fc_rows = [], [], []
test_scores, test_thresholds = {}, {}  # per-window test scores, for figures


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
    test_scores[name], test_thresholds[name] = np.asarray(s_test, dtype=np.float32), thr
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
X = {s: features(*data[s], 0).astype(np.float32) for s in data}  # float32: 3 splits x 84 cols fit in RAM
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

# Prerequisites logged after the first Phase 7 run: (a) HGB's built-in early stopping holds out random rows of
# autocorrelated series, so choose the number of trees on the val day instead; (b) drop absolute-level features,
# because the train/val/test days differ in traffic level.
LF = level_free(X["train"].columns)
print(f"level-free features: {len(LF)}")
yv, mv = data["val"][0]["y_attack_active"].to_numpy(), data["val"][0]["a_eval_mask"].to_numpy() == 1
for cols, tag in [(list(X["train"].columns), "all"), (LF, "level-free")]:
    name = f"hgb[val-tuned trees, {tag}]"
    t0 = time.perf_counter()
    hgb = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.1, max_leaf_nodes=63, early_stopping=False,
                                         random_state=SEED).fit(X["train"][cols], y_tr)
    fit_s = time.perf_counter() - t0
    best = (0, -1.0)
    for it, p in enumerate(hgb.staged_predict_proba(X["val"][cols]), start=1):
        if it % 25 == 0:
            f1 = best_f1_threshold(yv[mv], p[mv, 1])[1]
            if f1 > best[1]:
                best, sv = (it, f1), p[:, 1]
    test_iter = hgb.staged_predict_proba(X["test"][cols])
    t0 = time.perf_counter()
    for it, p in enumerate(test_iter, start=1):
        if it == best[0]:
            s_te = p[:, 1]
            break
    dt = time.perf_counter() - t0
    report(name, "tree_lagged", {"trees_chosen_on_val": best[0], "max_leaf_nodes": 63, "features": tag,
                                 "n_features": len(cols)}, sv, s_te, fit_s, dt)
rf = RandomForestClassifier(n_estimators=200, max_depth=14, min_samples_leaf=20, n_jobs=-1, random_state=SEED)
idx = rng.choice(len(y_tr), 400_000, replace=False)
t0 = time.perf_counter(); rf.fit(X["train"][LF].iloc[idx], y_tr[idx]); fit_s = time.perf_counter() - t0
sv = rf.predict_proba(X["val"][LF])[:, 1]
t0 = time.perf_counter(); s_te = rf.predict_proba(X["test"][LF])[:, 1]; dt = time.perf_counter() - t0
report("rf[level-free]", "tree_lagged", {"n_estimators": 200, "max_depth": 14, "min_samples_leaf": 20,
                                         "train_rows": 400_000, "features": "level-free", "n_features": len(LF)},
       sv, s_te, fit_s, dt)

pd.DataFrame(test_scores).to_parquet(SYN / f"phase7_test_scores_{STEP}s.parquet", index=False)
json.dump(test_thresholds, open(SYN / f"phase7_test_thresholds_{STEP}s.json", "w"), indent=2)
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
    Xh = {s: features(*data[s], h).astype(np.float32) for s in ["train", "test"]}
    reg = HistGradientBoostingRegressor(max_iter=300, max_leaf_nodes=63, early_stopping=True, random_state=SEED)
    t0 = time.perf_counter(); reg.fit(Xh["train"], data["train"][0]["x_requests"].to_numpy(float))
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter(); p = reg.predict(Xh["test"]); dt = time.perf_counter() - t0
    fc_metrics(te, p, h, "hgb_regressor", {"max_iter": 300, "max_leaf_nodes": 63}, fit_s, dt)

pd.DataFrame(fc_rows).to_csv(OUT / "forecast_test.csv", index=False)
LOG.close()
print("\nwrote", OUT)
