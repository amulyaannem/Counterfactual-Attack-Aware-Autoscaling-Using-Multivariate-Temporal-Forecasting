"""Phase 9: counterfactual legitimate-demand estimation (forecast -> residual -> attribution -> estimate).

  python src/models/13_counterfactual.py detect     # fit detectors and the legit-share regressor, cache scores
  python src/models/13_counterfactual.py evaluate   # choose span/variant on val, report on test, save estimates

Detector: the Phase 7 RandomForest (multivariate lag features), refit here so val scores exist for tuning; a
requests-only RandomForest is fitted too for the Phase 11 univariate ablation. Same splits and protocol as Phase 7.
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestClassifier

from common import CFG, ROOT, SEED, STEP, SYN, best_f1_threshold, features, load, wide
from counterfactual import estimate_legit

OUT = ROOT / CFG["paths"]["results"] / "phase9"
OUT.mkdir(parents=True, exist_ok=True)
SPLITS = ["train", "val", "test"]
VARIANTS = ["freeze", "client", "learned"]
SPANS = [12, 60, 240, 720]


def detect(_):
    data = {s: load(s) for s in SPLITS}
    X = {s: features(*data[s], 0).astype(np.float32) for s in SPLITS}
    y_tr = data["train"][0]["y_attack_active"].to_numpy()
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(y_tr), 400_000, replace=False)
    cols = {"rf_all": list(X["train"].columns), "rf_req": [c for c in X["train"].columns if c.startswith("x_requests")]}
    out = {s: pd.DataFrame(index=range(len(X[s]))) for s in ["val", "test"]}
    thr = {}
    for name, c in cols.items():
        t0 = time.perf_counter()
        rf = RandomForestClassifier(n_estimators=200, max_depth=14, min_samples_leaf=20, n_jobs=-1, random_state=SEED)
        rf.fit(X["train"][c].iloc[idx], y_tr[idx])
        for s in out:
            out[s][name] = rf.predict_proba(X[s][c])[:, 1].astype(np.float32)
        va = data["val"][0]
        thr[name], f1 = best_f1_threshold(va["y_attack_active"].to_numpy()[va["a_eval_mask"] == 1],
                                          out["val"][name].to_numpy()[va["a_eval_mask"] == 1])
        print(f"{name}: {len(c)} features, val F1 {f1:.3f}, fit {time.perf_counter() - t0:.0f}s", flush=True)
    # legitimate share of observed requests, learned on train (truth known by construction); attack windows only
    tr = data["train"][0]
    att = tr["y_attack_active"].to_numpy() == 1
    share = (tr["y_legit_requests"] / tr["x_requests"].clip(lower=1)).to_numpy()
    t0 = time.perf_counter()
    reg = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.1, early_stopping=False, random_state=SEED)
    reg.fit(X["train"][att], share[att])
    for s in out:
        out[s]["share"] = np.clip(reg.predict(X[s]), 0, 1).astype(np.float32)
    print(f"legit-share regressor: {att.sum()} attack windows, fit {time.perf_counter() - t0:.0f}s")
    for s, d in out.items():
        d.to_parquet(SYN / f"phase9_scores_{s}_{STEP}s.parquet", index=False)
    json.dump(thr, open(SYN / f"phase9_thresholds_{STEP}s.json", "w"), indent=1)


def arrays(split, det="rf_all", flag_source="detector"):
    df, n_s = load(split)
    sc = pd.read_parquet(SYN / f"phase9_scores_{split}_{STEP}s.parquet")
    thr = json.load(open(SYN / f"phase9_thresholds_{STEP}s.json"))[det]
    T = lambda v: np.asarray(v).reshape(n_s, -1).T  # (time, series)
    flag = T(df["y_attack_active"].to_numpy() == 1) if flag_source == "oracle" else T(sc[det].to_numpy() >= thr)
    return df, n_s, wide(df, n_s, "x_requests").to_numpy(), wide(df, n_s, "x_clients").to_numpy(), flag, T(sc["share"])


def errors(df, est_flat, name):
    """MAE of the legitimate-demand estimate against the truth, overall and by window type and family."""
    y, obs = df["y_legit_requests"].to_numpy(), df["x_requests"].to_numpy()
    e = np.abs(est_flat - y)
    att, surge = df["y_attack_active"].to_numpy() == 1, df["y_surge_active"].to_numpy() == 1
    app = att & (df["a_attack_type"].astype(str).to_numpy() == "app")
    row = {"estimator": name, "mae_all": e.mean(), "mae_attack": e[att].mean(), "mae_app_attack": e[app].mean(),
           "mae_surge_no_attack": e[surge & ~att].mean(), "mae_normal": e[~att & ~surge].mean(),
           "bias_attack": (est_flat - y)[att].mean(), "rel_err_app_attack": (e[app] / y[app]).mean(),
           "legit_underestimated_surge": np.clip(y - est_flat, 0, None)[surge & ~att].sum() / y[surge & ~att].sum()}
    fam = df["a_family"].astype(str) + " · " + df["a_attack_type"].astype(str)
    for f in sorted(fam[att].unique()):
        k = att & (fam == f).to_numpy()
        row[f"mae[{f}]"] = e[k].mean()
    return row


def flat(a):
    return a.T.ravel()


def evaluate(_):
    va, n_va, xr, xc, flag, share = arrays("val")
    grid = []
    for v in VARIANTS:
        for span in (SPANS if v != "learned" else [0]):
            est, _ = estimate_legit(xr, xc, flag, span, v, share)
            r = errors(va, flat(est), f"{v}[span {span}]")
            grid.append({"variant": v, "span": span, **{k: r[k] for k in ["mae_all", "mae_attack", "mae_surge_no_attack"]}})
    grid = pd.DataFrame(grid)
    grid.to_csv(OUT / "val_grid.csv", index=False)
    print(grid.round(2).to_string(index=False))
    best = {v: grid[grid.variant == v].sort_values("mae_attack").iloc[0] for v in VARIANTS}
    main = min(best.values(), key=lambda r: r["mae_attack"])
    print(f"\nchosen on val (lowest MAE in attack windows): {main['variant']}, span {main['span']}")

    te, n_te, xr, xc, flag, share = arrays("test")
    _, _, _, _, oflag, _ = arrays("test", flag_source="oracle")
    rows = [errors(te, te["x_requests"].to_numpy(), "reactive (observed = legit)")]
    keep = {}
    for v in VARIANTS:
        b = best[v]
        est, fc = estimate_legit(xr, xc, flag, int(b["span"]), v, share)
        rows.append(errors(te, flat(est), f"{v}[span {int(b['span'])}], RF flags"))
        keep[v] = est
        est_o, _ = estimate_legit(xr, xc, oflag, int(b["span"]), v, share)
        rows.append(errors(te, flat(est_o), f"{v}[span {int(b['span'])}], true flags"))
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "counterfactual_test.csv", index=False)
    pd.set_option("display.width", 250)
    print(res[["estimator", "mae_all", "mae_attack", "mae_app_attack", "mae_surge_no_attack", "bias_attack",
               "legit_underestimated_surge"]].round(3).to_string(index=False))
    print("\nMAE by attack family:")
    print(res.set_index("estimator")[[c for c in res.columns if c.startswith("mae[")]].round(1).T.to_string())
    pd.DataFrame({"flag": flat(flag), **{f"legit_{v}": flat(e).astype(np.float32) for v, e in keep.items()}}) \
        .to_parquet(SYN / f"phase9_estimates_test_{STEP}s.parquet", index=False)
    json.dump({"variant": main["variant"], "span": int(main["span"]),
               "spans": {v: int(best[v]["span"]) for v in VARIANTS}}, open(OUT / "chosen.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("detect").set_defaults(fn=detect)
    sub.add_parser("evaluate").set_defaults(fn=evaluate)
    a = ap.parse_args()
    a.fn(a)
