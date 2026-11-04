"""Phase 11: ablations and robustness (SIMULATED scaling results, test day TR3).

Each component of the Phase 9/10 method is removed in turn; estimation error and simulated scaling outcomes are
reported for every variant. Robustness: attack intensity (rho), surge size (peak ratio) and the simulator's own
assumptions (start-up delay, SYN cost, target utilisation).

  python src/sim/15_ablations.py
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "models"))
from common import CFG, ROOT, STEP, SYN  # noqa: E402
from counterfactual import estimate_legit  # noqa: E402


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


sim = module(HERE / "14_simulate.py", "simulate14")
cf = module(HERE.parent / "models" / "13_counterfactual.py", "counterfactual13")
OUT = ROOT / CFG["paths"]["results"] / "phase11"
OUT.mkdir(parents=True, exist_ok=True)
pd.set_option("display.width", 250)


def residual_flags(x_req, span, k=3.0):
    """Unsupervised flag: observed requests more than k trailing std above an EWMA forecast (no detector)."""
    w = pd.DataFrame(x_req)
    f = w.ewm(span=span, adjust=False).mean().shift(1)
    sd = w.diff().rolling(60, min_periods=30).std().shift(1)
    return ((w - f) / (sd + 1) > k).to_numpy()


def main():
    chosen = json.load(open(ROOT / CFG["paths"]["results"] / "phase9" / "chosen.json"))
    v, span = chosen["variant"], chosen["span"]
    ctx = sim.context()
    te, T = ctx["df"], ctx["T"]
    sc = pd.read_parquet(SYN / f"phase9_scores_test_{STEP}s.parquet")
    thr = json.load(open(SYN / f"phase9_thresholds_{STEP}s.json"))
    x_req, x_cl = ctx["x_req"], T(te["x_clients"].to_numpy())
    share = T(sc["share"])
    p_all = T(sc["rf_all"])

    # full method = the estimator chosen on val; "freeze" = forecast-residual attribution (all positive residual
    # in a flagged window is attack); residual-z flags = no supervised detector at all
    fs = chosen["spans"]["freeze"]
    uni = T(sc["rf_req"].to_numpy() >= thr["rf_req"])
    rz = residual_flags(x_req, fs)
    rows = {
        "full method": (ctx["flag"], v, span),
        "univariate detector (requests only)": (uni, v, span),
        "forecast-residual attribution (freeze)": (ctx["flag"], "freeze", fs),
        "no detector (residual z > 3 flags, freeze)": (rz, "freeze", fs),
        "perfect detection (true flags)": (ctx["att"], v, span),
        "perfect detection, forecast-residual": (ctx["att"], "freeze", fs),
    }
    flags = {n: r[0] for n, r in rows.items()}
    est = {n: estimate_legit(x_req, x_cl, f, sp, var, share)[0] for n, (f, var, sp) in rows.items()}
    est["no learned share (legit share = 1 - score)"] = np.where(ctx["flag"], (1 - p_all) * x_req, x_req)
    flags["no learned share (legit share = 1 - score)"] = ctx["flag"]

    sims = sim.baselines(ctx)
    for n in est:
        sims[n] = sim.ours(ctx, flags[n], est[n])
    sims["no MITIGATE (scale on estimate only)"] = sim.ours(ctx, ctx["flag"], est["full method"], mitigate=False)
    sims["MITIGATE, scale on observed"] = sim.ours(ctx, ctx["flag"], est["full method"], scale_on_estimate=False)
    res, fam = sim.summarize(sims, ctx)
    err = pd.DataFrame([cf.errors(te, cf.flat(e), n) for n, e in est.items()] +
                       [cf.errors(te, te["x_requests"].to_numpy(), "reactive")])
    err = err[["estimator", "mae_attack", "mae_app_attack", "mae_surge_no_attack", "legit_underestimated_surge"]]
    tab = res.merge(err, left_on="policy", right_on="estimator", how="left").drop(columns="estimator")
    tab.to_csv(OUT / "ablations_test.csv", index=False)
    fam.to_csv(OUT / "ablations_by_family_test.csv", index=False)
    cols = ["policy", "mae_attack", "mae_surge_no_attack", "replica_hours_per_series", "unneeded_scaleups_per_series",
            "sla_violation_attack_pct", "legit_lost_overload_pct", "legit_wrongly_mitigated_pct",
            "legit_wrongly_mitigated_surge_pct"]
    print("ABLATIONS (simulated scaling, test day):")
    print(tab[cols].round(3).to_string(index=False))

    # robustness across scenario parameters: per-series grouping by attack intensity and surge size
    rows = []
    rho, peak = T(te["a_rho"].to_numpy())[0], T(te["a_peak_ratio"].to_numpy())[0]
    for key, vals in [("rho", rho), ("peak_ratio", peak)]:
        for val in sorted(set(vals[~np.isnan(vals)])):
            k = vals == val
            for n in ["reactive", "naive predictive (Holt on observed)", "full method"]:
                s = {kk: vv[:, k] for kk, vv in sims[n].items()}
                m = sim.metrics(s, ctx["legit"][:, k], ctx["att"][:, k], ctx["surge"][:, k],
                                sims["oracle (true legit demand)"]["R"][:, k], n)
                e = np.abs(est["full method"][:, k] - ctx["legit"][:, k])[ctx["att"][:, k]].mean() if n == "full method" \
                    else np.nan
                rows.append({"param": key, "value": val, "series": int(k.sum()), "mae_attack": e, **m})
    rob = pd.DataFrame(rows)
    rob.to_csv(OUT / "robustness_scenarios_test.csv", index=False)
    print("\nROBUSTNESS by scenario parameter:")
    print(rob[["param", "value", "series", "policy", "mae_attack", "replica_hours_per_series",
               "sla_violation_attack_pct", "legit_lost_overload_pct", "legit_wrongly_mitigated_pct"]]
          .round(3).to_string(index=False))

    # sensitivity to the simulator's assumptions
    base = dict(sim.SIM)
    rows = []
    for param, values in [("startup_windows", [6, 12, 24]), ("syn_cost", [0.0, 0.2, 0.5]),
                          ("target_util", [0.5, 0.6, 0.75])]:
        for val in values:
            sim.SIM.update(base)
            sim.SIM[param] = val
            sim.H = sim.SIM["startup_windows"] + 1
            ctx["c"] = 1 + sim.SIM["syn_cost"]
            ctx["obs"] = ctx["x_req"] + sim.SIM["syn_cost"] * ctx["x_syn"]
            ss = sim.baselines(ctx)
            ss["full method"] = sim.ours(ctx, ctx["flag"], est["full method"])
            r, _ = sim.summarize(ss, ctx)
            orc = r.loc[r.policy.str.startswith("oracle"), "replica_hours_per_series"].iloc[0]
            for _, x in r.iterrows():
                rows.append({"param": param, "value": val, "policy": x["policy"],
                             "replica_hours_vs_oracle": x["replica_hours_per_series"] / orc,
                             "sla_violation_pct": x["sla_violation_pct"],
                             "sla_violation_attack_pct": x["sla_violation_attack_pct"],
                             "legit_lost_overload_pct": x["legit_lost_overload_pct"],
                             "legit_wrongly_mitigated_pct": x["legit_wrongly_mitigated_pct"]})
    sim.SIM.update(base)
    sens = pd.DataFrame(rows)
    sens.to_csv(OUT / "sensitivity_simulator_test.csv", index=False)
    print("\nSENSITIVITY to simulator assumptions:")
    print(sens.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
