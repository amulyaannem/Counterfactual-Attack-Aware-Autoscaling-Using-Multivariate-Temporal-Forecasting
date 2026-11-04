"""Phase 10: SIMULATED scaling evaluation. Every number this script writes is simulated, never measured.

Capacity model (config.yaml `sim`): each replica serves `req_per_replica` work units per 5 s window; a request costs 1
unit and a SYN `syn_cost` units. Replicas take `startup_windows` to become ready and are billed from the request.
Scale-down follows the maximum recommendation of the last `stabilization_windows` (Kubernetes-HPA style). Latency:
utilisation above `sla_util` breaks the SLA; work above capacity is dropped proportionally (legitimate share lost).
Mitigation is an idealised filter that drops attack traffic before legitimate traffic, admitting at most the
estimated legitimate requests x (1 + margin); legitimate requests above that budget are "wrongly mitigated".

Policies: reactive (last observed load), naive predictive (Holt forecast of observed load), ours (Phase 9 legitimate
estimate + MITIGATE on detection), oracle (true legitimate demand, perfect look-ahead and mitigation).

  python src/sim/14_simulate.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))
from common import CFG, ROOT, STEP, SYN, load, wide  # noqa: E402
from counterfactual import ACTIONS, action, holt  # noqa: E402

SIM = CFG["sim"]
OUT = ROOT / CFG["paths"]["results"] / "phase10"
OUT.mkdir(parents=True, exist_ok=True)
H = SIM["startup_windows"] + 1  # a decision at t is ready at t + startup + 1


def shift(a, k=1, fill=None):
    out = np.empty_like(a)
    out[k:] = a[:-k]
    out[:k] = a[0] if fill is None else fill
    return out


def replicas(D):
    """Requested and ready replicas from a work estimate D (time, series); the decision at t uses D[t-1]."""
    want = np.clip(np.ceil(shift(D) / (SIM["req_per_replica"] * SIM["target_util"])),
                   SIM["min_replicas"], SIM["max_replicas"])
    R = pd.DataFrame(want).rolling(SIM["stabilization_windows"], min_periods=1).max().to_numpy()
    ready = pd.DataFrame(R).rolling(H, min_periods=1).min().to_numpy()
    return want, R, ready


def simulate(D, x_req, x_syn, legit, mitigate=None, budget=None):
    want, R, ready = replicas(D)
    if mitigate is None:
        adm_req, adm_syn = x_req.astype(float), x_syn.astype(float)
    else:
        cap_req = budget * (1 + SIM["mitigation_margin"])
        adm_req = np.where(mitigate, np.minimum(x_req, cap_req), x_req)
        adm_syn = np.where(mitigate, np.minimum(x_syn, adm_req * (1 + SIM["mitigation_margin"])), x_syn)
    legit_adm = np.minimum(legit, adm_req)
    work = adm_req + SIM["syn_cost"] * adm_syn
    cap = ready * SIM["req_per_replica"]
    util = work / cap
    lost = legit_adm * np.clip(1 - cap / np.maximum(work, 1e-9), 0, 1)
    return {"want": want, "R": R, "ready": ready, "util": util, "mitigated": legit - legit_adm, "lost": lost,
            "mitigating": np.zeros_like(x_req, bool) if mitigate is None else mitigate}


def mean(a):
    return a.mean() if a.size else np.nan


def metrics(sim, legit, att, surge, R_ref, name):
    hours = sim["R"].sum(0) * STEP / 3600
    up = np.diff(sim["R"], axis=0, prepend=sim["R"][:1]) > 0
    viol = sim["util"] > SIM["sla_util"]
    L = legit.sum()
    sl = surge & ~att
    return {"policy": name, "replica_hours_per_series": hours.mean(),
            "excess_replica_hours_per_series": (np.clip(sim["R"] - R_ref, 0, None).sum(0) * STEP / 3600).mean(),
            "scaleups_per_series": up.sum(0).mean(),
            "unneeded_scaleups_per_series": (up & (sim["R"] > R_ref)).sum(0).mean(),
            "sla_violation_pct": 100 * viol.mean(), "sla_violation_attack_pct": 100 * mean(viol[att]),
            "sla_violation_surge_pct": 100 * mean(viol[sl]),
            "legit_lost_overload_pct": 100 * sim["lost"].sum() / L,
            "legit_wrongly_mitigated_pct": 100 * sim["mitigated"].sum() / L,
            "legit_wrongly_mitigated_surge_pct": 100 * sim["mitigated"][sl].sum() / legit[sl].sum() if sl.any() else np.nan,
            "mitigating_pct": 100 * sim["mitigating"].mean()}


def context(split="test"):
    """Arrays (time, series) shared by every policy, plus the Phase 9 estimate chosen on val."""
    df, n_s = load(split)
    T = lambda v: np.asarray(v).reshape(n_s, -1).T
    est = pd.read_parquet(SYN / f"phase9_estimates_{split}_{STEP}s.parquet")
    chosen = json.load(open(ROOT / CFG["paths"]["results"] / "phase9" / "chosen.json"))
    ctx = {"df": df, "T": T, "x_req": wide(df, n_s, "x_requests").to_numpy(), "x_syn": wide(df, n_s, "x_syn").to_numpy(),
           "legit": wide(df, n_s, "y_legit_requests").to_numpy(),
           "legit_syn": (wide(df, n_s, "x_syn") - wide(df, n_s, "y_attack_conn")).to_numpy(),
           "att": T(df["y_attack_active"].to_numpy() == 1), "surge": T(df["y_surge_active"].to_numpy() == 1),
           "fam": T((df["a_family"].astype(str) + " · " + df["a_attack_type"].astype(str)).to_numpy())[0],
           "flag": T(est["flag"].to_numpy()), "L_hat": T(est[f"legit_{chosen['variant']}"].to_numpy()),
           "c": 1 + SIM["syn_cost"]}  # estimated legitimate work per request (one connection per request)
    ctx["obs"] = ctx["x_req"] + SIM["syn_cost"] * ctx["x_syn"]
    return ctx


def baselines(ctx):
    x_req, x_syn, legit, obs = ctx["x_req"], ctx["x_syn"], ctx["legit"], ctx["obs"]
    lw = legit + SIM["syn_cost"] * ctx["legit_syn"]  # true legitimate work
    oracle_D = pd.DataFrame(lw)[::-1].rolling(H, min_periods=1).max()[::-1].shift(-1).ffill().to_numpy()
    return {
        "oracle (true legit demand)": simulate(oracle_D, x_req, x_syn, legit, np.ones_like(ctx["att"]), legit),
        "reactive": simulate(obs, x_req, x_syn, legit),
        "naive predictive (Holt on observed)": simulate(np.maximum(obs, holt(obs, H)), x_req, x_syn, legit),
    }


def ours(ctx, flag, L_hat, mitigate=True, scale_on_estimate=True):
    """The Phase 9 policy: scale on the forecast legitimate demand, MITIGATE while the detector fires."""
    # flagged windows: estimated legitimate work; otherwise the observed work (taken as legitimate)
    w = np.where(flag, L_hat * ctx["c"], ctx["obs"]) if scale_on_estimate else ctx["obs"]
    D = np.maximum(w, holt(w, H))
    if not mitigate:
        return simulate(D, ctx["x_req"], ctx["x_syn"], ctx["legit"])
    return simulate(D, ctx["x_req"], ctx["x_syn"], ctx["legit"], shift(flag, fill=False), shift(L_hat))


def summarize(sims, ctx):
    legit, att, surge, fam = ctx["legit"], ctx["att"], ctx["surge"], ctx["fam"]
    R_ref = sims["oracle (true legit demand)"]["R"]
    rows = [metrics(s, legit, att, surge, R_ref, n) for n, s in sims.items()]
    fam_rows = []
    for n, s in sims.items():
        for f in sorted(set(fam)):
            k = fam == f
            m = metrics({kk: v[:, k] for kk, v in s.items()}, legit[:, k], att[:, k], surge[:, k], R_ref[:, k], n)
            fam_rows.append({"family": f, **m})
    return pd.DataFrame(rows), pd.DataFrame(fam_rows)


if __name__ == "__main__":
    ctx = context()
    sims = baselines(ctx)
    sims["ours (counterfactual + MITIGATE)"] = o = ours(ctx, ctx["flag"], ctx["L_hat"])
    res, fam = summarize(sims, ctx)
    acts = action(shift(ctx["flag"], fill=False), o["R"], shift(o["R"]))
    acts = {a: float((acts == i).mean()) for i, a in enumerate(ACTIONS)}
    res.to_csv(OUT / "policies_test.csv", index=False)
    fam.to_csv(OUT / "policies_by_family_test.csv", index=False)
    json.dump(acts, open(OUT / "action_mix_ours.json", "w"), indent=1)
    pd.set_option("display.width", 250)
    print("SIMULATED results (test day TR3, 98 series):")
    print(res.round(3).to_string(index=False))
    print("\naction mix (ours):", {k: round(v, 4) for k, v in acts.items()})
    key = ["family", "policy", "replica_hours_per_series", "unneeded_scaleups_per_series", "sla_violation_attack_pct",
           "legit_lost_overload_pct", "legit_wrongly_mitigated_pct"]
    print("\nby family:")
    print(fam[key].round(2).to_string(index=False))
