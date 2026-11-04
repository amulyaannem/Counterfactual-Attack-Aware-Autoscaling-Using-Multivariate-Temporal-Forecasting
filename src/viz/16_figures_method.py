"""Figures for Phases 9-11: counterfactual estimation, SIMULATED scaling policies, ablations and robustness.

Run after 13_counterfactual.py, 14_simulate.py and 15_ablations.py. Same palette and style as 11_figures.py.
"""
import importlib.util
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "models"))
from common import CFG, ROOT, STEP  # noqa: E402

RES = ROOT / CFG["paths"]["results"]
OUT = RES / "figures"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "lines.linewidth": 1.5,
})
POLICY = {"oracle (true legit demand)": ("Oracle", MUTED), "reactive": ("Reactive", ORANGE),
          "naive predictive (Holt on observed)": ("Naive predictive", YELLOW),
          "ours (counterfactual + MITIGATE)": ("Ours", BLUE)}


def save(fig, name):
    fig.savefig(OUT / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


spec = importlib.util.spec_from_file_location("sim14", HERE.parent / "sim" / "14_simulate.py")
sim = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sim)

# ---- 11. counterfactual estimate and replicas on example series -----------------
ctx = sim.context()
sims = sim.baselines(ctx)
sims["ours (counterfactual + MITIGATE)"] = sim.ours(ctx, ctx["flag"], ctx["L_hat"])
df, T = ctx["df"], ctx["T"]
names = T(df["a_series"].astype(str).to_numpy())[0]
rho = T(df["a_rho"].to_numpy())[0]
picks = [next(i for i, n in enumerate(names) if "attack_only_app" in n and rho[i] == 3),
         next(i for i, n in enumerate(names) if "surge_plus_attack_app" in n and rho[i] == 3)]
fig, axes = plt.subplots(2, 2, figsize=(11, 5.6), sharex="col", gridspec_kw={"height_ratios": [1.3, 1]})
for col, k in enumerate(picks):
    att = np.flatnonzero(ctx["att"][:, k])
    lo, hi = max(att[0] - 360, 0), min(att[-1] + 480, len(att) and ctx["att"].shape[0])
    t = (np.arange(lo, hi) - att[0]) * STEP / 60
    ax = axes[0, col]
    fl = ctx["flag"][lo:hi, k]
    ax.fill_between(t, 0, 1, where=fl, transform=ax.get_xaxis_transform(), color=MAGENTA, alpha=0.18, lw=0,
                    label="detector fires (MITIGATE)")
    ax.plot(t, ctx["x_req"][lo:hi, k], c=ORANGE, lw=0.9, label="observed requests")
    ax.plot(t, ctx["legit"][lo:hi, k], c=INK, lw=1.0, ls="--", label="true legitimate (by construction)")
    ax.plot(t, ctx["L_hat"][lo:hi, k], c=BLUE, lw=1.1, label="estimated legitimate (Phase 9)")
    ax.set_title(["App-layer flood, ρ = 3", "Legitimate surge + app-layer flood, ρ = 3"][col])
    ax.set_ylabel("requests per 5 s")
    ax = axes[1, col]
    for n in ["oracle (true legit demand)", "reactive", "ours (counterfactual + MITIGATE)"]:
        lab, c = POLICY[n]
        ax.step(t, sims[n]["ready"][lo:hi, k], where="post", c=c, lw=1.2, label=lab)
    ax.set_ylabel("ready replicas (simulated)")
    ax.set_xlabel("minutes from attack onset")
axes[0, 0].legend(loc="upper right", fontsize=7.5)
axes[1, 0].legend(loc="upper right", fontsize=7.5)
fig.suptitle("Counterfactual demand and simulated replicas on two test series (synthetic surge and attack)",
             x=0.01, ha="left", fontsize=10.5, fontweight="bold")
fig.tight_layout()
save(fig, "11_counterfactual_example.png")

# ---- 12. estimation error by attack family --------------------------------------
cf = pd.read_csv(RES / "phase9" / "counterfactual_test.csv").set_index("estimator")
rows = {"Observed = legitimate (reactive)": "reactive (observed = legit)",
        "Ours, RF detector": next(i for i in cf.index if i.startswith("learned") and "RF flags" in i),
        "Ours, perfect detection": next(i for i in cf.index if i.startswith("learned") and "true flags" in i)}
fams = [c for c in cf.columns if c.startswith("mae[") and "syn" not in c]
fig, ax = plt.subplots(figsize=(9, 3.6))
x = np.arange(len(fams))
for j, ((lab, key), c) in enumerate(zip(rows.items(), [ORANGE, BLUE, AQUA])):
    ax.bar(x + (j - 1) * 0.27, cf.loc[key, fams], 0.25, color=c, label=lab)
ax.set_xticks(x, [f[4:-1].replace(" · app", "").replace("_", " ") for f in fams])
ax.set_ylabel("MAE vs true legitimate\n(requests per 5 s)")
ax.set_title("Legitimate-demand error during app-layer attacks, test day")
ax.legend(fontsize=8)
fig.tight_layout()
save(fig, "12_estimation_error.png")

# ---- 13. simulated policy comparison --------------------------------------------
pol = pd.read_csv(RES / "phase10" / "policies_test.csv").set_index("policy")
panels = [("excess_replica_hours_per_series", "Excess replica-hours\nvs oracle, per series-day"),
          ("sla_violation_attack_pct", "SLA-violating windows\nduring attacks (%)"),
          ("legit_lost_overload_pct", "Legitimate requests lost\nto overload (%)"),
          ("legit_wrongly_mitigated_pct", "Legitimate requests\nwrongly mitigated (%)")]
fig, axes = plt.subplots(1, 4, figsize=(11, 2.9))
order = [p for p in POLICY if p != "oracle (true legit demand)"]
for ax, (m, title) in zip(axes, panels):
    v = pol.loc[order, m]
    ax.barh(range(len(order)), v, color=[POLICY[p][1] for p in order])
    ax.set_yticks(range(len(order)), [POLICY[p][0] for p in order] if ax is axes[0] else [])
    ax.invert_yaxis()
    ax.set_title(title, fontsize=9)
    for i, val in enumerate(v):
        ax.text(val, i, f" {val:.3g}", va="center", fontsize=7.5, color=INK2)
    ax.set_xlim(0, v.max() * 1.35)
    ax.grid(axis="y", visible=False)
fig.suptitle("SIMULATED scaling outcomes, test day (oracle = 0 on every panel)", x=0.01, ha="left",
             fontsize=10.5, fontweight="bold")
fig.tight_layout()
save(fig, "13_policy_comparison.png")

# ---- 14. ablations --------------------------------------------------------------
ab = pd.read_csv(RES / "phase11" / "ablations_test.csv").set_index("policy")
order = ["reactive", "full method", "univariate detector (requests only)", "forecast-residual attribution (freeze)",
         "no learned share (legit share = 1 - score)", "no detector (residual z > 3 flags, freeze)",
         "no MITIGATE (scale on estimate only)", "MITIGATE, scale on observed", "perfect detection (true flags)"]
labels = ["Reactive (reference)", "Full method", "Univariate detector", "Forecast-residual attribution",
          "No learned share (1 − score)", "No detector (residual z-score)", "No MITIGATE", "MITIGATE, scale on observed",
          "Perfect detection"]
cols = [ORANGE, BLUE] + [MUTED] * 6 + [AQUA]
panels = [("excess_replica_hours_per_series", "Excess replica-hours"),
          ("sla_violation_attack_pct", "SLA violations during attacks (%)"),
          ("legit_wrongly_mitigated_pct", "Legit wrongly mitigated (%)")]
fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
for ax, (m, title) in zip(axes, panels):
    v = ab.loc[order, m]
    ax.barh(range(len(order)), v, color=cols)
    ax.set_yticks(range(len(order)), labels if ax is axes[0] else [])
    ax.invert_yaxis()
    ax.set_title(title, fontsize=9)
    for i, val in enumerate(v):
        ax.text(val, i, f" {val:.2g}", va="center", fontsize=7.5, color=INK2)
    ax.set_xlim(0, v.max() * 1.3)
    ax.grid(axis="y", visible=False)
fig.suptitle("Ablations: one component removed at a time (SIMULATED, test day)", x=0.01, ha="left",
             fontsize=10.5, fontweight="bold")
fig.tight_layout()
save(fig, "14_ablations.png")

# ---- 15. robustness to attack intensity -----------------------------------------
rb = pd.read_csv(RES / "phase11" / "robustness_scenarios_test.csv")
rb = rb[rb.param == "rho"]
fig, axes = plt.subplots(1, 2, figsize=(9, 3))
for ax, (m, title) in zip(axes, [("excess_replica_hours_per_series", "Excess replica-hours vs oracle"),
                                 ("sla_violation_attack_pct", "SLA violations during attacks (%)")]):
    for n, lab, c in [("reactive", "Reactive", ORANGE), ("naive predictive (Holt on observed)", "Naive predictive", YELLOW),
                      ("full method", "Ours", BLUE)]:
        d = rb[rb.policy == n]
        ax.plot(d["value"], d[m], marker="o", c=c, label=lab)
    ax.set_xscale("log")
    ax.minorticks_off()
    ax.set_xticks([0.5, 1, 3], ["0.5", "1", "3"])
    ax.set_xlabel("attack intensity ρ (attack peak / legitimate peak)")
    ax.set_title(title, fontsize=9.5)
axes[0].legend(fontsize=8)
axes[1].text(0.02, 0.95, "reactive and naive predictive coincide", transform=axes[1].transAxes, fontsize=7.5,
             color=INK2, va="top")
fig.suptitle("Gains grow with attack intensity (SIMULATED, test day, 28 series per ρ)", x=0.01, ha="left",
             fontsize=10.5, fontweight="bold")
fig.tight_layout()
save(fig, "15_robustness_rho.png")
