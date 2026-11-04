"""Figures for everything up to Phase 7: pipeline architecture, data, benchmark and baseline results.

Writes PNGs to results/figures/. Colours follow a fixed categorical order (validated for colour-vision deficiency
on adjacent pairs); text stays in neutral ink. Synthetic and simulated content is labelled as such.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
CFG = yaml.safe_load(open(ROOT / "config.yaml"))
RAW, PROC = ROOT / CFG["paths"]["raw"], ROOT / CFG["paths"]["processed"]
SYN, RES = ROOT / CFG["paths"]["synthetic"], ROOT / CFG["paths"]["results"]
OUT = RES / "figures"
OUT.mkdir(parents=True, exist_ok=True)
STEP = CFG["step_s"]

BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#fcfcfb"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "lines.linewidth": 1.5,
})
NAMES = {
    "static[x_syn_minus_ack]": "Threshold: SYN−ACK", "static[TSD rule SYN-ACK>0]": "TSD rule: SYN−ACK > 0",
    "static[x_syn]": "Threshold: SYN", "static[x_requests]": "Threshold: requests",
    "static[syn_per_req]": "Threshold: SYN per request", "ewma_z[x_syn]": "EWMA z: SYN",
    "ewma_z[x_requests]": "EWMA z: requests", "ewma_z[x_syn_minus_ack]": "EWMA z: SYN−ACK",
    "hgb[lag features]": "HistGradientBoosting", "rf[lag features]": "RandomForest",
}


def save(fig, name):
    fig.savefig(OUT / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


# ---- 1. architecture ---------------------------------------------------------
def box(ax, x, y, w, h, title, body, fc, ec, dashed=False, title_color=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fc, ec=ec,
                                lw=1.2, ls=(0, (4, 3)) if dashed else "-"))
    ax.text(x + 0.12, y + h - 0.16, title, ha="left", va="top", fontsize=8.6, weight="bold", color=title_color)
    ax.text(x + 0.12, y + h - 0.48, body, ha="left", va="top", fontsize=7.2, color=INK2, linespacing=1.35)


def arrow(ax, a, b, color=MUTED, dashed=False):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=10, lw=1.1, color=color,
                                 ls=(0, (4, 3)) if dashed else "-", shrinkA=2, shrinkB=2))


fig, ax = plt.subplots(figsize=(16, 9.6))
ax.set_xlim(0, 20); ax.set_ylim(0, 12); ax.axis("off")
REAL_FC, SYN_FC, PROC_FC, GATE_FC, FUT_FC = "#eaf2fc", "#fdeee7", "#f3f2ee", "#e5f6ef", SURF
cols = [(0.2, "Raw data (read-only)"), (4.3, "Phases 1–5: inspect and decide"),
        (8.6, "Phase 6: benchmark"), (12.9, "Phase 7: baselines"), (16.9, "Next (not built)")]
for x, t in cols:
    ax.text(x, 11.75, t, fontsize=10, weight="bold", color=INK, va="top")

box(ax, 0.2, 9.0, 3.6, 2.2, "CIC-IDS2017 DDoS (Dataset A)",
    "225,745 flows, 1-min timestamps\none attack event, single source\nno legitimate ground truth", REAL_FC, BLUE)
box(ax, 0.2, 6.3, 3.6, 2.4, "TSD-DDoS Jan / Mar",
    "from CIC-DDoS2019 pcaps, 5-s windows\nSYN, SYN-ACK, ACK, RST, TCP counts\nfiltered on SYN−ACK, not contiguous\nflood = flat plateau 4.9 / 7.8 min",
    REAL_FC, BLUE)
box(ax, 0.2, 3.4, 3.6, 2.6, "WorldCup98 TR1–TR3 (Zenodo)",
    "3 real days, 6.6M requests\nper-request time + client ID\nno legitimate flash crowd\nTR1: one 22-s burst (2 clients)",
    REAL_FC, BLUE)
box(ax, 0.2, 1.0, 3.6, 2.1, "Pending inputs",
    "pageviews.csv (surge shapes)\nAlibaba trace (load→CPU→latency)\nTSD v6 label files", FUT_FC, MUTED, dashed=True)

box(ax, 4.3, 9.0, 3.8, 2.2, "01–03 Dataset A analysis",
    "inspect, flag-don't-delete cleaning\nminute series: step function\nGate (Phase 3): FAILED → pivot", PROC_FC, MUTED)
box(ax, 4.3, 6.3, 3.8, 2.4, "01d–01h, 07 TSD analysis",
    "label rule reproduced: H = SYN−ACK\nflood magnitudes (plateau 0.76–1.07)\nACK/SYN pool from 4,401 normal\nwindows (median ≈14)",
    PROC_FC, MUTED)
box(ax, 4.3, 3.4, 3.8, 2.6, "05–06 WorldCup98 baselines",
    "decode 20-byte binary records\nmd5 verified, sum checks OK\n5-s bins: requests, distinct clients\n17,280 bins per day = real L(t)",
    PROC_FC, MUTED)
box(ax, 4.3, 1.0, 3.8, 2.1, "Phase 4–5 decisions",
    "no real legitimate surge anywhere\n→ known counterfactual by construction\nliterature check: SPARK, Yo-Yo, EDoS,\nflash-crowd vs DDoS", PROC_FC, MUTED)

box(ax, 8.6, 7.3, 3.9, 3.9, "08 build windows (5-s step)",
    "L(t) = real baseline + SYNTHETIC surge\n   peak ×2/×5/×10, rise 5/30 min\nA(t) = SYNTHETIC attack, ρ = A/L ∈ {0.5,1,3}\n   SYN flood · app-layer flood · yo-yo\n   mimic · smokescreen · surge+attack\ncounters: requests, SYN, ACK (noisy),\n   clients → x_ inputs / y_ truth / a_ info\n49 configs × 2 seeds = 98 series/split",
    SYN_FC, ORANGE)
box(ax, 8.6, 4.5, 3.9, 2.5, "Chronological split by day",
    "train TR2 (05-04) · val TR1 (05-20)\ntest TR3 (05-23), ≈1.69M rows each\nlog1p + StandardScaler fit on train\nTR1 burst masked from evaluation",
    PROC_FC, MUTED)
box(ax, 8.6, 1.9, 3.9, 2.3, "09 leakage gate: 11/11 PASS",
    "x_ only in features · no label copies\ncontiguous series · no cross-split ids\nscaler = train stats · purge ≥ 1 h", GATE_FC, AQUA)

box(ax, 12.9, 7.3, 3.6, 3.9, "Detection (y_attack_active)",
    "static thresholds: SYN−ACK, SYN,\n   requests, SYN/request, TSD rule\nEWMA z-score (span tuned on val)\nHistGradientBoosting, RandomForest\n   on 84 trailing features\nbest: RF  F1 0.90 (excl. mimic)\n   FPR 0.03%, delay 5 s",
    PROC_FC, MUTED)
box(ax, 12.9, 4.5, 3.6, 2.5, "Forecast (5 s / 1 min ahead)",
    "naive · trailing mean · EWMA · HGB\nbest: EWMA  MAE 19.5 / 23.7\nerror vs true legit during attack\n   ≈ reactive (≈180): the gap", PROC_FC, MUTED)
box(ax, 12.9, 1.9, 3.6, 2.3, "Logged outputs",
    "runs.jsonl: commit, config hash,\n   seed, params, threshold, timing\ndetection / family / forecast CSVs", PROC_FC, MUTED)

box(ax, 16.9, 8.6, 2.9, 2.6, "Phase 8",
    "LSTM / TCN only if they\nbeat RF on app-layer\nand yo-yo floods", FUT_FC, MUTED, dashed=True)
box(ax, 16.9, 5.0, 2.9, 3.3, "Phase 9 (core)",
    "forecast → residual\n→ attribute attack-like\nvs legitimate\n→ L̂(t), action:\nSCALE / HOLD / MITIGATE", FUT_FC, MUTED,
    dashed=True)
box(ax, 16.9, 1.9, 2.9, 2.8, "Phase 10–11",
    "SIMULATED capacity model\nresource-hours, SLA,\nwrongly mitigated users\nablations", FUT_FC, MUTED, dashed=True)

for y in [10.1, 7.5, 4.7]:
    arrow(ax, (3.8, y), (4.3, y))
arrow(ax, (8.1, 7.3), (8.6, 8.6), ORANGE)
arrow(ax, (8.1, 4.7), (8.6, 8.0), ORANGE)
arrow(ax, (8.1, 2.0), (8.6, 7.6), MUTED)
arrow(ax, (6.2, 9.0), (6.2, 8.7))
arrow(ax, (10.55, 7.3), (10.55, 7.0))
arrow(ax, (10.55, 4.5), (10.55, 4.2))
arrow(ax, (12.5, 9.2), (12.9, 9.2))
arrow(ax, (12.5, 5.7), (12.9, 5.7))
arrow(ax, (14.7, 7.3), (14.7, 7.0))
arrow(ax, (14.7, 4.5), (14.7, 4.2))
arrow(ax, (16.5, 9.2), (16.9, 9.6), MUTED, dashed=True)
arrow(ax, (16.5, 5.7), (16.9, 6.6), MUTED, dashed=True)
arrow(ax, (18.35, 5.0), (18.35, 4.7), MUTED, dashed=True)
arrow(ax, (2.0, 3.1), (2.0, 3.4), MUTED, dashed=True)
ax.legend(handles=[Patch(fc=REAL_FC, ec=BLUE, label="real data"), Patch(fc=SYN_FC, ec=ORANGE, label="synthetic generation"),
                   Patch(fc=PROC_FC, ec=MUTED, label="script / analysis (done)"),
                   Patch(fc=GATE_FC, ec=AQUA, label="gate passed"),
                   Patch(fc=FUT_FC, ec=MUTED, ls="--", label="pending / planned")],
          loc="lower center", bbox_to_anchor=(0.5, -0.04), ncol=5, fontsize=8)
ax.set_title("Counterfactual attack-aware autoscaling: pipeline through Phase 7", fontsize=12, loc="left", pad=6)
save(fig, "01_architecture.png")

# ---- 2. real data overview -----------------------------------------------------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw={"width_ratios": [1.6, 1]})
for (day, date), c in zip({"TR2": "05-04 (train)", "TR1": "05-20 (val)", "TR3": "05-23 (test)"}.items(),
                          [BLUE, ORANGE, AQUA]):
    b = pd.read_csv(PROC / f"{day}_baseline_{STEP}s.csv")
    per_min = b["requests"].to_numpy()[: len(b) // 12 * 12].reshape(-1, 12).sum(1)
    h = np.arange(len(per_min)) / 60
    a1.plot(h, per_min, lw=0.9, color=c, label=f"{day} 1998-{date}")
burst_h = 8733 * STEP / 3600
a1.annotate("TR1 burst: 2 clients,\n~10k requests in 22 s", xy=(burst_h, 7900), xytext=(burst_h + 2.2, 7000),
            fontsize=7.5, color=INK2, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
a1.set(xlabel="hours since 22:00 UTC", ylabel="requests per minute", xlim=(0, 24))
a1.set_title("WorldCup98: real legitimate baseline, three days")
a1.legend(loc="upper left", fontsize=8)
for f, c, lab in [("jan-12", BLUE, "Jan"), ("mar-11", ORANGE, "Mar")]:
    t = pd.read_csv(RAW / f"{f}-dataset-final-version-csv-ieeedataport.csv")
    t.columns = [x.strip() for x in t.columns]
    tcp = t["#TCP Packets"].to_numpy(float)
    fl = np.flatnonzero(tcp > 1e5)
    s, e = fl.min() - 15, fl.max() + 16
    a2.plot(np.arange(s, e) - fl.min(), tcp[s:e], color=c, label=f"{lab} ({len(fl)} flood windows)")
a2.set_yscale("log")
a2.set(xlabel="window index from flood start (rows, not contiguous time)", ylabel="TCP packets per 5-s window")
a2.set_title("TSD-DDoS floods: flat plateaus, no ramp")
a2.legend(fontsize=8, loc="center right")
fig.tight_layout()
save(fig, "02_real_data.png")

# ---- 3. handshake noise and SYN-flood separability -----------------------------
pool = []
for f in CFG["legit"]["ack_ratio_files"]:
    t = pd.read_csv(RAW / f)
    t.columns = [c.strip().replace("# ", "#") for c in t.columns]
    k = (t["#TCP Packets"] <= 1e5) & (t["#SYN Packets"] - t["#ACK Packets"] < 100) & (t["#SYN Packets"] >= 3)
    pool.append((t.loc[k, "#ACK Packets"] / t.loc[k, "#SYN Packets"]).to_numpy())
pool = np.concatenate(pool)
te = pd.read_parquet(SYN / f"windows_{STEP}s_test.parquet")
pos = te[(te["a_attack_type"] == "syn") & (te["y_attack_active"] == 1)]
neg = te[te["a_family"].isin(["none", "surge_only"])]
y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
auc = {f: roc_auc_score(y, np.r_[pos[f], neg[f]]) for f in ["x_syn_minus_ack", "x_syn", "x_requests"]}
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.6))
a1.hist(pool, bins=np.logspace(np.log10(max(pool.min(), 0.3)), np.log10(pool.max()), 50), color=BLUE,
        edgecolor=SURF, linewidth=0.6)
a1.set_xscale("log")
a1.axvline(np.median(pool), color=INK, lw=1, ls="--")
a1.set_ylim(0, a1.get_ylim()[1] * 1.12)
a1.text(np.median(pool) * 0.9, a1.get_ylim()[1] * 0.95, f"median {np.median(pool):.1f}", fontsize=8, color=INK2,
        ha="right", va="top")
a1.set(xlabel="ACK packets per SYN in a 5-s window (log)", ylabel="windows")
a1.set_title(f"TSD normal windows: ACK ≫ SYN (n = {len(pool):,})")
labels = ["SYN − ACK", "SYN count", "requests"]
before = [1.0, np.nan, np.nan]
after = [auc["x_syn_minus_ack"], auc["x_syn"], auc["x_requests"]]
xi = np.arange(3)
a2.bar(xi - 0.19, before, 0.36, color=MUTED, label="before (legit SYN−ACK = 0 by construction)", edgecolor=SURF)
a2.bar(xi + 0.19, after, 0.36, color=BLUE, label="after handshake noise", edgecolor=SURF)
for i, v in enumerate(after):
    a2.text(i + 0.19, v + 0.02, f"{v:.2f}", ha="center", fontsize=8, color=INK)
a2.text(-0.19, 1.02, "1.00", ha="center", fontsize=8, color=INK)
a2.axhline(0.5, color=MUTED, lw=0.8, ls=":")
a2.text(2.45, 0.51, "chance", fontsize=7, color=MUTED, ha="right")
a2.set_xticks(xi, labels)
a2.set(ylim=(0, 1.38), ylabel="ROC AUC, SYN-flood vs legitimate windows")
a2.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
a2.set_title("SYN − ACK no longer gives SYN floods away (test)")
a2.legend(fontsize=7.5, loc="upper center", ncol=2)
fig.tight_layout()
save(fig, "03_handshake_noise.png")

# ---- 4. benchmark composition ---------------------------------------------------
cnt = te.groupby("a_family", observed=True).agg(series=("a_series", "nunique"),
                                                attack=("y_attack_active", "sum"), surge=("y_surge_active", "sum"))
order = ["none", "surge_only", "attack_only", "surge_plus_attack", "yoyo", "mimic", "smokescreen"]
cnt = cnt.loc[order]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.4))
a1.barh(order[::-1], cnt["series"][::-1], color=BLUE, edgecolor=SURF)
for i, v in enumerate(cnt["series"][::-1]):
    a1.text(v + 0.3, i, str(v), va="center", fontsize=8)
a1.set(xlabel="series per split (49 configs × 2 seeds)")
a1.set_title("Scenario families (SYNTHETIC events)")
a1.grid(axis="y", visible=False)
yy = np.arange(len(order))[::-1]
a2.barh(yy + 0.19, cnt["surge"] * STEP / 3600, 0.36, color=AQUA, label="legitimate surge active", edgecolor=SURF)
a2.barh(yy - 0.19, cnt["attack"] * STEP / 3600, 0.36, color=ORANGE, label="attack active", edgecolor=SURF)
a2.set_yticks(yy, order)
a2.set(xlabel="hours of labelled activity in the test split")
a2.set_title("Labelled event time per family (test, TR3)")
a2.legend(fontsize=8, loc="lower right")
a2.grid(axis="y", visible=False)
fig.tight_layout()
save(fig, "04_benchmark_composition.png")

# ---- 5. detection scores --------------------------------------------------------
det = pd.read_csv(RES / "phase7" / "detection_test.csv")
det["name"] = det["model"].map(NAMES)
d = det.dropna(subset=["f1_no_mimic"]).sort_values("f1_no_mimic")
fig, ax = plt.subplots(figsize=(8.5, 4.4))
yy = np.arange(len(d))
ax.barh(yy + 0.19, d["f1_no_mimic"], 0.36, color=BLUE, label="F1 without mimic series", edgecolor=SURF)
ax.barh(yy - 0.19, d["f1"], 0.36, color=ORANGE, label="F1 on all series", edgecolor=SURF)
for i, (a, b) in enumerate(zip(d["f1_no_mimic"], d["f1"])):
    ax.text(a + 0.01, i + 0.19, f"{a:.2f}", va="center", fontsize=7.5)
    ax.text(b + 0.01, i - 0.19, f"{b:.2f}", va="center", fontsize=7.5, color=INK2)
ax.set_yticks(yy, d["name"])
ax.set(xlim=(0, 1.05), xlabel="window-level F1 on test (threshold chosen on val)")
ax.set_title("Attack detection: RandomForest is the strongest simple baseline")
ax.legend(loc="lower right", fontsize=8)
ax.grid(axis="y", visible=False)
fig.tight_layout()
save(fig, "05_detection_f1.png")

# ---- 6. false alarms overall vs during legitimate surges ------------------------
d = det.dropna(subset=["fpr_during_legit_surge"])
d = d[d["fpr"] < 0.5].sort_values("fpr_during_legit_surge")
fig, ax = plt.subplots(figsize=(8.5, 3.8))
yy = np.arange(len(d))
floor = 1e-5
for i, (a, b) in enumerate(zip(d["fpr"], d["fpr_during_legit_surge"])):
    ax.plot([max(a, floor), max(b, floor)], [i, i], color=GRID, lw=2, zorder=1)
ax.scatter(np.maximum(d["fpr"], floor), yy, s=46, color=BLUE, zorder=3, label="all legitimate windows",
           edgecolor=SURF, linewidth=1.5)
ax.scatter(np.maximum(d["fpr_during_legit_surge"], floor), yy, s=46, color=ORANGE, zorder=3, marker="D",
           label="inside legitimate surges", edgecolor=SURF, linewidth=1.5)
ax.set_xscale("log")
ax.set_yticks(yy, d["model"].map(NAMES))
ax.set(xlabel=f"false-positive rate on test (log; 0 shown at {floor:g})")
ax.set_title("False alarms: volume thresholds fire on legitimate surges")
ax.legend(loc="lower right", fontsize=8)
ax.grid(axis="y", visible=False)
fig.tight_layout()
save(fig, "06_false_alarms.png")

# ---- 7. per-family recall heatmap -----------------------------------------------
fam = pd.read_csv(RES / "phase7" / "detection_by_family_test.csv").dropna(subset=["window_recall"])
fam = fam[~fam["model"].isin(["static[x_syn_minus_ack]", "ewma_z[x_syn_minus_ack]"])]
fam["row"] = fam["family"] + " · " + fam["attack_type"]
mods = ["static[x_syn]", "static[x_requests]", "static[syn_per_req]", "ewma_z[x_syn]", "ewma_z[x_requests]",
        "hgb[lag features]", "rf[lag features]"]
rows = ["attack_only · syn", "yoyo · syn", "surge_plus_attack · syn", "smokescreen · syn",
        "attack_only · app", "yoyo · app", "surge_plus_attack · app", "smokescreen · app", "mimic · app"]
m = fam.pivot_table(index="row", columns="model", values="window_recall").loc[rows, mods]
fig, ax = plt.subplots(figsize=(9, 4.6))
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blue_seq", ["#f4f8fd"] + SEQ)
im = ax.imshow(m.to_numpy(), cmap=cmap, vmin=0, vmax=1, aspect="auto")
for i in range(m.shape[0]):
    for j in range(m.shape[1]):
        v = m.iat[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5, color=SURF if v > 0.6 else INK)
ax.set_xticks(range(len(mods)), [NAMES[x] for x in mods], rotation=30, ha="right")
ax.set_yticks(range(len(rows)), rows)
ax.grid(False)
for s in ax.spines.values():
    s.set_visible(False)
ax.axhline(3.5, color=SURF, lw=3)
cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cb.set_label("window recall (test)")
cb.outline.set_visible(False)
ax.set_title("Recall by attack family: SYN floods solved, app-layer and mimic remain hard")
fig.tight_layout()
save(fig, "07_recall_by_family.png")

# ---- 8. forecasting -------------------------------------------------------------
fc = pd.read_csv(RES / "phase7" / "forecast_test.csv")
ref = fc.loc[fc["model"].str.startswith("reactive"), "mae_vs_true_legit_during_attack"].iat[0]
fc = fc.dropna(subset=["mae"])
mnames = {"naive_last": "naive (last value)", "trailing_mean_12": "trailing mean (1 min)", "ewma": "EWMA",
          "hgb_regressor": "HGB regressor"}
mk = list(mnames)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 3.8))
xi = np.arange(len(mk))
for k, (h, c) in enumerate([(1, BLUE), (12, ORANGE)]):
    v = fc[fc["horizon_steps"] == h].set_index("model").loc[mk, "mae"]
    a1.bar(xi + (k - 0.5) * 0.38, v, 0.36, color=c, label=f"{h * STEP} s ahead", edgecolor=SURF)
    for i, val in enumerate(v):
        a1.text(i + (k - 0.5) * 0.38, val + 0.4, f"{val:.1f}", ha="center", fontsize=7.5)
a1.set_xticks(xi, [mnames[x] for x in mk])
a1.set(ylabel="MAE, requests per 5-s window")
a1.set_title("Forecasting observed load (test)")
a1.legend(fontsize=8)
for k, (h, c) in enumerate([(1, BLUE), (12, ORANGE)]):
    v = fc[fc["horizon_steps"] == h].set_index("model").loc[mk, "mae_vs_true_legit_during_attack"]
    a2.bar(xi + (k - 0.5) * 0.38, v, 0.36, color=c, label=f"{h * STEP} s ahead", edgecolor=SURF)
a2.axhline(ref, color=INK, lw=1, ls="--")
a2.text(len(mk) - 0.5, ref + 4, f"reactive: use observed load ({ref:.0f})", ha="right", fontsize=7.5, color=INK2)
a2.set_xticks(xi, [mnames[x] for x in mk])
a2.set(ylabel="MAE vs TRUE legitimate demand")
a2.set_title("During attacks, forecasts do not recover legitimate demand")
a2.set_ylim(0, 235)
a2.legend(fontsize=8, loc="upper left", ncol=2)
fig.tight_layout()
save(fig, "08_forecasting.png")

# ---- 9. example detections on test series ---------------------------------------
sc = pd.read_parquet(SYN / f"phase7_test_scores_{STEP}s.parquet")
thr = json.load(open(SYN / f"phase7_test_thresholds_{STEP}s.json"))
flags = sc["rf[lag features]"].to_numpy() >= thr["rf[lag features]"]
examples = [("attack_only", "app"), ("yoyo", "syn"), ("surge_plus_attack", "app"), ("mimic", "app")]
fig, axes = plt.subplots(len(examples), 1, figsize=(11, 2.3 * len(examples)))
for ax, (f, a) in zip(axes, examples):
    sel = np.flatnonzero((te["a_family"] == f).to_numpy() & (te["a_attack_type"] == a).to_numpy()
                         & (te["a_rho"] == 1).to_numpy() & (te["a_seed_rep"] == 0).to_numpy())
    sid = te["a_series"].iat[sel[0]]
    rows_ = np.flatnonzero((te["a_series"] == sid).to_numpy())
    s = te.iloc[rows_]
    o = int(s["a_onset_bin"].iat[0])
    span = slice(max(o - 240, 0), o + (1900 if f == "mimic" else 600))
    w, fl = s.iloc[span], flags[rows_][span]
    t = (w["a_bin"].to_numpy() - o) * STEP / 60
    att = w["y_attack_active"].to_numpy() == 1
    ax.fill_between(t, 0, 1, where=att, transform=ax.get_xaxis_transform(), color="#fdeee7", step="mid",
                    label="true attack window")
    ax.plot(t, w["x_syn"], color=BLUE, lw=0.8, label="observed SYN")
    ax.plot(t, w["x_requests"], color=ORANGE, lw=0.8, label="observed requests")
    ax.plot(t, w["y_legit_requests"], color=INK, lw=0.9, ls="--", label="true legitimate requests")
    ymax = ax.get_ylim()[1]
    ax.scatter(t[fl], np.full(fl.sum(), ymax * 1.02), marker="|", s=40, color=AQUA, label="RandomForest flag",
               clip_on=False)
    ax.set_title(f"{f} · {a} flood · ρ = 1  [{sid}]   recall {fl[att].mean():.2f}, false flags {fl[~att].sum()}",
                 fontsize=9)
    ax.set_ylabel(f"per {STEP} s")
axes[0].legend(ncol=5, fontsize=7.5, loc="upper left", bbox_to_anchor=(0, 1.45))
axes[-1].set_xlabel("minutes from event onset (test day TR3; SYNTHETIC events on real baseline)")
fig.tight_layout()
save(fig, "09_example_detections.png")

# ---- 10. window-size sensitivity ------------------------------------------------
ws = pd.read_csv(RES / "08_window_sensitivity.csv")
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.2))
for (day, c) in zip(["TR2", "TR1", "TR3"], [BLUE, ORANGE, AQUA]):
    g = ws[ws["day"] == day]
    a1.plot(g["step_s"], g["median_abs_diff_over_median"], marker="o", ms=5, color=c, label=day)
    a2.plot(g["step_s"], g["lag1_acf"], marker="o", ms=5, color=c, label=day)
for a, lab, title in [(a1, "step-to-step noise / level", "Noise falls with coarser windows"),
                      (a2, "lag-1 autocorrelation", "Smoothness rises with coarser windows")]:
    a.set_xscale("log")
    a.set_xticks([5, 10, 30, 60, 300], ["5 s", "10 s", "30 s", "60 s", "300 s"])
    a.minorticks_off()
    a.set(xlabel="window size", ylabel=lab)
    a.set_title(title)
a1.legend(fontsize=8)
fig.tight_layout()
fig.text(0.01, -0.03, "5 s chosen: a 5-min attack spans 60 windows (at 60 s only 5). TR1's 5-s autocorrelation is "
         "raised by its real 22-s burst.", fontsize=7.5, color=INK2)
save(fig, "10_window_sensitivity.png")
