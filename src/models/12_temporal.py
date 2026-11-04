"""Phase 8: temporal models (causal TCN, LSTM) for window-level attack detection.

Only justified if they beat the Phase 7 RandomForest on app-layer and yo-yo floods (DECISIONS.md section 17).
Same protocol as Phase 7: fit on train (TR2), checkpoint and threshold on val (TR1), report on test (TR3).
Inputs are causal and level-free: log counters minus their trailing 1-h mean, plus log ratios.

  python src/models/12_temporal.py train --model tcn --seed 0 [--iters 2000]
  python src/models/12_temporal.py summarize
"""
import argparse
import hashlib
import json
import os
import subprocess
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

from common import BASE, CFG, ROOT, SEED, STEP, SYN, best_f1_threshold, by_family, detection_metrics, load, wide

OUT = ROOT / CFG["paths"]["results"] / "phase8"
OUT.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(os.cpu_count())
CHANNELS = ["rel_log_requests", "rel_log_syn", "rel_log_ack", "rel_log_clients",
            "log_syn_per_req", "log_ack_per_syn", "log_req_per_client"]


def inputs(df, n_s):
    """(series, channels, time) float32; every value uses only the current and past windows."""
    logs = {c: np.log1p(wide(df, n_s, c)) for c in BASE}  # (time, series)
    ch = [(l - l.shift(1).rolling(720, min_periods=60).mean()).to_numpy().T for l in logs.values()]
    lr, ls, la, lc = (logs[c].to_numpy().T for c in BASE)
    ch += [ls - lr, la - ls, lr - lc]
    return np.nan_to_num(np.stack(ch, 1), nan=0.0).astype(np.float32)


def split_arrays(split):
    df, n_s = load(split)
    y = df["y_attack_active"].to_numpy(np.float32).reshape(n_s, -1)
    m = df["a_eval_mask"].to_numpy(np.float32).reshape(n_s, -1)
    return df, n_s, inputs(df, n_s), y, m


class Block(nn.Module):
    def __init__(self, ch, d, drop):
        super().__init__()
        self.pad = 2 * d
        self.c1, self.c2 = nn.Conv1d(ch, ch, 3, dilation=d), nn.Conv1d(ch, ch, 3, dilation=d)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        h = self.drop(torch.relu(self.c1(F.pad(x, (self.pad, 0)))))
        h = self.drop(torch.relu(self.c2(F.pad(h, (self.pad, 0)))))
        return x + h


class TCN(nn.Module):
    """Causal dilated TCN; receptive field 1 + 4 * sum(dilations) = 509 windows (~42 min, longer than any attack)."""
    def __init__(self, cin, ch=48, dil=(1, 2, 4, 8, 16, 32, 64), drop=0.1):
        super().__init__()
        self.inp, self.out = nn.Conv1d(cin, ch, 1), nn.Conv1d(ch, 1, 1)
        self.blocks = nn.Sequential(*[Block(ch, d, drop) for d in dil])

    def forward(self, x):  # (B, C, T) -> (B, T) logits
        return self.out(self.blocks(self.inp(x))).squeeze(1)


class LSTM(nn.Module):
    def __init__(self, cin, hidden=64):
        super().__init__()
        self.rnn, self.out = nn.LSTM(cin, hidden, batch_first=True), nn.Linear(hidden, 1)

    def forward(self, x):
        h, _ = self.rnn(x.transpose(1, 2))
        return self.out(h).squeeze(-1)


SPEC = {  # chunk length, warm-up windows excluded from the loss, batch size
    "tcn": {"cls": TCN, "chunk": 1024, "warm": 512, "batch": 32},
    "lstm": {"cls": LSTM, "chunk": 1024, "warm": 256, "batch": 32},
}


@torch.no_grad()
def predict(model, X, batch=16):
    model.eval()
    return np.concatenate([torch.sigmoid(model(torch.from_numpy(X[i:i + batch]))).numpy()
                           for i in range(0, len(X), batch)])


def train(args):
    spec = SPEC[args.model]
    torch.manual_seed(SEED + args.seed)
    rng = np.random.default_rng(SEED + args.seed)
    t_load = time.perf_counter()
    tr_df, n_tr, Xtr, ytr, _ = split_arrays("train")
    va_df, n_va, Xva, yva, mva = split_arrays("val")
    te_df, n_te, Xte, _, _ = split_arrays("test")
    mu, sd = Xtr.mean((0, 2), keepdims=True), Xtr.std((0, 2), keepdims=True) + 1e-6  # train-only scaling
    Xtr, Xva, Xte = [(x - mu) / sd for x in (Xtr, Xva, Xte)]
    print(f"inputs ready in {time.perf_counter() - t_load:.0f}s: train {Xtr.shape}")

    model = spec["cls"](Xtr.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    pos = np.argwhere(ytr == 1)
    L, W, B, T = spec["chunk"], spec["warm"], spec["batch"], Xtr.shape[2]
    best = {"f1": -1.0}
    t0 = time.perf_counter()
    for it in range(1, args.iters + 1):
        model.train()
        # half the chunks contain an attack-active window, half are uniform (attacks are ~1.7% of windows)
        s_idx, starts = [], []
        for b in range(B):
            if b % 2 == 0:
                s, t = pos[rng.integers(len(pos))]
                st = int(np.clip(t - rng.integers(W, L), 0, T - L))
            else:
                s, st = rng.integers(len(Xtr)), int(rng.integers(0, T - L))
            s_idx.append(s); starts.append(st)
        xb = torch.from_numpy(np.stack([Xtr[s, :, st:st + L] for s, st in zip(s_idx, starts)]))
        yb = torch.from_numpy(np.stack([ytr[s, st:st + L] for s, st in zip(s_idx, starts)]))
        logit = model(xb)[:, W:]
        loss = F.binary_cross_entropy_with_logits(logit, yb[:, W:], pos_weight=torch.tensor(args.pos_weight))
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if it % args.eval_every == 0 or it == args.iters:
            sv = predict(model, Xva).ravel()
            m = mva.ravel() == 1
            thr, f1 = best_f1_threshold(yva.ravel()[m], sv[m])
            print(f"it {it:>5}  loss {loss.item():.4f}  val F1 {f1:.3f}  ({time.perf_counter() - t0:.0f}s)", flush=True)
            if f1 > best["f1"]:
                best = {"f1": f1, "thr": thr, "it": it, "state": {k: v.clone() for k, v in model.state_dict().items()}}
    fit_s = time.perf_counter() - t0

    model.load_state_dict(best["state"])
    sv = predict(model, Xva).ravel()
    t1 = time.perf_counter(); st = predict(model, Xte).ravel(); infer_s = time.perf_counter() - t1
    name = f"{args.model}[seed {args.seed}]"
    met = detection_metrics(te_df, n_te, st >= best["thr"])
    rec = {"task": "detection", "model": name, "family": args.model, "seed": args.seed, "iters": args.iters,
           "best_iter": best["it"], "val_f1": best["f1"], "threshold": best["thr"], "lr": args.lr,
           "pos_weight": args.pos_weight, "params": int(sum(p.numel() for p in model.parameters())),
           "fit_s": fit_s, "infer_us_per_window": infer_s / st.size * 1e6, "channels": CHANNELS,
           "git_commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                                        text=True).stdout.strip(),
           "config_sha256": hashlib.sha256(open(ROOT / "config.yaml", "rb").read()).hexdigest()[:16], **met}
    with open(OUT / "runs.jsonl", "a") as f:
        f.write(json.dumps(rec, default=float) + "\n")
    pd.DataFrame({"val": sv}).to_parquet(SYN / f"phase8_val_{args.model}_s{args.seed}.parquet", index=False)
    pd.DataFrame({"test": st}).to_parquet(SYN / f"phase8_test_{args.model}_s{args.seed}.parquet", index=False)
    print(f"{name}: best iter {best['it']}, val F1 {best['f1']:.3f} | test F1 {met['f1']:.3f}  "
          f"F1-no-mimic {met['f1_no_mimic']:.3f}  FPR {met['fpr']:.4f}  surge-FPR {met['fpr_during_legit_surge']:.4f}")


def summarize(_):
    te, n_te = load("test")
    runs = [json.loads(l) for l in open(OUT / "runs.jsonl")]
    p7 = pd.read_parquet(SYN / f"phase7_test_scores_{STEP}s.parquet")
    thr7 = json.load(open(SYN / f"phase7_test_thresholds_{STEP}s.json"))
    scores = {m: (p7[m].to_numpy(), thr7[m]) for m in p7.columns if m.startswith(("rf", "hgb"))}
    for r in runs:
        s = pd.read_parquet(SYN / f"phase8_test_{r['family']}_s{r['seed']}.parquet")["test"].to_numpy()
        scores[r["model"]] = (s, r["threshold"])
    y = te["y_attack_active"].to_numpy() == 1
    m = te["a_eval_mask"].to_numpy() == 1
    fam = (te["a_family"].astype(str) + " · " + te["a_attack_type"].astype(str)).to_numpy()
    legit = ~y & m
    rows, fams = [], []
    for name, (s, thr) in scores.items():
        s = np.nan_to_num(s, nan=np.nanmin(s) - 1)
        met = detection_metrics(te, n_te, s >= thr)
        no_mimic = m & (te["a_family"].to_numpy() != "mimic")
        row = {"model": name, "ap_all": average_precision_score(y[m], s[m]),
               "ap_no_mimic": average_precision_score(y[no_mimic], s[no_mimic]),
               **{k: met[k] for k in ["f1", "f1_no_mimic", "precision", "recall", "fpr", "fpr_during_legit_surge",
                                      "event_recall", "median_delay_s"]}}
        # per attack family: that family's attack windows vs every legitimate window of the test day
        for f in sorted(set(fam[y])):
            k = (y & (fam == f)) | legit
            row[f"ap[{f}]"] = average_precision_score(y[k], s[k])
        rows.append(row)
        fams.extend(by_family(te, s >= thr, name))
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "comparison_test.csv", index=False)
    pd.DataFrame(fams).to_csv(OUT / "detection_by_family_test.csv", index=False)
    pd.set_option("display.width", 250)
    print(res[["model", "ap_all", "ap_no_mimic", "f1", "f1_no_mimic", "fpr", "fpr_during_legit_surge"]]
          .round(4).to_string(index=False))
    print("\nAP per attack family (attack windows of that family vs all legitimate windows):")
    print(res.set_index("model")[[c for c in res.columns if c.startswith("ap[")]].round(3).T.to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--model", choices=list(SPEC), required=True)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--iters", type=int, default=2000)
    t.add_argument("--eval-every", type=int, default=250)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--pos-weight", type=float, default=5.0)
    t.set_defaults(fn=train)
    sub.add_parser("summarize").set_defaults(fn=summarize)
    a = ap.parse_args()
    a.fn(a)
