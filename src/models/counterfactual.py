"""Phase 9 core method: forecast expected load, take the residual, attribute it, estimate legitimate demand.

All functions work on (time, series) arrays and are causal: the value at window t uses windows <= t only.
"""
import numpy as np

ACTIONS = np.array(["HOLD", "SCALE", "MITIGATE"])


def estimate_legit(x_req, x_cl, flag, span=60, variant="client", share=None):
    """Counterfactual legitimate requests per window.

    Unflagged windows are taken as legitimate. In a flagged window the residual r = observed - forecast is split:
      freeze: all positive residual is attack (legitimate demand held at the forecast);
      client: the part explained by new clients at the legitimate requests-per-client rate is legitimate,
              the rest is attack (app floods send many requests from few clients);
      learned: legitimate share predicted by a regressor (`share`, same shape), no forecast.
    The forecast is an EWMA of the *estimated* legitimate series, so attack traffic does not leak into it.
    Returns (legit estimate, forecast used at each window).
    """
    if variant == "learned":
        est = np.where(flag, np.clip(share, 0, 1) * x_req, x_req).astype(float)
        return est, est
    a = 2 / (span + 1)
    T, S = x_req.shape
    est, fc_out = np.empty((T, S)), np.empty((T, S))
    f, fc = x_req[0].astype(float), x_cl[0].astype(float)
    rpc = x_req[0] / np.maximum(x_cl[0], 1)
    for t in range(T):
        r, rc = x_req[t] - f, x_cl[t] - fc
        legit_growth = np.maximum(rc, 0) * rpc if variant == "client" else 0.0
        e = np.where(flag[t], np.clip(f + np.minimum(r, legit_growth), 0, x_req[t]), x_req[t])
        ec = np.where(flag[t], e / np.maximum(rpc, 1e-6), x_cl[t])
        fc_out[t], est[t] = f, e
        f, fc = f + a * (e - f), fc + a * (ec - fc)
        rpc = np.where(flag[t], rpc, rpc + a * (x_req[t] / np.maximum(x_cl[t], 1) - rpc))
    return est, fc_out


def holt(x, h, span_level=60, span_trend=240):
    """h-step-ahead Holt (level + damped-free trend) forecast of each column, causal."""
    a, b = 2 / (span_level + 1), 2 / (span_trend + 1)
    out = np.empty_like(x, dtype=float)
    lvl, tr = x[0].astype(float), np.zeros(x.shape[1])
    for t in range(len(x)):
        prev = lvl
        lvl = lvl + tr + a * (x[t] - lvl - tr)
        tr = tr + b * (lvl - prev - tr)
        out[t] = lvl + h * tr
    return np.maximum(out, 0)


def action(flag, desired, current):
    """SCALE / HOLD / MITIGATE per window: mitigation takes priority, scaling follows the legitimate estimate."""
    return np.where(flag, 2, np.where(desired != current, 1, 0))
