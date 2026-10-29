# Benchmark specification v0.1 (draft)

## Principle
Observed(t) = f(L(t), A(t)). L(t) is the true legitimate demand, known by construction.
Estimators see only Observed(t) (and optionally a context signal) and output L_hat(t).

## Components
- L(t): real baseline + synthetic surge.
  - Baseline: WorldCup98 TR2 (1998-05-04), per-second request counts rebinned to the step size.
    Real data, 1998 web traffic.
  - Surge: SYNTHETIC. Shape (rise, peak, decay) taken from pageviews.csv, scaled to a stated
    peak-to-baseline ratio. Label "synthetic" everywhere.
  - OPEN: TR1/TR3 may show a real surge (unverified).
- A(t): attack-like traffic, SYNTHETIC magnitude and timing.
  - Intensity is defined relative to L: rho = A_peak / L_peak. Sweep rho; do not use absolute TSD
    packet counts (20 Gbps replay is not commensurate with 1998 request rates).
  - TSD supplies only the distribution of flood-to-normal window-count ratios.
    OPEN: compute from TSD High windows (new script).
  - Dataset A informs the handshake-completing (application-layer) flood only.

## Observed counters (per step)
- requests completed, SYN count, ACK count (TSD uses SYN - ACK).
- Legitimate: each request opens k connections. Assumption k = 1 (HTTP/1.0 in 1998;
  OPEN: verify against the status field's version bits). Sensitivity-test k.
- SYN flood: SYN counted, no completing ACK, no request served.
- Application-layer flood: handshakes complete, requests served (H1 predicts attribution fails).
- Mimicking attack: same counters and shape as legitimate demand (expected unidentifiable).

## Capacity model (SYNTHETIC / simulated)
- n replicas, service rate mu per replica, utilisation u = load / (n * mu).
- Latency from a queueing curve rising steeply as u -> 1; SLA threshold on latency.
- Provisioning delay D, scale-down cooldown, min/max replicas.
- All parameters get a stated range and a sensitivity analysis. No claim of measurement.

## Scenarios
1 flash crowd only | 2 attack only | 3 flash crowd + attack | 4 Yo-Yo pulsing
5 legitimate-mimicking | 6 smokescreen timed to the surge

## Baselines
reactive | naive predictive | SPARK-style legitimacy gating (reimplemented) |
robust forecasting | forecast + residual + attribution (ours)

## Metrics
Demand-estimate error vs truth (the only non-simulated quantity),
resource-hours, unnecessary scale-ups, SLA violations, legitimate traffic wrongly mitigated.

## Identifiability (to formalise)
If the counters depend on L and A only through their sum, L is unidentifiable. Identifiability
requires a counter or context signal that separates them. Define when this holds per scenario.

## Reproducibility
Fixed seeds, one config file per scenario, no model training before labels, leakage and temporal
split are settled.