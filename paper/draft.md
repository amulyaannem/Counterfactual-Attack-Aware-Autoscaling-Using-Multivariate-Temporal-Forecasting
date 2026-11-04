# Counterfactual Legitimate-Demand Estimation for Attack-Aware Autoscaling

*Draft v0.1, 2026-11-04. Target venue not chosen (see the end). Every scaling number is SIMULATED.*

## Abstract

Autoscalers size capacity from observed load, so attack-like traffic is paid for as if it were demand, and
legitimacy filters risk dropping real flash crowds. We frame the problem as estimating the **counterfactual
legitimate demand**: the load that would have been observed without the attack. We build a semi-synthetic benchmark
in which the true legitimate demand is known by construction: real WorldCup98 web traffic plus synthetic legitimate
surges and synthetic attack-like traffic (SYN floods, handshake-completing application-layer floods, yo-yo pulsing,
smokescreen attacks timed to a surge, and floods that mimic the legitimate profile), with TCP handshake noise taken
from TSD-DDoS. A multivariate detector on four aggregate counters plus a learned legitimate-share attribution cuts the
error of the legitimate-demand estimate during attacks from 180 to 84 requests per 5 s window (27 with perfect
detection). In a documented capacity simulator, scaling on that estimate and mitigating while the detector fires
halves excess replica-hours relative to reactive autoscaling (4.7 → 2.5 per series-day) and cuts SLA-violating
windows during attacks from 1.10% to 0.39%, at the cost of wrongly mitigating 0.065% of legitimate requests.

## 1. Introduction

- Problem: reactive and predictive autoscalers treat every request as demand. Attack-like traffic inflates cost
  (economic denial of sustainability) and pulsing attacks exploit scaling delays.
- Gap: prior work gates or caps scaling on an attack signal (Section 2) but does not estimate *how much* of the load is
  legitimate, nor evaluate that estimate against a known truth and link it to scaling outcomes.
- Contributions:
  1. A semi-synthetic benchmark with known legitimate demand, chronological day-level splits and an automated leakage
     check (11 checks).
  2. An identifiability analysis: what aggregate counters can and cannot separate (SYN floods: yes; app-layer floods:
     partly; mimicking floods: no, by design).
  3. A method: detect → attribute → estimate legitimate demand → SCALE / HOLD / MITIGATE, with ablations.
  4. A simulated scaling evaluation reporting cost, SLA, and legitimate traffic wrongly mitigated.
- Wording: we say *attack-like traffic*, not bots or attackers; we never claim measured production results.

## 2. Related work (targeted search, not systematic; see DECISIONS.md section 8)

Legitimacy-gated predictive autoscaling (SPARK); random-forest attack prediction capping the Kubernetes HPA; DRL-based
defences (ScaleShield); Yo-Yo, Tandem and EDoS attacks on autoscalers; flash-crowd versus DDoS discrimination
(entropy, correlation, clustering, World Cup 98); offered-load estimation during DDoS (Anycast Agility); robust load
forecasting under attack (power systems). Open: anomaly attribution and causal/counterfactual forecasting literature.

## 3. Benchmark

- **Legitimate demand.** Real baseline: WorldCup98 request and distinct-client counts per 5 s window on three whole
  days (train 1998-05-04, validation 05-20, test 05-23). Synthetic surge on top: rise 5 or 30 min, 10 min plateau,
  30 min decay half-life, peak 2/5/10× the pre-event level, new clients at the day's requests-per-client rate.
- **Attack-like traffic.** Intensity relative to legitimate demand, ρ = attack peak / legitimate peak ∈ {0.5, 1, 3};
  5–8 min plateaus with TSD-derived jitter. SYN floods add SYNs only; app-layer floods complete handshakes, send
  requests from 50 sources and follow the legitimate ACK/SYN ratio; mimic floods copy the surge profile and client mix.
- **Counters.** Requests, SYN, ACK, distinct clients. ACK/SYN per window is drawn from TSD non-flood windows (median
  about 14), so the SYN−ACK gap is not a give-away (single-counter AUC 0.44 after this fix, 1.0 before).
- **Families.** none, surge only, attack only, surge + attack, yo-yo (6 × 2 min on / 3 min off), smokescreen (attack
  at the surge peak), mimic; × {SYN, app}; 2 seeds per configuration; 98 series per day.
- **Protocol.** Fit on train, choose thresholds, checkpoints and estimator variants on validation, report on test.
  Leakage check: 11/11 (no label columns as inputs, day-level splits, 1 h purge, causal features).

## 4. Detection (identifiability)

| Detector (test) | F1 | F1 w/o mimic | FPR | FPR in legit surges |
|---|---|---|---|---|
| Threshold on SYN per request | 0.21 | 0.65 | 0.0002 | 0.0000 |
| EWMA z-score on SYN | 0.24 | 0.34 | — | 0.07 |
| RandomForest, 84 causal lag features | **0.54** | **0.89** | **0.0003** | **0.0009** |
| Causal TCN (best of 2 seeds) | 0.43 | 0.74 | 0.0009 | 0.018 |
| LSTM (best of 2 seeds) | 0.39 | 0.43 | 0.007 | 0.13 |

SYN floods are separable (SYN per request; window recall 0.92–1.00). App-layer floods are the gap (RF window recall
0.56–0.64 alone, higher when stacked on a surge). Mimic floods are unidentifiable from these counters by construction
and are reported separately. Temporal models did not beat the RandomForest on app-layer or yo-yo families and were
less stable across seeds (Phase 8 gate not passed).

## 5. Method

For each window *t* (all steps causal):

1. **Detect**: RandomForest attack score on multivariate lag features; threshold chosen on validation by max F1.
2. **Attribute and estimate**: unflagged windows are legitimate (L̂ = observed). In a flagged window the legitimate
   share of observed requests is predicted by a gradient-boosted regressor trained on train-day attack windows
   (L̂ = share × observed). Alternatives evaluated: *forecast-residual* (EWMA forecast of the estimated legitimate
   series; positive residual in a flagged window is attack) and *client-consistent* residual splitting. The learned
   share had the lowest validation error in attack windows (565 vs 579 and 578; a narrow margin).
3. **Act**: MITIGATE while the detector fires (an idealised filter admits at most L̂ × 1.1 requests, attack first);
   SCALE when the replica target from a Holt forecast of estimated legitimate work changes; otherwise HOLD. On the
   test day: HOLD 97.3%, SCALE 2.0%, MITIGATE 0.68% of windows.

## 6. Results

### 6.1 Legitimate-demand estimate (test day, MAE in requests per 5 s during attack windows)

| Estimator | All attacks | App-layer attacks | Legit surges without attack |
|---|---|---|---|
| Observed = legitimate | 180.4 | 204.9 | 0.00 |
| Forecast-residual, RF flags | 112.1 | 123.9 | 0.02 |
| **Learned share, RF flags** | **83.6** | **94.7** | 0.09 |
| Learned share, true flags | 27.2 | 30.7 | 0.00 |

Detection quality, not attribution, is the main source of remaining error (83.6 → 27.2 with perfect detection).

### 6.2 Simulated scaling (test day, 98 series)

Capacity model: 50 requests per replica per 5 s, SYN cost 0.2 request, 60 s start-up, 5 min scale-down stabilisation,
target utilisation 0.6, SLA broken above utilisation 0.9.

| Policy | Excess replica-h vs oracle | Unneeded scale-ups | SLA viol. in attacks | Legit lost (overload) | Legit wrongly mitigated |
|---|---|---|---|---|---|
| Oracle (true legitimate demand) | 0 | 0 | 0% | 0% | 0% |
| Reactive | 4.67 | 4.8 | 1.10% | 0.065% | 0% |
| Naive predictive (Holt) | 5.20 | 5.7 | 1.10% | 0.065% | 0% |
| **Ours** | **2.51** | **3.3** | **0.39%** | **0.051%** | 0.065% |

Total legitimate loss is slightly higher for ours (0.116% vs 0.065%): the mitigation filter costs more legitimate
requests than the overload it prevents. Gains grow with attack intensity (excess replica-hours at ρ = 3: 10.7 → 5.6).

### 6.3 Ablations (simulated)

| Variant | Excess replica-h | SLA viol. in attacks | Legit wrongly mitigated |
|---|---|---|---|
| Full method | 2.51 | 0.39% | 0.065% |
| Univariate detector (requests only) | 3.48 | 0.62% | 0.32% (2.6% inside legit surges) |
| Forecast-residual attribution | 2.62 | 0.39% | 0.031% |
| Legit share = 1 − detector score | 2.51 | 0.46% | 0.74% |
| No detector (residual z > 3) | 5.01 | 1.07% | 0.065% |
| No MITIGATE | 2.51 | 8.64% | 0% |
| MITIGATE, scale on observed load | 5.20 | 0.24% | 0.065% |
| Perfect detection | 0.63 | 0.22% | 0.12% |

Multivariate input and a supervised detector are necessary; MITIGATE protects the SLA, and scaling on the estimate is
what saves replica-hours, so both are needed. The learned share has lower estimation error but the forecast-residual
variant mitigates half as much legitimate traffic at almost the same cost; with perfect detection the
forecast-residual variant is better on every simulated metric. Results keep their ranking when start-up delay
(30–120 s), SYN cost (0–0.5) and target utilisation (0.5–0.75) are varied.

## 7. Limitations and threats to validity

- **Synthetic components.** Surges and attacks are synthetic; only the baseline and the handshake noise come from real
  data. Results show what aggregate counters can separate under these generators, not field performance.
- **Simulated scaling.** No real CPU, latency, replicas or billing exist in the data; the capacity and latency models
  are simple and stated. The mitigation filter is idealised (drops attack traffic first).
- **One site, one era.** WorldCup98 is 1998 web traffic (HTTP/1.0, one connection per request assumed); three days.
- **Dataset overlap and independence.** TSD-DDoS is derived from CIC-DDoS2019 captures; it supplies noise and
  magnitude distributions only. Cross-dataset transfer (A → B/C) was not possible: no second legitimate source with
  real surges was available.
- **Generator knowledge.** The detector and the legitimate-share regressor are trained on the same generator families
  that appear in test (different days and seeds); unseen attack families are not tested.
- **Onset lag.** A single undetected onset window lets the policy scale up, and the 5 min stabilisation keeps that
  capacity (visible in figure 11). How much of the remaining excess this explains has not been measured.
- **Mimic floods** are unidentifiable from these counters by design; claims exclude them where stated.

## 8. Reproducibility

Code, `config.yaml` (all parameters, seed 20261030), the exact preprocessing and `scripts/run_all.sh` are in the
repository; runs log the git commit and config hash. Raw data sources and placement are listed in README.md.

## 9. Venue (to decide)

Candidates to check (page limits and deadlines NOT verified): a cloud-security workshop (e.g. ACM CCSW), a
network/service-management venue (e.g. IEEE/IFIP CNSM, NOMS workshops), or an applied-security conference
(e.g. IEEE CNS). The current scope fits a workshop or short paper better than a full top-venue paper.
