# DECISIONS.md

Project: Counterfactual Attack-Aware Autoscaling Using Multivariate Temporal Forecasting
Last updated: 2026-11-04 (Phase 11 ablations; Alibaba file still not inspected)

## 0. Status summary
- One paper. Core contribution: a semi-synthetic benchmark with a known counterfactual (true legitimate demand known by construction) for attack-aware autoscaling, plus an identifiability analysis. Second research question (external context signals) is contingent on a feasibility check.
- The original claim cannot be validated on real data from the files collected so far: no real legitimate surges, no real resource metrics, one real attack event per file, and TSD labels/series have structural problems (sections 4 and 5).
- Environment, Dataset A cleaning (Phase 2) and the minute-level series (Phase 3) are done. TSD-DDoS v6 has been inspected and its labelling rule read.
- Literature check (Phase 5) done as a targeted search, not a systematic review.
- Next: inspect the processed Alibaba microservices file, prototype with the Zenodo three-day WorldCup98 subset, then write the benchmark specification.

## 1. Conventions
- Raw data is read-only (`chmod a-w`) and never edited. Cleaned data goes to `data/processed/`, generated synthetic data to `data/synthetic/`.
- Column prefixes in aggregated series: `x_` model inputs, `y_` labels, `a_` analysis-only.
- Terminology: "attack-like traffic" / "attack-labelled", not "bots" or "attackers". Results from a simulator are labelled "simulated", never "measured". Anything built on synthetic surges is labelled "synthetic".
- Environment: Fedora 44, VS Code, Python venv at `.venv`. Dependencies so far: pandas, numpy, matplotlib, pyarrow (pinned in `requirements.txt`). scikit-learn will be added when modelling starts.

## 2. Datasets and provenance
| Name | Source | Notes |
|---|---|---|
| Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv (Dataset A) | CIC-IDS2017 (capture dated 2017-07-07), NOT CIC-DDoS2019 | 225,745 rows x 85 cols; Label: DDoS 128,027 (56.7%), BENIGN 97,718 (43.3%) |
| jan-12 / mar-11 (Datasets B, C) | TSD-DDoS, derived from CIC-DDoS2019 pcaps. Mendeley Data v6 (DOI 10.17632/gyggjxs36z.6, CC BY 4.0) | Pcaps replayed with tcpreplay (up to 20 Gbps), captured with Wireshark, 5-second non-overlapping windows. v6 also has 4 labelled files (detection and severity, train and test), code and SHA256SUMS. DataPort copies have counts identical to v6 and additionally carry `File number` |
| machine_usage.csv | IEEE DataPort, Alibaba production cluster data v2018 | 322 rows, one machine (M_1937), blank header names; columns probably time offset (s), CPU %, memory % (UNVERIFIED). Not linkable to network traffic; superseded for simulator calibration by the Alibaba microservices file |
| pageviews.csv | Kaggle, Wikipedia Pageviews | Daily top-100 articles, 2016-01-01 to 2026-10-19, 393,400 rows; 7 article names missing |
| Processed Alibaba microservices (v2022) file | Provenance NOT yet confirmed (which trace, what processing) | To inspect with 04_inspect_alibaba.py. See section 7 for caveats |
| WorldCup98 web logs | Internet Traffic Archive (ITA); Zenodo copy (record 5145855, published 2021-07-29) with three days of logs | Planned for prototyping with the three-day Zenodo subset; which three days it covers still to be checked. ITA terms: free to use and redistribute provided the copyright notice stays unmodified. Full ITA release: about 1.3 billion requests, 249 binary files over 92 days; each record has a timestamp in seconds, anonymised client ID, object ID, size and status. 1998 site: state this limitation |

TODO: confirm licences and citation details for machine_usage.csv, pageviews.csv and the Alibaba file. Verify SHA-256 of the v6 files against SHA256SUMS.txt (cd data/raw/tsd_mendeley_v6 && sha256sum *.csv).

## 3. Dataset A: findings
- Timestamps are minute resolution (seconds always 0, 93 distinct minutes, 60 s gaps). 1 s, 5 s and 10 s windows are impossible. No AM/PM in the file (afternoon assumed).
- Timeline: benign 03:30-03:55 (26 min), attack 03:56-04:16 (21 minute bins), benign 04:17-05:02 (46 min). One attack event, abrupt on and off.
- DDoS flows: 128,024 of 128,027 come from 172.16.0.1, all go to 192.168.10.50:80, all TCP. The attack is not distributed in this capture. Benign traffic includes UDP and protocol 0; DDoS is 100% TCP.
- Benign flows to the victim service (31,340) exist only during the attack and ALL come from 172.16.0.1. They are not independent legitimate users. Dataset A therefore has NO usable real legitimate-demand ground truth for the attacked service and no pre-attack baseline for it.
- Benign traffic changes during the attack: about 2,480 flows/min vs 726 before and 582 after; median flow duration about 3 s vs about 0.03 s; fewer distinct benign source IPs (687 vs 783 before, 1,117 after). Cause not established.
- Flow direction is sometimes swapped in the CSV, so service membership is defined on either endpoint.
- CIC-IDS-2017 has documented errors in the literature (Engelen et al.; Lanvin et al.; Liu et al., CNS 2022, who released a corrected dataset). Check before relying on Dataset A; availability of the corrected release not yet verified.

### Minute-level series (03_minute_series.py, 93 time steps)
- Flows to the victim service form a step function (0 before and after, about 8,000/min during). Nothing to forecast.
- Packets per minute rise about 3-4x during the attack, with a taper at the end.
- Median flow duration is the only signal with ramp dynamics: builds after onset, peaks about 04:06 at about 5 s, declines before the attack stops (cause unverified).
- Source entropy drops to about 1 bit during the attack (3-5 outside): single-source artefact. Unique source IPs are noisy, no clear signature. `y_attack_frac` reaches 0.82.
- Verdict: a fine illustrative figure, not a forecasting benchmark (one event, 93 points, confounded benign series).

## 4. Dataset A: cleaning decisions
| Decision | Reason | Changes data? | Original kept? | Leakage / bias note |
|---|---|---|---|---|
| Strip whitespace in column names and labels | 70 padded names | names only | raw file untouched | none |
| Flag rows: `had_inf` (34), `had_nan` (4), `neg_duration` (2), `is_duplicate` (2) | audit trail, no deletion | adds columns | yes | none |
| Set +/-inf to NaN in Flow Bytes/s and Flow Packets/s | inf from zero-duration flows | yes | yes, in `*_raw` columns | any imputation later must be fit on train only |
| Keep negative values and the Init_Win_bytes = -1 sentinel | artefacts or sentinels, not errors | no | n/a | deleting them would bias toward flows with a SYN seen |
| Keep 2 duplicate rows (flagged) | no label conflicts, negligible | no | n/a | none |
| List, do not drop, constant columns (10) and duplicate columns (8) | dropping is a modelling decision | no | yes | keeps feature selection label-free |
| IPs, ports, Flow ID, Protocol, `segment`, Timestamp are analysis-only | shortcuts to the label (one IP, one port, all TCP) | no | n/a | prevents memorization of the testbed |
| Exclude all TCP flag-count columns from features | `SYN Flag Count` is identical to `Fwd PSH Flags`, and `ECE Flag Count` to `RST Flag Count`, so names are unreliable | no | yes | do not describe x_syn / x_ack / x_rst as SYN/ACK/RST counts |

Duplicate pairs (dropped -> kept): SYN Flag Count -> Fwd PSH Flags; ECE Flag Count -> RST Flag Count; Avg Fwd Segment Size -> Fwd Packet Length Mean; Fwd Header Length.1 -> Fwd Header Length; Subflow Fwd Packets -> Total Fwd Packets; Subflow Fwd Bytes -> Total Length of Fwd Packets; Subflow Bwd Packets -> Total Backward Packets; Subflow Bwd Bytes -> Total Length of Bwd Packets.

Result of 02_clean_flows.py after the flag-exclusion patch: 225,745 rows, 95 columns, flags 2 / 34 / 4 / 2, segments victim_service 159,367 and background 66,378, 53 candidate features. Outputs: `data/processed/flows_clean.parquet`, `flows_manifest.json` (includes SHA-256 of the raw file).

## 5. TSD-DDoS findings and decisions
- Labelled files from Mendeley v6. Jan = detection-train (5,531 windows: 5,302 Normal, 229 DoS Attack). Mar = detection-test (5,892 windows: 5,479 Normal, 413 DoS Attack). Severity: Jan Low 159 / Moderate 11 / High 59; Mar Low 308 / Moderate 11 / High 94. Feature columns of the labelled files match the base files exactly.
- Labelling rule (labeling-script.ipynb): H = #SYN - #ACK per 5 s window. Attack iff H > 0, Normal iff H < 0. Severity: Low 0<H<10, Moderate 100<H<1000, High H>1e6.
- The script deletes windows with H = 0 and with H in [10,100] or [1000,1e6], and renumbers the serial column. The published files show exactly this pattern (zero windows in those bands, zero idle windows, serial numbers 1..N). Treated as: the published files are filtered on the label-defining quantity, so they are NOT a contiguous time series. To confirm with the TSD paper/authors.
- Labels are a deterministic function of two input columns (SYN, ACK). Classification on these files is circular and not evidence of detection ability.
- Only High windows are floods: Jan 59 windows (index 3041-3099, about 4.9 min), Mar 94 windows (index 1487-1580, about 7.8 min), TCP > 1e5, contiguous. Low windows (159 / 308) are mostly near-idle noise (median TCP 4 / 1, ACK = 0, 22-25% of near-idle windows labelled attack). Moderate (11 per day) are about 900-SYN windows next to the flood.
- Flood onset/decay windows between 1e3 and 1e6 are missing, so ramp shapes are lost.
- Jan and Mar are not independent: 9 of 11 Moderate windows are identical across days (a run of consecutive windows at the end of Jan's last file matches a run in Mar, same order). Flood cores differ in scale (median SYN about 9.8M Jan, about 13.7M Mar). Cause unknown (shared pcap content or replay).
- The series is stitched from many pcap files: Jan 818 files (median 1 window, max 2,124), Mar 146 files (median 2, max 1,984). Rolling/lag features must be computed within `File number` segments, but because of the deletions even those are not guaranteed contiguous.
- Normal baseline shifts between days (median SYN-ACK 2 in Jan vs 0 in Mar).
- Replay speed varies, so window counts depend on replay and not only on the original traffic. Severity labels are traffic-based inferences, not CPU/memory measurements (README).
- Merged files with `file_number` and an `in_flood` flag: `data/processed/tsd_jan.csv`, `tsd_mar.csv`.
- Decision: no forecasting, lag or lead-time experiments on the TSD base files. Use them only for (a) the marginal distribution of normal window counts and (b) flood magnitudes, as unordered windows. Define attack-like windows by volume (TCP > 1e5), not by the TSD label. Exclude Low from any attack class. Do not treat Jan->Mar as an independent holdout.
- Option if real baseline dynamics are needed: regenerate contiguous 5 s windows with the authors' feature_extraction.cpp (in code.zip) from the CICDDoS2019 pcaps without the H-filter (heavy; check sizes and licence), or use CIC-DDoS2019 flow CSVs with timestamps. Neither verified.

## 6. Extra datasets: intended roles
- pageviews.csv: source of empirical surge magnitudes and decay shapes for synthetic legitimate surges. Limits: daily resolution, top-100 truncation (baselines before a spike are censored), views are not network flows. Compressing days to minutes is an assumption needing sensitivity analysis.
- machine_usage.csv: not used for the simulator (single machine, irregular sampling, no link to traffic).
- WorldCup98 (Zenodo three-day subset first): the main candidate for real flash-crowd request traffic at per-second resolution with client diversity. Prototype on the three-day subset; decide on the full ITA release afterwards.
- Alibaba microservices (processed file): candidate source for load -> CPU -> latency curves in the simulator. See section 7.

## 7. Resource feasibility notes (searched 2026-10-28; secondary sources flagged)
- Alibaba traces: ITA-style repository lists microservices as v2021 and microarchitecture as v2022; a GitHub issue discusses a "v2022 microservice trace" whose CPU, memory and call-rate values are min-max scaled to [0,1] (original constants requested; no official answer seen). Documentation (v2021): call rate/response time every 30 s, container CPU/memory every 60 s; about 12 hours; call graphs sampled at 0.5%. One source mentions shuffled copies of the 2021 and 2022 traces: verify time order. A derived Zenodo dataset (Mehran et al.) provides microservice time, call rate and replica counts per 30 s from the traces (filtered subset).
- Use of the Alibaba file: only for relative load-CPU-latency curves in the simulator, with an explicit and sensitivity-tested assumption mapping benchmark requests to Alibaba call rates. Labelled simulated.
- Wikimedia: per-article hourly pageviews are NOT available through the REST API (daily/monthly only; hourly is per-project). Hourly per-article data may exist in raw dump files: unverified.
- GDELT DOC 2.0 (secondary sources): free, keyless, volume timelines as percentage or raw counts, updated every 15 minutes, history back to about 2017, strict rate limit. Check GDELT's own terms.
- Wikipedia (2017+) and GDELT do not overlap 1998, so they cannot be paired with WorldCup98. Possible pairing: hourly Wikipedia article views as a demand proxy plus GDELT volume as the independent context signal (proxy; unverified end to end). Alternative: the 1998 match schedule as the context signal for WorldCup98 (schedule source not yet identified).

## 8. Phase 5: literature and novelty (targeted search, not a systematic review)
- Existing: SPARK (arXiv 2603.26833; legitimacy-gated predictive autoscaling on Kubernetes, early work, predictive model listed as future work); arXiv 2601.13515 (random-forest attack prediction, lowers HPA pod cap); ScaleShield (DRL, microservices); Yo-Yo and Tandem attacks and EDoS against autoscalers (Bremler-Barr et al. and others); flash-crowd vs DDoS discrimination (entropy, correlation, clustering, World Cup 98 plus ML, CDN crowd-event fingerprinting); robust load forecasting under attack (power-systems literature); Anycast Agility (offered-load estimation during DDoS); Social Auto-Scaling (Smith et al., 2018: scaling from social-network hashtag buzz); Qu et al. survey on web autoscaling (external information to anticipate bursts); FAST (social-signal page-view forecasting).
- Not found (absence in a limited search, not proof): explicit legitimate-demand (counterfactual) estimation under attack, evaluated against a known counterfactual and linked to scaling outcomes, with or without external context.
- Open searches: anomaly attribution / root-cause analysis for multivariate time series; causal and counterfactual forecasting (synthetic control, causal impact); papers using TSD-DDoS; licence/format details for WorldCup98 beyond the ITA terms.

## 9. Relationship to the earlier literature-survey draft
- Earlier draft: "Context-Aware Autoscaling for Unseen Traffic Events" (introduction and literature survey written; abstract, methodology, results empty). Formulation: P(legitimate | X_t, Z_t) with external signals Z_t.
- Merged into this project as RQ2. Its research-gap sentence must be narrowed: context-aware scaling (Social Auto-Scaling, Qu et al.) assumes surges are legitimate; attack-aware scaling (SPARK and others) ignores external context; none found that estimates legitimate demand counterfactually using both ("to our knowledge").
- Housekeeping in the draft: references [3] and [11] are listed but not cited in the visible text; stray quote mark at end of Section II-C; author emails missing "@"; abstract and index terms blank; check AWS and Google documentation before keeping the claim that commercial predictive scaling relies "primarily" on infrastructure telemetry.
- The draft's objectives mention CPU, memory and response time: these exist only via the simulator (Alibaba-calibrated) and must be labelled simulated.

## 10. Research questions, hypotheses and benchmark plan
- RQ1: how well can legitimate demand be recovered from aggregate traffic counters alone, and when is it unidentifiable?
- RQ2 (contingent on feasibility): does an exogenous context signal improve the estimate, especially for attacks that mimic legitimate demand? Risk: an attacker can time a flood to a real event (smokescreen).
- H1 (unverified): handshake-count attribution works for handshake-incomplete floods (SYN flood, SYN-ACK about 0) and fails for application-layer floods that complete handshakes (as in Dataset A).
- Benchmark plan: observed traffic = legitimate component (real surge shapes from WorldCup98; optionally a Wikipedia/GDELT proxy) + attack component (TSD flood magnitudes and Dataset A as a handshake-completing flood), so the true legitimate component is known by construction.
- Scenario families: flash crowd only; attack only; flash crowd + attack; Yo-Yo style pulsing attack; attack mimicking the legitimate profile; smokescreen attack timed to a real event.
- Baselines: reactive; naive predictive; legitimacy-gated (SPARK-style, reimplemented); robust forecasting; own forecast + residual + attribution method.
- Metrics: error of the legitimate-demand estimate vs known truth; resource-hours; unnecessary scale-ups; SLA violations; legitimate traffic wrongly mitigated (all simulated); plus precision, recall, F1, false-positive rate and detection lead time where labels are defined by the benchmark.
- Venue expectation: borderline for a top venue, more realistic for a workshop or applied-security venue; the target venue has not been chosen.

## 11. Scripts and generated files
Scripts (src/preprocessing unless noted): 01_inspect.py, 01b_followup_checks.py, 01c_victim_service.py, 01d_inspect_tsd.py, 01e_benign_victim_flows.py, 01f_inspect_tsd_v6.py, 01g_validate_tsd.py, 01h_tsd_label_structure.py, 02_clean_flows.py; src/features/03_minute_series.py; 04_inspect_alibaba.py (pending run).
Phase 6 (src/benchmark): 08_build_windows.py, 09_leakage_check.py. Phase 7 (src/models): common.py (shared loading, features, metrics), 10_baselines.py. Phase 8: 12_temporal.py. Figures: src/viz/11_figures.py -> results/figures/. Phase 9: 13_counterfactual.py (+ counterfactual.py). Phase 10: src/sim/14_simulate.py. Phase 11: src/sim/15_ablations.py; figures 11-15: src/viz/16_figures_method.py.
Outputs: results/01_inspect_report.txt, results/01c_minute_counts.csv, results/03_minute_series.png, results/01g_tsd_overview.png; data/processed/flows_clean.parquet, flows_manifest.json, minute_series_A.csv, tsd_jan.csv, tsd_mar.csv.

## 12. Open items
- Run 04_inspect_alibaba.py and record the file's provenance, scaling, time order and sampling intervals.
- Download the Zenodo three-day WorldCup98 subset (record 5145855), confirm which days it covers, and write a decoder/inspection script for per-second request counts and client diversity.
- Verify v6 checksums; confirm with the TSD authors/paper that windows were deleted before release.
- Confirm licences and citations for machine_usage.csv, pageviews.csv and the Alibaba file.
- Feasibility check for RQ2: hourly per-article Wikipedia data (dumps), GDELT terms, or a WorldCup98 match schedule as the context signal.
- Remaining literature searches (anomaly attribution, causal/counterfactual forecasting, TSD follow-up papers); fix the housekeeping items in the earlier draft.
- Decide the target venue; choose the legitimate-surge generation method and the request-to-TCP-counter mapping; write the simulator's capacity model.

## 13. Worldcup1998
- 2026-10-29: WorldCup98 TR2 (1998-05-04, 1,262,614 requests) decoded OK. Smooth diurnal pattern,
- max/median 2.15 per minute, no flash crowd. Use as real baseline only. Flash-crowd component
- will be synthetic unless TR1/TR3 show a real surge. RQ2 feasibility reduced accordingly.
- 2026-10-29: WorldCup98 TR2.gz verified (7,170,735 B, gzip OK, md5 matches Zenodo).
- TR2_baseline_5s.csv built: 17,280 bins, sum check OK. Step size 5 s (default, matches TSD windows).
- TR1/TR3 downloading in background; not yet inspected.

## 14. TSD Flood
- 2026-10-29: TSD flood windows (TCP > 1e5; 59 Jan, 94 Mar) are flat plateaus (Jan 0.88-1.07, Mar 0.76-1.008 of flood median); 
- same set at thresholds 3e4/1e5/3e5 (Mar 95 at 3e4). Normal windows near-idle (median 60/42 packets). 
- TSD gives no usable attack shape or flood/legit ratio: attack shape and rho are synthetic and swept.

- 2026-10-29: TSD label rule reproduced by counts: H = SYN - #ACK > 0 gives 59+170 = 229 (Jan), 94+319 = 413 (Mar),
matching label totals (per-window match not checked). Flood windows have SYN-ACK/SYN ~ 4e-5 (replay setup, not
live-victim behaviour). In floods ACK ~ RST, so ACK is not a clean handshake-completion counter (pending check).


## 15. WorldCup98 TR1/TR3 (2026-10-30)
- TR1.gz and TR3.gz downloaded from Zenodo 5145855; md5 matches for all three files, gzip OK. Scripts 05/06 ROOT changed from the absolute Linux path to the repo root (same pattern as the other scripts).
- TR1 = 1998-05-19 22:00 -> 05-20 22:00 UTC, 3,041,727 requests, 36,131 clients, req/min median 2,109. TR3 = 1998-05-22 22:00 -> 05-23 22:00 UTC, 2,342,242 requests, 28,829 clients, req/min median 1,582. Both: smooth diurnal pattern, no zero-request seconds.
- No real legitimate flash crowd in TR1/TR2/TR3. The surge component stays SYNTHETIC.
- TR1 contains one real attack-like burst: 5-s bins 8733-8737 (1998-05-20 10:07:45-10:08:10 UTC), about 2,800 req/5 s vs median 172. Two clients sent about 5,035 requests each for a single object within 22 s (status raw value 66). Kept in the data, flagged `a_baseline_burst`, excluded from evaluation via `a_eval_mask`. Candidate real test case for the method later (not labelled as an attack: intent unknown).

## 16. Phase 6: windowed benchmark dataset (2026-10-30)
- Script: src/benchmark/08_build_windows.py, config in config.yaml, pins in requirements.txt (system Python 3.13 on the Windows machine; no venv there).
- Step 5 s (matches TSD windows; a 5-min TSD-length attack spans 60 rows). Sensitivity on real baselines in results/08_window_sensitivity.csv: CV falls 0.47 -> 0.39 (TR2) and lag-1 ACF rises 0.79 -> 0.95 from 5 s to 300 s; at 60 s an attack is only 5 rows. Coarser steps not built yet.
- Split by calendar day, chronological: train TR2 (05-04), val TR1 (05-20), test TR3 (05-23). Gap of days between splits, far above purge_bins = 720 (1 h lookback). Caveat: traffic level grows across days (median req/5 s 69 / 172 / 132), so val/test are a level shift from train; this is intended as a distribution-shift test, not removed.
- Legitimate demand L = real baseline + SYNTHETIC surge (linear rise 5 or 30 min, 10-min plateau, exponential decay half-life 30 min, peak ratio 2/5/10 of the trailing 1-h median). Parametric because pageviews.csv is not on this machine; replacing it with empirical shapes is open.
- Attack A: SYNTHETIC. Flat plateau with abrupt on/off and multiplicative jitter U(0.88, 1.07), from the TSD flood findings (section 14). Durations 5 and 8 min (TSD 4.9 / 7.8 min). Intensity rho = A_peak / L_peak in {0.5, 1, 3}.
- Counter model: k = 1 connection per legitimate request. SYN flood adds SYNs only. App-layer flood adds SYN, ACK and served requests from 50 sources. Mimic = app-layer flood with the surge shape and legitimate requests-per-client. Spoofed SYNs add no clients (clients counted from served requests, as in the WorldCup98 log).
- Families: none, surge_only, attack_only, surge_plus_attack, yoyo (2 on / 3 off x 6), mimic, smokescreen (attack starts at surge peak). 49 configs x 2 seeds = 98 series per split, 1.69M rows per split. Seeds from SeedSequence([seed, split, config, rep]).
- Columns: x_ (requests, syn, ack, syn_minus_ack, clients, req_per_client), y_ (legit_requests, legit_clients, attack_conn, attack_active, surge_active, attack_frac), a_ (series, ts, bin, onset, family, params, baseline, burst flag, eval mask).
- Scaler: log1p + StandardScaler fit on train only, stored in data/synthetic/scaler_5s.json (applied at load time, scaled data not stored).
- Gate: src/benchmark/09_leakage_check.py, 11/11 checks pass (results/09_leakage_check.txt).
- RESOLVED 2026-10-31 (was: legitimate SYN - ACK = 0 exactly, making SYN floods trivially identifiable). TSD ACK counts every ACK-flagged packet, not handshake completions: in TSD non-flood windows (TCP <= 1e5, SYN - ACK < 100, SYN >= 3; 4,401 windows from Jan + Mar) ACK/SYN has median about 14 and 5-95% range about 2-67. New counter model: legitimate SYN ~ Poisson(k * requests), legitimate ACK ~ Poisson(SYN * r_t) with r_t drawn i.i.d. per window from that pool. App-layer and mimic floods use the same pool (conservative: packet-level look-alikes). SYN flood still adds SYNs only; no RST counter is modelled (friend's note: in TSD floods ACK ~ RST).
- Caveats: r_t is i.i.d. (TSD windows are not contiguous, so no temporal correlation is available); the TSD release deleted H = 0 windows, so the pool is biased; TSD is a replayed 2019 testbed, WorldCup98 is 1998 HTTP. Sensitivity-test the pool and k.
- Effect on the test split (results/10_syn_flood_separability.txt): legitimate windows have median SYN - ACK of -1,714 and 2.0% have SYN - ACK > 0. Window-level AUC for SYN-flood windows vs legitimate-only windows: x_syn_minus_ack 0.44, x_syn 0.96, x_requests 0.67. TSD rule (SYN - ACK > 0): recall 0.10 at FPR 0.02 (rho 0.5 / 1 / 3: 0.05 / 0.07 / 0.18). App floods: best single-counter AUC 0.64. SYN floods remain detectable through SYN volume without served requests, which is the H1 mechanism, not an artefact. Leakage check rerun: 11/11 pass.

## 17. Phase 7: CPU-only baselines (2026-10-31)
- Script: src/models/10_baselines.py. Fit on train (TR2), thresholds and spans chosen on val (TR1) by max F1, all numbers on test (TR3). Every run logged to results/phase7/runs.jsonl (git commit, config hash, seed, params, threshold, timings). Tables: results/phase7/detection_test.csv, detection_by_family_test.csv, forecast_test.csv.
- Phase 6 fix found here: the surge decay was cut at 1e-3 of peak, so each mimic attack stayed "active" for about 5 h of negligible traffic (46,178 of 53,234 attack rows in test). Cutoff now `surge.tail_cutoff` = 0.05 (about 2.2 h after the plateau), applied to surges and mimic attacks. Data rebuilt; leakage check 11/11. Attack-active rows per split 29,574, surge-active 88,362 (identical counts across splits because durations are deterministic; onsets differ).
- Not run, with reasons: seasonal-naive (each series is one day, no previous day exists); ARIMA (EWMA is the optimal ARIMA(0,1,1) forecast, statsmodels not installed); XGBoost (not installed, sklearn HistGradientBoosting used as the boosted-tree baseline). Time of day (a_bin) is not used as a feature: it is an analysis-only column.
- Detection, test split (F1 on all series / F1 without mimic series, FPR, false-positive rate inside legitimate surges):
  - TSD rule SYN - ACK > 0: F1 0.03, recall 0.03, FPR 0.020. SYN - ACK alone is uninformative after the handshake-noise fix (best val threshold flags everything).
  - static x_syn: 0.29 / 0.30, FPR 0.013, surge-FPR 0.25. static x_requests: 0.26 / 0.24, surge-FPR 0.28 (volume thresholds fire on legitimate surges).
  - static SYN per request: 0.21 / 0.65, precision 0.93, FPR 0.0002, surge-FPR 0.0000. Catches every SYN-flood window and no app-layer flood window: the H1 mechanism.
  - EWMA z-score on x_syn (span 60-1440 bins chosen on val): 0.24 / 0.34, event recall 0.97, surge-FPR 0.07.
  - HistGradientBoosting on 84 trailing features: 0.26 / 0.51, FPR 0.005. Already weak on val (F1 0.24 vs RF 0.41); likely cause: built-in early stopping on random rows of autocorrelated series overfits the train day's traffic level. Not tuned yet.
  - RandomForest (400k-row subsample, 200 trees, depth 14): 0.54 / 0.90, precision 0.95, FPR 0.0003, surge-FPR 0.0009, event recall 0.92, median detection delay 5 s. Strongest baseline.
  - Per family (RF window recall): SYN floods 0.92-1.00; app-layer floods 0.56 (yoyo) to 0.64 (attack only), 0.91-1.00 when stacked on a surge; mimic 0.22 (by design).
- Forecast of x_requests, test: h = 1 (5 s) MAE naive 21.6, trailing mean 20.9, EWMA 19.5, HGB 21.1; h = 12 (1 min) MAE 29.2 / 24.0 / 23.7 / 25.7. EWMA is the best simple forecaster.
- Counterfactual reference: during attacks, |observed requests - true legitimate| = 180 (reactive estimate). Forecasters track the observed series, so their error against true legitimate demand stays at 157-184 (h = 1) and 130-175 (h = 12). Forecasting alone does not recover legitimate demand: this is the gap Phase 9 (forecast + residual + attribution) must close.
- Gate: simple-baseline strength is known. Remaining gap is app-layer floods (single-counter AUC about 0.64, RF window recall 0.56-0.64) and mimic (unidentifiable by design). Phase 8 (LSTM/TCN) is only justified if it beats RF on app-layer and yoyo families; first fix HGB validation (time-blocked early stopping) and check feature reliance on absolute traffic level, since train/val/test days differ in level (median 69 / 172 / 132 requests per 5 s).
- Addendum (2026-11-02), prerequisites set by the gate above, same splits and protocol:
  - Shared code moved to src/models/common.py (load, features, level_free, thresholds, detection_metrics, by_family) so Phases 7 and 8 use identical evaluation.
  - Tree features are now float32 (memory). RF is unchanged; untuned HGB moved from F1-no-mimic 0.51 to 0.45 (different histogram bins), confirming it is unstable.
  - HGB with time-blocked validation (no built-in early stopping, 400 iterations, tree count chosen on val every 25): 0.37 / 0.66, FPR 0.004. Better than untuned HGB, still far below RF.
  - Level-free features only (ratios to trailing means, z-scores, per-request/per-client ratios): HGB 0.42 / 0.64; RF 0.53 / 0.88, surge-FPR 0.0098 (10x worse than RF on all features). Removing absolute levels does not help, so the train/val/test level shift was not what limited the models.

## 18. Phase 8: temporal models (2026-11-02)
- Script: src/models/12_temporal.py (PyTorch 2.12 CPU). Same protocol as Phase 7: fit on train (TR2), checkpoint and threshold chosen on val (TR1) by max F1, report on test (TR3). Runs logged to results/phase8/runs.jsonl; comparison against the Phase 7 tree models in results/phase8/comparison_test.csv and detection_by_family_test.csv. Log: results/local_run/12_temporal.txt (not committed).
- Inputs: 7 causal, level-free channels per 5 s window: log1p of requests, SYN, ACK and clients minus their trailing 1 h mean, and log SYN/request, ACK/SYN, request/client. Standardised with train statistics only.
- Models: causal dilated TCN (48 channels, dilations 1-64, receptive field 509 windows, about 42 min; 98k parameters) and a one-layer LSTM (64 hidden; 19k parameters). Training: 1024-window chunks (first 512 / 256 windows used only as warm-up, excluded from the loss), batch 32, half the chunks forced to contain an attack, BCE with pos_weight 5, Adam lr 1e-3, 1200 iterations, val F1 checked every 200. Two seeds each. Fit 19-30 min per run on 16 CPU threads.
- Test results (F1 all / F1 without mimic, FPR, surge-FPR, AP without mimic):
  - RF [lag features] (Phase 7): 0.54 / 0.89, 0.0003, 0.0009, AP 0.93.
  - TCN seed 0: 0.29 / 0.55, 0.0018, 0.034, AP 0.57. TCN seed 1: 0.43 / 0.74, 0.0009, 0.018, AP 0.80.
  - LSTM seed 0: 0.28 / 0.12, 0.042, 0.32, AP 0.05. LSTM seed 1: 0.39 / 0.43, 0.007, 0.13, AP 0.39.
- AP per family (that family's attack windows vs all legitimate windows), RF vs best TCN seed: attack_only app 0.67 vs 0.30; yoyo app 0.59 vs 0.31; yoyo SYN 0.94 vs 0.97; smokescreen app 1.00 vs 0.63; mimic app 0.44 vs 0.35. The LSTM is below 0.07 on every app-layer family.
- Stability: val F1 fluctuates by 0.1-0.2 between checks and peaks at different iterations per seed (TCN best at 200 and 800, LSTM at 800 and 1000). Seed-to-seed spread on test F1-no-mimic is 0.19 (TCN) and 0.31 (LSTM), larger than any gain over the baselines.
- Cost: inference 5.4 us/window (TCN) and 14-16 us/window (LSTM) vs RF 2.2 us/window; RF fits in 3 min.
- Gate: NOT passed. Neither temporal model beats RF on app-layer or yo-yo floods; the TCN only matches RF on SYN-flood families, which the SYN-per-request ratio already separates. Decision: keep RF [lag features] as the Phase 9 detector; no further LSTM/TCN tuning unless Phase 9 needs a sequence model for the counterfactual demand forecast. The app-layer gap (window recall 0.56-0.64) appears to be an information limit of the four aggregate counters, not a modelling limit, and is reported as such.
- Not tried (would be needed before claiming temporal models cannot help): longer training with a learning-rate schedule, more seeds, a Transformer, and multi-task training with the forecast target.

## 19. Phase 9: counterfactual legitimate-demand estimate (2026-11-04)
- Scripts: src/models/counterfactual.py (estimators, Holt forecast, action rule), src/models/13_counterfactual.py (`detect`: refits the Phase 7 RF so validation scores exist, a requests-only RF for the Phase 11 ablation, and a legitimate-share regressor; `evaluate`: chooses the estimator on val, reports on test). Outputs: results/phase9/{val_grid.csv, counterfactual_test.csv, chosen.json}; cached scores and estimates in data/synthetic/phase9_*.
- Detector: RandomForest on the 84 causal lag features (same parameters, rows and seed as Phase 7; val F1 0.406). Threshold by max F1 on val.
- Estimators (unflagged windows are taken as legitimate in all of them):
  - freeze: EWMA forecast of the *estimated* legitimate series (attack traffic never enters the forecast); in a flagged window the positive residual is attack.
  - client: the residual explained by new clients at the legitimate requests-per-client rate is legitimate. Fails because an app flood's own 50 sources add about 50 x 2.9 requests of "legitimate growth" (bias +120).
  - learned: HistGradientBoosting regressor (300 iterations, no early stopping) predicts the legitimate share of observed requests from the same lag features, trained on train-day attack windows (truth known by construction); estimate = share x observed in flagged windows.
- Choice on val (MAE in attack windows): learned 565, client 578 (span 720), freeze 579 (span 12). Narrow margin; val errors are higher than test because detection on val is weaker.
- Test, MAE vs true legitimate requests per 5 s (all attack windows / app-layer attack windows / legitimate surges without attack): observed-as-legitimate 180.4 / 204.9 / 0; freeze 112.1 / 123.9 / 0.02; client 124.0 / 140.0 / 0.01; learned 83.6 / 94.7 / 0.09. With true attack flags: learned 27.2, freeze 30.6. Detection is the bottleneck, not attribution.
- SYN floods add no requests, so request-level demand is unaffected; freeze still removes noise-level positive residuals there (MAE 13-49), learned does not (0.1-3).
- Actions (SCALE / HOLD / MITIGATE) are produced by the Phase 10 policy (counterfactual.action): MITIGATE while the detector fires, SCALE when the replica target changes, otherwise HOLD.

## 20. Phase 10: SIMULATED scaling evaluation (2026-11-04)
- Script: src/sim/14_simulate.py; parameters in config.yaml `sim`. Outputs: results/phase10/{policies_test.csv, policies_by_family_test.csv, action_mix_ours.json}. All numbers are simulated, never measured.
- Capacity model: a replica serves 50 work units per 5 s; a request costs 1, a SYN 0.2 (connection set-up). Replicas are ready 60 s after the request and billed from the request. Scale-down follows the maximum recommendation of the last 5 min (HPA-style stabilisation). Target utilisation 0.6; utilisation above 0.9 counts as an SLA violation (M/M/1: latency above 10x service time); work above capacity is dropped, legitimate share counted as lost. Decisions at t use information up to t-1.
- Mitigation: idealised filter while MITIGATE is active (flag at t-1): admits at most the estimate x 1.1 requests, dropping attack traffic first, and SYNs beyond the admitted requests x 1.1 (SYN cookies). Legitimate requests above the budget are "wrongly mitigated".
- Policies: oracle (true legitimate work incl. its own SYNs, perfect look-ahead over the start-up delay, perfect mitigation); reactive (last observed work); naive predictive (max of observed and a Holt forecast at the start-up horizon); ours (same rule on estimated legitimate work in flagged windows, observed work otherwise, plus MITIGATE).
- Fix during the phase: the first version charged the oracle and ours with exactly one SYN per legitimate request while reactive saw the Poisson-noisy SYN counts; that inflated reactive's unneeded scale-ups (12 vs 1 on attack-free series). The oracle now uses the true legitimate SYNs and ours uses observed work when not flagged.
- Test day, 98 series (excess replica-hours over oracle per series-day / unneeded scale-ups per series / SLA-violating windows during attacks / legitimate lost to overload / legitimate wrongly mitigated): reactive 4.67 / 4.8 / 1.10% / 0.065% / 0; naive predictive 5.20 / 5.7 / 1.10% / 0.065% / 0; ours 2.51 / 3.3 / 0.39% / 0.051% / 0.065%. Total replica-hours are dominated by baseline traffic (about 218-221 per series-day), so differences are reported as excess over the oracle.
- Honest negative: total legitimate loss is higher for ours (0.116% vs 0.065%); the filter costs more legitimate requests than the overload it prevents.
- Action mix (ours): HOLD 97.3%, SCALE 2.0%, MITIGATE 0.68% of windows.
- Observed weakness: a single undetected onset window lets ours scale up and the 5 min stabilisation keeps that capacity (results/figures/11_counterfactual_example.png, left). Its share of the remaining excess is not measured. Candidate fix (not done): require a scale-up signal to persist for 2 windows.

## 21. Phase 11: ablations and robustness (2026-11-04)
- Script: src/sim/15_ablations.py. Outputs: results/phase11/{ablations_test.csv, ablations_by_family_test.csv, robustness_scenarios_test.csv, sensitivity_simulator_test.csv}. Figures 11-15 from src/viz/16_figures_method.py. SIMULATED.
- Note: the estimator chosen on val (learned share) does not use the forecast residual; the forecast-residual estimator is therefore an explicit ablation row, not hidden inside "full method".
- Ablations, test (estimate MAE in attacks / excess replica-h / SLA violations in attacks / legit wrongly mitigated):
  - full method: 83.6 / 2.51 / 0.39% / 0.065%.
  - univariate detector (RF on the 12 request features; val F1 0.21): 116.5 / 3.48 / 0.62% / 0.32% (2.6% of legitimate surge traffic wrongly mitigated). Multivariate input matters.
  - forecast-residual attribution (freeze): 112.1 / 2.62 / 0.39% / 0.031%. Higher estimate error, same scaling outcome, half the collateral.
  - legitimate share = 1 - detector score (no learned attribution): 137.4 / 2.51 / 0.46% / 0.74%. Attribution quality drives collateral.
  - no detector (EWMA residual z > 3 flags, freeze): 174.9 / 5.01 / 1.07% / 0.065%. No better than reactive.
  - no MITIGATE (scale on estimate only): - / 2.51 / 8.64% / 0. Worse SLA than reactive: capacity is sized for legitimate demand while the attack still reaches the servers.
  - MITIGATE but scale on observed: - / 5.20 / 0.24% / 0.065%. Best SLA, no cost saving.
  - perfect detection: 27.2 / 0.63 / 0.22% / 0.12%; perfect detection with forecast-residual: 30.6 / 0.30 / 0.15% / 0.15%.
- Robustness by attack intensity (excess replica-h reactive -> ours; SLA in attacks): rho 0.5: 1.9 -> 1.3, 0.30% -> 0.05%; rho 1: 3.7 -> 1.6, 1.17% -> 0.52%; rho 3: 10.7 -> 5.6, 1.84% -> 0.59%. By surge peak ratio only the 5x group contains attacks (combined scenarios), so no trend can be read there.
- Simulator sensitivity (start-up 30/60/120 s, SYN cost 0/0.2/0.5, target utilisation 0.5/0.6/0.75): ours keeps the lowest replica-hours (relative to oracle) and the lowest SLA violations during attacks in every setting; its wrongly-mitigated share stays 0.065% (independent of the capacity model). Longer start-up widens the gap (SLA in attacks at 120 s: reactive 2.52%, ours 0.91%).
- Cross-dataset (A -> B/C): not possible. TSD-DDoS and Dataset A have no real legitimate surges and are not independent of CIC-DDoS2019; they are used only for noise and magnitudes. Recorded as a threat to validity.
