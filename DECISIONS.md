# DECISIONS.md

Project: Counterfactual Attack-Aware Autoscaling Using Multivariate Temporal Forecasting
Last updated: 2026-10-28 (state before inspecting the Alibaba file)

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