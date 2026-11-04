# Counterfactual Attack-Aware Autoscaling Using Multivariate Temporal Forecasting

A semi-synthetic benchmark and method for estimating **legitimate demand** while attack-like traffic is present,
and for using that estimate to scale (and mitigate) instead of scaling on raw observed load.

Observed traffic = real legitimate baseline (WorldCup98 web logs) + **synthetic** legitimate surge + **synthetic**
attack-like traffic, so the true legitimate demand is known by construction. All scaling results come from a
documented simulator and are **simulated, never measured**.

- Design decisions, every result and its caveats: [DECISIONS.md](DECISIONS.md)
- Benchmark specification: [docs/BENCHMARK_SPEC.md](docs/BENCHMARK_SPEC.md)
- Paper draft: [paper/draft.md](paper/draft.md)
- Figures: [results/figures/](results/figures/)

## Pipeline

| Phase | Script | Output |
|---|---|---|
| 6 Benchmark | `src/benchmark/08_build_windows.py`, `09_leakage_check.py` | `data/synthetic/windows_5s_*.parquet`, leakage check 11/11 |
| 7 Baselines | `src/models/10_baselines.py` | `results/phase7/` |
| 8 Temporal models | `src/models/12_temporal.py` | `results/phase8/` (gate not passed; RF kept) |
| 9 Counterfactual estimate | `src/models/13_counterfactual.py` (+ `counterfactual.py`) | `results/phase9/` |
| 10 Scaling simulation | `src/sim/14_simulate.py` | `results/phase10/` |
| 11 Ablations, robustness | `src/sim/15_ablations.py` | `results/phase11/` |
| Figures | `src/viz/11_figures.py`, `src/viz/16_figures_method.py` | `results/figures/` |

Shared loading, features and metrics: `src/models/common.py`. Splits are whole days: train TR2 (1998-05-04),
validation TR1 (05-20), test TR3 (05-23). Everything is fitted on train, chosen on validation and reported on test.

## Data

Raw files are not in the repository (`data/` is git-ignored). Place them as follows:

- `data/raw/worldcup98/TR1.gz`, `TR2.gz`, `TR3.gz`: the three-day WorldCup98 subset (Zenodo record 5145855; original
  logs from the Internet Traffic Archive, whose copyright notice must be kept).
- `data/raw/jan-12-dataset-final-version-csv-ieeedataport.csv`, `mar-11-...csv`: TSD-DDoS (Mendeley Data v6,
  DOI 10.17632/gyggjxs36z.6, CC BY 4.0), used for the handshake ACK/SYN noise and flood magnitudes.

## Reproduce

```bash
pip install -r requirements.txt
bash scripts/run_all.sh
```

Every parameter and seed is in [config.yaml](config.yaml) (`seed: 20261030`). Runs append to `runs.jsonl` files with
the git commit and a hash of `config.yaml`. Tested on Windows 11, Python 3.13, CPU only. Set `SKIP_TEMPORAL=1` to skip
Phase 8 (PyTorch), which no later phase depends on.

## Main results (test day; scaling numbers SIMULATED)

- Legitimate-demand error during attacks (MAE, requests per 5 s): 180 when observed load is taken as legitimate,
  84 with the method, 27 with perfect detection: detection is the bottleneck.
- Versus reactive autoscaling: excess replica-hours over an oracle 4.7 → 2.5 per series-day, SLA-violating windows
  during attacks 1.10% → 0.39%, at the cost of 0.065% of legitimate requests wrongly mitigated.
- Ablations: the multivariate detector and MITIGATE matter most; scaling on the estimate is what saves
  replica-hours. A purely unsupervised residual detector is no better than reactive scaling.
