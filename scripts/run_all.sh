#!/usr/bin/env bash
# Rebuild the benchmark and every result from the raw files (see README.md, "Data").
# Seeds and every parameter live in config.yaml. CPU only; about 1.5 h on 16 cores (Phase 8 is most of it).
# Phase 8 needs PyTorch; skip it with SKIP_TEMPORAL=1 (Phase 9 onward does not depend on it).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}

for d in TR1 TR2 TR3; do $PY src/preprocessing/06_build_baseline.py "data/raw/worldcup98/$d.gz"; done
$PY src/benchmark/08_build_windows.py          # Phase 6: windowed semi-synthetic benchmark
$PY src/benchmark/09_leakage_check.py          # leakage gate: must print 11/11 PASS
$PY src/models/10_baselines.py                 # Phase 7: baselines
if [ "${SKIP_TEMPORAL:-0}" != 1 ]; then        # Phase 8: temporal models, 2 seeds each
  for m in tcn lstm; do for s in 0 1; do
    $PY src/models/12_temporal.py train --model $m --seed $s --iters 1200 --eval-every 200
  done; done
  $PY src/models/12_temporal.py summarize
fi
$PY src/models/13_counterfactual.py detect     # Phase 9: detectors and legit-share regressor
$PY src/models/13_counterfactual.py evaluate   # Phase 9: counterfactual estimate (chosen on val)
$PY src/sim/14_simulate.py                     # Phase 10: SIMULATED scaling policies
$PY src/sim/15_ablations.py                    # Phase 11: ablations and robustness
$PY src/viz/11_figures.py                      # figures 01-10
$PY src/viz/16_figures_method.py               # figures 11-15
