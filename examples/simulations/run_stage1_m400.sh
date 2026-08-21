#!/bin/bash
# T2 Stage 1 at the operating point: M=400, B0=10 screen, B1=100 top-up.
set -e
PY=/Users/allen/miniforge3/envs/waveome/bin/python
cd "$(dirname "$0")"
echo "=== phase 1: simulate + fit 400 outcomes, screen at B0=10 ==="
$PY sim_fdr_stage1_uniformity.py --M 400 --B 10 \
    --out-prefix sim_waveome_output/t2_m400
echo "=== phase 2: top up non-degenerate outcomes to B1=100 ==="
$PY sim_fdr_stage1_stratified.py --B1 100 \
    --prefix sim_waveome_output/t2_m400 \
    --out sim_waveome_output/t2_m400strat
echo "=== ALL DONE ==="
