#!/bin/bash
# Sweep of the penalty on log acceleration factors (issue #40), centralized fits
# without Fed-BioMed (fit_local_centralized.py) on the synthetic multi-seed data
# (ground truth available), followed by get_metrics.py (held-out personalization,
# fit quality and parameter recovery).
#
# Usage: run_alpha_penalty_sweep.sh [SEED ...]
#
# Results: <RESULTS_DIR>/<variant>/<FL_PROG_DATE>/<tag>/
# ADNI: DATASET=adni_harmonized CONDITION=iid TAG_SUFFIX=_scaled_CN50_min
#       DATA_DIR=<main checkout>/data/2026_11_08
# Environment overrides: SEEDS, DATASET, CONDITION, TAG_SUFFIX, DATA_DIR, RESULTS_DIR, VARIANTS,
# N_JOBS, N_THREADS, VENV_DIR, FL_PROG_DATE, EXTRA_ARGS

set -uo pipefail

SCRIPTS_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT_DIR=$(dirname "$SCRIPTS_DIR")
MAIN_DIR="$(dirname "$ROOT_DIR")/fl-prog" # data/results live in the main checkout
VENV_DIR=${VENV_DIR:-"${MAIN_DIR}/.venv"}
export PATH="${VENV_DIR}/bin:${PATH}"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export FL_PROG_DATE=${FL_PROG_DATE:-$(date +%Y_%m_%d)}
export RNG_SEED=3791 # model and personalization seed (as in run_synthetic_seeds.sh)

SEEDS=${SEEDS:-"1 2 3 4 5 6 7 8 9 10"}
[ "$#" -gt 0 ] && SEEDS="$*"
DATASET=${DATASET:-synth_adnilike}
CONDITION=${CONDITION:-overlap100}
TAG_SUFFIX=${TAG_SUFFIX:-} # e.g. _scaled_CN50_min for ADNI
DATA_DIR=${DATA_DIR:-"${MAIN_DIR}/data/2026_11_11"}
RESULTS_DIR=${RESULTS_DIR:-"${MAIN_DIR}/results/2026_11_23-alpha_penalty"}
N_JOBS=${N_JOBS:-10}
N_THREADS=${N_THREADS:-2}
# same settings as run_synthetic_seeds.sh, except the acceleration factor penalty
BASE_ARGS="--n-rounds 30 --n-updates 25 --learning-rate 0.05 --lambda-time-shifts 0.01 ${EXTRA_ARGS:-}"

# variant -> acceleration factor penalty args
declare -A VARIANT_ARGS=(
    [none]="--penalty-acceleration-factors l2 --lambda-acceleration-factors 0"
    [l1_0.001]="--penalty-acceleration-factors l1 --lambda-acceleration-factors 0.001"
    [l1_0.01]="--penalty-acceleration-factors l1 --lambda-acceleration-factors 0.01"
    [l1_0.1]="--penalty-acceleration-factors l1 --lambda-acceleration-factors 0.1"
    [l2_0.001]="--penalty-acceleration-factors l2 --lambda-acceleration-factors 0.001"
    [l2_0.01]="--penalty-acceleration-factors l2 --lambda-acceleration-factors 0.01"
    [l2_0.1]="--penalty-acceleration-factors l2 --lambda-acceleration-factors 0.1"
    [l2_1]="--penalty-acceleration-factors l2 --lambda-acceleration-factors 1"
    # experimental options of fit_local_centralized.py
    [l2_0.01-centered]="--penalty-acceleration-factors l2 --lambda-acceleration-factors 0.01 --center-time"
    [l2_0.1-centered]="--penalty-acceleration-factors l2 --lambda-acceleration-factors 0.1 --center-time"
    [l2_0.1-90rounds]="--penalty-acceleration-factors l2 --lambda-acceleration-factors 0.1 --n-rounds 90"
    [em]="--learn-acceleration-prior"
    [em-centered]="--learn-acceleration-prior --center-time"
    [em-90rounds]="--learn-acceleration-prior --n-rounds 90"
    [oracle-l2_0.1]="--penalty-acceleration-factors l2 --lambda-acceleration-factors 0.1 --oracle-population"
    [oracle-em]="--learn-acceleration-prior --oracle-population"
    # time shift penalty (acceleration factors: default L2 0.1)
    [ts_l2_0.001]="--lambda-time-shifts 0.001"
    [ts_l2_0.003]="--lambda-time-shifts 0.003"
    [ts_l2_0.01]="--lambda-time-shifts 0.01"
    [ts_l2_0.05]="--lambda-time-shifts 0.05"
    [ts_l2_0.1]="--lambda-time-shifts 0.1"
)
VARIANTS=${VARIANTS:-"none l1_0.001 l1_0.01 l1_0.1 l2_0.001 l2_0.01 l2_0.1 l2_1"}

fit_one() {
    local variant=$1 seed=$2 tag dpath_results run_tag
    tag="${DATASET}_${CONDITION}_s${seed}_heldout10pooled_seed${seed}${TAG_SUFFIX}"
    dpath_results="${RESULTS_DIR}/${variant}"
    export OMP_NUM_THREADS=$N_THREADS MKL_NUM_THREADS=$N_THREADS
    # shellcheck disable=SC2086
    run_tag=$(python "${SCRIPTS_DIR}/fit_local_centralized.py" --tag "$tag" \
        --data-dir "$DATA_DIR" --results-dir "$dpath_results" \
        $BASE_ARGS ${VARIANT_ARGS[$variant]} | tail -n 1) &&
        python "${SCRIPTS_DIR}/get_metrics.py" --tag "$tag" \
            --results-dir "$dpath_results" --run-tag "$run_tag" >/dev/null 2>&1 &&
        echo "[OK] ${variant} seed ${seed}" ||
        echo "[ERROR] ${variant} seed ${seed}"
}
export -f fit_one
export SCRIPTS_DIR DATASET CONDITION TAG_SUFFIX DATA_DIR RESULTS_DIR N_THREADS BASE_ARGS
# associative arrays cannot be exported: declared again in each parallel job
VARIANT_ARGS_SERIALIZED=$(declare -p VARIANT_ARGS)
export VARIANT_ARGS_SERIALIZED

for variant in $VARIANTS; do
    [ -n "${VARIANT_ARGS[$variant]+x}" ] || { echo "Unknown variant ${variant}"; exit 1; }
    # create the dated dir and "latest" link once, before running seeds in parallel
    mkdir -p "${RESULTS_DIR}/${variant}"
    python -c "from fl_prog.utils.io import get_dpath_latest; get_dpath_latest('${RESULTS_DIR}/${variant}', use_today=True)"
done

# shellcheck disable=SC2016
parallel --line-buffer --jobs "$N_JOBS" \
    'eval "$VARIANT_ARGS_SERIALIZED"; fit_one {1} {2}' ::: $VARIANTS ::: $SEEDS
