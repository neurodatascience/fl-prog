#!/bin/bash
# Multi-seed end-to-end ADNI experiment.
#
# The seed controls the data (pooled subject hold-out and, for federated setups,
# the site split), not model fitting (fixed MODEL_SEED for every model).
#
# Usage: run_adni_seeds.sh {prep|start-nodes|stop-nodes|fit|forecast|summarize|all} [SEED ...]
#
# "fit" also scores held-out subjects personalized on their first N visits, for each
# N in N_TIMEPOINTS. "forecast" recomputes the metrics of fitted models without
# refitting, e.g. with other N_TIMEPOINTS. Run it after "fit"; "all" doesn't include it.
#
# All runs write to results/<FL_PROG_DATE>/<tag>/.
#
# Environment overrides: SEEDS, FL_PROG_DATE, RESULTS_DIR, LOG_DIR, MAX_USER_THREADS,
# FBM_EXTRA_ARGS / DPMOST_EXTRA_ARGS / LEASPY_EXTRA_ARGS (e.g. for smoke tests),
# N_TIMEPOINTS, FORECAST_AFTER, FORECAST_JOBS, FORECAST_THREADS, FORECAST_EXTRA_ARGS

set -uo pipefail

SCRIPTS_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT_DIR=$(dirname "$SCRIPTS_DIR")
export PATH="${ROOT_DIR}/.venv/bin:${PATH}"

# ===== settings =====
SEEDS=${SEEDS:-"1 2 3 4 5 6 7 8 9 10"}
MODEL_SEED=3791
TEST_FRACTION=0.1
TEST_PCT=10
export FL_PROG_DATE=${FL_PROG_DATE:-$(date +%Y_%m_%d)}
RESULTS_DIR=${RESULTS_DIR:-"${ROOT_DIR}/results"}
LOG_DIR=${LOG_DIR:-"${RESULTS_DIR}/${FL_PROG_DATE}/logs-adni"}
NODE_PIDS_FILE="${LOG_DIR}/node_pids.txt"
MAX_USER_THREADS=${MAX_USER_THREADS:-3000}
NICE="nice -n 10"

IDPS="${ROOT_DIR}/data/_harmonized_data/ADNI_freesurfer7p1p1-forSantiago.csv-10-CN-0.5-3791-HBR_Sb_ct_DK_lifespan_79K_100sites-HBR_Sb_sa_lifespan_DK_46K_59sites-HBR_Sb_sc_lifespan_79K_100sites/harmonized.csv"
CONFIG="${ROOT_DIR}/data/_Marco_data/raw/config.json"
SCALING_CONFIG="${ROOT_DIR}/data/_Marco_data/raw/scaling-scaled_CN50_min.json"
ADNI_MERGE="/data/origami/nikhil/datasets/adni/derivatives/idp/experiment_data/ADNIMERGE.csv"
SCALING_SUFFIX="_scaled_CN50_min"

FBM_ARGS="--n-rounds 30 --n-updates 25 --learning-rate 0.05 \
--lambda-time-shifts 0.01 --penalty-acceleration-factors l2 --lambda-acceleration-factors 0.1 \
--with-acceleration --with-scaling --training-replies --aggregated-params \
--random-seed ${MODEL_SEED} ${FBM_EXTRA_ARGS:-}"
DPMOST_ARGS="--random-seed ${MODEL_SEED} ${DPMOST_EXTRA_ARGS:-}"
LEASPY_ARGS="--random-seed ${MODEL_SEED} ${LEASPY_EXTRA_ARGS:-}"

# get_metrics.py --personalize-n-timepoints and --forecast-after
N_TIMEPOINTS=${N_TIMEPOINTS:-"1 2 3"}
# shellcheck disable=SC2086 # split N_TIMEPOINTS
FORECAST_AFTER=${FORECAST_AFTER:-$(printf "%s\n" $N_TIMEPOINTS | sort -n | tail -n 1)}
FORECAST_JOBS=${FORECAST_JOBS:-10} # tags processed at the same time
FORECAST_THREADS=${FORECAST_THREADS:-2}
# other get_metrics.py args, e.g. "--personalize-seed 3791 --leaspy-seed 3791"
FORECAST_EXTRA_ARGS=${FORECAST_EXTRA_ARGS:-}
METRICS_ARGS="--forecast-after ${FORECAST_AFTER}"
for _n in $N_TIMEPOINTS; do METRICS_ARGS+=" --personalize-n-timepoints ${_n}"; done

# condition -> get_adni_data.py args
declare -A COND_ARGS=(
    [iid]="--iid"
    [noniid_site]="--non-iid --non-iid-strategy site"
    [noniid_diag]="--non-iid --non-iid-strategy diagnosis"
)
CONDITIONS="iid noniid_site noniid_diag"

# Fed-BioMed lane -> "fedbiomed dir|condition|setup"
declare -A FBM_LANES=(
    [iid-federated]="fedbiomed|iid|federated"
    [iid-centralized]="fedbiomed-centralized|iid|centralized"
    [noniid-site]="fedbiomed-noniid-site|noniid_site|federated"
    [noniid-diag]="fedbiomed-noniid-diag|noniid_diag|federated"
)
# nodes to start in each Fed-BioMed dir
declare -A FBM_NODES=(
    [fedbiomed]="1 2 3 4 5"
    [fedbiomed-centralized]="centralized"
    [fedbiomed-noniid-site]="1 2 3 4 5"
    [fedbiomed-noniid-diag]="1 2 3 4 5"
)

# ===== helpers =====
log() { echo "[$(date '+%F %T')] $*"; }

set_threads() {
    local n=$1
    export OMP_NUM_THREADS=$n MKL_NUM_THREADS=$n OPENBLAS_NUM_THREADS=$n \
        NUMEXPR_NUM_THREADS=$n POLARS_MAX_THREADS=$n
}

base_tag() { echo "adni_harmonized_${1}_s${2}"; }
final_tag() { echo "$(base_tag "$1" "$2")_heldout${TEST_PCT}pooled_seed${2}${SCALING_SUFFIX}"; }

wait_for_threads() {
    local n
    while true; do
        n=$(ps -L -u "$USER" --no-headers | wc -l)
        [ "$n" -lt "$MAX_USER_THREADS" ] && return 0
        log "Waiting: ${n} user threads >= ${MAX_USER_THREADS}"
        sleep 60
    done
}

run() {
    echo "[RUN] $*"
    "$@"
}

# ===== prep =====
prep_one() {
    local cond=$1 seed=$2
    local tag split_tag final
    tag=$(base_tag "$cond" "$seed")
    split_tag="${tag}_heldout${TEST_PCT}pooled_seed${seed}"
    final=$(final_tag "$cond" "$seed")

    if [ -f "${ROOT_DIR}/data/${FL_PROG_DATE}/${final}/${final}.json" ]; then
        log "Data for ${final} already exists, skipping prep"
    else
        # shellcheck disable=SC2086
        run "${SCRIPTS_DIR}/get_adni_data.py" --tag "$tag" ${COND_ARGS[$cond]} \
            --rng-seed "$seed" --idps "$IDPS" --config "$CONFIG" \
            --adni-merge "$ADNI_MERGE" || return 1
        run "${SCRIPTS_DIR}/split_train_test.py" --tag "$tag" --mode subject \
            --test-fraction "$TEST_FRACTION" --random-seed "$seed" --pooled \
            --overwrite || return 1
        run "${SCRIPTS_DIR}/merge_data.py" --tag "$split_tag" || return 1
        run "${SCRIPTS_DIR}/scale_adni_data.py" --tag "$split_tag" \
            --config "$SCALING_CONFIG" || return 1
    fi

    local lane dpath_fbm lane_cond
    for lane in "${!FBM_LANES[@]}"; do
        IFS='|' read -r dpath_fbm lane_cond _ <<<"${FBM_LANES[$lane]}"
        [ "$lane_cond" = "$cond" ] || continue
        run "${SCRIPTS_DIR}/add_datasets_to_nodes.py" --tag "$final" \
            --nodes-dir "${ROOT_DIR}/${dpath_fbm}" || return 1
    done
}

check_test_sets() {
    local seed=$1
    python - "$seed" "$ROOT_DIR/data/${FL_PROG_DATE}" $(for c in $CONDITIONS; do final_tag "$c" "$seed"; done) <<'EOF'
import sys
from pathlib import Path

import pandas as pd

seed, dpath, *tags = sys.argv[1:]
subject_sets = {}
for tag in tags:
    df = pd.read_csv(Path(dpath) / tag / f"{tag}-test.tsv", sep="\t", dtype={"rid": str})
    subject_sets[tag] = set(df["rid"])
sizes = {tag: len(s) for tag, s in subject_sets.items()}
if len({frozenset(s) for s in subject_sets.values()}) != 1:
    sys.exit(f"[ERROR] seed {seed}: test sets differ across conditions: {sizes}")
print(f"[OK] seed {seed}: identical test sets across conditions ({sizes})")
EOF
}

cmd_prep() {
    set_threads 4
    local seed cond
    for seed in $SEEDS; do
        for cond in $CONDITIONS; do
            log "===== prep ${cond} seed ${seed} ====="
            prep_one "$cond" "$seed" || { log "[ERROR] prep failed: ${cond} seed ${seed}"; return 1; }
        done
        check_test_sets "$seed" || return 1
    done
}

# ===== nodes =====
cmd_start_nodes() {
    set_threads 1
    mkdir -p "${LOG_DIR}/nodes"
    : >"$NODE_PIDS_FILE"
    local dpath_fbm node
    for dpath_fbm in "${!FBM_NODES[@]}"; do
        for node in ${FBM_NODES[$dpath_fbm]}; do
            wait_for_threads
            # own session/process group, so that stopping also stops node subprocesses
            setsid $NICE fedbiomed node -p "${ROOT_DIR}/${dpath_fbm}/node-${node}" start \
                >"${LOG_DIR}/nodes/${dpath_fbm}-node-${node}.log" 2>&1 </dev/null &
            echo "$! ${dpath_fbm}/node-${node}" >>"$NODE_PIDS_FILE"
            log "Started ${dpath_fbm}/node-${node} (PGID $!)"
        done
    done
}

cmd_stop_nodes() {
    [ -f "$NODE_PIDS_FILE" ] || { log "No node PID file"; return 0; }
    local pgid name
    while read -r pgid name; do
        kill -INT -- "-${pgid}" 2>/dev/null && log "Stopping ${name} (PGID ${pgid})"
    done <"$NODE_PIDS_FILE"
    sleep 15
    while read -r pgid name; do
        # force-stop anything left in the process group
        kill -TERM -- "-${pgid}" 2>/dev/null && log "Force-stopped ${name} (PGID ${pgid})"
    done <"$NODE_PIDS_FILE"
    rm -f "$NODE_PIDS_FILE"
}

# ===== fit =====
fit_fbm_lane() {
    local lane=$1 dpath_fbm cond setup seed tag run_tag
    IFS='|' read -r dpath_fbm cond setup <<<"${FBM_LANES[$lane]}"
    mkdir -p "${LOG_DIR}/${lane}"
    set_threads 1
    for seed in $SEEDS; do
        tag=$(final_tag "$cond" "$seed")
        wait_for_threads
        log "[${lane}] seed ${seed}: ${tag}"
        {
            # this run's results file only (other lanes can write to the same tag)
            # shellcheck disable=SC2086
            run_tag=$(python "${SCRIPTS_DIR}/run_fedbiomed.py" --tag "$tag" \
                --fedbiomed-dir "${ROOT_DIR}/${dpath_fbm}" --setup "$setup" \
                $FBM_ARGS --print-run-tag | tail -n 1) &&
                # shellcheck disable=SC2086
                run $NICE "${SCRIPTS_DIR}/run_fedbiomed.py" --tag "$tag" \
                    --fedbiomed-dir "${ROOT_DIR}/${dpath_fbm}" --setup "$setup" \
                    --results-dir "$RESULTS_DIR" $FBM_ARGS &&
                # shellcheck disable=SC2086
                run $NICE "${SCRIPTS_DIR}/get_metrics.py" --tag "$tag" \
                    --results-dir "$RESULTS_DIR" --run-tag "$run_tag" $METRICS_ARGS
        } >"${LOG_DIR}/${lane}/seed${seed}.log" 2>&1 ||
            log "[ERROR] [${lane}] seed ${seed} failed, see ${LOG_DIR}/${lane}/seed${seed}.log"
    done
    log "[${lane}] done"
}

fit_baseline_one() {
    # $1: model (dpmost|leaspy), $2: seed
    local model=$1 seed=$2 tag args
    tag=$(final_tag iid "$seed")
    [ "$model" = dpmost ] && args=$DPMOST_ARGS || args=$LEASPY_ARGS
    wait_for_threads
    log "[${model}] seed ${seed}: ${tag}"
    {
        # shellcheck disable=SC2086
        run $NICE "${SCRIPTS_DIR}/run_${model}.py" --tag "$tag" \
            --results-dir "$RESULTS_DIR" $args &&
            run $NICE "${SCRIPTS_DIR}/get_metrics.py" --tag "$tag" \
                --results-dir "$RESULTS_DIR" --run-tag "${model}-*" $METRICS_ARGS
    } >"${LOG_DIR}/${model}/seed${seed}.log" 2>&1 ||
        { log "[ERROR] [${model}] seed ${seed} failed, see ${LOG_DIR}/${model}/seed${seed}.log"; return 1; }
}

fit_baseline_lane() {
    local model=$1
    mkdir -p "${LOG_DIR}/${model}"
    set_threads 4
    export -f fit_baseline_one wait_for_threads log run final_tag base_tag
    export SCRIPTS_DIR RESULTS_DIR LOG_DIR DPMOST_ARGS LEASPY_ARGS NICE MAX_USER_THREADS \
        TEST_PCT SCALING_SUFFIX METRICS_ARGS
    # shellcheck disable=SC2086
    parallel --line-buffer --jobs 2 fit_baseline_one "$model" ::: $SEEDS
    log "[${model}] done"
}

cmd_fit() {
    mkdir -p "$LOG_DIR"
    # create the dated results dir/symlink before running jobs in parallel
    ( cd "$ROOT_DIR" && python -c "from fl_prog.utils.io import get_dpath_latest; get_dpath_latest('${RESULTS_DIR}', use_today=True)" ) || return 1
    local pids=() lane
    for lane in "${!FBM_LANES[@]}"; do
        fit_fbm_lane "$lane" &
        pids+=($!)
    done
    fit_baseline_lane dpmost &
    pids+=($!)
    fit_baseline_lane leaspy &
    pids+=($!)
    wait "${pids[@]}"
    log "All lanes done"
}

# ===== forecast =====
# recompute the metrics of all fitted models of a tag (no nodes needed)
forecast_one() {
    # $1: tag
    local tag=$1
    wait_for_threads
    log "[forecast] ${tag}"
    # shellcheck disable=SC2086
    run $NICE "${SCRIPTS_DIR}/get_metrics.py" --tag "$tag" \
        --results-dir "$RESULTS_DIR" $METRICS_ARGS $FORECAST_EXTRA_ARGS \
        >"${LOG_DIR}/forecast/${tag}.log" 2>&1 ||
        { log "[ERROR] [forecast] ${tag} failed, see ${LOG_DIR}/forecast/${tag}.log"; return 1; }
}

cmd_forecast() {
    mkdir -p "${LOG_DIR}/forecast"
    set_threads "$FORECAST_THREADS"
    log "[forecast] N_TIMEPOINTS=${N_TIMEPOINTS} FORECAST_AFTER=${FORECAST_AFTER}"
    export -f forecast_one wait_for_threads log run
    export SCRIPTS_DIR RESULTS_DIR LOG_DIR NICE MAX_USER_THREADS METRICS_ARGS \
        FORECAST_EXTRA_ARGS
    local cond seed
    for cond in $CONDITIONS; do
        for seed in $SEEDS; do
            final_tag "$cond" "$seed"
        done
    done | parallel --line-buffer --jobs "$FORECAST_JOBS" forecast_one
    log "[forecast] done"
}

cmd_summarize() {
    run python "${SCRIPTS_DIR}/collect_seed_metrics.py" \
        --results-dir "${RESULTS_DIR}/${FL_PROG_DATE}"
}

# ===== main =====
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 {prep|start-nodes|stop-nodes|fit|forecast|summarize|all} [SEED ...]"
    exit 1
fi
COMMAND=$1
shift
[ "$#" -gt 0 ] && SEEDS="$*"

cd "$ROOT_DIR" || exit 1
log "FL_PROG_DATE=${FL_PROG_DATE} LOG_DIR=${LOG_DIR} SEEDS=${SEEDS}"
case $COMMAND in
prep) cmd_prep ;;
start-nodes) cmd_start_nodes ;;
stop-nodes) cmd_stop_nodes ;;
fit) cmd_fit ;;
forecast) cmd_forecast ;;
summarize) cmd_summarize ;;
all)
    cmd_prep || exit 1
    cmd_start_nodes
    sleep 30
    cmd_fit
    cmd_stop_nodes
    cmd_summarize
    ;;
*)
    echo "Unknown command: ${COMMAND}"
    exit 1
    ;;
esac
