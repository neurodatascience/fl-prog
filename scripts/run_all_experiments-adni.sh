#!/bin/bash


SCRIPTS_DIR=`dirname "$0"`
TAGS_AND_ARGS_FILE="${SCRIPTS_DIR}/tags_and_args.tsv"

# iterations
I_START=1
I_STOP=10

# tag \t simulation args \t federation args
# NOTE: use script to create node for new sites if changing number of sites
cat > ${TAGS_AND_ARGS_FILE} <<'EOF'
adni_iid	--iid	--learning-rate 0.05 --n-rounds 4 --n-updates 25 --time-shift-range 0 3 --lambda 10
adni_noniid	--non-iid	--learning-rate 0.05 --n-rounds 4 --n-updates 25 --time-shift-range 0 3 --lambda 10
EOF

# awk call is to update tags and args for multiple iterations (adding random seed)
# {1}: tag
# {2}: simulation args
# {3}: federation args
awk -v I_START="$I_START" -v I_STOP="$I_STOP" -F'\t' '{for (i=I_START; i<=I_STOP; i++) print $1 "_" i "\t" $2 " --rng-seed " i "\t" $3}' ${TAGS_AND_ARGS_FILE} | \
parallel --colsep '\t' --jobs 1 "${SCRIPTS_DIR}/run_single_experiment-adni.sh {1} {2} {3}"

if [ $? -ne 0 ]
then
    echo "[ERROR] Exiting: error code $?"
    exit $?
fi
rm ${TAGS_AND_ARGS_FILE}
