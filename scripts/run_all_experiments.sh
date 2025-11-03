#!/bin/bash


SCRIPTS_DIR=`dirname "$0"`
TAGS_AND_ARGS_FILE="${SCRIPTS_DIR}/tags_and_args.tsv"

I_START=6
I_STOP=20
cat > ${TAGS_AND_ARGS_FILE} <<'EOF'
50percent_3sites_50subjects	--n-subjects 50 --n-subjects 50 --n-subjects 50 --t0-min 0 --t0-max 0.5 --t0-min 0.25 --t0-max 0.75 --t0-min 0.5 --t0-max 1.0	--n-rounds 8
EOF
# 0percent_3sites_50subjects	--n-subjects 50 --n-subjects 50 --n-subjects 50 --t0-min 0 --t0-max 0.33 --t0-min 0.33 --t0-max 0.66 --t0-min 0.66 --t0-max 1.0	--n-rounds 8
# 100percent_3sites_50subjects	--n-subjects 50 --n-subjects 50 --n-subjects 50 --t0-min 0 --t0-max 1.0 --t0-min 0 --t0-max 1.0 --t0-min 0 --t0-max 1.0	--n-rounds 8

# {1}: tag
# {2}: simulation args
# {3}: federation args
awk -v I_START="$I_START" -v I_STOP="$I_STOP" -F'\t' '{for (i=I_START; i<=I_STOP; i++) print $1 "_" i "\t" $2 " --rng-seed " i "\t" $3}' ${TAGS_AND_ARGS_FILE} | \
parallel --colsep '\t' --jobs 1 "${SCRIPTS_DIR}/run_single_experiment.sh {1} {2} {3}"

if [ $? -ne 0 ]
then
    echo "[ERROR] Exiting: error code $?"
    exit $?
fi
rm ${TAGS_AND_ARGS_FILE}
