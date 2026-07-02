#!/bin/bash


SCRIPTS_DIR=`dirname "$0"`
TAGS_AND_ARGS_FILE="${SCRIPTS_DIR}/tags_and_args.tsv"

# iterations
I_START=1
I_STOP=50

# 000percent	--n-subjects 439 --t0-min 0 --t0-max 0.2 --t0-min 0.2 --t0-max 0.4 --t0-min 0.4 --t0-max 0.6 --t0-min 0.6 --t0-max 0.8 --t0-min 0.8 --t0-max 1.0	--learning-rate 0.05 --n-rounds 4 --n-updates 25 --time-shift-range 0 3 --lambda 10
# 050percent	--n-subjects 439 --t0-min 0.00 --t0-max 0.33 --t0-min 0.17 --t0-max 0.50 --t0-min 0.33 --t0-max 0.67 --t0-min 0.50 --t0-max 0.83 --t0-min 0.67 --t0-max 1.00	--learning-rate 0.05 --n-rounds 4 --n-updates 25 --time-shift-range 0 3 --lambda 10
# 100percent	--n-subjects 439 --t0-min 0 --t0-max 1.0 --t0-min 0 --t0-max 1.0 --t0-min 0 --t0-max 1.0 --t0-min 0 --t0-max 1.0 --t0-min 0 --t0-max 1.0	--learning-rate 0.05 --n-rounds 4 --n-updates 25 --time-shift-range 0 3 --lambda 10

# tag \t simulation args \t federation args
# NOTE: use script to create node for new sites if changing number of sites
cat > ${TAGS_AND_ARGS_FILE} <<'EOF'
000percent_3sites_50subjects	--n-subjects 50 --n-subjects 50 --n-subjects 50 --t0-min 0 --t0-max 0.33 --t0-min 0.33 --t0-max 0.66 --t0-min 0.66 --t0-max 1.0	--n-rounds 10
050percent_3sites_50subjects	--n-subjects 50 --n-subjects 50 --n-subjects 50 --t0-min 0 --t0-max 0.5 --t0-min 0.25 --t0-max 0.75 --t0-min 0.5 --t0-max 1.0	--n-rounds 10
100percent_3sites_50subjects	--n-subjects 50 --n-subjects 50 --n-subjects 50 --t0-min 0 --t0-max 1.0 --t0-min 0 --t0-max 1.0 --t0-min 0 --t0-max 1.0	--n-rounds 10
EOF
n = 5
# 050 percent:
for i in range(n):
    print(f'--t0-min {i/(n+1):.2f} --t0-max {(i+2)/(n+1):.2f}', end=' ')
# 000 percent:
for i in range(n):
    print(f'--t0-min {i/n:.2f} --t0-max {(i+1)/n:.2f}', end=' ')

# 100percent_8sites_50subjects_5biomarkers	--t0-min 0.00 --t0-max 1.00 --t0-min 0.00 --t0-max 1.00 --t0-min 0.00 --t0-max 1.00 --t0-min 0.00 --t0-max 1.00 --t0-min 0.00 --t0-max 1.00 --t0-min 0.00 --t0-max 1.00 --t0-min 0.00 --t0-max 1.00 --t0-min 0.00 --t0-max 1.00 --n-biomarkers 5 --n-subjects 50	--n-rounds 8
# 050percent_8sites_50subjects_5biomarkers	--t0-min 0.00 --t0-max 0.22 --t0-min 0.11 --t0-max 0.33 --t0-min 0.22 --t0-max 0.44 --t0-min 0.33 --t0-max 0.56 --t0-min 0.44 --t0-max 0.67 --t0-min 0.56 --t0-max 0.78 --t0-min 0.67 --t0-max 0.89 --t0-min 0.78 --t0-max 1.00 --n-biomarkers 5 --n-subjects 50	--n-rounds 8
# 000percent_8sites_50subjects_5biomarkers	--t0-min 0.00 --t0-max 0.12 --t0-min 0.12 --t0-max 0.25 --t0-min 0.25 --t0-max 0.38 --t0-min 0.38 --t0-max 0.50 --t0-min 0.50 --t0-max 0.62 --t0-min 0.62 --t0-max 0.75 --t0-min 0.75 --t0-max 0.88 --t0-min 0.88 --t0-max 1.00 --n-biomarkers 5 --n-subjects 50	--n-rounds 8
# 000percent_2sites_50subjects_5biomarkers	--t0-min 0.00 --t0-max 0.50 --t0-min 0.50 --t0-max 1.00 --n-biomarkers 5 --n-subjects 50	--n-rounds 8
# 050percent_2sites_50subjects_5biomarkers	--t0-min 0.00 --t0-max 0.66 --t0-min 0.33 --t0-max 1.00 --n-biomarkers 5 --n-subjects 50	--n-rounds 8
# 100percent_2sites_50subjects_5biomarkers	--t0-min 0.00 --t0-max 1.00 --t0-min 0.00 --t0-max 1.00 --n-biomarkers 5 --n-subjects 50	--n-rounds 8

# 050percent_5sites_50subjects_20biomarkers	--t0-min 0.00 --t0-max 0.33 --t0-min 0.17 --t0-max 0.50 --t0-min 0.33 --t0-max 0.67 --t0-min 0.50 --t0-max 0.83 --t0-min 0.67 --t0-max 1.00 --n-biomarkers 20 --n-subjects 50	--n-rounds 8
# 100percent_5sites_50subjects_20biomarkers	--t0-min 0.00 --t0-max 0.50 --t0-min 0.50 --t0-max 1.00 --t0-min 0.00 --t0-max 0.50 --t0-min 0.50 --t0-max 1.00 --t0-min 0.50 --t0-max 1.00 --n-biomarkers 20 --n-subjects 50	--n-rounds 8
# 000percent_5sites_50subjects_20biomarkers	--t0-min 0.00 --t0-max 0.20 --t0-min 0.20 --t0-max 0.40 --t0-min 0.40 --t0-max 0.60 --t0-min 0.60 --t0-max 0.80 --t0-min 0.80 --t0-max 1.00 --n-biomarkers 20 --n-subjects 50	--n-rounds 8

# 000percent_5sites_50subjects_5biomarkers	--t0-min 0.00 --t0-max 0.20 --t0-min 0.20 --t0-max 0.40 --t0-min 0.40 --t0-max 0.60 --t0-min 0.60 --t0-max 0.80 --t0-min 0.80 --t0-max 1.00 --n-biomarkers 5 --n-subjects 50	--n-rounds 8
# 050percent_5sites_50subjects_5biomarkers	--t0-min 0.00 --t0-max 0.33 --t0-min 0.17 --t0-max 0.50 --t0-min 0.33 --t0-max 0.67 --t0-min 0.50 --t0-max 0.83 --t0-min 0.67 --t0-max 1.00 --n-biomarkers 5 --n-subjects 50	--n-rounds 8
# 100percent_5sites_50subjects_5biomarkers	--t0-min 0.00 --t0-max 0.50 --t0-min 0.50 --t0-max 1.00 --t0-min 0.00 --t0-max 0.50 --t0-min 0.50 --t0-max 1.00 --t0-min 0.50 --t0-max 1.00 --n-biomarkers 5 --n-subjects 50	--n-rounds 8

# 000percent_2sites_50subjects_20biomarkers	--t0-min 0.00 --t0-max 0.50 --t0-min 0.50 --t0-max 1.00 --n-biomarkers 20 --n-subjects 50	--n-rounds 8
# 050percent_2sites_50subjects_20biomarkers	--t0-min 0.00 --t0-max 0.66 --t0-min 0.33 --t0-max 1.00 --n-biomarkers 20 --n-subjects 50	--n-rounds 8
# 100percent_2sites_50subjects_20biomarkers	--t0-min 0.00 --t0-max 1.00 --t0-min 0.00 --t0-max 1.00 --n-biomarkers 20 --n-subjects 50	--n-rounds 8

# 100percent_8sites_50subjects_20biomarkers	--t0-min 0.00 --t0-max 1.00 --t0-min 0.00 --t0-max 1.00 --t0-min 0.00 --t0-max 1.00 --t0-min 0.00 --t0-max 1.00 --t0-min 0.00 --t0-max 1.00 --t0-min 0.00 --t0-max 1.00 --t0-min 0.00 --t0-max 1.00 --t0-min 0.00 --t0-max 1.00 --n-biomarkers 20 --n-subjects 50	--n-rounds 8
# 050percent_8sites_50subjects_20biomarkers	--t0-min 0.00 --t0-max 0.22 --t0-min 0.11 --t0-max 0.33 --t0-min 0.22 --t0-max 0.44 --t0-min 0.33 --t0-max 0.56 --t0-min 0.44 --t0-max 0.67 --t0-min 0.56 --t0-max 0.78 --t0-min 0.67 --t0-max 0.89 --t0-min 0.78 --t0-max 1.00 --n-biomarkers 20 --n-subjects 50	--n-rounds 8
# 000percent_8sites_50subjects_20biomarkers	--t0-min 0.00 --t0-max 0.12 --t0-min 0.12 --t0-max 0.25 --t0-min 0.25 --t0-max 0.38 --t0-min 0.38 --t0-max 0.50 --t0-min 0.50 --t0-max 0.62 --t0-min 0.62 --t0-max 0.75 --t0-min 0.75 --t0-max 0.88 --t0-min 0.88 --t0-max 1.00 --n-biomarkers 20 --n-subjects 50	--n-rounds 8

# 0percent_3sites_50subjects	--n-subjects 50 --n-subjects 50 --n-subjects 50 --t0-min 0 --t0-max 0.33 --t0-min 0.33 --t0-max 0.66 --t0-min 0.66 --t0-max 1.0	--n-rounds 8
# 50percent_3sites_50subjects	--n-subjects 50 --n-subjects 50 --n-subjects 50 --t0-min 0 --t0-max 0.5 --t0-min 0.25 --t0-max 0.75 --t0-min 0.5 --t0-max 1.0	--n-rounds 8
# 100percent_3sites_50subjects	--n-subjects 50 --n-subjects 50 --n-subjects 50 --t0-min 0 --t0-max 1.0 --t0-min 0 --t0-max 1.0 --t0-min 0 --t0-max 1.0	--n-rounds 8

# awk call is to update tags and args for multiple iterations (adding random seed)
# {1}: tag
# {2}: simulation args
# {3}: federation args
awk -v I_START="$I_START" -v I_STOP="$I_STOP" -F'\t' '{for (i=I_START; i<=I_STOP; i++) print $1 "_" i "\t" $2 " --rng-seed " i "\t" $3}' ${TAGS_AND_ARGS_FILE} | \
parallel --colsep '\t' --jobs 1 "${SCRIPTS_DIR}/run_single_experiment-simulations.sh {1} {2} {3}"

if [ $? -ne 0 ]
then
    echo "[ERROR] Exiting: error code $?"
    exit $?
fi
rm ${TAGS_AND_ARGS_FILE}
