#!/bin/bash

# ===== FUNCTIONS =====
run_command() {
    COMMAND=$1
    echo "[RUN] ${COMMAND}"
    eval "${COMMAND}"
    if [ $? -ne 0 ]; then
        echo "[ERROR] Exiting: error code $?"
        exit $?
    fi
}

# ===== MAIN =====
if [ "$#" -ne 3 ]
then
    echo "Usage: $0 TAG SIMULATION_ARGS FEDERATION_ARGS"
    exit 1
fi

TAG=$1
SIMULATION_ARGS=$2
FEDERATION_ARGS=$3

SCRIPTS_DIR=`dirname "$0"`

echo "===== ${TAG} ====="
run_command "${SCRIPTS_DIR}/simulate_data.py --tag ${TAG} $SIMULATION_ARGS"
run_command "${SCRIPTS_DIR}/merge_data.py --tag ${TAG}"
run_command "${SCRIPTS_DIR}/add_datasets_to_nodes.py --tag ${TAG}"
run_command "${SCRIPTS_DIR}/run_fedbiomed.py --tag ${TAG} $FEDERATION_ARGS"


