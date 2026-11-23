from enum import Enum

# CLI
CLICK_CONTEXT_SETTINGS = {
    "help_option_names": ["-h", "--help"],
    "show_default": True,
}

# directory naming conventions
DNAME_LATEST = "latest"
DATE_FORMAT = "%Y_%m_%d"

# federation
NODE_PREFIX = "node_"
NODE_ID_CENTRALIZED = "centralized"
NODE_ID_TEST = "test"  # not an actual node

# Leaspy
LEASPY_COL_SUBJECT = "ID"
LEASPY_COL_TIMEPOINT = "TIME"

# DPMoSt
DPMOST_COL_SUBJECT = "subj_id"
DPMOST_COL_TIMEPOINT = "time"

# normative modelling
COL_ADAPTATION = "adaptation"

# plotting
SITE_COLORS = ["#1f3b73", "#b2182b", "#e6ab02", "#4d4d4d", "#7b3294"]

# synthetic data conditions (site splits, see run_synthetic_seeds.sh) -> label
SYNTHETIC_CONDITION_LABELS = {
    "overlap100": "100% overlap",
    "overlap000": "0% overlap",
    "chunks_full": "0% overlap + full",
    "chunks_earlylate": "0% overlap + early/late",
    "chunks_earlylate_sep": "0% overlap + separate early/late",
    "cross": "100% overlap + cross-sectional",
}


class Setup(str, Enum):
    CENTRALIZED = "centralized"
    FEDERATED = "federated"


# model
class Penalty(str, Enum):
    L1 = "l1"
    L2 = "l2"
