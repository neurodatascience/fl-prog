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


class Setup(str, Enum):
    CENTRALIZED = "centralized"
    FEDERATED = "federated"


# model
class Penalty(str, Enum):
    L1 = "l1"
    L2 = "l2"
