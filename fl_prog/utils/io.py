import json
from pathlib import Path

import numpy as np


def serialize_data(obj: object):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Type {type(obj)} not serializable")


def save_json(dpath: Path, data: dict, indent: int = 4):
    with open(dpath, "w") as file_json:
        json.dump(data, file_json, indent=indent, default=serialize_data)
