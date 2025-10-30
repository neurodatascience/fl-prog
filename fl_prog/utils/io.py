import datetime
import json
import os
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from fl_prog.utils.constants import DNAME_LATEST, DATE_FORMAT

DPATH_PROJECT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DPATH_DATA = DPATH_PROJECT / "data"
DEFAULT_DPATH_FEDBIOMED = DPATH_PROJECT / "fedbiomed"
DEFAULT_DPATH_RESULTS = DPATH_PROJECT / "results"


def get_dpath_latest(dpath_parent, use_today=False):
    dpath_parent = Path(dpath_parent)
    dpath_latest = dpath_parent / DNAME_LATEST

    dpath_today = dpath_parent / datetime.datetime.today().strftime(DATE_FORMAT)
    if dpath_latest.exists():
        if use_today and dpath_latest.resolve() != dpath_today.resolve():
            if dpath_latest.is_symlink():
                dpath_latest.unlink()
            else:
                raise RuntimeError(f"{dpath_latest=} exists but is not a symlink")

    if not dpath_latest.exists():
        if dpath_latest.is_symlink():
            dpath_latest.unlink()
        dpath_today.mkdir(parents=True, exist_ok=True)
        dpath_latest.symlink_to(dpath_today, target_is_directory=True)

    return dpath_latest.resolve()


def save_json(dpath: Path, data: dict, indent: int = 4):
    with open(dpath, "w") as file_json:
        json.dump(data, file_json, indent=indent, default=serialize_data)


def serialize_data(obj: object):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Type {type(obj)} not serializable")


@contextmanager
def working_directory(dpath):
    dpath_old = Path.cwd()
    os.chdir(dpath)
    try:
        yield
    finally:
        os.chdir(dpath_old)
