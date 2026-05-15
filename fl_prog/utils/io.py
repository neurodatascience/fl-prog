import datetime
import json
import os
from contextlib import contextmanager
from pathlib import Path

import torch
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


def _get_json_field(
    fpath_json,
    field_name: str | list[str],
    content_description: str | list[str],
):
    fpath_json = Path(fpath_json)

    if isinstance(field_name, str):
        field_name = [field_name]
    if isinstance(content_description, str):
        content_description = [content_description]

    if not len(field_name) == len(content_description):
        raise ValueError(
            "field_name and content_description must have the same length"
            f", got {field_name=} ({len(field_name)}) and {content_description=} ({len(content_description)})."
        )

    if not fpath_json.exists():
        raise FileNotFoundError(
            f"Expected a JSON file at {fpath_json} with a '{field_name[0]}' entry"
            f" {content_description[0]}."
        )
    json_data = json.loads(fpath_json.read_text())

    field_value = None
    while field_name or content_description:
        current_field_name = field_name.pop(0)
        current_content_description = content_description.pop(0)

        field_value = json_data.get(current_field_name)
        if not field_value:
            raise ValueError(
                f"{fpath_json} must contain a '{current_field_name}' entry"
                f" {current_content_description}."
            )
        json_data = field_value
    return field_value


def get_node_id_map(fpath_json) -> dict[str, str]:
    return _get_json_field(
        fpath_json,
        "node_id_map",
        "where keys are data filenames and values are node IDs",
    )


def save_json(dpath: Path, data: dict, indent: int = 4):
    with open(dpath, "w") as file_json:
        json.dump(data, file_json, indent=indent, default=serialize_data)


def serialize_data(obj: object):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.data.numpy().tolist()
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
