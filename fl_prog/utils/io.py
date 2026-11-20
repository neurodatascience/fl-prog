import datetime
import enum
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from declearn.model.api import Vector
from declearn.utils import Aggregate

from fl_prog.utils.constants import (
    DATE_FORMAT,
    DNAME_LATEST,
    DPMOST_COL_SUBJECT,
    DPMOST_COL_TIMEPOINT,
    LEASPY_COL_SUBJECT,
    LEASPY_COL_TIMEPOINT,
    NODE_ID_CENTRALIZED,
    NODE_ID_TEST,
)

DPATH_PROJECT = Path(__file__).resolve().parent.parent.parent
ENV_VAR_DATE = "FL_PROG_DATE"
DEFAULT_DPATH_DATA = DPATH_PROJECT / "data"
DEFAULT_DPATH_FEDBIOMED = DPATH_PROJECT / "fedbiomed"
DEFAULT_DPATH_RESULTS = DPATH_PROJECT / "results"
DPATH_DPMOST = DPATH_PROJECT / "vendored" / "dpmost" / "models" / "dpmost"


def infer_sep(fpath: Path) -> str:
    if fpath.suffix == ".tsv":
        return "\t"
    elif fpath.suffix == ".csv":
        return ","
    else:
        raise ValueError(
            f"Could not infer separator from file extension {fpath.suffix}."
        )


def add_dpmost_to_path():
    if str(DPATH_DPMOST) not in sys.path:
        sys.path.append(str(DPATH_DPMOST))


def get_dpath_latest(dpath_parent, use_today=False):
    dpath_parent = Path(dpath_parent)
    dpath_latest = dpath_parent / DNAME_LATEST

    # override "today" (e.g. to keep long runs from moving "latest" at midnight)
    date_str = os.environ.get(
        ENV_VAR_DATE
    ) or datetime.datetime.now().astimezone().strftime(DATE_FORMAT)
    dpath_today = dpath_parent / date_str
    if (
        dpath_latest.exists()
        and use_today
        and dpath_latest.resolve() != dpath_today.resolve()
    ):
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


def load_json(fpath_json: Path) -> dict | list:
    return json.loads(fpath_json.read_text())


def save_json(fpath: Path, data: dict, indent: int = 4):
    with open(fpath, "w") as file_json:
        json.dump(
            data, file_json, indent=indent, allow_nan=False, default=serialize_data
        )


def serialize_data(obj: object):
    if isinstance(obj, set):
        return sorted(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return serialize_data(obj.data.numpy())
    elif isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, enum.Enum):
        return obj.name
    # declearn objects in training replies
    elif isinstance(obj, Vector):
        return obj.coefs
    elif isinstance(obj, Aggregate):
        return obj.to_dict()
    raise TypeError(f"Type {type(obj)} not serializable")


@contextmanager
def working_directory(dpath):
    dpath_old = Path.cwd()
    os.chdir(dpath)
    try:
        yield
    finally:
        os.chdir(dpath_old)


def format_df_for_leaspy(
    df: pd.DataFrame, col_subject: str, col_timepoint: str, cols_biomarker: list[str]
) -> pd.DataFrame:
    df[col_subject] = df[col_subject].astype(str)
    df = df.rename(
        columns={
            col_subject: LEASPY_COL_SUBJECT,
            col_timepoint: LEASPY_COL_TIMEPOINT,
        }
    )
    df = df.set_index([LEASPY_COL_SUBJECT, LEASPY_COL_TIMEPOINT])
    df = df.loc[:, cols_biomarker]
    return df


def format_df_for_dpmost(
    df: pd.DataFrame, col_subject: str, col_timepoint: str, cols_biomarker: list[str]
) -> pd.DataFrame:
    df[col_subject] = df[col_subject].astype(str)
    df = df.rename(
        columns={
            col_subject: DPMOST_COL_SUBJECT,
            col_timepoint: DPMOST_COL_TIMEPOINT,
        }
    )
    df = df.loc[:, [DPMOST_COL_SUBJECT, DPMOST_COL_TIMEPOINT] + cols_biomarker]
    return df


def load_simulated_sites(dpath: Path, json_data: dict) -> pd.DataFrame:
    """Load the site data of a ``simulate_data.py`` output, one row per visit, with
    the site and the subject's true t0, acceleration factor and disease time
    (t0 + acceleration factor * timepoint).

    Merged (centralized) data and held-out test subjects are ignored.
    """
    cols = json_data["cols"]
    col_subject, col_timepoint = cols["col_subject"], cols["col_timepoint"]
    params = json_data["params"]
    subjects_by_node = json_data["subjects_by_node"]

    # true per-subject params: one list per entry of subjects_by_node (in order,
    # excluding the centralized node), each in the same subject order
    node_ids = [
        node_id for node_id in subjects_by_node if node_id != NODE_ID_CENTRALIZED
    ]
    if len(node_ids) != len(params["time_shifts"]):
        raise ValueError(
            f"Expected one params['time_shifts'] entry per node in {node_ids}, got "
            f"{len(params['time_shifts'])}"
        )
    df_params = pd.concat(
        [
            pd.DataFrame(
                {
                    col_subject: [
                        str(subject) for subject in subjects_by_node[node_id]
                    ],
                    "t0": time_shifts,
                    "acceleration_factor": acceleration_factors,
                }
            )
            for node_id, time_shifts, acceleration_factors in zip(
                node_ids,
                params["time_shifts"],
                params["acceleration_factors"],
                strict=True,
            )
        ],
        ignore_index=True,
    )

    dfs = []
    for fname, node_id in json_data["node_id_map"].items():
        if node_id in (NODE_ID_CENTRALIZED, NODE_ID_TEST):
            continue
        # subject IDs as strings (JSON and TSV types can differ)
        df = pd.read_csv(dpath / fname, sep="\t", dtype={col_subject: str})
        df = df.merge(df_params, on=col_subject, how="left", validate="many_to_one")
        if df["t0"].isna().any():
            raise ValueError(f"Subjects in {fname} without true parameters")
        df["site"] = node_id
        df["disease_time"] = df["t0"] + df["acceleration_factor"] * df[col_timepoint]
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)
