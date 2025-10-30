#!/usr/bin/env python
from pathlib import Path
import configparser
import re
import json
import subprocess
import tempfile
import warnings

import click
import pandas as pd

from fl_prog.utils.constants import (
    CLICK_CONTEXT_SETTINGS,
    DNAME_LATEST,
    FNAME_NODE_CONFIG,
)
from fl_prog.utils.io import save_json, DEFAULT_DPATH_DATA, DEFAULT_DPATH_FEDBIOMED

DEFAULT_COL_SUBJECT = "subject"

DATA_SUFFIX_TO_NODE_MAP = {
    "1": "node-1",
    "2": "node-2",
    "3": "node-3",
    "merged": "node-centralized",
}


def _get_info(fpath_tsv) -> str:
    match = re.search(r"-(.+)\.", Path(fpath_tsv).name)
    if not match:
        raise ValueError(f"Could not extract dataset name from {fpath_tsv}")
    dataset_name = match.group(1)
    if dataset_name not in DATA_SUFFIX_TO_NODE_MAP:
        raise ValueError(f"Unrecognized dataset name: {dataset_name}")

    return DATA_SUFFIX_TO_NODE_MAP[dataset_name]


def _data_already_added(dpath_node: Path, fpath_tsv: Path, wipe: bool = False) -> bool:
    fpath_config = dpath_node / "etc" / "config.ini"
    if not fpath_config.exists():
        raise FileNotFoundError(
            f"Node config file not found: {fpath_config}"
            ". Make sure node has been initialized."
        )

    config = configparser.ConfigParser()
    config.read(str(fpath_config))
    fpath_db = (fpath_config.parent / config["default"]["db"]).resolve()
    if not fpath_db.exists():
        warnings.warn(f"Node database file not found: {fpath_db}")
        return False

    if wipe:
        fpath_db.unlink()
        return False

    try:
        db_json = json.loads(fpath_db.read_text())
        df_datasets = pd.DataFrame(db_json["Datasets"]).T
    except (json.JSONDecodeError, KeyError):
        warnings.warn(f"Error parsing the database file: {fpath_db}")
        return False

    if df_datasets.empty:
        return False

    return str(fpath_tsv) in df_datasets["path"].to_list()


def _create_config(fpath_tsv: Path, col_subject: str) -> dict:
    df = pd.read_csv(fpath_tsv, sep="\t")
    if col_subject not in df.columns:
        raise ValueError(f"Subject column '{col_subject}' not found in {fpath_tsv}")
    config = {
        "n_participants": df[col_subject].nunique(),
    }
    return config


def _add_dataset_to_node(
    fpath_tsv: Path,
    dpath_nodes: Path,
    tag: str,
    col_subject: str = DEFAULT_COL_SUBJECT,
    wipe: bool = False,
):
    node_name = _get_info(fpath_tsv)
    dpath_node = dpath_nodes / node_name

    fpath_config = dpath_node / FNAME_NODE_CONFIG
    if fpath_config.exists() and not wipe:
        config = json.loads(fpath_config.read_text())
    else:
        config = {}
    config[tag] = _create_config(fpath_tsv, col_subject)
    save_json(fpath_config, config)
    print(f"Created/updated node config file: {fpath_config}")
    print(config)

    if _data_already_added(dpath_node, fpath_tsv, wipe=wipe):
        print(f"{fpath_tsv.name} is already in node {dpath_node.name}. Skipping")
        return

    dataset_info = {
        "path": str(fpath_tsv),
        "data_type": "csv",
        "description": "",
        "tags": tag,
        "name": node_name,
    }
    with tempfile.NamedTemporaryFile(mode="+wt") as file_json:
        file_json.write(json.dumps(dataset_info))
        file_json.flush()
        subprocess.run(
            [
                "fedbiomed",
                "node",
                "-p",
                str(dpath_node),
                "dataset",
                "add",
                "--file",
                file_json.name,
            ],
            check=True,
        )


@click.command(context_settings=CLICK_CONTEXT_SETTINGS)
@click.option("--tag", type=str, required=True)
@click.option(
    "--data-dir",
    "dpath_data",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    default=DEFAULT_DPATH_DATA,
)
@click.option(
    "--nodes-dir",
    "dpath_nodes",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    default=DEFAULT_DPATH_FEDBIOMED,
)
@click.option(
    "--col-subject",
    type=str,
    default=DEFAULT_COL_SUBJECT,
    help="Column name for subject IDs",
)
@click.option("--wipe/--no-wipe", default=False, help="Wipe existing data in nodes")
def add_datasets_to_nodes(
    tag: str, dpath_data: Path, dpath_nodes: Path, col_subject: str, wipe: bool
):
    fpaths_tsv = (dpath_data / DNAME_LATEST).glob(f"{tag}*.tsv")
    for fpath_tsv in sorted(fpaths_tsv):
        fpath_tsv = fpath_tsv.absolute()
        print(f"----- {fpath_tsv} -----")
        _add_dataset_to_node(fpath_tsv, dpath_nodes, tag, col_subject, wipe)


if __name__ == "__main__":
    add_datasets_to_nodes()
