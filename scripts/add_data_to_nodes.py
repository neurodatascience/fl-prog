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

from fl_prog.utils.constants import DNAME_LATEST, FNAME_NODE_CONFIG
from fl_prog.utils.io import save_json

DEFAULT_COL_SUBJECT = "subject"

DATA_SUFFIX_TO_NODE_TAGS_MAP = {
    "1": ("node-1", ["iid"]),
    "2": ("node-2", ["iid"]),
    "3": ("node-3", ["iid"]),
    "merged": ("node-centralized", ["iid", "centralized"]),
}


def _get_info(fpath_tsv) -> str:
    match = re.search(r"-(.+)\.", Path(fpath_tsv).name)
    if not match:
        raise ValueError(f"Could not extract dataset name from {fpath_tsv}")
    dataset_name = match.group(1)
    if dataset_name not in DATA_SUFFIX_TO_NODE_TAGS_MAP:
        raise ValueError(f"Unrecognized dataset name: {dataset_name}")

    node_name, tags = DATA_SUFFIX_TO_NODE_TAGS_MAP[dataset_name]

    return node_name, tags


def _data_already_added(dpath_node: Path, fpath_tsv: Path) -> bool:
    fpath_config = dpath_node / "etc" / "config.ini"
    if not fpath_config.exists():
        raise FileNotFoundError(f"Node config file not found: {fpath_config}")

    config = configparser.ConfigParser()
    config.read(str(fpath_config))
    fpath_db = (fpath_config.parent / config["default"]["db"]).resolve()
    if not fpath_db.exists():
        warnings.warn(f"Node database file not found: {fpath_db}")
        return False

    db_json = json.loads(fpath_db.read_text())
    df_datasets = pd.DataFrame(db_json["Datasets"]).T

    if df_datasets.empty:
        return False

    return str(fpath_tsv) in df_datasets["path"].to_list()


def _create_config_file(fpath_tsv: Path, col_subject: str) -> dict:
    df = pd.read_csv(fpath_tsv, sep="\t")
    if col_subject not in df.columns:
        raise ValueError(f"Subject column '{col_subject}' not found in {fpath_tsv}")
    config = {
        "n_participants": df[col_subject].nunique(),
    }
    return config


def _add_data_to_node(
    fpath_tsv: Path, dpath_nodes: Path, col_subject: str = DEFAULT_COL_SUBJECT
):
    node_name, tags = _get_info(fpath_tsv)
    dpath_node = dpath_nodes / node_name

    fpath_config = dpath_node / FNAME_NODE_CONFIG
    config = _create_config_file(fpath_tsv, col_subject)
    save_json(fpath_config, config)
    print(f"Created node config file: {fpath_config}")

    if _data_already_added(dpath_node, fpath_tsv):
        print(f"{fpath_tsv.name} is already in node {dpath_node.name}. Skipping")
        return

    dataset_info = {
        "path": str(fpath_tsv),
        "data_type": "csv",
        "description": "",
        "tags": ",".join(tags),
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


@click.command()
@click.argument(
    "dpath_data",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    envvar="DPATH_DATA",
)
@click.argument(
    "dpath_nodes",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    envvar="DPATH_FEDBIOMED",
)
@click.option(
    "--col-subject",
    type=str,
    default=DEFAULT_COL_SUBJECT,
    help="Column name for subject IDs",
)
def add_data_to_nodes(dpath_data: Path, dpath_nodes: Path, col_subject: str):
    fpaths_tsv = (dpath_data / DNAME_LATEST).glob("*.tsv")
    for fpath_tsv in sorted(fpaths_tsv):
        print(f"----- {fpath_tsv} -----")
        _add_data_to_node(fpath_tsv, dpath_nodes, col_subject)


if __name__ == "__main__":
    add_data_to_nodes()
