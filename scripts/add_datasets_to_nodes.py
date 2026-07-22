#!/usr/bin/env python
from pathlib import Path
import json
import subprocess
import sys
import tempfile
import warnings

import click

from fl_prog.utils.constants import CLICK_CONTEXT_SETTINGS
from fl_prog.utils.fbm_node import get_fpath_db, load_node_db
from fl_prog.utils.io import (
    get_dpath_latest,
    get_node_id_map,
    DEFAULT_DPATH_DATA,
    DEFAULT_DPATH_FEDBIOMED,
)


def _data_already_added(
    dpath_node: Path, fpath_tsv: Path, tag: str, wipe: bool = False
) -> bool:
    fpath_db = get_fpath_db(dpath_node)

    if wipe:
        fpath_db.unlink()
        return False

    try:
        df_datasets = load_node_db(fpath_db=fpath_db)
    except RuntimeError as exception:
        warnings.warn(exception)
        return False

    if df_datasets.empty:
        return False

    if str(fpath_tsv) in df_datasets["path"].to_list():
        return True
    elif any(tag in tags for tags in df_datasets["tags"]):
        click.secho(
            "Error: Other dataset with the same tag already exists in the node. Use --wipe to overwrite.",
            fg="red",
            bold=True,
        )
        sys.exit(1)
    else:
        return False


def _add_dataset_to_node(
    fpath_tsv: Path,
    dpath_node: Path,
    tag: str,
    wipe: bool = False,
):
    if _data_already_added(dpath_node, fpath_tsv, tag, wipe=wipe):
        print(f"{fpath_tsv.name} is already in node {dpath_node.name}. Skipping")
        return

    dataset_info = {
        "path": str(fpath_tsv),
        "data_type": "csv",
        "description": f"{dpath_node.name} {tag}",
        "tags": tag,
        "name": f"{dpath_node.name} {tag}",
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
@click.option("--wipe/--no-wipe", default=False, help="Wipe existing data in nodes")
def add_datasets_to_nodes(tag: str, dpath_data: Path, dpath_nodes: Path, wipe: bool):
    dpath_tag = get_dpath_latest(dpath_data) / tag
    fpaths_tsv = dpath_tag.glob(f"{tag}*.tsv")
    fpath_json = dpath_tag / f"{tag}.json"

    node_id_map = get_node_id_map(fpath_json)

    for fpath_tsv in sorted(fpaths_tsv):
        fpath_tsv = fpath_tsv.absolute()
        dpath_node = dpath_nodes / f"node-{node_id_map[fpath_tsv.name]}"
        print(f"----- {fpath_tsv} -----")
        _add_dataset_to_node(fpath_tsv, dpath_node, tag, wipe)


if __name__ == "__main__":
    add_datasets_to_nodes()
