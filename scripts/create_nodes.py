#!/usr/bin/env python
from pathlib import Path
import configparser

import click
import subprocess

from fl_prog.utils.constants import CLICK_CONTEXT_SETTINGS, NODE_PREFIX
from fl_prog.utils.io import (
    DEFAULT_DPATH_DATA,
    DEFAULT_DPATH_FEDBIOMED,
    get_dpath_latest,
    get_node_id_map,
)


def _create_node(dpath_nodes: Path, node_id: str):
    dpath_node = dpath_nodes / f"node-{node_id}"
    subprocess.run(
        [
            "fedbiomed",
            "component",
            "create",
            "-p",
            str(dpath_node),
            "-c",
            "NODE",
            "--exist-ok",
        ],
        check=True,
    )

    fpath_config = dpath_node / "etc" / "config.ini"
    config = configparser.ConfigParser()
    config.read(str(fpath_config))

    config["default"]["id"] = f"{NODE_PREFIX}{node_id}"
    print(f"Setting node ID to {config['default']['id']}")

    original_db = Path(config["default"]["db"])
    if not original_db.exists():
        new_db = original_db.parent / f"db_{NODE_PREFIX}{node_id}.json"
        config["default"]["db"] = str(new_db)
        print(f"Setting node database path to {config['default']['db']}")

    with fpath_config.open("w") as file_config:
        config.write(file_config)


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
    type=click.Path(path_type=Path, file_okay=False),
    default=DEFAULT_DPATH_FEDBIOMED,
)
def create_nodes(tag: str, dpath_data: Path, dpath_nodes: Path):
    dpath_nodes.mkdir(parents=True, exist_ok=True)

    fpath_json: Path = get_dpath_latest(dpath_data) / f"{tag}.json"
    node_id_map = get_node_id_map(fpath_json)

    for _, node_id in node_id_map.items():
        _create_node(dpath_nodes, node_id)


if __name__ == "__main__":
    create_nodes()
