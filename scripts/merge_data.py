#!/usr/bin/env python

from pathlib import Path

import click
import json
import pandas as pd

from fl_prog.utils.constants import CLICK_CONTEXT_SETTINGS, NODE_ID_CENTRALIZED
from fl_prog.utils.io import get_dpath_latest, save_json, DEFAULT_DPATH_DATA


def _get_fname_merged(tag: str) -> str:
    return f"{tag}-merged.tsv"


@click.command(context_settings=CLICK_CONTEXT_SETTINGS)
@click.option("--tag", type=str, required=True)
@click.option(
    "--data-dir",
    "dpath_data",
    type=click.Path(path_type=Path, file_okay=False, dir_okay=True),
    default=DEFAULT_DPATH_DATA,
)
def merge_data(dpath_data, tag):
    dpath_out = get_dpath_latest(dpath_data)
    fname_merged = _get_fname_merged(tag)

    fpath_json = dpath_out / f"{tag}.json"
    json_data = json.loads(fpath_json.read_text())
    json_data["node_id_map"][fname_merged] = NODE_ID_CENTRALIZED

    fpaths_tsv = [
        fpath
        for fpath in sorted(dpath_out.glob(f"{tag}*.tsv"))
        if fpath.name != fname_merged
    ]
    dfs = [pd.read_csv(fpath, sep="\t") for fpath in fpaths_tsv]
    df = pd.concat(dfs)

    fpath_out = dpath_out / fname_merged
    df.to_csv(fpath_out, sep="\t", index=False)
    print(f"Saved merged data to {fpath_out}")

    save_json(fpath_json, json_data)
    print(f"Updated node ID map in {fpath_json}")


if __name__ == "__main__":
    merge_data()
