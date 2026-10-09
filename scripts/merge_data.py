#!/usr/bin/env python

import json
from pathlib import Path

import click
import pandas as pd

from fl_prog.utils.constants import CLICK_CONTEXT_SETTINGS, NODE_ID_CENTRALIZED
from fl_prog.utils.io import DEFAULT_DPATH_DATA, get_dpath_latest, save_json

DEFAULT_SAME_SCALING_ACROSS_SITES = True


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
@click.option(
    "--shared-scaling/--local-scaling",
    "same_scaling_across_sites",
    is_flag=True,
    default=DEFAULT_SAME_SCALING_ACROSS_SITES,
    help="Whether to use the merged data as the scaling reference for all sites (shared) or to use each site's own data as the scaling reference (local).",
)
def merge_data(
    dpath_data, tag, same_scaling_across_sites: bool = DEFAULT_SAME_SCALING_ACROSS_SITES
):
    dpath_out = get_dpath_latest(dpath_data) / tag
    fname_merged = _get_fname_merged(tag)

    fpath_json = dpath_out / f"{tag}.json"
    json_data = json.loads(fpath_json.read_text())
    json_data["node_id_map"][fname_merged] = NODE_ID_CENTRALIZED

    col_subject = json_data["cols"]["col_subject"]
    col_subject_index = json_data["cols"]["col_subject_index"]

    fpaths_tsv = []
    for fpath in sorted(dpath_out.glob(f"{tag}*.tsv")):
        if fpath.name in json_data["do_not_merge"]:
            print(f"Skipping {fpath.name}")
            continue
        fpaths_tsv.append(fpath)

    dfs = [
        pd.read_csv(fpath, sep="\t", dtype={col_subject: str}) for fpath in fpaths_tsv
    ]
    df = pd.concat(dfs)

    subjects = df[col_subject].unique().tolist()
    df[col_subject_index] = df[col_subject].map(lambda x: subjects.index(x))

    json_data["subjects_by_node"][NODE_ID_CENTRALIZED] = subjects
    json_data["do_not_merge"].append(fname_merged)

    json_data["scaling_references"][fname_merged] = fname_merged
    for fname, scaling_reference in json_data["scaling_references"].items():
        if scaling_reference is None or same_scaling_across_sites:
            json_data["scaling_references"][fname] = fname_merged

    fpath_out = dpath_out / fname_merged
    df.to_csv(fpath_out, sep="\t", index=False)
    print(f"Saved merged data (shape {df.shape}) to {fpath_out}")

    save_json(fpath_json, json_data)
    print(f"Updated node ID map in {fpath_json}")


if __name__ == "__main__":
    merge_data()
