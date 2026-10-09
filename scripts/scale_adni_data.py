#!/usr/bin/env python
import sys
from pathlib import Path

import click
import pandas as pd

from fl_prog.data_scaling import get_min_max_by_measure, scale_min_max
from fl_prog.utils.constants import CLICK_CONTEXT_SETTINGS
from fl_prog.utils.io import (
    DEFAULT_DPATH_DATA,
    get_dpath_latest,
    load_json,
    save_json,
)


def scale_adni_data(tag: str, dpath_data: Path, fpath_scaling_config: Path):
    old_tag = tag
    dpath_out_old = get_dpath_latest(dpath_data) / old_tag
    fpath_json_old = dpath_out_old / f"{old_tag}.json"
    scaling_config = load_json(fpath_scaling_config)
    tag = f"{tag}{scaling_config['suffix']}"
    dpath_out_new = get_dpath_latest(dpath_data, use_today=True) / tag

    settings = locals().copy()

    scaling_methods = scaling_config["scaling_methods"]

    settings["tag"] = tag

    json_data_old = load_json(fpath_json_old)
    node_id_map_old = json_data_old["node_id_map"]
    scaling_references = json_data_old["scaling_references"]
    col_subject = json_data_old["cols"]["col_subject"]
    cols_biomarker = json_data_old["cols"]["cols_biomarker"]
    col_group = json_data_old["cols"]["col_group"]
    config_old = json_data_old["settings"]["config"]
    settings["config"] = config_old

    min_max_by_measure_map = {}  # scaling reference -> min max by measure

    node_id_map_new = {}
    for fname_to_scale, fname_reference in scaling_references.items():
        if fname_reference is None:
            click.secho(
                f"ERROR: No scaling reference for {fname_to_scale}.",
                fg="red",
                bold=True,
            )
            sys.exit(1)

        fname_scaled = fname_to_scale.replace(old_tag, tag)
        try:
            node_id = node_id_map_old[fname_to_scale]
            node_id_map_new[fname_scaled] = node_id
        except KeyError:
            click.secho(
                f"WARNING: {fname_to_scale} not found in node_id_map_old. Make sure this file doesn't need to be added to a node.",
                fg="yellow",
                bold=True,
            )

        fpath_to_scale = dpath_out_old / fname_to_scale
        fpath_reference = dpath_out_old / fname_reference

        df_to_scale = pd.read_csv(fpath_to_scale, sep="\t", dtype={col_subject: str})
        df_reference = pd.read_csv(fpath_reference, sep="\t", dtype={col_subject: str})

        if fname_reference not in min_max_by_measure_map:
            min_max_by_measure_map[fname_reference] = get_min_max_by_measure(
                df_reference, cols_biomarker, scaling_methods, col_group=col_group
            )
        min_max_by_measure = min_max_by_measure_map[fname_reference]

        df_scaled = scale_min_max(df_to_scale, min_max_by_measure)

        dpath_out_new.mkdir(parents=True, exist_ok=True)
        fpath_scaled = dpath_out_new / fname_scaled
        df_scaled.to_csv(fpath_scaled, sep="\t", index=False)
        print(f"Scaled {fname_to_scale} using {fname_reference} -> {fpath_scaled}")

    json_data_new = {}
    json_data_new["settings"] = settings
    json_data_new["node_id_map"] = node_id_map_new
    json_data_new["cols"] = json_data_old["cols"]
    json_data_new["subjects_by_node"] = json_data_old["subjects_by_node"]
    json_data_new["need_scaling"] = False
    json_data_new["scaling_references"] = scaling_references
    json_data_new["do_not_merge"] = json_data_old["do_not_merge"]
    json_data_new["min_max_by_measure_map"] = min_max_by_measure_map
    fpath_json_new = dpath_out_new / f"{tag}.json"
    save_json(fpath_json_new, json_data_new)
    print(f"Saved new JSON data to {fpath_json_new}")
    print(f"Use new tag --tag {tag} in next steps")


@click.command(context_settings=CLICK_CONTEXT_SETTINGS)
@click.option("--tag", type=str, required=True)
@click.option(
    "--data-dir",
    "dpath_data",
    type=click.Path(path_type=Path, file_okay=False, dir_okay=True),
    default=DEFAULT_DPATH_DATA,
)
@click.option(
    "--config",
    "fpath_scaling_config",
    type=click.Path(path_type=Path, file_okay=True, dir_okay=False),
    required=True,
    envvar="ADNI_SCALING_CONFIG_FILE",
)
def main(*args, **kwargs):
    """
    Split into train/test sets by leaving out the last timepoint for each subject.

    Run before merging the data across sites.
    """
    scale_adni_data(*args, **kwargs)


if __name__ == "__main__":
    main()
