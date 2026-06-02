#!/usr/bin/env python

from pathlib import Path

import click
import pandas as pd

from fl_prog.utils.constants import CLICK_CONTEXT_SETTINGS
from fl_prog.utils.io import load_json, save_json, get_dpath_latest, DEFAULT_DPATH_DATA

DEFAULT_MIN_N_TIMEPOINTS = 2


def split_train_test(
    tag: str,
    dpath_data: Path,
    min_n_timepoints: int = DEFAULT_MIN_N_TIMEPOINTS,
):
    new_tag = f"{tag}-train"
    dpath_out_old = get_dpath_latest(dpath_data) / tag
    fpath_json_old = dpath_out_old / f"{tag}.json"
    dpath_out_new = get_dpath_latest(dpath_data, use_today=True) / new_tag
    dpath_out_new.mkdir(parents=True, exist_ok=True)

    settings = locals().copy()

    json_data_old = load_json(fpath_json_old)
    node_id_map_old = json_data_old["node_id_map"]
    col_subject = json_data_old["cols"]["col_subject"]
    col_timepoint = json_data_old["cols"]["col_timepoint"]

    dfs_test = []
    node_id_map_new = {}
    subjects_by_node = {}
    for fname_site, node_id in node_id_map_old.items():

        if fname_site.endswith("-merged.tsv"):
            continue

        subjects_by_node[node_id] = []

        fpath_site = dpath_out_old / fname_site
        df_site = pd.read_csv(fpath_site, sep="\t", dtype={col_subject: str})

        dfs_train = []
        for subject, df_subject in df_site.groupby(col_subject):

            if len(df_subject) >= min_n_timepoints:
                df_subject = df_subject.sort_values(col_timepoint, ascending=True)
                dfs_test.append(df_subject.iloc[[-1]])
                dfs_train.append(df_subject.iloc[:-1])
            else:
                dfs_train.append(df_subject)

            subjects_by_node[node_id].append(subject)

        df_train = pd.concat(dfs_train)

        fname_site_new = fname_site.replace(tag, new_tag)
        node_id_map_new[fname_site_new] = node_id_map_old[fname_site]

        fpath_site_new = dpath_out_new / fname_site_new
        df_train.to_csv(fpath_site_new, sep="\t", index=False)
        print(f"{df_site.shape} -> {df_train.shape}: {fpath_site_new}")

    df_test = pd.concat(dfs_test)

    # sanity check
    if df_test[col_subject].nunique() != len(df_test):
        print(df_test[col_subject].value_counts())
        raise ValueError("Some subjects have multiple timepoints in the test set.")

    fpath_test = dpath_out_new / f"{tag}-test.tsv"
    df_test.to_csv(fpath_test, sep="\t", index=False)
    print(f"Saved test set ({df_test.shape}) to {fpath_test}")

    json_data_new = {}
    json_data_new["settings"] = settings
    json_data_new["node_id_map"] = node_id_map_new
    json_data_new["cols"] = json_data_old["cols"]
    json_data_new["subjects_by_node"] = subjects_by_node
    fpath_json_new = dpath_out_new / f"{new_tag}.json"
    save_json(fpath_json_new, json_data_new)
    print(f"Saved new JSON data to {fpath_json_new}")


@click.command(context_settings=CLICK_CONTEXT_SETTINGS)
@click.option("--tag", type=str, required=True)
@click.option(
    "--data-dir",
    "dpath_data",
    type=click.Path(path_type=Path, file_okay=False, dir_okay=True),
    default=DEFAULT_DPATH_DATA,
)
@click.option(
    "--min-timepoints",
    "min_n_timepoints",
    type=click.IntRange(min=2, min_open=False),
    default=DEFAULT_MIN_N_TIMEPOINTS,
)
def main(*args, **kwargs):
    split_train_test(*args, **kwargs)


if __name__ == "__main__":
    main()
