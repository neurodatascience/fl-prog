#!/usr/bin/env python

from pathlib import Path
from typing import Optional
from collections.abc import Iterable

import click
import numpy as np
import pandas as pd

from fl_prog.utils.constants import CLICK_CONTEXT_SETTINGS
from fl_prog.utils.io import save_json, get_dpath_latest, DEFAULT_DPATH_DATA

DEFAULT_MEASURES = (
    "rh_parahippocampal_thickness",
    "lh_parahippocampal_thickness",
    "rh_entorhinal_thickness",
    "lh_entorhinal_thickness",
    "rh_middletemporal_thickness",
    "lh_middletemporal_thickness",
    "rh_inferiorparietal_thickness",
    "lh_inferiorparietal_thickness",
    "rh_precuneus_thickness",
    "lh_precuneus_thickness",
)
DEFAULT_MERGE_HEMISPHERES = True
DEFAULT_N_SITES = 5

COL_SUBJECT_ORIGINAL = "participant_id"
COL_SESSION = "session_id"
COL_SUBJECT = "participant_id_int"
COL_TIMEPOINT = "months"

SESSION_TIMEPOINT_MAP = {
    "bl": 0.0,
    "m24": 24.0,
}


def _merge_hemispheres(df: pd.DataFrame) -> pd.DataFrame:
    measures = set(col.removeprefix("rh_").removeprefix("lh_") for col in df.columns)
    for measure in sorted(measures):
        col_rh = f"rh_{measure}"
        col_lh = f"lh_{measure}"
        if col_rh in df.columns and col_lh in df.columns:
            df[measure] = df[[col_rh, col_lh]].mean(axis=1)
            df = df.drop(columns=[col_rh, col_lh])
    return df


def _get_fname_out(tag, i: Optional[int] = None, suffix: str = ".tsv") -> str:
    if i is not None:
        tag = f"{tag}-{i}"
    return f"{tag}{suffix}"


def get_fs_data(
    tag: str,
    fpath_idps: Path,
    measures: Iterable[str],
    merge_hemispheres: bool,
    n_sites: int,
    dpath_data: Path,
    rng_seed: int = None,
):
    dpath_out = get_dpath_latest(dpath_data, use_today=True) / tag
    dpath_out.mkdir(parents=True, exist_ok=True)

    json_data = {"settings": locals()}

    rng = np.random.default_rng(rng_seed)

    df_idp = pd.read_csv(
        fpath_idps, sep="\t", index_col=[COL_SUBJECT_ORIGINAL, COL_SESSION]
    )
    df_idp = df_idp.dropna(axis="index", how="any")
    df_idp = df_idp.sort_index()
    df_idp = df_idp.loc[:, measures]

    if merge_hemispheres:
        df_idp = _merge_hemispheres(df_idp)

    cols_biomarkers = list(df_idp.columns)

    # scale so that min-max is 0-1 for each biomarker
    extrema = {}
    for col in df_idp.columns:
        min_val = df_idp[col].min()
        max_val = df_idp[col].max()
        extrema[col] = (min_val, max_val)
        df_idp[col] = (df_idp[col] - min_val) / (max_val - min_val)

    df_idp = df_idp.reset_index(level=COL_SESSION)
    df_idp[COL_TIMEPOINT] = df_idp[COL_SESSION].map(SESSION_TIMEPOINT_MAP)

    participant_ids = sorted(
        df_idp.index.get_level_values(COL_SUBJECT_ORIGINAL).unique().tolist()
    )
    df_idp[COL_SUBJECT] = df_idp.index.get_level_values(COL_SUBJECT_ORIGINAL).map(
        lambda x: participant_ids.index(x)
    )

    json_data["participant_ids"] = participant_ids.copy()

    rng.shuffle(participant_ids)

    node_id_map = {}
    for i_site, site_participant_ids in enumerate(
        np.array_split(participant_ids, n_sites),
        start=1,
    ):
        df_site = df_idp.loc[site_participant_ids]
        fname_tsv = _get_fname_out(tag, i=i_site)
        fpath_tsv = dpath_out / fname_tsv
        df_site.to_csv(fpath_tsv, sep="\t", index=True)
        print(
            f"Site {i_site}: {df_site.shape}, {len(site_participant_ids)} participants, {fpath_tsv}"
        )
        node_id_map[fname_tsv] = str(i_site)

    json_data["node_id_map"] = node_id_map

    json_data["cols"] = {
        "col_subject": COL_SUBJECT,
        "col_timepoint": COL_TIMEPOINT,
        "cols_biomarker": sorted(
            cols_biomarkers,
        ),
    }

    json_data["extrema"] = extrema

    fpath_json = dpath_out / _get_fname_out(tag, suffix=".json")
    save_json(fpath_json, json_data)
    print(f"Saved settings, col names, and node ID map to {fpath_json}")


@click.command(context_settings=CLICK_CONTEXT_SETTINGS)
@click.option("--tag", type=str, required=True)
@click.option(
    "--idps",
    "fpath_idps",
    type=click.Path(path_type=Path, file_okay=True, dir_okay=False),
    required=True,
    envvar="ADNI_IDP_FILE",
)
@click.option("--measures", "-m", multiple=True, default=DEFAULT_MEASURES)
@click.option(
    "--merge-hemis/--no-merge-hemis",
    "merge_hemispheres",
    is_flag=True,
    default=DEFAULT_MERGE_HEMISPHERES,
)
@click.option("--n-sites", "-n", type=int, default=DEFAULT_N_SITES)
@click.option(
    "--data-dir",
    "dpath_data",
    type=click.Path(path_type=Path, file_okay=False, dir_okay=True),
    default=DEFAULT_DPATH_DATA,
)
@click.option("--rng-seed", type=int, default=None, envvar="RNG_SEED")
def main(*args, **kwargs):
    get_fs_data(*args, **kwargs)


if __name__ == "__main__":
    main()
