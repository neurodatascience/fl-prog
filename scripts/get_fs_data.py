#!/usr/bin/env python

from pathlib import Path
from typing import Optional

import click
import numpy as np
import pandas as pd

from fl_prog.freesurfer import get_df_idp, COL_SUBJECT, COL_TIMEPOINT
from fl_prog.utils.constants import CLICK_CONTEXT_SETTINGS
from fl_prog.utils.io import load_json, save_json, get_dpath_latest, DEFAULT_DPATH_DATA

DEFAULT_N_SITES = 5


def _get_fname_out(tag, i: Optional[int] = None, suffix: str = ".tsv") -> str:
    if i is not None:
        tag = f"{tag}-{i}"
    return f"{tag}{suffix}"


def _scale_min_max(df, min, max, measures) -> pd.DataFrame:
    df.loc[:, measures] = (df[measures] - min) / (max - min)
    return df


def _flip(df, measures) -> pd.DataFrame:
    """Flip 1->0 to 0->1."""
    df.loc[:, measures] = 1 - df[measures]
    return df


def get_fs_data(
    tag: str,
    fpath_idps: Path,
    n_sites: int,
    dpath_data: Path,
    fpath_config: Path,
    rng_seed: int = None,
):
    dpath_out = get_dpath_latest(dpath_data, use_today=True) / tag
    dpath_out.mkdir(parents=True, exist_ok=True)

    config = load_json(fpath_config)

    json_data = {"settings": locals().copy()}

    rng = np.random.default_rng(rng_seed)

    merge_hemispheres: bool = config.get("merge_hemispheres", True)
    col_subject_original: str = config["col_subject_original"]
    col_session_original: str = config["col_session_original"]
    session_timepoint_map: dict[str, float] = config["session_timepoint_map"]
    measures: list[str] = config["measures"]
    flip: bool = config.get("flip", False)
    max_time: float = config.get("max_time", None)
    min_max_by_measure: dict[list[float]] = config.get("min_max_by_measure", None)

    df_idp = get_df_idp(
        fpath_idps,
        merge_hemispheres,
        col_subject_original,
        col_session_original,
        session_timepoint_map,
        measures,
    )

    cols_biomarkers = list(set(df_idp.columns) - {COL_SUBJECT, COL_TIMEPOINT})

    if min_max_by_measure is not None:
        min_values = pd.DataFrame(
            data=[x[0] for x in min_max_by_measure.values()],
            index=min_max_by_measure.keys(),
        ).squeeze()
        max_values = pd.DataFrame(
            data=[x[1] for x in min_max_by_measure.values()],
            index=min_max_by_measure.keys(),
        ).squeeze()
        df_idp = _scale_min_max(df_idp, min_values, max_values, cols_biomarkers)

    if flip:
        df_idp = _flip(df_idp, cols_biomarkers)

    if max_time is not None:
        df_idp = df_idp.query(f"{COL_TIMEPOINT} <= {max_time}")
        df_idp.loc[:, COL_TIMEPOINT] = df_idp[COL_TIMEPOINT] / max_time

    participant_ids = sorted(
        df_idp.index.get_level_values(col_subject_original).unique().tolist()
    )
    rng.shuffle(participant_ids)

    node_id_map = {}
    subjects_by_node = {}
    for i_site, site_participant_ids in enumerate(
        np.array_split(participant_ids, n_sites),
        start=1,
    ):
        node_id = str(i_site)
        subjects_by_node[node_id] = site_participant_ids.tolist()
        df_site = df_idp.loc[site_participant_ids]

        # reindex
        df_site[COL_SUBJECT] = df_site.index.get_level_values(col_subject_original).map(
            lambda x: subjects_by_node[node_id].index(x)
        )

        fname_tsv = _get_fname_out(tag, i=i_site)
        fpath_tsv = dpath_out / fname_tsv
        df_site.to_csv(fpath_tsv, sep="\t", index=True)
        print(
            f"Site {i_site}: {df_site.shape}, {len(site_participant_ids)} participants, {fpath_tsv}"
        )
        node_id_map[fname_tsv] = node_id

    json_data["node_id_map"] = node_id_map

    json_data["cols"] = {
        "col_subject": col_subject_original,
        "col_subject_index": COL_SUBJECT,
        "col_timepoint": COL_TIMEPOINT,
        "cols_biomarker": sorted(
            cols_biomarkers,
        ),
    }

    json_data["subjects_by_node"] = subjects_by_node

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
@click.option("--n-sites", "-n", type=int, default=DEFAULT_N_SITES)
@click.option(
    "--data-dir",
    "dpath_data",
    type=click.Path(path_type=Path, file_okay=False, dir_okay=True),
    default=DEFAULT_DPATH_DATA,
)
@click.option(
    "--config",
    "fpath_config",
    type=click.Path(path_type=Path, file_okay=True, dir_okay=False),
    required=True,
    envvar="ADNI_CONFIG_FILE",
)
@click.option("--rng-seed", type=int, default=None, envvar="RNG_SEED")
def main(*args, **kwargs):
    get_fs_data(*args, **kwargs)


if __name__ == "__main__":
    main()
