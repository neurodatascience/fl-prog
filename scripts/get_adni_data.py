#!/usr/bin/env python

import enum
from collections.abc import Iterable
from pathlib import Path
from typing import Optional

import click
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from fl_prog.freesurfer import get_df_idp, COL_SUBJECT, COL_TIMEPOINT
from fl_prog.utils.constants import CLICK_CONTEXT_SETTINGS
from fl_prog.utils.io import load_json, save_json, get_dpath_latest, DEFAULT_DPATH_DATA


class NonIIDStrategy(enum.Enum):
    SITE = "site"
    DIAGNOSIS = "diagnosis"


DEFAULT_N_SITES = 5
DEFAULT_IID = True
DEFAULT_NON_IID_STRATEGY = NonIIDStrategy.SITE

COL_SUBJECT_ADNIMERGE = "RID"
COL_SESSION_ADNIMERGE = "VISCODE"
COL_SITE_ADNIMERGE = "SITE"
COL_AGE_ADNIMERGE = "AGE"


def _normalize_adni_rid(value) -> str:
    """Normalize ADNI RID values so CSV/string/int representations match."""
    if pd.isna(value):
        return value

    value = str(value).strip()
    if value.endswith(".0"):
        value = value[:-2]

    return value


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


def _split_participants_into_sites(
    participant_ids: list, n_sites: int, site_map: dict | None = None
) -> list[np.ndarray]:
    if site_map is None:
        site_map = {
            participant_id: i for i, participant_id in enumerate(participant_ids)
        }

    participant_ids = np.array(participant_ids)

    splitter = GroupKFold(n_splits=n_sites)
    participant_ids_split = []
    for _, idx_test in splitter.split(
        participant_ids,
        groups=[site_map[participant_id] for participant_id in participant_ids],
    ):
        participant_ids_split.append(participant_ids[idx_test])

    return participant_ids_split


def _add_adnimerge_measures(
    df_idp: pd.DataFrame,
    df_adnimerge: pd.DataFrame,
    measures_adnimerge: Iterable[str],
) -> pd.DataFrame:
    original_index_names = df_idp.index.names
    df_idp.index.names = [COL_SUBJECT_ADNIMERGE, COL_SESSION_ADNIMERGE]

    df_adnimerge_subset = df_adnimerge.set_index(
        [COL_SUBJECT_ADNIMERGE, COL_SESSION_ADNIMERGE]
    )[list(measures_adnimerge)]

    df_merged = df_idp.merge(
        df_adnimerge_subset,
        left_index=True,
        right_index=True,
        validate="1:1",
    )
    df_merged = df_merged.dropna(axis="index", how="any", subset=measures_adnimerge)
    df_merged.index.names = original_index_names
    return df_merged


def _add_adni_age_column(
    df_idp: pd.DataFrame,
    fpath_adni_merge: Path | None,
    col_subject_original: str,
) -> pd.DataFrame:
    """
    Add biological AGE from ADNIMERGE to the FreeSurfer dataframe.

    ADNIMERGE.AGE is constant across longitudinal rows for the same RID, so it is
    treated as baseline chronological age and copied directly.
    """
    if fpath_adni_merge is None:
        raise ValueError(
            "AGE requires ADNIMERGE. Pass --adni-merge or set ADNI_MERGE_FILE."
        )

    df_adnimerge = pd.read_csv(fpath_adni_merge, dtype={COL_SUBJECT_ADNIMERGE: str})

    required_cols = {COL_SUBJECT_ADNIMERGE, COL_AGE_ADNIMERGE}
    missing_cols = required_cols - set(df_adnimerge.columns)
    if missing_cols:
        raise ValueError(
            f"{fpath_adni_merge} is missing columns: {sorted(missing_cols)}"
        )

    df_age = df_adnimerge[[COL_SUBJECT_ADNIMERGE, COL_AGE_ADNIMERGE]].copy()
    df_age[COL_SUBJECT_ADNIMERGE] = df_age[COL_SUBJECT_ADNIMERGE].map(
        _normalize_adni_rid
    )
    df_age[COL_AGE_ADNIMERGE] = pd.to_numeric(
        df_age[COL_AGE_ADNIMERGE], errors="coerce"
    )

    # ADNIMERGE has repeated rows per RID. AGE should be constant within RID.
    age_nunique = (
        df_age.dropna().groupby(COL_SUBJECT_ADNIMERGE)[COL_AGE_ADNIMERGE].nunique()
    )
    inconsistent_rids = age_nunique[age_nunique > 1].index.tolist()
    if inconsistent_rids:
        print(
            "Warning: some RID values have multiple AGE values in ADNIMERGE. "
            f"Using the first non-missing AGE. Example RIDs: {inconsistent_rids[:10]}"
        )

    age_by_rid = (
        df_age.dropna(subset=[COL_SUBJECT_ADNIMERGE, COL_AGE_ADNIMERGE])
        .drop_duplicates(subset=[COL_SUBJECT_ADNIMERGE])
        .set_index(COL_SUBJECT_ADNIMERGE)[COL_AGE_ADNIMERGE]
    )

    index_names = df_idp.index.names
    df = df_idp.reset_index()

    if col_subject_original not in df.columns:
        raise ValueError(
            f"Expected subject column/index level {col_subject_original!r} "
            "in FreeSurfer dataframe."
        )

    df[COL_AGE_ADNIMERGE] = (
        df[col_subject_original].map(_normalize_adni_rid).map(age_by_rid)
    )

    n_missing = int(df[COL_AGE_ADNIMERGE].isna().sum())
    if n_missing:
        print(
            f"Warning: {n_missing} rows have missing {COL_AGE_ADNIMERGE} after ADNIMERGE join."
        )

    return df.set_index(index_names)


def get_adni_data(
    tag: str,
    fpath_idps: Path,
    n_sites: int,
    dpath_data: Path,
    fpath_config: Path,
    fpath_adni_merge: Path | None = None,
    iid: bool = DEFAULT_IID,
    rng_seed: int = None,
    non_iid_strategy: str = DEFAULT_NON_IID_STRATEGY,
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
    measures_adnimerge: list[str] = config.get("measures_adnimerge", [])
    flip: bool = config.get("flip", False)
    max_time: float = config.get("max_time", None)
    min_max_by_measure: dict[list[float]] = config.get("min_max_by_measure", None)

    if fpath_adni_merge is not None:
        df_adnimerge = pd.read_csv(
            fpath_adni_merge,
            dtype={
                COL_SUBJECT_ADNIMERGE: str,
                COL_SESSION_ADNIMERGE: str,
                COL_SITE_ADNIMERGE: str,
            },
        )
        df_adnimerge[COL_SESSION_ADNIMERGE] = df_adnimerge[
            COL_SESSION_ADNIMERGE
        ].str.upper()
    else:
        df_adnimerge = None

    df_idp = get_df_idp(
        fpath_idps,
        merge_hemispheres,
        col_subject_original,
        col_session_original,
        session_timepoint_map,
        measures,
    )
    if measures_adnimerge:
        if df_adnimerge is None:
            raise ValueError(
                "fpath_adni_merge must be provided if requesting ADNIMERGE measures"
            )
        df_idp = _add_adnimerge_measures(df_idp, df_adnimerge, measures_adnimerge)
    # Add biological AGE information
    df_idp = _add_adni_age_column(
        df_idp=df_idp,
        fpath_adni_merge=fpath_adni_merge,
        col_subject_original=col_subject_original,
    )

    if not iid:
        if df_adnimerge is None:
            raise ValueError(
                "fpath_adni_merge must be provided if requesting non-IID split"
            )

        match non_iid_strategy:
            case NonIIDStrategy.SITE:
                site_map = (
                    df_adnimerge.drop_duplicates(
                        [COL_SUBJECT_ADNIMERGE, COL_SITE_ADNIMERGE]
                    )
                    .set_index(COL_SUBJECT_ADNIMERGE)[COL_SITE_ADNIMERGE]
                    .to_dict()
                )
            case NonIIDStrategy.DIAGNOSIS:
                site_map = (
                    df_adnimerge.drop_duplicates([COL_SUBJECT_ADNIMERGE, "DX_bl"])
                    .set_index(COL_SUBJECT_ADNIMERGE)["DX_bl"]
                    .to_dict()
                )
            case _:
                raise ValueError(f"Invalid non_iid_strategy: {non_iid_strategy}")
    else:
        site_map = None

    cols_biomarkers = list(
        set(df_idp.columns) - {COL_SUBJECT, COL_TIMEPOINT, COL_AGE_ADNIMERGE}
    )

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
        _split_participants_into_sites(participant_ids, n_sites, site_map=site_map),
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
        "col_age": COL_AGE_ADNIMERGE,
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
@click.option("--iid/--non-iid", default=DEFAULT_IID)
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
@click.option(
    "--adni-merge",
    "fpath_adni_merge",
    type=click.Path(path_type=Path, file_okay=True, dir_okay=False),
    envvar="ADNI_MERGE_FILE",
)
@click.option("--rng-seed", type=int, default=None, envvar="RNG_SEED")
@click.option(
    "--non-iid-strategy",
    type=click.Choice(NonIIDStrategy, case_sensitive=False),
    default=DEFAULT_NON_IID_STRATEGY,
)
def main(*args, **kwargs):
    get_adni_data(*args, **kwargs)


if __name__ == "__main__":
    main()
