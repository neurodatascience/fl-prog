from collections.abc import Iterable
from pathlib import Path

import pandas as pd

COL_SUBJECT_ORIGINAL = "participant_id"
COL_SESSION = "session_id"
COL_SUBJECT = "participant_id_int"
COL_TIMEPOINT = "months"


def _merge_hemispheres(df: pd.DataFrame) -> pd.DataFrame:
    measures = set(col.removeprefix("rh_").removeprefix("lh_") for col in df.columns)
    for measure in sorted(measures):
        col_rh = f"rh_{measure}"
        col_lh = f"lh_{measure}"
        if col_rh in df.columns and col_lh in df.columns:
            df[measure] = df[[col_rh, col_lh]].mean(axis=1)
            df = df.drop(columns=[col_rh, col_lh])
    return df


def get_df_idp(
    fpath_idps: Path,
    merge_hemispheres: bool,
    session_timepoint_map: dict[str, float],
    measures: Iterable[str] | None = None,
) -> pd.DataFrame:
    df_idp = pd.read_csv(
        fpath_idps, sep="\t", index_col=[COL_SUBJECT_ORIGINAL, COL_SESSION]
    )
    df_idp = df_idp.dropna(axis="index", how="any")
    df_idp = df_idp.sort_index()
    if measures is not None:
        df_idp = df_idp.loc[:, measures]

    if merge_hemispheres:
        df_idp = _merge_hemispheres(df_idp)

    df_idp[COL_TIMEPOINT] = df_idp.index.get_level_values(COL_SESSION).map(
        session_timepoint_map
    )
    participant_ids = sorted(
        df_idp.index.get_level_values(COL_SUBJECT_ORIGINAL).unique().tolist()
    )
    df_idp[COL_SUBJECT] = df_idp.index.get_level_values(COL_SUBJECT_ORIGINAL).map(
        lambda x: participant_ids.index(x)
    )

    return df_idp
