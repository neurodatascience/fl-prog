from collections.abc import Iterable
from pathlib import Path

import pandas as pd

COL_SUBJECT = "participant_id_int"
COL_TIMEPOINT = "months_scaled"


def _merge_hemispheres(df: pd.DataFrame) -> pd.DataFrame:
    measures = set(col.removeprefix("rh_").removeprefix("lh_") for col in df.columns)
    for measure in sorted(measures):
        col_rh = f"rh_{measure}"
        col_lh = f"lh_{measure}"
        if col_rh in df.columns and col_lh in df.columns:
            df[measure] = df[[col_rh, col_lh]].mean(axis=1)
            df = df.drop(columns=[col_rh, col_lh])
    return df


def infer_sep(fpath: Path) -> str:
    if fpath.suffix == ".tsv":
        return "\t"
    elif fpath.suffix == ".csv":
        return ","
    else:
        raise ValueError(
            f"Could not infer separator from file extension {fpath.suffix}."
        )


def get_df_idp(
    fpath_idps: Path,
    merge_hemispheres: bool,
    col_subject_original: str,
    col_session_original: str,
    session_timepoint_map: dict[str, float],
    measures: Iterable[str] | None = None,
) -> pd.DataFrame:
    df_idp = pd.read_csv(
        fpath_idps,
        sep=infer_sep(fpath_idps),
        index_col=[col_subject_original, col_session_original],
        dtype={col_subject_original: str, col_session_original: str},
    )
    df_idp = df_idp.dropna(axis="index", how="any")
    df_idp = df_idp.sort_index()
    if measures is not None:
        df_idp = df_idp.loc[:, measures]

    if merge_hemispheres:
        df_idp = _merge_hemispheres(df_idp)

    session_ids = df_idp.index.get_level_values(col_session_original).unique()
    for session_id in session_ids:
        if session_id not in session_timepoint_map:
            raise ValueError(
                f"Session ID {session_id} not found in session_timepoint_map."
                f" Make sure all session IDs are present: {session_ids}."
            )
    df_idp[COL_TIMEPOINT] = df_idp.index.get_level_values(col_session_original).map(
        session_timepoint_map
    )
    participant_ids = sorted(
        df_idp.index.get_level_values(col_subject_original).unique().tolist()
    )
    df_idp[COL_SUBJECT] = df_idp.index.get_level_values(col_subject_original).map(
        lambda x: participant_ids.index(x)
    )

    return df_idp
