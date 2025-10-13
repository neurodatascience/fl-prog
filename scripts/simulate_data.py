#!/usr/bin/env python

from pathlib import Path
from typing import Optional

import click
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from fl_prog.simulation import simulate_all_subjects
from fl_prog.utils.constants import CLICK_CONTEXT_SETTINGS
from fl_prog.utils.io import save_json, get_dpath_latest

DEFAULT_N_SUBJECTS = 50
DEFAULT_N_SPLITS = 3
DEFAULT_N_MAX_TIMEPOINTS = 3
DEFAULT_N_BIOMARKERS = 5
DEFAULT_SHIFT_TIME = True
DEFAULT_K_MIN = 5.0
DEFAULT_K_MAX = 10.0
DEFAULT_X0_MIN = 0.0
DEFAULT_X0_MAX = 1.0
DEFAULT_T0_MIN = 0.0
DEFAULT_T0_MAX = 1.0
DEFAULT_SIGMA = 0.1

COL_SUBJECT = "subject"
COL_TIMEPOINT = "timepoint"
COL_BIOMARKER_PREFIX = "biomarker_"


def _build_df(timepoints, biomarkers, n_biomarkers) -> pd.DataFrame:
    subjects = np.repeat(np.arange(len(biomarkers)), [len(bm) for bm in biomarkers])

    df_data = pd.DataFrame(
        data={
            COL_SUBJECT: subjects,
            COL_TIMEPOINT: np.concatenate(timepoints),
            **{
                f"{COL_BIOMARKER_PREFIX}{i}": np.concatenate(biomarkers)[:, i]
                for i in range(n_biomarkers)
            },
        }
    )

    return df_data


def _get_fname_out(i_split: Optional[int] = None, suffix: str = ".tsv") -> str:
    prefix = "simulated_data"
    if i_split is not None:
        prefix = f"{prefix}-{i_split+1}"
    return f"{prefix}{suffix}"


@click.command(context_settings=CLICK_CONTEXT_SETTINGS)
@click.argument(
    "dpath_data",
    type=click.Path(path_type=Path, file_okay=False, dir_okay=True),
    envvar="DPATH_DATA",
)
@click.option("--n-subjects", type=click.IntRange(min=1), default=DEFAULT_N_SUBJECTS)
@click.option("--n-splits", type=click.IntRange(min=1), default=DEFAULT_N_SPLITS)
@click.option(
    "--n-max-timepoints", type=click.IntRange(min=1), default=DEFAULT_N_MAX_TIMEPOINTS
)
@click.option(
    "--n-biomarkers", type=click.IntRange(min=1), default=DEFAULT_N_BIOMARKERS
)
@click.option("--shift-time/--no-shift-time", default=DEFAULT_SHIFT_TIME)
@click.option("--k-min", type=float, default=DEFAULT_K_MIN)
@click.option("--k-max", type=float, default=DEFAULT_K_MAX)
@click.option("--x0-min", type=float, default=DEFAULT_X0_MIN)
@click.option("--x0-max", type=float, default=DEFAULT_X0_MAX)
@click.option("--t0-min", type=float, default=DEFAULT_T0_MIN)
@click.option("--t0-max", type=float, default=DEFAULT_T0_MAX)
@click.option(
    "--sigma", type=click.FloatRange(min=0, min_open=True), default=DEFAULT_SIGMA
)
@click.option("--rng-seed", type=int, default=None, envvar="RNG_SEED")
def simulate_data(
    dpath_data: Path,
    n_subjects: int = DEFAULT_N_SUBJECTS,
    n_splits: int = DEFAULT_N_SPLITS,
    n_max_timepoints: int = DEFAULT_N_MAX_TIMEPOINTS,
    n_biomarkers: int = DEFAULT_N_BIOMARKERS,
    shift_time: bool = DEFAULT_SHIFT_TIME,
    k_min: float = DEFAULT_K_MIN,
    k_max: float = DEFAULT_K_MAX,
    x0_min: float = DEFAULT_X0_MIN,
    x0_max: float = DEFAULT_X0_MAX,
    t0_min: float = DEFAULT_T0_MIN,
    t0_max: float = DEFAULT_T0_MAX,
    sigma: float = DEFAULT_SIGMA,
    rng_seed: int = None,
):
    dpath_out = get_dpath_latest(dpath_data, use_today=True)
    dpath_out.mkdir(parents=True, exist_ok=True)

    json_data = {"settings": locals()}
    timepoints, biomarkers, time_shifts, k_values, x0_values = simulate_all_subjects(
        n_subjects=n_subjects,
        max_n_timepoints=n_max_timepoints,
        n_biomarkers=n_biomarkers,
        shift_time=shift_time,
        k_min=k_min,
        k_max=k_max,
        x0_min=x0_min,
        x0_max=x0_max,
        t0_min=t0_min,
        t0_max=t0_max,
        sigma=sigma,
        rng=np.random.default_rng(rng_seed),
    )
    json_data["params"] = {
        "time_shifts": time_shifts,
        "k_values": k_values,
        "x0_values": x0_values,
    }

    df_data = _build_df(timepoints, biomarkers, n_biomarkers)

    dfs_split: list[pd.DataFrame] = []
    if n_splits > 1:
        for _, idx_test in GroupKFold(n_splits=DEFAULT_N_SPLITS).split(
            df_data.index, groups=df_data[COL_SUBJECT]
        ):
            df_split: pd.DataFrame = df_data.loc[idx_test]
            df_split = df_split.sort_values(by=COL_SUBJECT)
            dfs_split.append(df_split)
    else:
        dfs_split.append(df_data)

    for i_split, df_split in enumerate(dfs_split):
        fpath_tsv = dpath_out / _get_fname_out(i_split)
        df_split.to_csv(fpath_tsv, sep="\t", index=False)
        print(f"Saved simulated data to {fpath_tsv}")

    fpath_json = dpath_out / _get_fname_out(suffix=".json")
    save_json(fpath_json, json_data)
    print(f"Saved simulation settings and parameters to {fpath_json}")


if __name__ == "__main__":
    simulate_data()
