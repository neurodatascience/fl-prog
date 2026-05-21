#!/usr/bin/env python

from pathlib import Path
from typing import Optional

import click
import numpy as np
import pandas as pd

from fl_prog.simulation import simulate_all_subjects
from fl_prog.utils.constants import CLICK_CONTEXT_SETTINGS
from fl_prog.utils.io import save_json, get_dpath_latest, DEFAULT_DPATH_DATA

DEFAULT_N_BIOMARKERS = 5
DEFAULT_SHIFT_TIME = True
DEFAULT_N_SUBJECTS_ALL = (20, 20, 20)
DEFAULT_N_MAX_TIMEPOINTS_ALL = (3,)
DEFAULT_T0_MIN_ALL = (0.0,)
DEFAULT_T0_MAX_ALL = (1.0,)
DEFAULT_SIGMA_ALL = (0.1,)
DEFAULT_K_MIN = 5.0
DEFAULT_K_MAX = 10.0
DEFAULT_X0_MIN = 0.0
DEFAULT_X0_MAX = 1.0
DEFAULT_VERTICAL_SHIFT_MIN = 0
DEFAULT_VERTICAL_SHIFT_MAX = 0
DEFAULT_SCALING_FACTOR_MIN = 1
DEFAULT_SCALING_FACTOR_MAX = 1

COL_SUBJECT = "subject"
COL_SUBJECT_INDEX = "subject_index"
COL_TIMEPOINT = "timepoint"
COL_BIOMARKER_PREFIX = "biomarker_"


def _process_list_args(*list_args):
    def _get_list_lengths(list_args):
        return [len(list_arg) for list_arg in list_args]

    list_args = list(list_args)
    initial_list_lengths = _get_list_lengths(list_args)

    # broadcast single-value args to match the length of the longest arg
    max_list_length = max(initial_list_lengths)
    for i, list_arg in enumerate(list_args):
        if len(list_arg) == 1:
            list_args[i] = list_arg * max_list_length

    if len(set(_get_list_lengths(list_args))) != 1:
        raise ValueError(
            "Arguments --n-subjects, --n-max-timepoints, --t0-min, --t0-max, and "
            f"--sigma must have the same number of values. Got {initial_list_lengths}."
        )

    return list_args


def _build_df(timepoints, biomarkers, n_biomarkers, n_subjects_so_far) -> pd.DataFrame:
    subjects = np.repeat(
        np.arange(len(biomarkers)) + n_subjects_so_far, [len(bm) for bm in biomarkers]
    )

    df_data = pd.DataFrame(
        data={
            COL_SUBJECT: subjects,
            COL_SUBJECT_INDEX: subjects - n_subjects_so_far,
            COL_TIMEPOINT: np.concatenate(timepoints),
            **{
                f"{COL_BIOMARKER_PREFIX}{i}": np.concatenate(biomarkers)[:, i]
                for i in range(n_biomarkers)
            },
        }
    )

    return df_data


def _get_fname_out(tag, i: Optional[int] = None, suffix: str = ".tsv") -> str:
    if i is not None:
        tag = f"{tag}-{i+1}"
    return f"{tag}{suffix}"


@click.command(context_settings=CLICK_CONTEXT_SETTINGS)
@click.option("--tag", type=str, required=True)
@click.option(
    "--data-dir",
    "dpath_data",
    type=click.Path(path_type=Path, file_okay=False, dir_okay=True),
    default=DEFAULT_DPATH_DATA,
)
@click.option(
    "--n-biomarkers", type=click.IntRange(min=1), default=DEFAULT_N_BIOMARKERS
)
@click.option("--shift-time/--no-shift-time", default=DEFAULT_SHIFT_TIME)
@click.option(
    "--n-subjects",
    "n_subjects_all",
    type=click.IntRange(min=1),
    multiple=True,
    default=DEFAULT_N_SUBJECTS_ALL,
)
@click.option(
    "--n-max-timepoints",
    "n_max_timepoints_all",
    type=click.IntRange(min=1),
    multiple=True,
    default=DEFAULT_N_MAX_TIMEPOINTS_ALL,
)
@click.option(
    "--t0-min", "t0_min_all", type=float, multiple=True, default=DEFAULT_T0_MIN_ALL
)
@click.option(
    "--t0-max", "t0_max_all", type=float, multiple=True, default=DEFAULT_T0_MAX_ALL
)
@click.option(
    "--sigma",
    "sigma_all",
    type=click.FloatRange(min=0, min_open=True),
    multiple=True,
    default=DEFAULT_SIGMA_ALL,
)
@click.option("--k-min", type=float, default=DEFAULT_K_MIN)
@click.option("--k-max", type=float, default=DEFAULT_K_MAX)
@click.option("--x0-min", type=float, default=DEFAULT_X0_MIN)
@click.option("--x0-max", type=float, default=DEFAULT_X0_MAX)
@click.option("--vertical-shift-min", type=float, default=DEFAULT_VERTICAL_SHIFT_MIN)
@click.option("--vertical-shift-max", type=float, default=DEFAULT_VERTICAL_SHIFT_MAX)
@click.option("--scaling-factor-min", type=float, default=DEFAULT_SCALING_FACTOR_MIN)
@click.option("--scaling-factor-max", type=float, default=DEFAULT_SCALING_FACTOR_MAX)
@click.option("--rng-seed", type=int, default=None, envvar="RNG_SEED")
def simulate_data(
    tag: str,
    dpath_data: Path,
    n_biomarkers: int = DEFAULT_N_BIOMARKERS,
    shift_time: bool = DEFAULT_SHIFT_TIME,
    n_subjects_all: int = DEFAULT_N_SUBJECTS_ALL,
    n_max_timepoints_all: int = DEFAULT_N_MAX_TIMEPOINTS_ALL,
    t0_min_all: float = DEFAULT_T0_MIN_ALL,
    t0_max_all: float = DEFAULT_T0_MAX_ALL,
    sigma_all: float = DEFAULT_SIGMA_ALL,
    k_min: float = DEFAULT_K_MIN,
    k_max: float = DEFAULT_K_MAX,
    x0_min: float = DEFAULT_X0_MIN,
    x0_max: float = DEFAULT_X0_MAX,
    vertical_shift_min: float = DEFAULT_VERTICAL_SHIFT_MIN,
    vertical_shift_max: float = DEFAULT_VERTICAL_SHIFT_MAX,
    scaling_factor_min: float = DEFAULT_SCALING_FACTOR_MIN,
    scaling_factor_max: float = DEFAULT_SCALING_FACTOR_MAX,
    rng_seed: int = None,
):
    dpath_out = get_dpath_latest(dpath_data, use_today=True) / tag
    dpath_out.mkdir(parents=True, exist_ok=True)

    n_subjects_all, n_max_timepoints_all, t0_min_all, t0_max_all, sigma_all = (
        _process_list_args(
            n_subjects_all, n_max_timepoints_all, t0_min_all, t0_max_all, sigma_all
        )
    )

    json_data = {"settings": locals()}

    rng = np.random.default_rng(rng_seed)
    k_values = rng.uniform(k_min, k_max, size=n_biomarkers)
    x0_values = rng.uniform(x0_min, x0_max, size=n_biomarkers)
    vertical_shifts = rng.uniform(
        vertical_shift_min, vertical_shift_max, size=n_biomarkers
    )
    scaling_factors = rng.uniform(
        scaling_factor_min, scaling_factor_max, size=n_biomarkers
    )

    timepoints_all = []
    biomarkers_all = []
    time_shifts_all = []

    n_subjects_so_far = 0
    node_id_map = {}
    subjects_by_node = {}
    for i_dataset, (n_subjects, n_max_timepoints, t0_min, t0_max, sigma) in enumerate(
        zip(
            n_subjects_all,
            n_max_timepoints_all,
            t0_min_all,
            t0_max_all,
            sigma_all,
            strict=True,
        )
    ):
        node_id = f"{i_dataset+1}"
        timepoints, biomarkers, time_shifts = simulate_all_subjects(
            n_subjects=n_subjects,
            k_values=k_values,
            x0_values=x0_values,
            vertical_shifts=vertical_shifts,
            scaling_factors=scaling_factors,
            max_n_timepoints=n_max_timepoints,
            shift_time=shift_time,
            t0_min=t0_min,
            t0_max=t0_max,
            sigma=sigma,
            rng=rng,
        )
        timepoints_all.append(timepoints)
        biomarkers_all.append(biomarkers)
        time_shifts_all.append(time_shifts)

        df_data = _build_df(timepoints, biomarkers, n_biomarkers, n_subjects_so_far)

        subjects_by_node[node_id] = df_data[COL_SUBJECT].unique().tolist()

        fname_tsv = _get_fname_out(tag, i_dataset)
        fpath_tsv = dpath_out / fname_tsv
        df_data.to_csv(fpath_tsv, sep="\t", index=False)
        print(f"Saved simulated data to {fpath_tsv}")

        n_subjects_so_far += n_subjects
        node_id_map[fname_tsv] = node_id

    json_data["params"] = {
        "time_shifts": time_shifts_all,
        "k_values": k_values,
        "x0_values": x0_values,
        "vertical_shifts": vertical_shifts,
        "scaling_factors": scaling_factors,
        "sigmas": sigma_all,
    }
    json_data["node_id_map"] = node_id_map

    json_data["cols"] = {
        "col_subject": COL_SUBJECT,
        "col_subject_index": COL_SUBJECT_INDEX,
        "col_timepoint": COL_TIMEPOINT,
        "cols_biomarker": sorted(
            list(set(df_data.columns) - {COL_SUBJECT, COL_TIMEPOINT, COL_SUBJECT_INDEX})
        ),
    }
    json_data["subjects_by_node"] = subjects_by_node

    fpath_json = dpath_out / _get_fname_out(tag, suffix=".json")
    save_json(fpath_json, json_data)
    print(f"Saved simulation settings, parameters and node ID map to {fpath_json}")


if __name__ == "__main__":
    simulate_data()
