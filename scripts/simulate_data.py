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
DEFAULT_N_MAX_TIMEPOINTS_ALL = (3, 3, 3)
DEFAULT_T0_MIN_ALL = (0.0, 0.0, 0.0)
DEFAULT_T0_MAX_ALL = (1.0, 1.0, 1.0)
DEFAULT_SIGMA_ALL = (0.1, 0.1, 0.1)
DEFAULT_K_MIN = 5.0
DEFAULT_K_MAX = 10.0
DEFAULT_X0_MIN = 0.0
DEFAULT_X0_MAX = 1.0

COL_SUBJECT = "subject"
COL_TIMEPOINT = "timepoint"
COL_BIOMARKER_PREFIX = "biomarker_"


def _build_df(timepoints, biomarkers, n_biomarkers, n_subjects_so_far) -> pd.DataFrame:
    subjects = np.repeat(
        np.arange(len(biomarkers)) + n_subjects_so_far, [len(bm) for bm in biomarkers]
    )

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
    rng_seed: int = None,
):
    dpath_out = get_dpath_latest(dpath_data, use_today=True)
    dpath_out.mkdir(parents=True, exist_ok=True)

    json_data = {"settings": locals()}

    list_lengths = [
        len(n_subjects_all),
        len(n_max_timepoints_all),
        len(t0_min_all),
        len(t0_max_all),
        len(sigma_all),
    ]
    if len(set(list_lengths)) != 1:
        raise ValueError(
            "Arguments --n-subjects, --n-max-timepoints, --t0-min, --t0-max, and "
            f"--sigma must have the same number of values. Got {list_lengths}."
        )

    rng = np.random.default_rng(rng_seed)
    k_values = rng.uniform(k_min, k_max, size=n_biomarkers)
    x0_values = rng.uniform(x0_min, x0_max, size=n_biomarkers)

    timepoints_all = []
    biomarkers_all = []
    time_shifts_all = []

    n_subjects_so_far = 0
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
        timepoints, biomarkers, time_shifts = simulate_all_subjects(
            n_subjects=n_subjects,
            k_values=k_values,
            x0_values=x0_values,
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
        fpath_tsv = dpath_out / _get_fname_out(tag, i_dataset)
        df_data.to_csv(fpath_tsv, sep="\t", index=False)
        print(f"Saved simulated data to {fpath_tsv}")

        n_subjects_so_far += n_subjects

    json_data["params"] = {
        "time_shifts": time_shifts_all,
        "k_values": k_values,
        "x0_values": x0_values,
        "sigmas": sigma_all,
    }

    fpath_json = dpath_out / _get_fname_out(tag, suffix=".json")
    save_json(fpath_json, json_data)
    print(f"Saved simulation settings and parameters to {fpath_json}")


if __name__ == "__main__":
    simulate_data()
