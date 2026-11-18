#!/usr/bin/env python

from collections.abc import Iterable
from pathlib import Path

import click
import numpy as np
import pandas as pd

from fl_prog.simulation import assign_sites_by_overlap, simulate_all_subjects
from fl_prog.utils.constants import CLICK_CONTEXT_SETTINGS
from fl_prog.utils.io import DEFAULT_DPATH_DATA, get_dpath_latest, load_json, save_json

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
DEFAULT_ACCELERATION_FACTOR_STD = 0
DEFAULT_SCALING_FACTOR_MIN = 1
DEFAULT_SCALING_FACTOR_MAX = 1
DEFAULT_NA_RATIO = 0.0

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
        raise ValueError("List length mismatch: " + str(initial_list_lengths))

    return list_args


def _build_df(
    timepoints,
    biomarkers,
    n_biomarkers,
    n_subjects_so_far,
    na_ratio: float = 0.0,
    subject_ids: Iterable[int] | None = None,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    if subject_ids is None:
        subject_ids = np.arange(len(biomarkers)) + n_subjects_so_far
    n_timepoints = [len(bm) for bm in biomarkers]
    subjects = np.repeat(np.asarray(subject_ids), n_timepoints)
    subject_indices = np.repeat(np.arange(len(biomarkers)), n_timepoints)

    biomarkers_dict = {
        f"{COL_BIOMARKER_PREFIX}{i}": np.concatenate(biomarkers)[:, i]
        for i in range(n_biomarkers)
    }

    # inject missing values
    if na_ratio > 0:
        if rng is None:
            rng = np.random.default_rng(None)
        for biomarker_name, biomarker_values in biomarkers_dict.items():
            n_missing = int(len(biomarker_values) * na_ratio)
            missing_indices = rng.choice(
                len(biomarker_values), size=n_missing, replace=False
            )
            biomarkers_dict[biomarker_name][missing_indices] = np.nan

    df_data = pd.DataFrame(
        data={
            COL_SUBJECT: subjects,
            COL_SUBJECT_INDEX: subject_indices,
            COL_TIMEPOINT: np.concatenate(timepoints),
            **biomarkers_dict,
        }
    )

    return df_data


def _get_fname_out(tag, i: int | None = None, suffix: str = ".tsv") -> str:
    if i is not None:
        tag = f"{tag}-{i + 1}"
    return f"{tag}{suffix}"


def simulate_data(
    tag: str,
    dpath_data: Path,
    fpath_config: Path | None = None,
    n_biomarkers: int = DEFAULT_N_BIOMARKERS,
    shift_time: bool = DEFAULT_SHIFT_TIME,
    n_subjects_all: Iterable[int] = DEFAULT_N_SUBJECTS_ALL,
    n_max_timepoints_all: Iterable[int] = DEFAULT_N_MAX_TIMEPOINTS_ALL,
    t0_min_all: Iterable[float] = DEFAULT_T0_MIN_ALL,
    t0_max_all: Iterable[float] = DEFAULT_T0_MAX_ALL,
    sigma_all: Iterable[float] = DEFAULT_SIGMA_ALL,
    k_min: float = DEFAULT_K_MIN,
    k_max: float = DEFAULT_K_MAX,
    x0_min: float = DEFAULT_X0_MIN,
    x0_max: float = DEFAULT_X0_MAX,
    acceleration_factor_std: float = DEFAULT_ACCELERATION_FACTOR_STD,
    scaling_factor_min: float = DEFAULT_SCALING_FACTOR_MIN,
    scaling_factor_max: float = DEFAULT_SCALING_FACTOR_MAX,
    na_ratio: float = DEFAULT_NA_RATIO,
    n_sites: int | None = None,
    overlap: float | None = None,
    rng_seed: int | None = None,
):
    """Simulate data for multiple sites.

    By default, each site is simulated independently (with its own number of
    subjects and t0 range). If ``n_sites`` and ``overlap`` are given, a single
    population is simulated and then split into ``n_sites`` equally sized sites
    based on t0.
    """
    pooled = n_sites is not None or overlap is not None
    if pooled and (n_sites is None or overlap is None):
        raise ValueError("--n-sites and --overlap must be used together.")
    if pooled and any(
        len(arg) != 1
        for arg in (n_subjects_all, n_max_timepoints_all, t0_min_all, t0_max_all)
    ):
        raise ValueError(
            "--n-subjects, --n-max-timepoints, --t0-min, and --t0-max must have a "
            "single value when using --n-sites/--overlap."
        )

    dpath_out = get_dpath_latest(dpath_data, use_today=True) / tag
    dpath_out.mkdir(parents=True, exist_ok=True)

    try:
        n_subjects_all, n_max_timepoints_all, t0_min_all, t0_max_all = (
            _process_list_args(
                n_subjects_all, n_max_timepoints_all, t0_min_all, t0_max_all
            )
        )
    except ValueError as exception:
        raise ValueError(
            "Arguments --n-subjects, --n-max-timepoints, --t0-min, and --t0-max "
            f"must have the same number of values: {exception}."
        ) from exception

    if fpath_config is not None:
        print(f"Loading config from {fpath_config}")
        config = load_json(fpath_config)
    else:
        config = {}

    # fixed biomarker-specific parameters (override sampled values)
    fixed_params = {
        key: np.asarray(config[key], dtype=float)
        for key in ("k_values", "x0_values", "scaling_factors", "sigmas")
        if key in config
    }
    if fixed_params:
        lengths = {key: len(values) for key, values in fixed_params.items()}
        if len(set(lengths.values())) != 1:
            raise ValueError(
                f"Biomarker parameters in config have different lengths: {lengths}"
            )
        n_biomarkers = next(iter(lengths.values()))
        print(
            f"Using {n_biomarkers} biomarkers with fixed parameters from config: {sorted(fixed_params)}"
        )
        if "sigmas" in fixed_params:
            sigma_all = tuple(fixed_params["sigmas"].tolist())

    if len(sigma_all) == 1:
        sigma_all = sigma_all * n_biomarkers
    if len(sigma_all) != n_biomarkers:
        raise ValueError(
            "Argument --sigma must have either 1 value or the same number of values "
            f"as --n-biomarkers ({n_biomarkers}), got {len(sigma_all)}."
        )

    settings = dict(locals())
    for key in ("config", "fixed_params", "lengths"):
        settings.pop(key, None)
    json_data = {"settings": settings}

    n_timepoints_distribution = config.get("n_timepoints_distribution", None)
    time_at_timepoint = config.get("time_at_timepoint", None)

    rng = np.random.default_rng(rng_seed)

    # biomarker-specific parameters
    k_values = rng.uniform(k_min, k_max, size=n_biomarkers)
    x0_values = rng.uniform(x0_min, x0_max, size=n_biomarkers)
    scaling_factors = rng.uniform(
        scaling_factor_min, scaling_factor_max, size=n_biomarkers
    )
    k_values = fixed_params.get("k_values", k_values)
    x0_values = fixed_params.get("x0_values", x0_values)
    scaling_factors = fixed_params.get("scaling_factors", scaling_factors)

    timepoints_all = []
    biomarkers_all = []
    time_shifts_all = []
    acceleration_factors_all = []

    n_subjects_so_far = 0
    node_id_map = {}
    subjects_by_node = {}
    if pooled:
        (
            timepoints_pool,
            biomarkers_pool,
            time_shifts_pool,
            acceleration_factors_pool,
        ) = simulate_all_subjects(
            n_subjects=n_subjects_all[0],
            k_values=k_values,
            x0_values=x0_values,
            scaling_factors=scaling_factors,
            sigmas=sigma_all,
            max_n_timepoints=n_max_timepoints_all[0],
            n_timepoints_distribution=n_timepoints_distribution,
            time_at_timepoint=time_at_timepoint,
            shift_time=shift_time,
            t0_min=t0_min_all[0],
            t0_max=t0_max_all[0],
            acceleration_factor_std=acceleration_factor_std,
            rng=rng,
        )
        # true t0 (= time shift if shift_time, else the first timepoint)
        t0_pool = np.array(
            [
                shift if shift_time else tp[0]
                for shift, tp in zip(time_shifts_pool, timepoints_pool, strict=True)
            ]
        )
        # use a related but separate RNG instance for the split to keep the original
        # RNG stream independent from the overlap value
        sites = assign_sites_by_overlap(
            t0_pool,
            n_sites,
            overlap,
            rng=np.random.default_rng(None if rng_seed is None else [rng_seed, 1]),
        )
        n_subjects_all = []
        for i_dataset in range(n_sites):
            node_id = f"{i_dataset + 1}"
            idx = np.flatnonzero(sites == i_dataset)
            timepoints = [timepoints_pool[i] for i in idx]
            biomarkers = [biomarkers_pool[i] for i in idx]
            timepoints_all.append(timepoints)
            biomarkers_all.append(biomarkers)
            time_shifts_all.append(time_shifts_pool[idx])
            acceleration_factors_all.append(acceleration_factors_pool[idx])
            n_subjects_all.append(len(idx))

            df_data = _build_df(
                timepoints,
                biomarkers,
                n_biomarkers,
                n_subjects_so_far,
                na_ratio=na_ratio,
                subject_ids=idx,
                rng=rng,
            )
            subjects_by_node[node_id] = df_data[COL_SUBJECT].unique().tolist()

            fname_tsv = _get_fname_out(tag, i_dataset)
            fpath_tsv = dpath_out / fname_tsv
            df_data.to_csv(fpath_tsv, sep="\t", index=False)
            print(
                f"Saved simulated data to {fpath_tsv} ({len(idx)} subjects, "
                f"t0 range [{t0_pool[idx].min():.2f}, {t0_pool[idx].max():.2f}])"
            )
            node_id_map[fname_tsv] = node_id

    # one independent simulation per site (not used in pooled mode)
    site_args = (
        []
        if pooled
        else zip(
            n_subjects_all,
            n_max_timepoints_all,
            t0_min_all,
            t0_max_all,
            strict=True,
        )
    )
    for i_dataset, (n_subjects, n_max_timepoints, t0_min, t0_max) in enumerate(
        site_args
    ):
        node_id = f"{i_dataset + 1}"
        timepoints, biomarkers, time_shifts, acceleration_factors = (
            simulate_all_subjects(
                n_subjects=n_subjects,
                k_values=k_values,
                x0_values=x0_values,
                scaling_factors=scaling_factors,
                sigmas=sigma_all,
                max_n_timepoints=n_max_timepoints,
                n_timepoints_distribution=n_timepoints_distribution,
                time_at_timepoint=time_at_timepoint,
                shift_time=shift_time,
                t0_min=t0_min,
                t0_max=t0_max,
                acceleration_factor_std=acceleration_factor_std,
                rng=rng,
            )
        )
        timepoints_all.append(timepoints)
        biomarkers_all.append(biomarkers)
        time_shifts_all.append(time_shifts)
        acceleration_factors_all.append(acceleration_factors)

        df_data = _build_df(
            timepoints,
            biomarkers,
            n_biomarkers,
            n_subjects_so_far,
            na_ratio=na_ratio,
            rng=rng,
        )

        subjects_by_node[node_id] = df_data[COL_SUBJECT].unique().tolist()

        fname_tsv = _get_fname_out(tag, i_dataset)
        fpath_tsv = dpath_out / fname_tsv
        df_data.to_csv(fpath_tsv, sep="\t", index=False)
        print(f"Saved simulated data to {fpath_tsv}")

        n_subjects_so_far += n_subjects
        node_id_map[fname_tsv] = node_id

    json_data["params"] = {
        "time_shifts": time_shifts_all,
        "acceleration_factors": acceleration_factors_all,
        "k_values": k_values,
        "x0_values": x0_values,
        "scaling_factors": scaling_factors,
        "sigmas": sigma_all,
    }
    json_data["node_id_map"] = node_id_map
    if "biomarker_names" in config:
        # descriptive names for biomarker_0, biomarker_1, ...
        json_data["biomarker_names"] = config["biomarker_names"]

    json_data["cols"] = {
        "col_subject": COL_SUBJECT,
        "col_subject_index": COL_SUBJECT_INDEX,
        "col_timepoint": COL_TIMEPOINT,
        "cols_biomarker": sorted(
            set(df_data.columns) - {COL_SUBJECT, COL_TIMEPOINT, COL_SUBJECT_INDEX}
        ),
    }
    json_data["subjects_by_node"] = subjects_by_node
    json_data["need_scaling"] = False

    fpath_json = dpath_out / _get_fname_out(tag, suffix=".json")
    save_json(fpath_json, json_data)
    print(f"Saved simulation settings, parameters and node ID map to {fpath_json}")


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
    "fpath_config",
    type=click.Path(path_type=Path, file_okay=True, dir_okay=False),
    envvar="SIMULATION_CONFIG_FILE",
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
    type=click.FloatRange(min=0, min_open=False),
    multiple=True,
    default=DEFAULT_SIGMA_ALL,
)
@click.option("--k-min", type=float, default=DEFAULT_K_MIN)
@click.option("--k-max", type=float, default=DEFAULT_K_MAX)
@click.option("--x0-min", type=float, default=DEFAULT_X0_MIN)
@click.option("--x0-max", type=float, default=DEFAULT_X0_MAX)
@click.option(
    "--acceleration-factor-std", type=float, default=DEFAULT_ACCELERATION_FACTOR_STD
)
@click.option("--scaling-factor-min", type=float, default=DEFAULT_SCALING_FACTOR_MIN)
@click.option("--scaling-factor-max", type=float, default=DEFAULT_SCALING_FACTOR_MAX)
@click.option("--rng-seed", type=int, default=None, envvar="RNG_SEED")
@click.option(
    "--na-ratio",
    type=click.FloatRange(min=0, max=1, max_open=True),
    default=DEFAULT_NA_RATIO,
)
@click.option(
    "--n-sites",
    type=click.IntRange(min=1),
    default=None,
    help=(
        "Simulate a single population and split it into this many sites based on t0."
        " Requires --overlap and single values for --n-subjects/--t0-min/--t0-max."
    ),
)
@click.option(
    "--overlap",
    type=click.FloatRange(min=0, max=1),
    default=None,
    help=(
        "Fraction of each site's t0 window shared with the next site. Used with --n-sites."
    ),
)
def main(**params):
    simulate_data(**params)


if __name__ == "__main__":
    main()
