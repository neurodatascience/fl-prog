#!/usr/bin/env python
"""Simulate longitudinal biomarker data for multiple sites."""

from collections.abc import Iterable, Sequence
from fractions import Fraction
from pathlib import Path
from typing import NamedTuple

import click
import numpy as np
import pandas as pd

from fl_prog.simulation import Window, assign_sites_by_windows, simulate_all_subjects
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

# biomarker-specific parameters that can be fixed in the config file
KEYS_FIXED_PARAMS = ("k_values", "x0_values", "scaling_factors", "sigmas")

# suffix of a --site spec for a site with only the first visit of each subject
SITE_SPEC_CROSS = "cross"


class SiteSpec(NamedTuple):
    windows: list[Window]  # on the t0 rank scale [0, 1]
    cross_sectional: bool


def _parse_site_spec(spec: str) -> SiteSpec:
    """Parse a --site spec: ``LO-HI[,LO-HI...][:cross]``.

    Bounds can be decimals or fractions (e.g. ``1/6-1/2``).
    """
    windows_str, sep, suffix = spec.partition(":")
    if sep and suffix != SITE_SPEC_CROSS:
        raise ValueError(
            f"Invalid site spec {spec!r}: expected suffix ':{SITE_SPEC_CROSS}', "
            f"got ':{suffix}'"
        )
    windows = []
    for window_str in windows_str.split(","):
        try:
            start, end = (float(Fraction(x.strip())) for x in window_str.split("-"))
        except ValueError as exception:
            raise ValueError(
                f"Invalid window {window_str!r} in site spec {spec!r}: expected LO-HI"
            ) from exception
        if not 0 <= start < end <= 1:
            raise ValueError(
                f"Invalid window {window_str!r} in site spec {spec!r}: "
                "need 0 <= LO < HI <= 1"
            )
        if windows and start < windows[-1][1]:
            raise ValueError(
                f"Windows in site spec {spec!r} must be sorted and not overlap"
            )
        windows.append((start, end))
    return SiteSpec(windows=windows, cross_sectional=bool(sep))


class SiteData(NamedTuple):
    subject_ids: np.ndarray  # unique across sites
    timepoints: list[np.ndarray]
    biomarkers: list[np.ndarray]
    time_shifts: np.ndarray
    acceleration_factors: np.ndarray


def _broadcast(*list_args: Iterable) -> list[list]:
    """Repeat single-value args to match the longest arg."""
    list_args = [list(arg) for arg in list_args]
    max_length = max(len(arg) for arg in list_args)
    list_args = [arg * max_length if len(arg) == 1 else arg for arg in list_args]
    if len({len(arg) for arg in list_args}) != 1:
        raise ValueError(
            "Arguments --n-subjects, --n-max-timepoints, --t0-min, and --t0-max "
            "must have 1 value or the same number of values, got "
            f"{[len(arg) for arg in list_args]}."
        )
    return list_args


def _get_biomarker_params(
    config: dict,
    n_biomarkers: int,
    sigma_all: tuple[float, ...],
    k_range: tuple[float, float],
    x0_range: tuple[float, float],
    scaling_factor_range: tuple[float, float],
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Get k, x0, scaling factor and sigma for each biomarker.

    Any value in the config is used as-is. The rest are sampled or taken from
    input args. ``n_biomarkers`` is ignored if the number of biomarkers can
    be inferred from the config.
    """
    fixed = {
        key: np.asarray(config[key], dtype=float)
        for key in KEYS_FIXED_PARAMS
        if key in config
    }
    if fixed:
        lengths = {key: len(values) for key, values in fixed.items()}
        if len(set(lengths.values())) != 1:
            raise ValueError(
                f"Biomarker parameters in config have different lengths: {lengths}"
            )
        n_biomarkers = next(iter(lengths.values()))
        print(
            f"Using {n_biomarkers} biomarkers with fixed parameters from config: "
            f"{sorted(fixed)}"
        )

    if "sigmas" not in fixed and len(sigma_all) not in (1, n_biomarkers):
        raise ValueError(
            "Argument --sigma must have either 1 value or the same number of values "
            f"as the number of biomarkers ({n_biomarkers}), got {len(sigma_all)}."
        )

    # only sample parameters that are not fixed
    samplers = {
        "k_values": lambda: rng.uniform(*k_range, size=n_biomarkers),
        "x0_values": lambda: rng.uniform(*x0_range, size=n_biomarkers),
        "scaling_factors": lambda: rng.uniform(
            *scaling_factor_range, size=n_biomarkers
        ),
        "sigmas": lambda: np.broadcast_to(sigma_all, n_biomarkers).astype(float),
    }
    return {
        key: fixed[key] if key in fixed else sample()
        for key, sample in samplers.items()
    }


def _simulate_independent_sites(
    n_subjects_all: Iterable[int],
    n_max_timepoints_all: Iterable[int],
    t0_min_all: Iterable[float],
    t0_max_all: Iterable[float],
    sim_kwargs: dict,
    rng: np.random.Generator,
) -> list[SiteData]:
    sites = []
    n_subjects_so_far = 0
    for n_subjects, n_max_timepoints, t0_min, t0_max in zip(
        *_broadcast(n_subjects_all, n_max_timepoints_all, t0_min_all, t0_max_all),
        strict=True,
    ):
        timepoints, biomarkers, time_shifts, acceleration_factors = (
            simulate_all_subjects(
                n_subjects=n_subjects,
                max_n_timepoints=n_max_timepoints,
                t0_min=t0_min,
                t0_max=t0_max,
                rng=rng,
                **sim_kwargs,
            )
        )
        sites.append(
            SiteData(
                subject_ids=np.arange(n_subjects) + n_subjects_so_far,
                timepoints=timepoints,
                biomarkers=biomarkers,
                time_shifts=time_shifts,
                acceleration_factors=acceleration_factors,
            )
        )
        n_subjects_so_far += n_subjects
    return sites


def _simulate_pooled_sites(
    n_subjects: int,
    n_max_timepoints: int,
    t0_min: float,
    t0_max: float,
    windows_per_site: Sequence[Sequence[Window]],
    sim_kwargs: dict,
    rng: np.random.Generator,
) -> list[SiteData]:
    timepoints, biomarkers, time_shifts, acceleration_factors = simulate_all_subjects(
        n_subjects=n_subjects,
        max_n_timepoints=n_max_timepoints,
        t0_min=t0_min,
        t0_max=t0_max,
        rng=rng,
        **sim_kwargs,
    )
    # true t0 (= time shift if shift_time, else the first timepoint)
    t0 = (
        time_shifts
        if sim_kwargs["shift_time"]
        else np.array([tp[0] for tp in timepoints])
    )
    # use a related but separate RNG instance for the split to keep the original
    # RNG stream independent from the site windows
    site_indices = assign_sites_by_windows(t0, windows_per_site, rng=rng.spawn(1)[0])

    sites = []
    for i_site in range(len(windows_per_site)):
        idx = np.flatnonzero(site_indices == i_site)
        print(
            f"Site {i_site + 1}: {len(idx)} subjects, "
            f"t0 range [{t0[idx].min():.2f}, {t0[idx].max():.2f}]"
        )
        sites.append(
            SiteData(
                subject_ids=idx,
                timepoints=[timepoints[i] for i in idx],
                biomarkers=[biomarkers[i] for i in idx],
                time_shifts=time_shifts[idx],
                acceleration_factors=acceleration_factors[idx],
            )
        )
    return sites


def _build_df(
    site: SiteData,
    cols_biomarker: list[str],
    na_ratio: float = 0.0,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Build a long-format dataframe (one row per subject/timepoint) for a site."""
    n_timepoints = [len(bm) for bm in site.biomarkers]
    biomarkers = np.concatenate(site.biomarkers)

    # inject missing values
    if na_ratio > 0:
        if rng is None:
            rng = np.random.default_rng(None)
        n_missing = int(len(biomarkers) * na_ratio)
        for i_biomarker in range(biomarkers.shape[1]):
            missing_indices = rng.choice(len(biomarkers), size=n_missing, replace=False)
            biomarkers[missing_indices, i_biomarker] = np.nan

    return pd.DataFrame(
        data={
            # subject ID (unique across sites)
            COL_SUBJECT: np.repeat(site.subject_ids, n_timepoints),
            # index of the subject within the site (0 to n_subjects_site - 1)
            COL_SUBJECT_INDEX: np.repeat(np.arange(len(n_timepoints)), n_timepoints),
            COL_TIMEPOINT: np.concatenate(site.timepoints),
            **dict(zip(cols_biomarker, biomarkers.T, strict=True)),
        }
    )


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
    sites: Iterable[str] = (),
    rng_seed: int | None = None,
):
    """Simulate data for multiple sites.

    Two modes:
    - Independent sites (default): each site is simulated separately, with its own
      number of subjects, maximum number of timepoints and t0 range.
    - Pooled (``sites``, one spec per site): a single population is simulated and
      split into equally sized sites based on t0 windows (see ``_parse_site_spec``).
      Cross-sectional sites are listed in the output JSON; their data keep all
      visits.
    """
    sites = list(sites)
    settings = dict(locals())  # function arguments only

    site_specs = [_parse_site_spec(spec) for spec in sites]
    pooled = len(site_specs) > 0
    if pooled and any(
        len(arg) != 1
        for arg in (n_subjects_all, n_max_timepoints_all, t0_min_all, t0_max_all)
    ):
        raise ValueError(
            "--n-subjects, --n-max-timepoints, --t0-min, and --t0-max must have a "
            "single value when using --site."
        )

    if fpath_config is not None:
        print(f"Loading config from {fpath_config}")
        config = load_json(fpath_config)
    else:
        config = {}

    rng = np.random.default_rng(rng_seed)

    params = _get_biomarker_params(
        config,
        n_biomarkers,
        tuple(sigma_all),
        k_range=(k_min, k_max),
        x0_range=(x0_min, x0_max),
        scaling_factor_range=(scaling_factor_min, scaling_factor_max),
        rng=rng,
    )
    cols_biomarker = [
        f"{COL_BIOMARKER_PREFIX}{i}" for i in range(len(params["k_values"]))
    ]

    sim_kwargs = {
        **params,
        "n_timepoints_distribution": config.get("n_timepoints_distribution"),
        "time_at_timepoint": config.get("time_at_timepoint"),
        "shift_time": shift_time,
        "acceleration_factor_std": acceleration_factor_std,
    }
    if pooled:
        sites_data = _simulate_pooled_sites(
            n_subjects_all[0],
            n_max_timepoints_all[0],
            t0_min_all[0],
            t0_max_all[0],
            [spec.windows for spec in site_specs],
            sim_kwargs,
            rng,
        )
    else:
        sites_data = _simulate_independent_sites(
            n_subjects_all,
            n_max_timepoints_all,
            t0_min_all,
            t0_max_all,
            sim_kwargs,
            rng,
        )

    dpath_out = get_dpath_latest(dpath_data, use_today=True) / tag
    dpath_out.mkdir(parents=True, exist_ok=True)

    node_id_map = {}
    subjects_by_node = {}
    cross_sectional_nodes = []
    for i_site, site in enumerate(sites_data):
        node_id = f"{i_site + 1}"
        if pooled and site_specs[i_site].cross_sectional:
            cross_sectional_nodes.append(node_id)
        df_data = _build_df(site, cols_biomarker, na_ratio=na_ratio, rng=rng)
        fname_tsv = _get_fname_out(tag, i_site)
        fpath_tsv = dpath_out / fname_tsv
        df_data.to_csv(fpath_tsv, sep="\t", index=False)
        print(f"Saved simulated data to {fpath_tsv}")
        node_id_map[fname_tsv] = node_id
        subjects_by_node[node_id] = site.subject_ids.tolist()

    json_data = {
        "settings": settings,
        "params": {
            "time_shifts": [site.time_shifts for site in sites_data],
            "acceleration_factors": [site.acceleration_factors for site in sites_data],
            **params,
        },
        "node_id_map": node_id_map,
        "cols": {
            "col_subject": COL_SUBJECT,
            "col_subject_index": COL_SUBJECT_INDEX,
            "col_timepoint": COL_TIMEPOINT,
            "cols_biomarker": cols_biomarker,
        },
        "subjects_by_node": subjects_by_node,
        # training subjects at these nodes keep only their first visit after the split
        "cross_sectional_nodes": cross_sectional_nodes,
        "need_scaling": False,
    }
    if "biomarker_names" in config:
        # descriptive names for biomarker_0, biomarker_1, ...
        json_data["biomarker_names"] = config["biomarker_names"]

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
    "--site",
    "sites",
    type=str,
    multiple=True,
    help=(
        "One site of a single simulated population, repeated once per site. Format:"
        " LO-HI[,LO-HI...][:cross], with windows on the t0 rank scale from 0 to 1,"
        " e.g. '0-1/4,3/4-1' for early and late subjects. ':cross' keeps only the"
        " first visit of training subjects. Requires single values for"
        " --n-subjects, --t0-min and --t0-max."
    ),
)
def main(**params):
    simulate_data(**params)


if __name__ == "__main__":
    main()
