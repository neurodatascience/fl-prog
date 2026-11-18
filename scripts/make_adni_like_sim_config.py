#!/usr/bin/env python
"""Create an ADNI-like simulation config for ``simulate_data.py``."""

import re
from collections import Counter, defaultdict
from pathlib import Path

import click
import numpy as np
import pandas as pd

from fl_prog.utils.constants import CLICK_CONTEXT_SETTINGS
from fl_prog.utils.io import load_json, save_json

# keys in estimated params JSON -> keys in simulation config
PARAM_KEYS = {
    "estimated_k_values": "k_values",
    "estimated_x0_values": "x0_values",
    "estimated_scaling_factors": "scaling_factors",
    "estimated_sigma": "sigmas",
}
PATTERN_HEMISPHERE = re.compile(r"^(Left-|Right-|lh_|rh_)")


def _get_visit_structure(
    dpath_adni: Path, decimals: int = 4
) -> tuple[dict[str, int], list[float]]:
    tag = dpath_adni.name
    json_data = load_json(dpath_adni / f"{tag}.json")
    col_subject = json_data["cols"]["col_subject"]
    col_timepoint = json_data["cols"]["col_timepoint"]

    fpaths = [dpath_adni / f"{tag}-merged.tsv"]
    fpath_test = dpath_adni / f"{tag}-test.tsv"
    if fpath_test.exists():
        fpaths.append(fpath_test)
    df = pd.concat(
        [pd.read_csv(fpath, sep="\t", dtype={col_subject: str}) for fpath in fpaths]
    )
    if df.duplicated([col_subject, col_timepoint]).any():
        raise ValueError(f"Duplicate subject/timepoint rows in {fpaths}")
    print(f"Visit structure from {df[col_subject].nunique()} subjects in {fpaths}")

    n_timepoints = df.groupby(col_subject).size()
    n_timepoints_distribution = {
        str(n): int(count) for n, count in sorted(Counter(n_timepoints).items())
    }
    # visit schedule: most frequent visit times (since the first visit), one for
    # each possible timepoint
    time_since_first = (
        df[col_timepoint] - df.groupby(col_subject)[col_timepoint].transform("min")
    ).round(decimals)
    time_at_timepoint = sorted(
        time_since_first.value_counts().index[: n_timepoints.max()].tolist()
    )
    return n_timepoints_distribution, time_at_timepoint


def _get_biomarker_params(
    fpaths_params: list[Path], setup: str, cols_biomarker: list[str]
) -> tuple[list[str], dict[str, list[float]]]:
    values = defaultdict(list)
    for fpath in fpaths_params:
        results = load_json(fpath)["results"][setup]
        for key in PARAM_KEYS:
            values[key].append(results[key])
    means = {key: np.mean(vals, axis=0) for key, vals in values.items()}
    sds = {key: np.std(vals, axis=0) for key, vals in values.items()}
    for key in PARAM_KEYS:
        if len(means[key]) != len(cols_biomarker):
            raise ValueError(
                f"{key} has {len(means[key])} values but there are "
                f"{len(cols_biomarker)} biomarkers"
            )
        print(f"{key}: max SD across {len(fpaths_params)} runs: {sds[key].max():.3f}")

    # collapse left/right biomarkers (keep order of first occurrence)
    groups = defaultdict(list)
    for i, col in enumerate(cols_biomarker):
        groups[PATTERN_HEMISPHERE.sub("", col)].append(i)
    biomarker_names = list(groups)
    params = {
        key_out: [float(np.mean(means[key_in][idx])) for idx in groups.values()]
        for key_in, key_out in PARAM_KEYS.items()
    }
    return biomarker_names, params


@click.command(context_settings=CLICK_CONTEXT_SETTINGS)
@click.option(
    "--adni-data-dir",
    "dpath_adni",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    required=True,
    help="ADNI data directory (named after its tag) with merged and test TSVs",
)
@click.argument(
    "fpaths_params",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    nargs=-1,
    required=True,
)
@click.option("--setup", type=str, default="centralized", show_default=True)
@click.option(
    "--output",
    "fpath_out",
    type=click.Path(path_type=Path, dir_okay=False),
    required=True,
)
def make_adni_like_sim_config(
    dpath_adni: Path, fpaths_params: tuple[Path], setup: str, fpath_out: Path
):
    """Create config from ADNI data and estimated params JSON(s) FPATHS_PARAMS.

    FPATHS_PARAMS are ``*-estimated_params.json`` files written by
    ``run_fedbiomed.py`` for runs on ADNI data.
    """
    tag = dpath_adni.name
    cols_biomarker = load_json(dpath_adni / f"{tag}.json")["cols"]["cols_biomarker"]
    n_timepoints_distribution, time_at_timepoint = _get_visit_structure(dpath_adni)
    biomarker_names, params = _get_biomarker_params(
        list(fpaths_params), setup, cols_biomarker
    )

    config = {
        "description": (
            "ADNI-like simulation config. Visit structure from ADNI data; biomarker "
            f"parameters averaged over {len(fpaths_params)} {setup} model fits on ADNI"
            " (left/right biomarkers averaged)."
        ),
        "dpath_adni": dpath_adni,
        "fpaths_params": fpaths_params,
        "n_timepoints_distribution": n_timepoints_distribution,
        "time_at_timepoint": time_at_timepoint,
        "biomarker_names": biomarker_names,
        **params,
    }
    fpath_out.parent.mkdir(parents=True, exist_ok=True)
    save_json(fpath_out, config)
    print(f"Saved config to {fpath_out}")
    df = pd.DataFrame(params, index=biomarker_names).round(3)
    print(df.to_string())
    print(f"n_timepoints_distribution: {n_timepoints_distribution}")
    print(f"time_at_timepoint: {time_at_timepoint}")


if __name__ == "__main__":
    make_adni_like_sim_config()
