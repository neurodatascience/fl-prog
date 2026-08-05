#!/usr/bin/env python

from pathlib import Path

import click
import pandas as pd

from fl_prog.utils.constants import CLICK_CONTEXT_SETTINGS
from fl_prog.utils.io import (
    DEFAULT_DPATH_RESULTS,
    get_dpath_latest,
    load_json,
)


def _get_fpath_json_results(tag: str, dpath_results: Path) -> Path:
    dpath_results_latest = get_dpath_latest(dpath_results)
    return dpath_results_latest / tag / f"{tag}-estimated_params.json"


def _get_dpath_data(fpath_json_results: Path) -> Path:
    json_results = load_json(fpath_json_results)
    return Path(json_results["settings"]["dpath_data"])


def _save_tsv(df_metrics: pd.DataFrame, fpath_out: Path):
    fpath_out.parent.mkdir(parents=True, exist_ok=True)
    df_metrics.to_csv(fpath_out, sep="\t", index=False)
    print(f"Saved metrics to {fpath_out}")


def assess_fit(
    tag: str,
    dpath_results: Path,
):
    fpath_json_results = _get_fpath_json_results(tag, dpath_results)
    print(f"fpath_json_results: {fpath_json_results}")

    dpath_data = _get_dpath_data(fpath_json_results)
    print(f"dpath_data: {dpath_data}")

    df_metrics = pd.DataFrame(
        columns=["setup", "set_name", "col_biomarker", "metric", "value"]
    )

    fpath_out = fpath_json_results.with_name(f"{tag}-fit_quality.tsv")
    _save_tsv(df_metrics, fpath_out)

    return df_metrics


@click.command(context_settings=CLICK_CONTEXT_SETTINGS)
@click.option("--tag", type=str, required=True)
@click.option(
    "--results-dir",
    "dpath_results",
    type=click.Path(path_type=Path, file_okay=False, dir_okay=True),
    default=DEFAULT_DPATH_RESULTS,
)
def main(**params):
    assess_fit(**params)


if __name__ == "__main__":
    main()
