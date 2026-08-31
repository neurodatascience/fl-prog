#!/usr/bin/env python
import json
import tempfile
from pathlib import Path

import click
import numpy as np
import pandas as pd
from leaspy.io.data import Data
from leaspy.models import LogisticModel, ModelName

from fl_prog.utils.constants import CLICK_CONTEXT_SETTINGS
from fl_prog.utils.io import (
    DEFAULT_DPATH_DATA,
    DEFAULT_DPATH_RESULTS,
    format_df_for_leaspy,
    get_dpath_latest,
    load_json,
    save_json,
)

DEFAULT_N_ITER = 20000


def _results_exist(fpath_out: Path) -> bool:
    if not fpath_out.exists():
        return False

    json_results = load_json(fpath_out)
    return "results" in json_results


def _get_results(fitted_model: LogisticModel) -> dict:
    with tempfile.NamedTemporaryFile(
        mode="+wt", suffix=".json", delete=True
    ) as tmp_file:
        fitted_model.save(tmp_file.name)
        tmp_file.flush()
        return load_json(Path(tmp_file.name))


def run_leaspy(
    tag: str,
    dpath_data: Path,
    dpath_results: Path,
    n_iter: int = DEFAULT_N_ITER,
    random_seed: int | None = None,
    overwrite: bool = False,
):
    dpath_out = get_dpath_latest(dpath_results, use_today=True) / tag
    run_tag = "-".join(
        [
            "leaspy",
            str(n_iter),
            str(random_seed) if random_seed is not None else "no_seed",
        ]
    )
    fpath_out = dpath_out / f"{run_tag}-estimated_params.json"
    if _results_exist(fpath_out) and not overwrite:
        click.secho(
            f"{fpath_out} already exists. Use --overwrite to overwrite.",
            fg="red",
            bold=True,
        )
        return

    dpath_data = get_dpath_latest(dpath_data) / tag
    fpath_config = dpath_data / f"{tag}.json"
    try:
        config = json.loads(fpath_config.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        raise RuntimeError(f"Expected a JSON file at {fpath_config}")

    model_args = {
        "name": ModelName.LOGISTIC,
        "dimension": len(config["cols"]["cols_biomarker"]),
        "source_dimension": int(
            np.ceil(np.sqrt(len(config["cols"]["cols_biomarker"])))
        ),
        "obs_models": "gaussian-diagonal",
    }

    fit_args = {
        "algorithm": "mcmc_saem",
        "n_iter": n_iter,
        "seed": random_seed,
    }

    json_data = {"settings": locals()}

    fpath_merged = dpath_data / f"{tag}-merged.tsv"
    df_data = format_df_for_leaspy(
        df=pd.read_csv(fpath_merged, sep="\t"),
        col_subject=config["cols"]["col_subject"],
        col_timepoint=config["cols"]["col_timepoint"],
        cols_biomarker=config["cols"]["cols_biomarker"],
    )
    data = Data.from_dataframe(df_data, data_type="visit")

    model = LogisticModel(**model_args)
    model.fit(data, **fit_args)

    dpath_out.mkdir(parents=True, exist_ok=True)
    save_json(fpath_out, json_data)

    json_data["results"] = {}
    json_data["results"]["centralized"] = _get_results(model)
    save_json(fpath_out, json_data)

    print(f"Saved results to {fpath_out}")


@click.command(context_settings=CLICK_CONTEXT_SETTINGS)
@click.option("--tag", type=str, required=True)
@click.option(
    "--data-dir",
    "dpath_data",
    type=click.Path(path_type=Path, file_okay=False, dir_okay=True),
    default=DEFAULT_DPATH_DATA,
)
@click.option(
    "--results-dir",
    "dpath_results",
    type=click.Path(path_type=Path, file_okay=False, dir_okay=True),
    default=DEFAULT_DPATH_RESULTS,
)
@click.option("--n-iter", type=int, default=DEFAULT_N_ITER)
@click.option("--random-seed", type=int, envvar="RNG_SEED")
@click.option("--overwrite/--no-overwrite", default=False)
def main(**params):
    run_leaspy(**params)


if __name__ == "__main__":
    main()
