#!/usr/bin/env python
import json
import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd
import torch

from fl_prog.utils.constants import CLICK_CONTEXT_SETTINGS
from fl_prog.utils.io import (
    DEFAULT_DPATH_DATA,
    DEFAULT_DPATH_RESULTS,
    format_df_for_dpmost,
    get_dpath_latest,
    load_json,
    save_json,
)

PATH_TO_DPMOST = (
    Path(__file__).parent.parent.resolve() / "vendored" / "dpmost" / "models" / "dpmost"
)

sys.path.append(str(PATH_TO_DPMOST))
from DPMoSt import DPMoSt

DEFAULT_N_OUTER_ITER = 1000


def _results_exist(fpath_out_json: Path, fpath_model_out: Path) -> bool:
    if not fpath_out_json.exists():
        return False

    if fpath_model_out.exists():
        return True

    json_results = load_json(fpath_out_json)
    return "results" in json_results


def _get_results(model: DPMoSt, fpath_model: Path) -> dict:
    model.save(str(fpath_model.with_suffix("")))
    # model.est_theta: x0, k, scaling_factor
    sigmoid_params = np.vstack(model.est_theta)
    return {
        "estimated_k_values": sigmoid_params[:, 1],
        "estimated_x0_values": sigmoid_params[:, 0],
        "estimated_scaling_factors": sigmoid_params[:, 2],
        "estimated_sigma": np.vstack(model.est_noise).squeeze(),
        "estimated_time_shifts": {
            "node_centralized": model.time_shift.detach().numpy(),
        },
        "estimated_acceleration_factors": {
            "node_centralized": np.ones(model.n_subjects),
        },
    }


def run_dpmost(
    tag: str,
    dpath_data: Path,
    dpath_results: Path,
    n_outer_iter: int = DEFAULT_N_OUTER_ITER,
    random_seed: int | None = None,
    overwrite: bool = False,
):
    if random_seed is not None:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    dpath_out = get_dpath_latest(dpath_results, use_today=True) / tag
    run_tag = "-".join(
        [
            "dpmost",
            str(n_outer_iter),
            str(random_seed) if random_seed is not None else "no_seed",
        ]
    )
    fpath_out_json = dpath_out / f"{run_tag}-estimated_params.json"
    fpath_out_model = dpath_out / f"{run_tag}-model.pkl"
    if _results_exist(fpath_out_json, fpath_out_model) and not overwrite:
        click.secho(
            f"{fpath_out_json} already exists. Use --overwrite to overwrite.",
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
        "n_clusters": 1,
        "verbose": True,
        "do_normalisation": False,
        "time_shift_eval": True,
        "noise_std_eval": True,
        "theta_eval": True,
        "xi_eval": True,
        "pi_eval": True,
    }

    optimize_args = {
        "n_outer_iterations": n_outer_iter,
        "n_final_iterations": 50,
        "n_inner_iterations_time_shift": 30,
        "n_inner_iterations_theta": 30,
        "n_inner_iterations_noise": 30,
        "lr_theta": 1e-1,
        "lr_noise": 1e-1,
        "lr_time_shift": 1e-2,
        "stopping_criteria": True,
        "threshold": 1e-3,
    }

    json_data = {"settings": locals()}

    fpath_merged = dpath_data / f"{tag}-merged.tsv"
    df_data = format_df_for_dpmost(
        df=pd.read_csv(fpath_merged, sep="\t"),
        col_subject=config["cols"]["col_subject"],
        col_timepoint=config["cols"]["col_timepoint"],
        cols_biomarker=config["cols"]["cols_biomarker"],
    )

    model = DPMoSt(data=df_data, **model_args)
    model.optimise(**optimize_args)

    dpath_out.mkdir(parents=True, exist_ok=True)
    save_json(fpath_out_json, json_data)

    json_data["results"] = {}
    json_data["results"]["centralized"] = _get_results(model, fpath_out_model)
    save_json(fpath_out_json, json_data)

    print(f"Saved results to {fpath_out_json}")


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
@click.option("--n-outer-iter", type=int, default=DEFAULT_N_OUTER_ITER)
@click.option("--random-seed", type=int, envvar="RNG_SEED")
@click.option("--overwrite/--no-overwrite", default=False)
def main(**params):
    run_dpmost(**params)


if __name__ == "__main__":
    main()
