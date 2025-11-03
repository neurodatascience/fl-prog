#!/usr/bin/env python
import json
from pathlib import Path
from typing import Iterable, Optional

import click

from fl_prog.aggregator import SelectiveFedAverage
from fl_prog.model import LogisticRegressionModelWithShift
from fl_prog.training_plan import FLProgTrainingPlan
from fl_prog.utils.constants import (
    CLICK_CONTEXT_SETTINGS,
    FNAME_WEIGHTS,
    NODE_ID_CENTRALIZED,
    NODE_PREFIX,
)
from fl_prog.utils.io import (
    get_dpath_latest,
    get_node_id_map,
    save_json,
    working_directory,
    DEFAULT_DPATH_DATA,
    DEFAULT_DPATH_FEDBIOMED,
    DEFAULT_DPATH_RESULTS,
)

DEFAULT_N_ROUNDS = 10
DEFAULT_N_UPDATES = 70
DEFAULT_BATCH_SIZE = 100000
DEFAULT_LEARNING_RATE = 0.01


def _get_model_args(
    tag: str,
    col_subject_id: str,
    col_time: str,
    cols_biomarker: Optional[Iterable[str]],
):
    return {
        "colnames": {
            "col_subject_id": col_subject_id,
            "col_time": col_time,
            "cols_biomarker": cols_biomarker,
        },
        "lr_with_shift": {
            "n_features": len(cols_biomarker),
        },
        "tag": tag,
    }


def _get_training_args(
    n_updates: int = DEFAULT_N_UPDATES,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
):
    return {
        "num_updates": n_updates,
        "loader_args": {"batch_size": batch_size, "shuffle": False},
        "optimizer_args": {"lr": learning_rate},
    }


def _run_experiment(
    dpath_fbm,
    nodes: Iterable[str],
    tags: Iterable[str],
    model_args: dict,
    n_rounds: int,
    training_args: dict,
):
    for fpath_weights in Path(dpath_fbm).rglob(f"{FNAME_WEIGHTS}"):
        fpath_weights.unlink()
        print(f"Removed {fpath_weights}")

    with working_directory(dpath_fbm):

        from fedbiomed.researcher.federated_workflows import Experiment

        experiment = Experiment(
            nodes=list(nodes),
            tags=tags,
            training_plan_class=FLProgTrainingPlan,
            model_args=model_args,
            round_limit=n_rounds,
            training_args=training_args,
            aggregator=SelectiveFedAverage(["time_shifts"]),
            node_selection_strategy=None,
        )
        experiment.run()

    fbm_model: LogisticRegressionModelWithShift = experiment.training_plan().model()
    final_params = experiment.aggregated_params()[experiment.round_limit() - 1][
        "params"
    ]

    return {
        "estimated_k_values": fbm_model.get_k_values(
            final_params["log_k_values"]
        ).data.numpy(),
        "estimated_x0_values": final_params["x0_values"].data.numpy(),
        "estimated_time_shifts": final_params["time_shifts"].data.numpy(),
        "estimated_sigma": fbm_model.get_sigma(
            final_params["log_sigma_sq"]
        ).data.numpy(),
    }


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
@click.option(
    "--fedbiomed-dir",
    "dpath_fbm",
    type=click.Path(path_type=Path, exists=True, file_okay=False, dir_okay=True),
    default=DEFAULT_DPATH_FEDBIOMED,
)
@click.option(
    "--node-centralized", "node_id_centralized", type=str, default=NODE_ID_CENTRALIZED
)
@click.option("--n-rounds", type=click.IntRange(min=1), default=DEFAULT_N_ROUNDS)
@click.option("--n-updates", type=click.IntRange(min=1), default=DEFAULT_N_UPDATES)
@click.option("--batch-size", type=click.IntRange(min=1), default=DEFAULT_BATCH_SIZE)
@click.option(
    "--learning-rate",
    type=click.FloatRange(min=0, min_open=True),
    default=DEFAULT_LEARNING_RATE,
)
@click.option("--overwrite/--no-overwrite", default=False)
def run_fedbiomed(
    tag: str,
    dpath_fbm: Path,
    dpath_data: Path,
    dpath_results: Path,
    node_id_centralized: str = NODE_ID_CENTRALIZED,
    n_rounds: int = DEFAULT_N_ROUNDS,
    n_updates: int = DEFAULT_N_UPDATES,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    overwrite: bool = False,
):
    dpath_out = get_dpath_latest(dpath_results, use_today=True) / tag
    fpath_out = dpath_out / f"{tag}-estimated_params.json"
    if fpath_out.exists() and not overwrite:
        print(f"{fpath_out} already exists. Use --overwrite to overwrite.")
        return

    fpath_config = get_dpath_latest(dpath_data) / tag / f"{tag}.json"
    try:
        config = json.loads(fpath_config.read_text())
    except Exception:
        raise RuntimeError(f"Expected a JSON file at {fpath_config}")
    try:
        cols = config["cols"]
        col_subject_id = cols["col_subject"]
        col_time = cols["col_timepoint"]
        cols_biomarker = cols["cols_biomarker"]
    except KeyError:
        raise RuntimeError(
            f'{fpath_config} should have a "cols" entry with keys: '
            '"col_subject" (str), "col_timepoint" (str), "cols_biomarker" (list[str])'
        )

    tags = [tag]
    model_args = _get_model_args(tag, col_subject_id, col_time, cols_biomarker)
    training_args = _get_training_args(n_updates, batch_size, learning_rate)

    node_id_map = get_node_id_map(fpath_config)
    nodes_federated = sorted(
        [
            f"{NODE_PREFIX}{node_id}"
            for node_id in set(node_id_map.values()) - {node_id_centralized}
        ]
    )

    json_data = {"settings": locals()}

    # centralized
    results_centralized = _run_experiment(
        dpath_fbm,
        nodes=[f"{NODE_PREFIX}{node_id_centralized}"],
        tags=tags,
        model_args=model_args,
        n_rounds=n_rounds,
        training_args=training_args,
    )

    # federated
    results_federated = _run_experiment(
        dpath_fbm,
        nodes=nodes_federated,
        tags=tags,
        model_args=model_args,
        n_rounds=n_rounds,
        training_args=training_args,
    )

    json_data["results"] = {
        "centralized": results_centralized,
        "federated": results_federated,
    }

    dpath_out.mkdir(parents=True, exist_ok=True)
    save_json(fpath_out, json_data)
    print(f"Saved results to {fpath_out}")


if __name__ == "__main__":
    run_fedbiomed()
