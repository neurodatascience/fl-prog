#!/usr/bin/env python
from pathlib import Path
from typing import Iterable, Optional

import click

from fl_prog.aggregator import SelectiveFedAverage
from fl_prog.model import LogisticRegressionModelWithShift
from fl_prog.training_plan import FLProgTrainingPlan
from fl_prog.utils.constants import CLICK_CONTEXT_SETTINGS, FNAME_WEIGHTS
from fl_prog.utils.io import (
    get_dpath_latest,
    save_json,
    working_directory,
    DEFAULT_DPATH_FEDBIOMED,
    DEFAULT_DPATH_RESULTS,
)

DEFAULT_NODE_MEGA = "NODE_CENTRALIZED"
DEFAULT_NODES_FEDERATED = ("NODE_1", "NODE_2", "NODE_3")

DEFAULT_N_BIOMARKERS = 5
DEFAULT_COL_SUBJECT_ID = "subject"
DEFAULT_COL_TIME = "timepoint"
DEFAULT_COLS_BIOMARKER = [f"biomarker_{i}" for i in range(DEFAULT_N_BIOMARKERS)]

DEFAULT_N_ROUNDS = 10
DEFAULT_N_UPDATES = 70
DEFAULT_BATCH_SIZE = 100000
DEFAULT_LEARNING_RATE = 0.01


def _get_model_args(
    tag: str,
    n_biomarkers: int = DEFAULT_N_BIOMARKERS,
    col_subject_id: str = DEFAULT_COL_SUBJECT_ID,
    col_time: str = DEFAULT_COL_TIME,
    cols_biomarker: Optional[Iterable[str]] = DEFAULT_COLS_BIOMARKER,
):
    return {
        "colnames": {
            "col_subject_id": col_subject_id,
            "col_time": col_time,
            "cols_biomarker": cols_biomarker,
        },
        "lr_with_shift": {
            "n_features": n_biomarkers,
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
@click.option("--node-mega", type=str, default=DEFAULT_NODE_MEGA)
@click.option(
    "--nodes-federated",
    type=str,
    multiple=True,
    default=DEFAULT_NODES_FEDERATED,
)
@click.option("--col-subject-id", type=str, default=DEFAULT_COL_SUBJECT_ID)
@click.option("--col-time", type=str, default=DEFAULT_COL_TIME)
@click.option(
    "--cols-biomarker",
    type=str,
    multiple=True,
    default=DEFAULT_COLS_BIOMARKER,
)
@click.option(
    "--n-biomarkers", type=click.IntRange(min=1), default=DEFAULT_N_BIOMARKERS
)
@click.option("--n-rounds", type=click.IntRange(min=1), default=DEFAULT_N_ROUNDS)
@click.option("--n-updates", type=click.IntRange(min=1), default=DEFAULT_N_UPDATES)
@click.option("--batch-size", type=click.IntRange(min=1), default=DEFAULT_BATCH_SIZE)
@click.option(
    "--learning-rate",
    type=click.FloatRange(min=0, min_open=True),
    default=DEFAULT_LEARNING_RATE,
)
def run_fedbiomed(
    tag: str,
    dpath_fbm: Path,
    dpath_results: Path,
    node_mega: str = DEFAULT_NODE_MEGA,
    nodes_federated: Iterable[str] = DEFAULT_NODES_FEDERATED,
    col_subject_id: str = DEFAULT_COL_SUBJECT_ID,
    col_time: str = DEFAULT_COL_TIME,
    cols_biomarker: str = DEFAULT_COLS_BIOMARKER,
    n_biomarkers: int = DEFAULT_N_BIOMARKERS,
    n_rounds: int = DEFAULT_N_ROUNDS,
    n_updates: int = DEFAULT_N_UPDATES,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
):
    tags = [tag]
    model_args = _get_model_args(
        tag, n_biomarkers, col_subject_id, col_time, cols_biomarker
    )
    training_args = _get_training_args(n_updates, batch_size, learning_rate)
    dpath_out = get_dpath_latest(dpath_results, use_today=True)

    json_data = {"settings": locals()}

    # centralized
    results_centralized = _run_experiment(
        dpath_fbm,
        nodes=[node_mega],
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
    fpath_out = dpath_out / f"{tag}-estimated_params.json"
    save_json(fpath_out, json_data)
    print(f"Saved results to {fpath_out}")


if __name__ == "__main__":
    run_fedbiomed()
