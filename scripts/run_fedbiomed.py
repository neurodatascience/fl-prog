#!/usr/bin/env python
import json
import shutil
from pathlib import Path
from collections.abc import Iterable

import click
import numpy as np
import pandas as pd
from declearn.optimizer.modules import ScaffoldServerModule
from fedbiomed.common.optimizers.optimizer import Optimizer
from fedbiomed.common.serializer import Serializer

from fedbiomed.researcher.aggregators.fedavg import FedAverage
from fedbiomed.researcher.aggregators.scaffold import Scaffold

from fl_prog.model import LogisticRegressionModelWithShift
from fl_prog.training_plan import FLProgTrainingPlan
from fl_prog.utils.constants import (
    CLICK_CONTEXT_SETTINGS,
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
DEFAULT_N_UPDATES = 100
DEFAULT_BATCH_SIZE = 100000  # all data (no batching)
DEFAULT_LEARNING_RATE = 0.1
DEFAULT_LAMBDA = 1
DEFAULT_EXPECTED_TIME_SHIFT_RANGE = (0.0, 0.0)
DEFAULT_AGGREGATOR_NAME = "fedavg"

VALID_AGGREGATOR_NAMES = ["fedavg", "fedprox", "scaffold"]
DNAME_TENSORBOARD = "tensorboard"


def _get_n_participants_map(
    dpath_data: Path, node_id_map: dict, col_subject_id: str
) -> dict[str, int]:
    n_participants_map = {}
    for fpath_tsv, node_id in node_id_map.items():
        df = pd.read_csv(dpath_data / fpath_tsv, sep="\t")
        if col_subject_id not in df.columns:
            raise ValueError(
                f"Subject column '{col_subject_id}' not found in {fpath_tsv}"
            )
        n_participants_map[f"{NODE_PREFIX}{node_id}"] = df[col_subject_id].nunique()
    return n_participants_map


def _get_model_args(
    dpath_data: Path,
    col_subject_id: str,
    col_time: str,
    cols_biomarker: Iterable[str],
    node_id_map: dict[str, str],
    lambda_: float = DEFAULT_LAMBDA,
    estimated_time_shift_range: Iterable[float] = DEFAULT_EXPECTED_TIME_SHIFT_RANGE,
):
    return {
        "colnames": {
            "col_subject_id": col_subject_id,
            "col_time": col_time,
            "cols_biomarker": cols_biomarker,
        },
        "lr_with_shift": {
            "n_features": len(cols_biomarker),
            "lambda_": lambda_,
            "expected_time_shift_range": estimated_time_shift_range,
        },
        "node_specific_args": {
            "n_participants": _get_n_participants_map(
                dpath_data, node_id_map, col_subject_id
            )
        },
    }


def _get_training_args(
    n_updates: int = DEFAULT_N_UPDATES,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    aggregator_name: str = DEFAULT_AGGREGATOR_NAME,
    random_seed: int | None = None,
):
    training_args = {
        "num_updates": n_updates,
        "loader_args": {"batch_size": batch_size, "shuffle": False},
        "optimizer_args": {"lr": learning_rate, "aggregator_name": aggregator_name},
    }

    if random_seed is not None:
        training_args["random_seed"] = random_seed

    if aggregator_name == "fedprox":
        training_args["fedprox_mu"] = 1.0
    return training_args


def _get_aggregator(aggregator_name: str):
    return {
        "fedavg": FedAverage(),
        "fedprox": FedAverage(),  # FedProx is FedAvg with additional training arg
        "scaffold": Scaffold(server_lr=1),
    }[aggregator_name]


def _get_agg_optimizer(aggregator_name: str):
    if aggregator_name == "scaffold":
        return Optimizer(lr=0.8, modules=[ScaffoldServerModule()])
    return None


def _run_experiment(
    dpath_fbm: Path,
    nodes: Iterable[str],
    tags: Iterable[str],
    model_args: dict,
    training_args: dict,
    n_rounds: int = DEFAULT_N_ROUNDS,
    aggregator_name: str = DEFAULT_AGGREGATOR_NAME,
    with_tensorboard: bool = False,
    dpath_tensorboard: Path | None = None,
    save_training_replies: bool = False,
    save_all_aggregated_params: bool = False,
):

    training_args = training_args.copy()

    with working_directory(dpath_fbm):

        from fedbiomed.researcher.federated_workflows import Experiment

        experiment = Experiment(
            nodes=list(nodes),
            tags=tags,
            training_plan_class=FLProgTrainingPlan,
            model_args=model_args,
            round_limit=n_rounds,
            training_args=training_args,
            aggregator=_get_aggregator(aggregator_name),
            agg_optimizer=_get_agg_optimizer(aggregator_name),
            node_selection_strategy=None,
            tensorboard=with_tensorboard,
        )
        for _ in range(n_rounds):
            if "random_seed" in training_args:
                training_args["random_seed"] += 1
            experiment.set_training_args(training_args)
            experiment.run_once()

    fbm_model: LogisticRegressionModelWithShift = experiment.training_plan().model()
    final_params = experiment.aggregated_params()[experiment.round_limit() - 1][
        "params"
    ]

    # get node-side time shift values
    time_shifts = {}
    for node in nodes:
        node_state_file = experiment._node_state_agent.get_last_node_states()[node]

        fpath_persistent_params = (
            dpath_fbm
            / node.replace("_", "-")
            / "var"
            / f"node_state_{node}"
            / f"experiment_id_{experiment.id}"
            / f"persistent_model_weights_{experiment.round_current() - 1}_{node_state_file}"
        )
        print(f"Loading time shifts for node {node} from {fpath_persistent_params}")

        final_local_persistent = Serializer.load(fpath_persistent_params)
        time_shifts[node] = final_local_persistent["time_shifts"].data.numpy()

    if "vertical_shifts" in final_params:
        vertical_shifts = final_params["vertical_shifts"].data.numpy()
    else:
        vertical_shifts = np.zeros_like(final_params["x0_values"].data.numpy())
    if "parametrizations.scaling_factors.original" in final_params:
        scaling_factors = fbm_model.get_scaling_factors(
            final_params["parametrizations.scaling_factors.original"]
        ).data.numpy()
    else:
        scaling_factors = np.ones_like(final_params["x0_values"].data.numpy())

    results = {
        "estimated_k_values": fbm_model.get_k_values(
            final_params["parametrizations.k_values.original"]
        ).data.numpy(),
        "estimated_x0_values": final_params["x0_values"].data.numpy(),
        "estimated_vertical_shifts": vertical_shifts,
        "estimated_scaling_factors": scaling_factors,
        "estimated_sigma": fbm_model.get_sigma(
            final_params["parametrizations.sigma.original"]
        ).data.numpy(),
        "estimated_time_shifts": time_shifts,
    }

    if save_training_replies:
        results["training_replies"] = experiment.training_replies()

    if save_all_aggregated_params:
        results["aggregated_params"] = experiment.aggregated_params()

    # move loss data (overwriting if needed)
    if with_tensorboard:
        dpath_tensorboard.mkdir(exist_ok=True)
        for dpath_node_src in Path(experiment.tensorboard_results_path).glob("*"):
            dpath_node_dest = dpath_tensorboard / dpath_node_src.name
            if dpath_node_dest.exists():
                shutil.rmtree(dpath_node_dest)
            shutil.move(dpath_node_src, dpath_node_dest)

    return results


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
@click.option(
    "--lambda",
    "lambda_",
    type=click.FloatRange(min=0),
    default=DEFAULT_LAMBDA,
    help="Regularization strength for time shifts",
)
@click.option(
    "--time-shift-range",
    "estimated_time_shift_range",
    type=click.Tuple((float, float)),
    nargs=2,
    default=DEFAULT_EXPECTED_TIME_SHIFT_RANGE,
    help="Expected range of time shifts (for initialization and regularization)",
)
@click.option(
    "--aggregator",
    "aggregator_name",
    type=click.Choice(VALID_AGGREGATOR_NAMES),
    default=DEFAULT_AGGREGATOR_NAME,
)
@click.option("--tensorboard/--no-tensorboard", "with_tensorboard", default=True)
@click.option(
    "--training-replies/--no-training-replies", "save_training_replies", default=False
)
@click.option(
    "--aggregated-params/--no-aggregated-params",
    "save_all_aggregated_params",
    default=False,
)
@click.option("--random-seed", type=int, envvar="RNG_SEED")
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
    lambda_: float = DEFAULT_LAMBDA,
    estimated_time_shift_range: tuple[float, float] = DEFAULT_EXPECTED_TIME_SHIFT_RANGE,
    aggregator_name: str = DEFAULT_AGGREGATOR_NAME,
    with_tensorboard: bool = False,
    save_training_replies: bool = False,
    save_all_aggregated_params: bool = False,
    random_seed: int | None = None,
    overwrite: bool = False,
):
    dpath_out = get_dpath_latest(dpath_results, use_today=True) / tag
    fpath_out = dpath_out / f"{tag}-estimated_params.json"
    if fpath_out.exists() and not overwrite:
        print(f"{fpath_out} already exists. Use --overwrite to overwrite.")
        return

    dpath_data = get_dpath_latest(dpath_data) / tag
    fpath_config = dpath_data / f"{tag}.json"
    try:
        config = json.loads(fpath_config.read_text())
    except Exception:
        raise RuntimeError(f"Expected a JSON file at {fpath_config}")

    node_id_map = get_node_id_map(fpath_config)

    try:
        model_args = _get_model_args(
            dpath_data,
            config["cols"]["col_subject_index"],
            config["cols"]["col_timepoint"],
            config["cols"]["cols_biomarker"],
            node_id_map,
            lambda_=lambda_,
            estimated_time_shift_range=estimated_time_shift_range,
        )
    except KeyError:
        raise RuntimeError(
            f'{fpath_config} should have a "cols" entry with keys: '
            '"col_subject" (str), "col_timepoint" (str), "cols_biomarker" (list[str])'
        )

    training_args = _get_training_args(
        n_updates, batch_size, learning_rate, aggregator_name, random_seed
    )

    nodes_federated = sorted(
        [
            f"{NODE_PREFIX}{node_id}"
            for node_id in set(node_id_map.values()) - {node_id_centralized}
        ]
    )

    if with_tensorboard:
        dpath_tensorboard = dpath_out / DNAME_TENSORBOARD

    json_data = {"settings": locals()}

    tags = [tag]

    dpath_out.mkdir(parents=True, exist_ok=True)
    save_json(fpath_out, json_data)

    json_data["results"] = {}

    # federated
    results_federated = _run_experiment(
        dpath_fbm,
        nodes=nodes_federated,
        tags=tags,
        model_args=model_args,
        training_args=training_args,
        n_rounds=n_rounds,
        aggregator_name=aggregator_name,
        with_tensorboard=with_tensorboard,
        dpath_tensorboard=dpath_tensorboard,
        save_training_replies=save_training_replies,
        save_all_aggregated_params=save_all_aggregated_params,
    )

    json_data["results"]["federated"] = results_federated
    save_json(fpath_out, json_data)

    # centralized
    results_centralized = _run_experiment(
        dpath_fbm,
        nodes=[f"{NODE_PREFIX}{node_id_centralized}"],
        tags=tags,
        model_args=model_args,
        training_args=training_args,
        n_rounds=n_rounds,
        aggregator_name=aggregator_name,
        with_tensorboard=with_tensorboard,
        dpath_tensorboard=dpath_tensorboard,
        save_training_replies=save_training_replies,
        save_all_aggregated_params=save_all_aggregated_params,
    )

    json_data["results"]["centralized"] = results_centralized
    save_json(fpath_out, json_data)
    print(f"Saved results to {fpath_out}")


if __name__ == "__main__":
    run_fedbiomed()
