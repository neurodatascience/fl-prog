#!/usr/bin/env python

from pathlib import Path
from typing import Any

import click
import numpy as np
import pandas as pd
import torch

from fl_prog.metrics import (
    PER_SUBJECT_FIELDS,
    compute_per_subject_recovery,
    compute_predictive_metrics,
    compute_recovery_metrics,
)
from fl_prog.model import LogisticRegressionModelWithShift, ModelParams
from fl_prog.utils.constants import CLICK_CONTEXT_SETTINGS, NODE_PREFIX
from fl_prog.utils.io import (
    DEFAULT_DPATH_RESULTS,
    get_dpath_latest,
    load_json,
)

TRUE_FEATURE_KEYS = {
    "k_values": "k_values",
    "x0_values": "x0_values",
    "scaling_factors": "scaling_factors",
    "sigma": "sigmas",
}

ESTIMATED_FEATURE_KEYS = {
    "k_values": "estimated_k_values",
    "x0_values": "estimated_x0_values",
    "scaling_factors": "estimated_scaling_factors",
    "sigma": "estimated_sigma",
}

CENTRALIZED_NODE = "centralized"


def _build_model_params(
    feature_arrays: dict[str, np.ndarray],
    per_node: dict[str, dict[str, list[float]]],
    subjects_by_node: dict[str, list[str]],
) -> ModelParams:
    """Assemble a ModelParams from per-feature arrays and per-node lists.

    ``per_node`` maps node ids (without the ``node_`` prefix) to dicts with
    ``time_shifts`` and ``acceleration_factors`` lists, each aligned to
    ``subjects_by_node[node_id]``. Nodes must be given in merged-data subject
    order (site order); their lists are validated and concatenated in that
    order.
    """
    time_shifts = []
    acceleration_factors = []
    for node_id, node_params in per_node.items():
        subjects = subjects_by_node[node_id]
        for param_name in PER_SUBJECT_FIELDS:
            if len(node_params[param_name]) != len(subjects):
                raise ValueError(
                    f"{param_name} for node {node_id} have length "
                    f"{len(node_params[param_name])} but {len(subjects)} subjects"
                )
        time_shifts.append(np.asarray(node_params["time_shifts"], dtype=float))
        acceleration_factors.append(
            np.asarray(node_params["acceleration_factors"], dtype=float)
        )

    return ModelParams(
        **feature_arrays,
        time_shifts=np.concatenate(time_shifts),
        acceleration_factors=np.concatenate(acceleration_factors),
    )


def _true_params_by_subject(
    subjects_by_node: dict[str, list[str]], params: dict[str, Any]
) -> ModelParams:
    """Align true simulation parameters to the canonical subject order.

    True per-subject parameters are stored as lists per site in file order.
    Site ``i`` (0-indexed) corresponds to node ``str(i + 1)``, whose subjects
    are listed in ``subjects_by_node`` in the same order as in the site TSV.
    """
    feature_arrays = {
        field: np.asarray(params[source_key], dtype=float)
        for field, source_key in TRUE_FEATURE_KEYS.items()
    }

    per_node = {
        str(i_site + 1): {
            "time_shifts": time_shifts_site,
            "acceleration_factors": acc_site,
        }
        for i_site, (time_shifts_site, acc_site) in enumerate(
            zip(params["time_shifts"], params["acceleration_factors"], strict=True)
        )
    }

    return _build_model_params(feature_arrays, per_node, subjects_by_node)


def _estimated_params_by_setup(
    json_results: dict, subjects_by_node: dict[str, list[str]]
) -> dict[str, ModelParams]:
    """Align estimated parameters to the canonical subject order, per setup.

    Estimated per-subject parameters are keyed by node (``node_1``, ...,
    ``node_centralized``); subjects are listed in ``subjects_by_node`` in the
    same order as in the estimated arrays. Node keys are processed in JSON
    order, which matches the merged data's subject order.
    """
    params_by_setup: dict[str, ModelParams] = {}
    for setup, result in json_results["results"].items():
        feature_arrays = {
            field: np.asarray(result[source_key], dtype=float)
            for field, source_key in ESTIMATED_FEATURE_KEYS.items()
        }

        estimated_time_shifts = result["estimated_time_shifts"]
        estimated_acceleration_factors = result["estimated_acceleration_factors"]
        per_node = {
            node_id.removeprefix(NODE_PREFIX): {
                "time_shifts": shifts,
                "acceleration_factors": estimated_acceleration_factors[node_id],
            }
            for node_id, shifts in estimated_time_shifts.items()
        }

        params_by_setup[setup] = _build_model_params(
            feature_arrays, per_node, subjects_by_node
        )
    return params_by_setup


def _load_df_data(
    dpath_data: Path, tag: str, cols: dict, test: bool = False
) -> pd.DataFrame:
    if test:
        suffix = "-test"
    else:
        suffix = "-merged"
    fpath_data = dpath_data / f"{tag}{suffix}.tsv"
    return pd.read_csv(fpath_data, sep="\t", dtype={cols["col_subject"]: str})


def _predict(params: ModelParams, t: np.ndarray, subject_ids: np.ndarray) -> np.ndarray:
    """Model forward pass.

    Returns predictions of shape (n_entries, n_features)
    """
    model = LogisticRegressionModelWithShift(
        n_participants=len(params.time_shifts),
        n_features=len(params.k_values),
    )

    t = torch.tensor(t, dtype=torch.float)
    subject_ids = torch.tensor(subject_ids, dtype=torch.long)

    return model.forward(t, subject_ids, params=params).detach().numpy()


def _save_tsv(df_metrics: pd.DataFrame, fpath_out: Path):
    fpath_out.parent.mkdir(parents=True, exist_ok=True)
    df_metrics.to_csv(fpath_out, sep="\t", index=False)
    print(f"Saved metrics to {fpath_out}")


def get_metrics_single_run(fpath_json_results: Path, tag: str):
    json_results = load_json(fpath_json_results)

    dpath_data = Path(json_results["settings"]["dpath_data"])
    print(f"dpath_data: {dpath_data}")

    json_data = load_json(Path(json_results["settings"]["fpath_config"]))

    cols = json_data["cols"]
    subjects_by_node = json_data["subjects_by_node"]
    try:
        true_params = json_data["params"]
    except KeyError:
        true_params = None
        click.secho(
            f"True parameters not found in {json_results['settings']['fpath_config']}."
            " Skipping recovery metrics.",
        )

    df_data_train = _load_df_data(dpath_data, tag, cols, test=False)
    df_data_test: pd.DataFrame | None = _load_df_data(dpath_data, tag, cols, test=True)

    # same for train/test data
    col_subject = cols["col_subject"]
    n_biomarkers = len(cols["cols_biomarker"])
    n_subjects = df_data_train[col_subject].nunique()
    print(f"n_biomarkers: {n_biomarkers}")
    print(f"n_subjects: {n_subjects}")
    print(
        "n_subjects_by_node: "
        + ", ".join(
            f"{node_id}: {len(subjects)}"
            for node_id, subjects in subjects_by_node.items()
        )
    )

    if not set(cols["cols_biomarker"]).issubset(df_data_train.columns):
        raise ValueError(
            "cols_biomarker contains columns not present in the merged data"
        )
    subjects_in_nodes = sorted(
        {
            str(subject)
            for node_id, subjects in subjects_by_node.items()
            if node_id != CENTRALIZED_NODE
            for subject in subjects
        },
    )
    if subjects_in_nodes != sorted(df_data_train[col_subject].unique()):
        raise ValueError("subjects_by_node does not match subjects in merged data")

    if true_params is not None:
        true_params_by_subject = _true_params_by_subject(subjects_by_node, true_params)
        if true_params_by_subject.time_shifts.shape != (n_subjects,):
            raise ValueError(
                f"true time_shifts have shape {true_params_by_subject.time_shifts.shape} "
                f"but expected ({n_subjects},)"
            )
        if true_params_by_subject.acceleration_factors.shape != (n_subjects,):
            raise ValueError(
                f"true acceleration_factors have shape "
                f"{true_params_by_subject.acceleration_factors.shape} "
                f"but expected ({n_subjects},)"
            )
        print(
            f"true time_shifts range: [{true_params_by_subject.time_shifts.min():.3f}, "
            f"{true_params_by_subject.time_shifts.max():.3f}]"
        )
        print(
            f"true acceleration_factors range: [{true_params_by_subject.acceleration_factors.min():.3f}, "
            f"{true_params_by_subject.acceleration_factors.max():.3f}]"
        )

    estimated_by_setup = _estimated_params_by_setup(json_results, subjects_by_node)
    for setup, estimated_params in estimated_by_setup.items():
        if estimated_params.time_shifts.shape != (n_subjects,):
            raise ValueError(
                f"{setup}: estimated time_shifts have shape "
                f"{estimated_params.time_shifts.shape} but expected ({n_subjects},)"
            )
        if estimated_params.acceleration_factors.shape != (n_subjects,):
            raise ValueError(
                f"{setup}: estimated acceleration_factors have shape "
                f"{estimated_params.acceleration_factors.shape} but expected "
                f"({n_subjects},)"
            )
        print(
            f"{setup}: estimated time_shifts range: "
            f"[{estimated_params.time_shifts.min():.3f}, "
            f"{estimated_params.time_shifts.max():.3f}], "
            f"estimated acceleration_factors range: "
            f"[{estimated_params.acceleration_factors.min():.3f}, "
            f"{estimated_params.acceleration_factors.max():.3f}]"
        )

    cols_biomarker = cols["cols_biomarker"]
    t_train = df_data_train[cols["col_timepoint"]].to_numpy(dtype=float)
    subject_ids_train = df_data_train[cols["col_subject_index"]].to_numpy(dtype=int)
    y_true_train = df_data_train[cols_biomarker].to_numpy(dtype=float)

    if df_data_test is not None:
        t_test = df_data_test[cols["col_timepoint"]].to_numpy(dtype=float)
        subject_ids_test = df_data_test[cols["col_subject_index"]].to_numpy(dtype=int)
        y_true_test = df_data_test[cols_biomarker].to_numpy(dtype=float)

    if subject_ids_train.size and (
        subject_ids_train.min() < 0 or subject_ids_train.max() >= n_subjects
    ):
        raise ValueError(
            "subject indices fall outside the merged data's subject range "
            f"[0, {n_subjects})"
        )

    rows: list[dict[str, str | float]] = []
    rows_recovery: list[dict[str, str | float]] = []
    for setup, estimated_params in estimated_by_setup.items():
        y_pred_train = _predict(estimated_params, t_train, subject_ids_train)
        rows.extend(
            compute_predictive_metrics(
                y_true_train,
                y_pred_train,
                cols_biomarker,
                setup=setup,
                set_name="train",
            )
        )
        if df_data_test is not None:
            y_pred_test = _predict(estimated_params, t_test, subject_ids_test)
            rows.extend(
                compute_predictive_metrics(
                    y_true_test,
                    y_pred_test,
                    cols_biomarker,
                    setup=setup,
                    set_name="test",
                )
            )
        if true_params is not None:
            rows_recovery.extend(
                compute_recovery_metrics(
                    true_params_by_subject,
                    estimated_params,
                    cols_biomarker,
                    setup=setup,
                )
            )
            rows_recovery.extend(
                compute_per_subject_recovery(
                    true_params_by_subject, estimated_params, setup=setup
                )
            )

    df_metrics = pd.DataFrame(
        rows, columns=["setup", "set_name", "col_biomarker", "metric", "value"]
    )

    df_recovery = pd.DataFrame(
        rows_recovery,
        columns=["setup", "set_name", "col_biomarker", "metric", "value"],
    )

    return df_metrics, df_recovery


def get_metrics(
    tag: str,
    dpath_results: Path,
):
    dpath_results_latest = get_dpath_latest(dpath_results)
    for fpath_json_results in (dpath_results_latest / tag).glob(
        "*-estimated_params.json"
    ):
        run_tag = fpath_json_results.stem.removesuffix("-estimated_params")
        print(f"fpath_json_results: {fpath_json_results}")

        df_metrics, df_recovery = get_metrics_single_run(fpath_json_results, tag)

        fpath_metrics_out = fpath_json_results.with_name(f"{run_tag}-fit_quality.tsv")
        _save_tsv(df_metrics, fpath_metrics_out)

        if not df_recovery.empty:
            fpath_recovery_out = fpath_json_results.with_name(
                f"{run_tag}-param_recovery.tsv"
            )
            _save_tsv(df_recovery, fpath_recovery_out)


@click.command(context_settings=CLICK_CONTEXT_SETTINGS)
@click.option("--tag", type=str, required=True)
@click.option(
    "--results-dir",
    "dpath_results",
    type=click.Path(path_type=Path, file_okay=False, dir_okay=True),
    default=DEFAULT_DPATH_RESULTS,
)
def main(**params):
    get_metrics(**params)


if __name__ == "__main__":
    main()
