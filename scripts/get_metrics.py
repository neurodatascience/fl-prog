#!/usr/bin/env python

from collections.abc import Iterable
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import click
import numpy as np
import pandas as pd
from scipy.stats import linregress, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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

PER_SUBJECT_FIELDS = ("time_shifts", "acceleration_factors")
CORRELATED_FIELDS = ("k_values", "x0_values")
CENTRALIZED_NODE = "centralized"


@dataclass
class ModelParams:
    """Parameters aligned to a canonical subject order.

    Per-feature arrays have shape (n_features,); per-subject arrays have
    shape (n_subjects,), aligned to the merged data's subject order.
    """

    k_values: np.ndarray
    x0_values: np.ndarray
    scaling_factors: np.ndarray
    sigma: np.ndarray
    time_shifts: np.ndarray
    acceleration_factors: np.ndarray


def _get_dpath_data(json_results: dict) -> Path:
    return Path(json_results["settings"]["dpath_data"])


def _load_json_data(json_results: dict) -> dict:
    fpath_json_data = Path(json_results["settings"]["fpath_config"])
    return load_json(fpath_json_data)


def _get_cols(json_data: dict) -> dict[str, Any]:
    return json_data["cols"]


def _get_subjects_by_node(json_data: dict) -> dict[str, list[str]]:
    return json_data["subjects_by_node"]


def _get_true_params(json_data: dict) -> dict[str, Any]:
    return json_data["params"]


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


def _load_df_data(json_data: dict, dpath_data: Path) -> pd.DataFrame:
    tag = json_data["settings"]["tag"]
    cols = _get_cols(json_data)
    fpath_data = dpath_data / f"{tag}-merged.tsv"
    return pd.read_csv(fpath_data, sep="\t", dtype={cols["col_subject"]: str})


def _predict(p: ModelParams, t: np.ndarray, subject_ids: np.ndarray) -> np.ndarray:
    """Model forward pass, vectorized over entries.

    Returns predictions of shape (n_entries, n_features), matching
    LogisticRegressionModelWithShift.forward::
        pred = scaling * sigmoid(k * ((t + shift) * acc - x0)) + vertical
    """
    t = np.asarray(t, dtype=float)
    subject_ids = np.asarray(subject_ids, dtype=int)

    shift = p.time_shifts[subject_ids]
    acceleration = p.acceleration_factors[subject_ids]
    shifted_t = t * acceleration + shift

    linear_combination = p.k_values * (shifted_t[:, np.newaxis] - p.x0_values)
    output = p.scaling_factors * (1 / (1 + np.exp(-linear_combination)))
    return output


def _metric_rows(
    setup: str,
    set_name: str,
    col_biomarker: str,
    items: Iterable[tuple[str, float]],
) -> list[dict[str, str | float]]:
    """Build metric rows carrying the shared output schema."""
    return [
        {
            "setup": setup,
            "set_name": set_name,
            "col_biomarker": col_biomarker,
            "metric": metric,
            "value": value,
        }
        for metric, value in items
    ]


def _compute_predictive_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cols_biomarker: list[str],
    setup: str,
) -> list[dict[str, str | float]]:
    """Predictive fit metrics on observed data, per biomarker.

    Rows with a NaN in ``y_true`` are masked out per biomarker, mirroring the
    missing-data handling in the model loss.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    rows: list[dict[str, str | float]] = []
    y_true_all = []
    y_pred_all = []
    for i_biomarker, col_biomarker in enumerate(cols_biomarker):
        mask = ~np.isnan(y_true[:, i_biomarker])
        y_true_biomarker = y_true[mask, i_biomarker]
        y_pred_biomarker = y_pred[mask, i_biomarker]
        residuals = y_true_biomarker - y_pred_biomarker

        y_true_all.append(y_true_biomarker)
        y_pred_all.append(y_pred_biomarker)

        rows.extend(
            _metric_rows(
                setup,
                "data",
                col_biomarker,
                [
                    ("r2_score", r2_score(y_true_biomarker, y_pred_biomarker)),
                    (
                        "mean_squared_error",
                        mean_squared_error(y_true_biomarker, y_pred_biomarker),
                    ),
                    (
                        "mean_absolute_error",
                        mean_absolute_error(y_true_biomarker, y_pred_biomarker),
                    ),
                    ("residual_mean", residuals.mean()),
                    ("residual_std", residuals.std()),
                ],
            )
        )

    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)
    rows.extend(
        _metric_rows(
            setup,
            "data",
            "all",
            [
                ("r2_score", r2_score(y_true_all, y_pred_all)),
                ("mean_squared_error", mean_squared_error(y_true_all, y_pred_all)),
                ("mean_absolute_error", mean_absolute_error(y_true_all, y_pred_all)),
            ],
        )
    )
    return rows


def _pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    return np.corrcoef(x, y)[0, 1]


def _spearman_r(x: np.ndarray, y: np.ndarray) -> float:
    return spearmanr(x, y).statistic


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    try:
        return linregress(x, y).slope
    except ValueError:
        return np.nan


def _compute_recovery_metrics(
    param_true: ModelParams,
    param_estimated: ModelParams,
    cols_biomarker: list[str],
    setup: str,
) -> list[dict[str, str | float]]:
    """Per-feature recovery errors between true and estimated parameters.

    For each feature, pooled ``all`` rows carry mean-abs-relative/mean-abs
    errors (plus pearson/spearman correlations for k_values and x0_values),
    and a nested per-biomarker part adds relative and absolute errors per
    biomarker. Relative errors are NaN where the true value is zero.
    """
    rows: list[dict[str, str | float]] = []
    for field in fields(ModelParams):
        if field.name in PER_SUBJECT_FIELDS:
            continue
        true_values = np.asarray(getattr(param_true, field.name), dtype=float)
        est_values = np.asarray(getattr(param_estimated, field.name), dtype=float)

        if len(true_values) != len(cols_biomarker):
            raise ValueError(
                f"{field.name} has {len(true_values)} values but "
                f"{len(cols_biomarker)} biomarkers"
            )

        mask = true_values != 0
        relative_errors = np.full(true_values.shape, np.nan)
        relative_errors[mask] = (est_values[mask] - true_values[mask]) / true_values[
            mask
        ]
        absolute_errors = est_values - true_values

        pooled_items = [
            (
                f"{field.name}_mean_abs_relative_error",
                np.nanmean(np.abs(relative_errors)),
            ),
            (f"{field.name}_mae", np.mean(np.abs(absolute_errors))),
        ]
        if field.name in CORRELATED_FIELDS:
            pooled_items.extend(
                [
                    (f"{field.name}_pearson_r", _pearson_r(true_values, est_values)),
                    (
                        f"{field.name}_spearman_r",
                        _spearman_r(true_values, est_values),
                    ),
                ]
            )
        rows.extend(_metric_rows(setup, "recovery_per_biomarker", "all", pooled_items))

        for i_biomarker, col_biomarker in enumerate(cols_biomarker):
            rows.extend(
                _metric_rows(
                    setup,
                    "recovery_per_biomarker",
                    col_biomarker,
                    [
                        (f"{field.name}_relative_error", relative_errors[i_biomarker]),
                        (f"{field.name}_absolute_error", absolute_errors[i_biomarker]),
                    ],
                )
            )
    return rows


def _compute_per_subject_recovery(
    param_true: ModelParams,
    param_estimated: ModelParams,
    setup: str,
) -> list[dict[str, str | float]]:
    """Per-subject recovery metrics for time shifts and acceleration factors.

    Correlations isolate ordering (gauge-robust); the slope (est vs true)
    shows magnitude recovery, with the intercept absorbing recentering.
    Acceleration factors additionally get a scale median (median est/true) and
    a mean-abs-relative error, since they are multiplicative and positive.
    """
    rows: list[dict[str, str | float]] = []
    for field_name in PER_SUBJECT_FIELDS:
        true_values = np.asarray(getattr(param_true, field_name), dtype=float)
        est_values = np.asarray(getattr(param_estimated, field_name), dtype=float)

        items = [
            (f"{field_name}_pearson_r", _pearson_r(true_values, est_values)),
            (f"{field_name}_spearman_r", _spearman_r(true_values, est_values)),
            (f"{field_name}_slope", _slope(true_values, est_values)),
            (f"{field_name}_mae", np.mean(np.abs(est_values - true_values))),
        ]
        if field_name == "acceleration_factors":
            mask = true_values != 0
            ratios = np.full(true_values.shape, np.nan)
            ratios[mask] = est_values[mask] / true_values[mask]
            items.extend(
                [
                    (f"{field_name}_scale_median", np.nanmedian(ratios)),
                    (
                        f"{field_name}_mean_abs_relative_error",
                        np.nanmean(np.abs(ratios - 1)),
                    ),
                ]
            )

        rows.extend(_metric_rows(setup, "recovery_per_subject", "all", items))
    return rows


def _save_tsv(df_metrics: pd.DataFrame, fpath_out: Path):
    fpath_out.parent.mkdir(parents=True, exist_ok=True)
    df_metrics.to_csv(fpath_out, sep="\t", index=False)
    print(f"Saved metrics to {fpath_out}")


def get_metrics_single_run(fpath_json_results: Path):
    json_results = load_json(fpath_json_results)

    dpath_data = _get_dpath_data(json_results)
    print(f"dpath_data: {dpath_data}")

    json_data = _load_json_data(json_results)

    cols = _get_cols(json_data)
    subjects_by_node = _get_subjects_by_node(json_data)
    true_params = _get_true_params(json_data)

    df_data = _load_df_data(json_data, dpath_data)

    col_subject = cols["col_subject"]
    n_biomarkers = len(cols["cols_biomarker"])
    n_subjects = df_data[col_subject].nunique()
    print(f"n_biomarkers: {n_biomarkers}")
    print(f"n_subjects: {n_subjects}")
    print(
        "n_subjects_by_node: "
        + ", ".join(
            f"{node_id}: {len(subjects)}"
            for node_id, subjects in subjects_by_node.items()
        )
    )

    if not set(cols["cols_biomarker"]).issubset(df_data.columns):
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
    if subjects_in_nodes != sorted(df_data[col_subject].unique()):
        raise ValueError("subjects_by_node does not match subjects in merged data")

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
    t = df_data[cols["col_timepoint"]].to_numpy(dtype=float)
    subject_ids = df_data[cols["col_subject_index"]].to_numpy(dtype=int)
    y_true = df_data[cols_biomarker].to_numpy(dtype=float)

    if subject_ids.size and (subject_ids.min() < 0 or subject_ids.max() >= n_subjects):
        raise ValueError(
            "subject indices fall outside the merged data's subject range "
            f"[0, {n_subjects})"
        )

    rows: list[dict[str, str | float]] = []
    rows_recovery: list[dict[str, str | float]] = []
    for setup, estimated_params in estimated_by_setup.items():
        y_pred = _predict(estimated_params, t, subject_ids)
        rows.extend(
            _compute_predictive_metrics(y_true, y_pred, cols_biomarker, setup=setup)
        )
        rows_recovery.extend(
            _compute_recovery_metrics(
                true_params_by_subject,
                estimated_params,
                cols_biomarker,
                setup=setup,
            )
        )
        rows_recovery.extend(
            _compute_per_subject_recovery(
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

        df_metrics, df_recovery = get_metrics_single_run(fpath_json_results)

        fpath_out = fpath_json_results.with_name(f"{run_tag}-fit_quality.tsv")
        _save_tsv(df_metrics, fpath_out)

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
