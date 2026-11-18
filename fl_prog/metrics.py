import dataclasses
from collections.abc import Iterable
from dataclasses import fields

import numpy as np
import pandas as pd
from scipy.stats import linregress, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from fl_prog.model import ModelParams

PER_SUBJECT_FIELDS = ("time_shifts", "acceleration_factors")
CORRELATED_FIELDS = ("k_values", "x0_values")
# recovery after removing the arbitrary origin of the disease-time axis only
SUFFIX_SHIFTED = "_shifted"
# recovery after also removing a stretch of the axis (shape of the trajectories)
SUFFIX_ALIGNED = "_aligned"
# rows over all biomarkers: mean of the per-biomarker values of these metrics
BIOMARKER_ALL = "all"
METRICS_AVERAGED_OVER_BIOMARKERS = (
    "r2_score",
    "mean_squared_error",
    "mean_absolute_error",
    "residual_mean",
    "residual_std",
)


def _pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    return np.corrcoef(x, y)[0, 1]


def _spearman_r(x: np.ndarray, y: np.ndarray) -> float:
    return spearmanr(x, y).statistic


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    try:
        return linregress(x, y).slope
    except ValueError:
        return np.nan


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


def compute_predictive_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cols_biomarker: list[str],
    setup: str,
    set_name: str,
) -> list[dict[str, str | float]]:
    """Predictive fit metrics on observed data, per biomarker, and their mean over
    biomarkers (``BIOMARKER_ALL`` rows, see ``METRICS_AVERAGED_OVER_BIOMARKERS``;
    not pooled over biomarkers, so that each biomarker has the same weight).

    Rows with a NaN in ``y_true`` are masked out per biomarker, mirroring the
    missing-data handling in the model loss.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    rows: list[dict[str, str | float]] = []
    for i_biomarker, col_biomarker in enumerate(cols_biomarker):
        mask = ~np.isnan(y_true[:, i_biomarker])
        y_true_biomarker = y_true[mask, i_biomarker]
        y_pred_biomarker = y_pred[mask, i_biomarker]
        residuals = y_true_biomarker - y_pred_biomarker

        rows.extend(
            _metric_rows(
                setup,
                set_name,
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

    rows.extend(
        _metric_rows(
            setup,
            set_name,
            BIOMARKER_ALL,
            [
                (
                    metric,
                    float(
                        np.mean(
                            [row["value"] for row in rows if row["metric"] == metric]
                        )
                    ),
                )
                for metric in METRICS_AVERAGED_OVER_BIOMARKERS
            ],
        )
    )
    return rows


def average_over_biomarkers(df_metrics: pd.DataFrame) -> pd.DataFrame:
    """Recompute the ``BIOMARKER_ALL`` rows of a long table of predictive metrics
    (columns include ``col_biomarker``, ``metric`` and ``value``) as the mean of
    the per-biomarker values (as in ``compute_predictive_metrics``), e.g. for
    results computed when these rows were pooled over biomarkers."""
    df_biomarkers = df_metrics.loc[df_metrics["col_biomarker"] != BIOMARKER_ALL]
    keys = [col for col in df_metrics.columns if col not in ("col_biomarker", "value")]
    df_mean = (
        df_biomarkers.loc[
            df_biomarkers["metric"].isin(METRICS_AVERAGED_OVER_BIOMARKERS)
        ]
        .groupby(keys, dropna=False, sort=False)["value"]
        .mean()
        .reset_index()
        .assign(col_biomarker=BIOMARKER_ALL)
    )
    return pd.concat([df_biomarkers, df_mean[df_metrics.columns]], ignore_index=True)


def get_time_axis_alignment(
    param_true: ModelParams, param_estimated: ModelParams
) -> tuple[float, float]:
    """Affine map between the true and estimated disease-time axes.

    The model is invariant to an affine change of the disease-time axis
    t -> scale * t + offset, with time shifts -> scale * time shifts + offset,
    x0 -> scale * x0 + offset, k -> k / scale and acceleration factors ->
    scale * acceleration factors. The map is estimated by regressing the
    estimated time shifts on the true ones (same subjects).

    Returns (scale, offset), NaN if it cannot be estimated.
    """
    true_values = np.asarray(param_true.time_shifts, dtype=float)
    est_values = np.asarray(param_estimated.time_shifts, dtype=float)
    try:
        result = linregress(true_values, est_values)
    except ValueError:
        return np.nan, np.nan
    return float(result.slope), float(result.intercept)


def get_midpoint_offset(param_true: ModelParams, param_estimated: ModelParams) -> float:
    """Offset of the estimated disease-time axis relative to the true one, as the
    mean difference between estimated and true biomarker midpoints (x0).

    Only the origin of the disease-time axis is arbitrary (the model is exactly
    invariant to translations); its scale is tied to calendar time through the
    acceleration factors, so it is not removed here.
    """
    return float(
        np.mean(
            np.asarray(param_estimated.x0_values, dtype=float)
            - np.asarray(param_true.x0_values, dtype=float)
        )
    )


def align_time_axis(params: ModelParams, scale: float, offset: float) -> ModelParams:
    """Express parameters on the true disease-time axis (inverse of the map from
    ``get_time_axis_alignment``). Scaling factors and sigma are unchanged."""
    return dataclasses.replace(
        params,
        k_values=np.asarray(params.k_values, dtype=float) * scale,
        x0_values=(np.asarray(params.x0_values, dtype=float) - offset) / scale,
        time_shifts=(np.asarray(params.time_shifts, dtype=float) - offset) / scale,
        acceleration_factors=np.asarray(params.acceleration_factors, dtype=float)
        / scale,
    )


def compute_time_axis_metrics(
    scale: float, offset: float, midpoint_offset: float, setup: str
) -> list[dict[str, str | float]]:
    """Rows with the estimated-vs-true disease-time axis map: scale and offset from
    ``get_time_axis_alignment`` and offset from ``get_midpoint_offset``."""
    return _metric_rows(
        setup,
        "time_axis",
        "all",
        [
            ("time_axis_scale", scale),
            ("time_axis_offset", offset),
            ("time_axis_midpoint_offset", midpoint_offset),
        ],
    )


def compute_recovery_metrics(
    param_true: ModelParams,
    param_estimated: ModelParams,
    cols_biomarker: list[str],
    setup: str,
    set_name: str = "recovery_per_biomarker",
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
        rows.extend(_metric_rows(setup, set_name, "all", pooled_items))

        for i_biomarker, col_biomarker in enumerate(cols_biomarker):
            rows.extend(
                _metric_rows(
                    setup,
                    set_name,
                    col_biomarker,
                    [
                        (f"{field.name}_relative_error", relative_errors[i_biomarker]),
                        (f"{field.name}_absolute_error", absolute_errors[i_biomarker]),
                    ],
                )
            )
    return rows


def compute_per_subject_recovery(
    param_true: ModelParams,
    param_estimated: ModelParams,
    setup: str,
    set_name: str = "recovery_per_subject",
) -> list[dict[str, str | float]]:
    """Per-subject recovery metrics for time shifts and acceleration factors.

    Metrics are: Pearson and Spearman correlations, slope (estimated vs true),
    mean absolute error (MAE). Acceleration factors have two additional metrics
    (scale median (median estimated/true) and mean-abs-relative error), since they
    are multiplicative and positive.
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

        rows.extend(_metric_rows(setup, set_name, "all", items))
    return rows
