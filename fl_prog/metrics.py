from collections.abc import Iterable
from dataclasses import fields

import numpy as np
from scipy.stats import linregress, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from fl_prog.model import ModelParams

PER_SUBJECT_FIELDS = ("time_shifts", "acceleration_factors")
CORRELATED_FIELDS = ("k_values", "x0_values")


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

    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)
    rows.extend(
        _metric_rows(
            setup,
            set_name,
            "all",
            [
                ("r2_score", r2_score(y_true_all, y_pred_all)),
                ("mean_squared_error", mean_squared_error(y_true_all, y_pred_all)),
                ("mean_absolute_error", mean_absolute_error(y_true_all, y_pred_all)),
            ],
        )
    )
    return rows


def compute_recovery_metrics(
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


def compute_per_subject_recovery(
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
