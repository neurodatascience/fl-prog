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
# rows over all biomarkers: mean of the per-biomarker values of these metrics
BIOMARKER_ALL = "all"
METRICS_AVERAGED_OVER_BIOMARKERS = (
    "r2_score",
    "mean_squared_error",
    "mean_absolute_error",
    "residual_mean",
    "residual_std",
)

SET_NAME_TRAIN = "train"
SET_NAME_TEST = "test"  # personalized on all visits
SET_NAME_TEST_FIT = "test_fit"  # visits used for personalization
SET_NAME_TEST_FORECAST = "test_forecast"  # visits after the first forecast_after
COL_N_TIMEPOINTS = "n_timepoints"  # visits used for personalization (empty: all)
COL_FORECAST_AFTER = "forecast_after"
LABEL_TRAIN = "Train"


def get_forecast_label(n_timepoints: int) -> str:
    """Return the set label for ``n_timepoints``, e.g. "Test (2 timepoints)"."""
    return f"Test ({n_timepoints} timepoint{'s' if n_timepoints > 1 else ''})"


def select_forecast_after(
    df: pd.DataFrame, forecast_after: int | None = None
) -> pd.DataFrame:
    """Select the rows of ``df`` scored after the first ``forecast_after`` visits.

    If ``forecast_after`` is None, use the largest value in ``df``.
    """
    values = sorted(int(value) for value in df[COL_FORECAST_AFTER].dropna().unique())
    if not values:
        raise ValueError("No forecast results (forecast_after is empty)")
    if forecast_after is None:
        forecast_after = values[-1]
        if len(values) > 1:
            print(
                f"WARNING: found forecast_after values {values}, using {forecast_after}"
            )
    elif forecast_after not in values:
        raise ValueError(
            f"No results with forecast_after={forecast_after} (found {values})"
        )
    print(f"Forecast results: scored after the first {forecast_after} visit(s)")
    return df.loc[df[COL_FORECAST_AFTER] == forecast_after]


def get_fit_sets(
    df_metrics: pd.DataFrame, forecast_after: int | None = None
) -> tuple[pd.DataFrame, list[str]]:
    """Select the train and test_forecast rows of ``df_metrics`` and relabel them.

    Returns the selected rows, with ``set_name`` replaced by labels like "Train" and
    "Test (2 timepoints)", and the list of labels in plotting order.
    """
    is_all_visits = df_metrics[COL_N_TIMEPOINTS].isna()
    df_train = df_metrics.loc[
        is_all_visits & (df_metrics["set_name"] == SET_NAME_TRAIN)
    ].assign(set_name=LABEL_TRAIN)
    df_test = df_metrics.loc[
        ~is_all_visits & (df_metrics["set_name"] == SET_NAME_TEST_FORECAST)
    ]
    if df_test.empty:
        raise ValueError(
            f"No {SET_NAME_TEST_FORECAST!r} rows: run get_metrics.py with "
            "--personalize-n-timepoints"
        )
    df_test = select_forecast_after(df_test, forecast_after)
    n_values = sorted(int(n) for n in df_test[COL_N_TIMEPOINTS].unique())
    df_test = df_test.assign(
        set_name=df_test[COL_N_TIMEPOINTS].astype(int).map(get_forecast_label)
    )
    labels = [LABEL_TRAIN, *map(get_forecast_label, n_values)]
    return pd.concat([df_train, df_test], ignore_index=True), labels


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


def get_midpoint_offset(param_true: ModelParams, param_estimated: ModelParams) -> float:
    """Offset of the estimated disease-time axis relative to the true one, as the
    mean difference between estimated and true biomarker midpoints (x0)."""
    return float(
        np.mean(
            np.asarray(param_estimated.x0_values, dtype=float)
            - np.asarray(param_true.x0_values, dtype=float)
        )
    )


def shift_time_axis(params: ModelParams, offset: float) -> ModelParams:
    """Translate the disease-time axis by ``-offset``.

    x0 and time shifts are shifted, other parameters are unchanged."""
    return dataclasses.replace(
        params,
        x0_values=np.asarray(params.x0_values, dtype=float) - offset,
        time_shifts=np.asarray(params.time_shifts, dtype=float) - offset,
    )


def compute_time_axis_metrics(
    midpoint_offset: float, setup: str
) -> list[dict[str, str | float]]:
    """Row with the offset of the estimated disease-time axis relative to the true
    one."""
    return _metric_rows(
        setup,
        "time_axis",
        "all",
        [("time_axis_midpoint_offset", midpoint_offset)],
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
