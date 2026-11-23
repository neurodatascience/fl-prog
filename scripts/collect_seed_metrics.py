#!/usr/bin/env python
"""Collect metrics from a multi-seed run (see ``run_adni_seeds.sh`` and
``run_synthetic_seeds.sh``).

Expected layout::

    <results-dir>/<tag>/<model type>-*-fit_quality*.tsv
    <results-dir>/<tag>/<model type>-*-param_recovery*.tsv (synthetic data only)

where ``<tag>`` looks like ``<dataset>_<condition>_s<seed>_...``.
"""

import re
from pathlib import Path

import click
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from fl_prog.plotting import get_categorical_palette, save_fig
from fl_prog.utils.constants import CLICK_CONTEXT_SETTINGS, SYNTHETIC_CONDITION_LABELS

DATASET_ADNI = "adni_harmonized"
DATASET_SYNTHETIC = "synth_fullcov"
# same simulation with ADNI-like (reduced) disease coverage
DATASET_SYNTHETIC_ADNILIKE = "synth_adnilike"
DATASETS = [DATASET_ADNI, DATASET_SYNTHETIC, DATASET_SYNTHETIC_ADNILIKE]

# (model type, data condition, setup[, variant]) -> model label
# variant: Fed-BioMed aggregator/optimizer from the file name (see _get_variant),
# omitted for the default (FedAvg + Adam)
MODELS = {
    # ADNI
    ("fedbiomed", "iid", "centralized"): "Fed-BioMed centralized",
    ("fedbiomed", "iid", "federated"): "Fed-BioMed IID",
    ("fedbiomed", "noniid_site", "federated"): "Fed-BioMed non-IID (site)",
    ("fedbiomed", "noniid_diag", "federated"): "Fed-BioMed non-IID (diag.)",
    ("dpmost", "iid", "centralized"): "DPMoSt",
    ("leaspy", "iid", "centralized"): "Leaspy",
    # synthetic
    ("fedbiomed", "overlap100", "centralized"): "Fed-BioMed centralized",
    **{
        ("fedbiomed", condition, "federated"): f"Fed-BioMed {label}"
        for condition, label in SYNTHETIC_CONDITION_LABELS.items()
    },
    # other aggregators/optimizers (run_synthetic_seeds.sh VARIANT_CONDITIONS)
    (
        "fedbiomed",
        "overlap000",
        "federated",
        "scaffold_native_sgd",
    ): "Fed-BioMed 0% overlap (SCAFFOLD)",
    (
        "fedbiomed",
        "overlap000",
        "federated",
        "fedavg_sgd",
    ): "Fed-BioMed 0% overlap (FedAvg, SGD)",
    (
        "fedbiomed",
        "overlap000",
        "federated",
        "scaffold_adam",
    ): "Fed-BioMed 0% overlap (SCAFFOLD, Adam)",
    ("dpmost", "overlap100", "centralized"): "DPMoSt",
    ("leaspy", "overlap100", "centralized"): "Leaspy",
}
METRICS = {"mean_absolute_error": "MAE", "r2_score": "R²"}
# (set name, metric) -> label
RECOVERY_METRICS = {
    ("recovery_per_biomarker", "k_values_mean_abs_relative_error"): "k rel. error",
    (
        "recovery_per_biomarker_shifted",
        "x0_values_mean_abs_relative_error",
    ): "x0 rel. error (shifted)",
    (
        "recovery_per_biomarker",
        "scaling_factors_mean_abs_relative_error",
    ): "Scaling rel. error",
    ("recovery_per_subject", "time_shifts_pearson_r"): "Time shift r (train)",
    ("recovery_per_subject_test", "time_shifts_pearson_r"): "Time shift r (test)",
    (
        "recovery_per_subject_test_shifted",
        "time_shifts_mae",
    ): "Time shift MAE (test, shifted)",
    (
        "recovery_per_subject",
        "acceleration_factors_spearman_r",
    ): "Accel. factor \u03c1 (train)",  # rho
}
BIOMARKER_ALL = "all"
SET_NAME_ORDER = ["train", "test"]
# held-out subjects personalized on their first N visits
SET_NAME_TEST_ALL = "test"
SET_NAME_TEST_FIT = "test_fit"
SET_NAMES_FORECAST = [SET_NAME_TEST_FIT, "test_forecast"]
# prefix of the recovery set names of held-out subjects (also "_shifted")
PREFIX_SET_NAME_RECOVERY_TEST = "recovery_per_subject_test"
COL_N_TIMEPOINTS = "n_timepoints"
COL_FORECAST_AFTER = "forecast_after"
PATTERN_TAG = re.compile(
    rf"^(?P<dataset>{'|'.join(DATASETS)})_(?P<condition>.+?)_s(?P<seed>\d+)_"
)
# Fed-BioMed run tag: fedbiomed[-<setups>]-...-<scaling|no_scaling>-<aggregator[_optimizer][_locallrX]>-<seed>-
PATTERN_FBM_AGGREGATOR = re.compile(
    r"-(?:no_)?scaling-(?P<aggregator>[^-]+)-(?:\d+|no_seed)-"
)
# for extracting N and K for personalization and forecast timepoints respectivaly
PATTERN_N_TIMEPOINTS = re.compile(r"-first(?P<n>\d+)_after(?P<k>\d+)\.tsv$")
# suffixes to strip
PATTERN_VARIANT_SUFFIXES_IGNORED = re.compile(
    r"((?<!scaffold)_adam)?(_locallr[\d.]+)?$"
)
DEFAULT_FBM_AGGREGATOR = "fedavg"
DNAME_FIGURES = "figures"
PREFIX_OUT = "fit_metrics-seeds"
PREFIX_OUT_RECOVERY = "param_recovery-seeds"


def _get_variant(fname: str) -> str | None:
    """Fed-BioMed aggregator/optimizer variant (None for the default FedAvg + Adam)."""
    match = PATTERN_FBM_AGGREGATOR.search(fname)
    if match is None:
        return None
    variant = PATTERN_VARIANT_SUFFIXES_IGNORED.sub("", match.group("aggregator"))
    return None if variant == DEFAULT_FBM_AGGREGATOR else variant


def _get_n_timepoints(fname: str) -> tuple[float, float]:
    """Get (N, K) of a metrics file personalized on the first N visits and scored
    after the first K (NaN, NaN for personalization on all visits)."""
    match = PATTERN_N_TIMEPOINTS.search(fname)
    if match is None:
        return float("nan"), float("nan")
    return int(match.group("n")), int(match.group("k"))


def _collect(dpath_results: Path, dataset: str, pattern_fname: str) -> pd.DataFrame:
    dfs = []
    for fpath in sorted(dpath_results.glob(f"*/{pattern_fname}")):
        tag = fpath.parent.name
        match = PATTERN_TAG.search(tag)
        if match is None or match.group("dataset") != dataset:
            continue  # not a multi-seed run of this dataset
        model_type = fpath.name.split("-", 1)[0]
        condition = match.group("condition")
        df = pd.read_csv(fpath, sep="\t")
        for setup, df_setup in df.groupby("setup"):
            variant = _get_variant(fpath.name)
            key = (model_type, condition, setup)
            model = MODELS.get(key if variant is None else (*key, variant))
            if model is None:
                click.secho(
                    f"WARNING: ignoring {model_type}/{condition}/{setup}/{variant} "
                    f"({fpath})",
                    fg="yellow",
                )
                continue
            df_setup = df_setup.copy()
            df_setup.insert(0, "model", model)
            df_setup.insert(1, "condition", condition)
            df_setup.insert(2, "seed", int(match.group("seed")))
            df_setup.insert(3, "tag", tag)
            df_setup[COL_N_TIMEPOINTS], df_setup[COL_FORECAST_AFTER] = (
                _get_n_timepoints(fpath.name)
            )
            df_setup["fpath"] = str(fpath)
            dfs.append(df_setup)
    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    # a model/condition/seed must come from a single file (e.g. not from both a run
    # with all setups and a run with only one of them, or from two personalization
    # seeds), otherwise it would be counted more than once
    keys = ["model", "condition", "seed", COL_N_TIMEPOINTS, COL_FORECAST_AFTER]
    n_files = df.groupby(keys, dropna=False)["fpath"].nunique()
    if (n_files > 1).any():
        duplicates = n_files[n_files > 1].index.tolist()
        fpaths = df.loc[
            df.set_index(keys).index.isin(n_files[n_files > 1].index), "fpath"
        ]
        raise click.ClickException(
            f"Results for {duplicates} come from more than one file: "
            + ", ".join(sorted(set(fpaths)))
        )
    return df


def _split_n_timepoints(
    df: pd.DataFrame, set_names_forecast: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split ``df`` by how held-out subjects were personalized.

    Returns the rows where held-out subjects were personalized on all visits
    (``n_timepoints`` is NaN), and the rows where they were personalized on their
    first N visits only, restricted to ``set_names_forecast``. Other rows with N
    set (e.g. train) are the same for any N and are dropped.
    """
    is_all = df[COL_N_TIMEPOINTS].isna()
    is_forecast = ~is_all & df["set_name"].isin(set_names_forecast)
    return df.loc[is_all].copy(), df.loc[is_forecast].copy()


def _select_forecast_after(
    df: pd.DataFrame, forecast_after: int | None
) -> pd.DataFrame:
    """Get rows scored after the first ``forecast_after`` visits (default: largest)."""
    values = sorted(df[COL_FORECAST_AFTER].unique())
    if forecast_after is None:
        forecast_after = values[-1]
        if len(values) > 1:
            click.secho(
                f"WARNING: found forecast_after values {values}, using "
                f"{forecast_after} (see --forecast-after)",
                fg="yellow",
            )
    elif forecast_after not in values:
        raise click.UsageError(
            f"No results with forecast_after={forecast_after} (found {values})"
        )
    print(f"Forecast results: scored after the first {forecast_after} visit(s)")
    return df.loc[df[COL_FORECAST_AFTER] == forecast_after]


def _infer_dataset(dpath_results: Path) -> str:
    datasets = {
        match.group("dataset")
        for dpath in dpath_results.iterdir()
        if (match := PATTERN_TAG.search(dpath.name)) is not None
    }
    if len(datasets) != 1:
        raise click.UsageError(
            f"Found datasets {sorted(datasets)} in {dpath_results}, use --dataset"
        )
    return datasets.pop()


def _get_model_order(df: pd.DataFrame) -> list[str]:
    models = set(df["model"])
    return list(dict.fromkeys(m for m in MODELS.values() if m in models))


def _summarize(df: pd.DataFrame, cols: list[str], fpath_out: Path) -> pd.DataFrame:
    model_order = _get_model_order(df)
    df_summary = (
        df.groupby(["model", *cols])["value"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    df_summary["model"] = pd.Categorical(df_summary["model"], model_order)
    df_summary = df_summary.sort_values([*cols, "model"])
    df_summary.to_csv(fpath_out, sep="\t", index=False)
    print(f"Saved summary to {fpath_out}")
    print(df_summary.to_string(index=False))
    return df_summary


def _boxplot(ax, data: pd.DataFrame, model_order: list[str], palette):
    sns.boxplot(
        data=data,
        x="model",
        y="value",
        hue="model",
        order=model_order,
        hue_order=model_order,
        palette=palette,
        showfliers=False,
        ax=ax,
    )
    sns.stripplot(
        data=data,
        x="model",
        y="value",
        order=model_order,
        color="black",
        size=3,
        ax=ax,
    )
    ax.set_xlabel("")
    ax.tick_params(axis="x", labelrotation=45)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")


def _plot_fit_metrics(df_all: pd.DataFrame, fpath_out: Path):
    model_order = _get_model_order(df_all)
    palette = get_categorical_palette(model_order)
    set_names = [s for s in SET_NAME_ORDER if s in set(df_all["set_name"])]
    fig, axes = plt.subplots(
        nrows=len(METRICS),
        ncols=len(set_names),
        figsize=(4.5 * len(set_names), 3.5 * len(METRICS)),
        squeeze=False,
        sharey="row",
    )
    for i_row, (metric, metric_label) in enumerate(METRICS.items()):
        for i_col, set_name in enumerate(set_names):
            ax = axes[i_row, i_col]
            data = df_all.query("metric == @metric and set_name == @set_name")
            _boxplot(ax, data, model_order, palette)
            ax.set_title(f"{set_name.capitalize()} {metric_label}")
            ax.set_ylabel(metric_label if i_col == 0 else "")
    fig.tight_layout()
    save_fig(fig, fpath_out)


def _plot_recovery(
    df_recovery: pd.DataFrame, fpath_out: Path, metrics: dict = RECOVERY_METRICS
):
    model_order = _get_model_order(df_recovery)
    palette = get_categorical_palette(model_order)
    panels = [
        (key, label)
        for key, label in metrics.items()
        if not df_recovery.query("set_name == @key[0] and metric == @key[1]").empty
    ]
    if not panels:
        click.secho("WARNING: no recovery metrics to plot", fg="yellow")
        return
    n_cols = 3
    n_rows = -(-len(panels) // n_cols)
    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(4.5 * n_cols, 3.5 * n_rows),
        squeeze=False,
    )
    axes = axes.ravel()
    for ax, ((set_name, metric), label) in zip(axes, panels, strict=False):
        data = df_recovery.query("set_name == @set_name and metric == @metric")
        _boxplot(ax, data, model_order, palette)
        ax.set_title(label)
        ax.set_ylabel("")
    for ax in axes[len(panels) :]:
        ax.set_visible(False)
    fig.tight_layout()
    save_fig(fig, fpath_out)


def _lineplot_vs_n_timepoints(
    ax,
    data: pd.DataFrame,
    model_order: list[str],
    palette,
    data_reference: pd.DataFrame | None = None,
):
    """Mean +/- std over seeds against N, with each model's value when personalized
    on all visits (mean over seeds) as a dashed reference line."""
    n_values = sorted(data[COL_N_TIMEPOINTS].unique())
    for model in model_order:
        data_model = data.loc[data["model"] == model]
        if data_model.empty:
            continue
        stats = data_model.groupby(COL_N_TIMEPOINTS)["value"].agg(["mean", "std"])
        ax.errorbar(
            stats.index,
            stats["mean"],
            yerr=stats["std"],
            color=palette[model],
            marker="o",
            capsize=3,
            label=model,
        )
        if data_reference is not None:
            values_reference = data_reference.loc[
                data_reference["model"] == model, "value"
            ]
            if not values_reference.empty:
                ax.axhline(
                    values_reference.mean(),
                    color=palette[model],
                    linestyle="--",
                    linewidth=1,
                )
    ax.set_xticks(n_values)
    ax.set_xticklabels([str(int(n)) for n in n_values])
    ax.set_xlabel("Personalization visits (N)")


def _plot_fit_metrics_vs_n_timepoints(
    df_forecast: pd.DataFrame, df_all_visits: pd.DataFrame, fpath_out: Path
):
    model_order = _get_model_order(df_forecast)
    palette = get_categorical_palette(model_order)
    set_names = [s for s in SET_NAMES_FORECAST if s in set(df_forecast["set_name"])]
    fig, axes = plt.subplots(
        nrows=len(METRICS),
        ncols=len(set_names),
        figsize=(4.5 * len(set_names), 3.5 * len(METRICS)),
        squeeze=False,
        sharey="row",
    )
    for i_row, (metric, metric_label) in enumerate(METRICS.items()):
        for i_col, set_name in enumerate(set_names):
            ax = axes[i_row, i_col]
            _lineplot_vs_n_timepoints(
                ax,
                df_forecast.query("metric == @metric and set_name == @set_name"),
                model_order,
                palette,
                # all-visit scores are in-sample: not a reference for forecasts
                data_reference=(
                    df_all_visits.query(
                        "metric == @metric and set_name == @SET_NAME_TEST_ALL"
                    )
                    if set_name == SET_NAME_TEST_FIT
                    else None
                ),
            )
            ax.set_title(f"{set_name} {metric_label}")
            ax.set_ylabel(metric_label if i_col == 0 else "")
    axes[0, -1].legend(fontsize="small")
    fig.suptitle(
        "Held-out subjects (dashed: personalized and scored on all visits, "
        "all subjects)"
    )
    fig.tight_layout()
    save_fig(fig, fpath_out)


def _plot_recovery_vs_n_timepoints(
    df_forecast: pd.DataFrame, df_all_visits: pd.DataFrame, fpath_out: Path
):
    model_order = _get_model_order(df_forecast)
    palette = get_categorical_palette(model_order)
    panels = [
        (key, label)
        for key, label in RECOVERY_METRICS.items()
        if not df_forecast.query("set_name == @key[0] and metric == @key[1]").empty
    ]
    if not panels:
        click.secho("WARNING: no held-out recovery metrics to plot", fg="yellow")
        return
    fig, axes = plt.subplots(
        ncols=len(panels), figsize=(4.5 * len(panels), 3.5), squeeze=False
    )
    for ax, ((set_name, metric), label) in zip(axes[0], panels, strict=True):
        query = "set_name == @set_name and metric == @metric"
        _lineplot_vs_n_timepoints(
            ax,
            df_forecast.query(query),
            model_order,
            palette,
            data_reference=df_all_visits.query(query),
        )
        ax.set_title(label)
        ax.set_ylabel("")
    axes[0, -1].legend(fontsize="small")
    fig.tight_layout()
    save_fig(fig, fpath_out)


@click.command(context_settings=CLICK_CONTEXT_SETTINGS)
@click.option(
    "--results-dir",
    "dpath_results",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    required=True,
    help="Dated results directory, e.g. results/2026_11_08",
)
@click.option(
    "--dataset",
    type=click.Choice(DATASETS),
    default=None,
    help="Dataset to collect (inferred if the results directory has only one)",
)
@click.option(
    "--forecast-after",
    type=click.IntRange(min=1),
    default=None,
    help="For held-out subjects personalized on their first N visits: plot results "
    "scored after the first K visits (default: largest K found)",
)
def main(dpath_results: Path, dataset: str | None, forecast_after: int | None):
    if dataset is None:
        dataset = _infer_dataset(dpath_results)
    # keep original output names for ADNI
    suffix = "" if dataset == DATASET_ADNI else f"-{dataset}"

    dpath_out = dpath_results / DNAME_FIGURES
    dpath_out.mkdir(parents=True, exist_ok=True)

    # fit quality
    df = _collect(dpath_results, dataset, "*-fit_quality*.tsv")
    if df.empty:
        raise RuntimeError(
            f"No multi-seed fit_quality files for {dataset} found in {dpath_results}"
        )
    fpath_all = dpath_out / f"{PREFIX_OUT}{suffix}-all.tsv"
    df.to_csv(fpath_all, sep="\t", index=False)
    print(f"Saved {len(df)} rows to {fpath_all}")

    df_all, df_forecast = _split_n_timepoints(
        df.loc[(df["col_biomarker"] == BIOMARKER_ALL) & df["metric"].isin(METRICS)],
        SET_NAMES_FORECAST,
    )
    _summarize(
        df_all, ["set_name", "metric"], dpath_out / f"{PREFIX_OUT}{suffix}-summary.tsv"
    )
    _plot_fit_metrics(df_all, dpath_out / f"{PREFIX_OUT}{suffix}.svg")

    # held-out subjects personalized on their first N visits
    if not df_forecast.empty:
        df_forecast = _select_forecast_after(df_forecast, forecast_after)
        _summarize(
            df_forecast,
            [COL_N_TIMEPOINTS, COL_FORECAST_AFTER, "set_name", "metric"],
            dpath_out / f"{PREFIX_OUT}{suffix}-forecast-summary.tsv",
        )
        _plot_fit_metrics_vs_n_timepoints(
            df_forecast, df_all, dpath_out / f"{PREFIX_OUT}{suffix}-forecast.svg"
        )

    # parameter recovery (only if ground truth is known)
    df_recovery = _collect(dpath_results, dataset, "*-param_recovery*.tsv")
    if df_recovery.empty:
        print("No parameter recovery files found")
        return
    fpath_all = dpath_out / f"{PREFIX_OUT_RECOVERY}{suffix}-all.tsv"
    df_recovery.to_csv(fpath_all, sep="\t", index=False)
    print(f"Saved {len(df_recovery)} rows to {fpath_all}")
    df_recovery, df_recovery_forecast = _split_n_timepoints(
        df_recovery.loc[df_recovery["col_biomarker"] == BIOMARKER_ALL],
        [
            set_name
            for set_name, _ in RECOVERY_METRICS
            if set_name.startswith(PREFIX_SET_NAME_RECOVERY_TEST)
        ],
    )
    _summarize(
        df_recovery,
        ["set_name", "metric"],
        dpath_out / f"{PREFIX_OUT_RECOVERY}{suffix}-summary.tsv",
    )
    _plot_recovery(df_recovery, dpath_out / f"{PREFIX_OUT_RECOVERY}{suffix}.svg")

    if not df_recovery_forecast.empty:
        df_recovery_forecast = _select_forecast_after(
            df_recovery_forecast, forecast_after
        )
        _summarize(
            df_recovery_forecast,
            [COL_N_TIMEPOINTS, COL_FORECAST_AFTER, "set_name", "metric"],
            dpath_out / f"{PREFIX_OUT_RECOVERY}{suffix}-forecast-summary.tsv",
        )
        _plot_recovery_vs_n_timepoints(
            df_recovery_forecast,
            df_recovery,
            dpath_out / f"{PREFIX_OUT_RECOVERY}{suffix}-forecast.svg",
        )


if __name__ == "__main__":
    main()
