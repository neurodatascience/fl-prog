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

from fl_prog.metrics import average_over_biomarkers
from fl_prog.plotting import get_categorical_palette, save_fig
from fl_prog.utils.constants import CLICK_CONTEXT_SETTINGS

DATASET_ADNI = "adni_harmonized"
DATASET_SYNTHETIC = "synth_adnilike"
# same simulation with wider true time shifts (disease coverage up to late biomarkers)
DATASET_SYNTHETIC_FULLCOV = "synth_fullcov"
DATASETS = [DATASET_ADNI, DATASET_SYNTHETIC, DATASET_SYNTHETIC_FULLCOV]

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
    ("fedbiomed", "overlap100", "federated"): "Fed-BioMed 100% overlap",
    ("fedbiomed", "overlap050", "federated"): "Fed-BioMed 50% overlap",
    (
        "fedbiomed",
        "overlap050",
        "federated",
        "scaffold_native_sgd",
    ): "Fed-BioMed 50% overlap (SCAFFOLD)",
    (
        "fedbiomed",
        "overlap050",
        "federated",
        "fedavg_sgd",
    ): "Fed-BioMed 50% overlap (FedAvg, SGD)",
    (
        "fedbiomed",
        "overlap050",
        "federated",
        "scaffold_adam",
    ): "Fed-BioMed 50% overlap (SCAFFOLD, Adam)",
    ("fedbiomed", "overlap000", "federated"): "Fed-BioMed 0% overlap",
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
PATTERN_TAG = re.compile(
    rf"^(?P<dataset>{'|'.join(DATASETS)})_(?P<condition>.+?)_s(?P<seed>\d+)_"
)
# Fed-BioMed run tag: fedbiomed[-<setups>]-...-<scaling|no_scaling>-<aggregator[_optimizer][_locallrX]>-<seed>-
PATTERN_FBM_AGGREGATOR = re.compile(
    r"-(?:no_)?scaling-(?P<aggregator>[^-]+)-(?:\d+|no_seed)-"
)
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
            df_setup["fpath"] = str(fpath)
            dfs.append(df_setup)
    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    # a model/condition/seed must come from a single file (e.g. not from both a run
    # with all setups and a run with only one of them, or from two personalization
    # seeds), otherwise it would be counted more than once
    n_files = df.groupby(["model", "condition", "seed"])["fpath"].nunique()
    if (n_files > 1).any():
        duplicates = n_files[n_files > 1].index.tolist()
        raise click.ClickException(
            f"Results for {duplicates} come from more than one file: "
            + ", ".join(
                sorted(
                    set(
                        df.set_index(["model", "condition", "seed"])
                        .loc[duplicates, "fpath"]
                        .tolist()
                    )
                )
            )
        )
    return df


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
def main(dpath_results: Path, dataset: str | None):
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

    # mean over biomarkers (also for results where "all" was pooled)
    df = average_over_biomarkers(df)
    df_all = df.loc[
        (df["col_biomarker"] == BIOMARKER_ALL) & df["metric"].isin(METRICS)
    ].copy()
    _summarize(
        df_all, ["set_name", "metric"], dpath_out / f"{PREFIX_OUT}{suffix}-summary.tsv"
    )
    _plot_fit_metrics(df_all, dpath_out / f"{PREFIX_OUT}{suffix}.svg")

    # parameter recovery (only if ground truth is known)
    df_recovery = _collect(dpath_results, dataset, "*-param_recovery*.tsv")
    if df_recovery.empty:
        print("No parameter recovery files found")
        return
    fpath_all = dpath_out / f"{PREFIX_OUT_RECOVERY}{suffix}-all.tsv"
    df_recovery.to_csv(fpath_all, sep="\t", index=False)
    print(f"Saved {len(df_recovery)} rows to {fpath_all}")
    df_recovery = df_recovery.loc[df_recovery["col_biomarker"] == BIOMARKER_ALL]
    _summarize(
        df_recovery,
        ["set_name", "metric"],
        dpath_out / f"{PREFIX_OUT_RECOVERY}{suffix}-summary.tsv",
    )
    _plot_recovery(df_recovery, dpath_out / f"{PREFIX_OUT_RECOVERY}{suffix}.svg")


if __name__ == "__main__":
    main()
