#!/usr/bin/env python
"""Collect fit metrics from a multi-seed run (see ``run_adni_seeds.sh``).

Expected layout (after ``consolidate_seed_results.py``)::

    <results-dir>/<tag>/<model type>-*-fit_quality*.tsv

where ``<tag>`` looks like ``adni_harmonized_<condition>_s<seed>_...``.
"""

import re
from pathlib import Path

import click
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from fl_prog.plotting import get_categorical_palette, save_fig
from fl_prog.utils.constants import CLICK_CONTEXT_SETTINGS

# (model type, data condition, setup) -> model label
MODELS = {
    ("fedbiomed", "iid", "centralized"): "Fed-BioMed centralized",
    ("fedbiomed", "iid", "federated"): "Fed-BioMed IID",
    ("fedbiomed", "noniid_site", "federated"): "Fed-BioMed non-IID (site)",
    ("fedbiomed", "noniid_diag", "federated"): "Fed-BioMed non-IID (diag.)",
    ("dpmost", "iid", "centralized"): "DPMoSt",
    ("leaspy", "iid", "centralized"): "Leaspy",
}
METRICS = {"mean_absolute_error": "MAE", "r2_score": "R²"}
BIOMARKER_ALL = "all"
SET_NAME_ORDER = ["train", "test"]
PATTERN_TAG = re.compile(r"^adni_harmonized_(?P<condition>.+?)_s(?P<seed>\d+)_")
DNAME_FIGURES = "figures"
PREFIX_OUT = "fit_metrics-seeds"


def _collect(dpath_results: Path) -> pd.DataFrame:
    dfs = []
    for fpath in sorted(dpath_results.glob("*/*-fit_quality*.tsv")):
        tag = fpath.parent.name
        match = PATTERN_TAG.search(tag)
        if match is None:
            continue  # not a multi-seed run
        model_type = fpath.name.split("-", 1)[0]
        condition = match.group("condition")
        df = pd.read_csv(fpath, sep="\t")
        for setup, df_setup in df.groupby("setup"):
            model = MODELS.get((model_type, condition, setup))
            if model is None:
                click.secho(
                    f"WARNING: ignoring {model_type}/{condition}/{setup} ({fpath})",
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
        raise RuntimeError(f"No multi-seed fit_quality files found in {dpath_results}")
    return pd.concat(dfs, ignore_index=True)


@click.command(context_settings=CLICK_CONTEXT_SETTINGS)
@click.option(
    "--results-dir",
    "dpath_results",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    required=True,
    help="Dated results directory, e.g. results/2026_11_08",
)
def main(dpath_results: Path):
    dpath_out = dpath_results / DNAME_FIGURES
    dpath_out.mkdir(parents=True, exist_ok=True)

    df = _collect(dpath_results)
    fpath_all = dpath_out / f"{PREFIX_OUT}-all.tsv"
    df.to_csv(fpath_all, sep="\t", index=False)
    print(f"Saved {len(df)} rows to {fpath_all}")

    model_order = [m for m in MODELS.values() if m in set(df["model"])]
    df_all = df.loc[
        (df["col_biomarker"] == BIOMARKER_ALL) & df["metric"].isin(METRICS)
    ].copy()

    df_summary = (
        df_all.groupby(["model", "set_name", "metric"])["value"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    df_summary["model"] = pd.Categorical(df_summary["model"], model_order)
    df_summary = df_summary.sort_values(["metric", "set_name", "model"])
    fpath_summary = dpath_out / f"{PREFIX_OUT}-summary.tsv"
    df_summary.to_csv(fpath_summary, sep="\t", index=False)
    print(f"Saved summary to {fpath_summary}")
    print(df_summary.to_string(index=False))

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
            ax.set_title(f"{set_name.capitalize()} {metric_label}")
            ax.set_xlabel("")
            ax.set_ylabel(metric_label if i_col == 0 else "")
            ax.tick_params(axis="x", labelrotation=45)
            for label in ax.get_xticklabels():
                label.set_horizontalalignment("right")
    fig.tight_layout()
    save_fig(fig, dpath_out / f"{PREFIX_OUT}.svg")


if __name__ == "__main__":
    main()
