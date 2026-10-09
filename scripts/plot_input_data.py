#!/usr/bin/env python
import math
import warnings
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from fl_prog.utils.constants import CLICK_CONTEXT_SETTINGS
from fl_prog.utils.io import (
    DEFAULT_DPATH_DATA,
    get_dpath_latest,
    get_node_id_map,
    infer_sep,
    load_json,
)

N_COLS = 3
PANEL_SIZE = 2.5
DPI = 300
ALPHA_REF = 0.5  # alpha for a dataset with ALPHA_N_SUBJECTS_REF subjects
ALPHA_N_SUBJECTS_REF = 350
ALPHA_MIN = 0.05

GROUP_ORDER = ["CN", "SMC", "EMCI", "LMCI", "AD"]

sns.set_theme(context="paper", style="ticks")


def _get_alpha(n_subjects: int) -> float:
    # inverse square-root scaling: fade lines as subjects increase, but not too much
    alpha = ALPHA_REF * np.sqrt(ALPHA_N_SUBJECTS_REF / max(n_subjects, 1))
    return float(np.clip(alpha, ALPHA_MIN, 1.0))


def _load_df(fpath: Path, col_subject: str, col_timepoint: str) -> pd.DataFrame:
    df = pd.read_csv(fpath, sep=infer_sep(fpath))
    return df.sort_values([col_subject, col_timepoint])


def _plot_node(
    df: pd.DataFrame,
    col_subject: str,
    col_timepoint: str,
    cols_biomarker: list[str],
    col_group: str,
    hue_order: list,
) -> plt.Figure:
    alpha = _get_alpha(df[col_subject].nunique())

    n_rows = math.ceil(len(cols_biomarker) / N_COLS)
    fig, axes = plt.subplots(
        n_rows,
        N_COLS,
        figsize=(PANEL_SIZE * N_COLS, PANEL_SIZE * n_rows),
        squeeze=False,
        layout="constrained",
    )
    axes = axes.flatten()

    for i_ax, (ax, col_biomarker) in enumerate(zip(axes, cols_biomarker)):
        sns.lineplot(
            data=df,
            x=col_timepoint,
            y=col_biomarker,
            hue=col_group,
            hue_order=hue_order,
            units=col_subject,
            estimator=None,
            linewidth=0.5,
            alpha=alpha,
            legend=(i_ax == 0),
            ax=ax,
        )
        # keep text/axes as vectors but rasterize the (many) subject lines
        for line in ax.get_lines():
            line.set_rasterized(True)
        ax.set_ylabel("")
        ax.set_title(col_biomarker)

    # one legend for the whole figure
    legend = axes[0].get_legend()
    if legend is not None:
        handles = legend.legend_handles
        labels = [text.get_text() for text in legend.get_texts()]
        legend.remove()
        fig_legend = fig.legend(
            handles, labels, title=col_group, loc="outside right center"
        )
        for handle in fig_legend.legend_handles:
            handle.set_alpha(1)
            handle.set_linewidth(1)

    for ax in axes[len(cols_biomarker) :]:
        ax.set_visible(False)

    return fig


@click.command(context_settings=CLICK_CONTEXT_SETTINGS)
@click.option("--tag", type=str, required=True)
@click.option(
    "--data-dir",
    "dpath_data",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    default=DEFAULT_DPATH_DATA,
)
def plot_input_data(tag: str, dpath_data: Path):
    """Plot model input data: one figure per node, one panel per biomarker."""
    dpath_tag = get_dpath_latest(dpath_data) / tag
    fpath_json = dpath_tag / f"{tag}.json"

    node_id_map = get_node_id_map(fpath_json)
    cols = load_json(fpath_json)["cols"]
    col_subject = cols["col_subject"]
    col_timepoint = cols["col_timepoint"]
    cols_biomarker = cols["cols_biomarker"]
    col_group = cols["col_group"]

    dfs = {}
    for filename, node_id in node_id_map.items():
        fpath = dpath_tag / filename
        if not fpath.exists():
            warnings.warn(f"{fpath} not found, skipping node {node_id}")
            continue
        dfs[node_id] = _load_df(fpath, col_subject, col_timepoint)

    for node_id, df in dfs.items():
        fig = _plot_node(
            df,
            col_subject=col_subject,
            col_timepoint=col_timepoint,
            cols_biomarker=cols_biomarker,
            col_group=col_group,
            hue_order=GROUP_ORDER,
        )
        fpath_out = dpath_tag / f"{tag}-node_{node_id}.svg"
        fig.savefig(fpath_out, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        click.secho(
            f"Saved {fpath_out} (n_subjects={df[col_subject].nunique()}, alpha={_get_alpha(df[col_subject].nunique()):.2f})",
            fg="green",
        )


if __name__ == "__main__":
    plot_input_data()
