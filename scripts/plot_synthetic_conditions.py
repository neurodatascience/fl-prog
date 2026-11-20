#!/usr/bin/env python
"""Plot the synthetic-data conditions (e.g. overlap between sites).

One panel per tag (``simulate_data.py`` output), each showing the disease-time span of every
subject (sorted by t0), coloured by site. Held-out test subjects, if any, are not shown.
"""

from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D

from fl_prog.plotting import get_categorical_palette
from fl_prog.utils.constants import CLICK_CONTEXT_SETTINGS, SITE_COLORS
from fl_prog.utils.io import (
    DEFAULT_DPATH_DATA,
    get_dpath_latest,
    load_json,
    load_simulated_sites,
)

FIGSIZE = (6.4, 3.0)  # half the width of a 16:9 slide (13.33 in)
DPI = 300
LINE_WIDTH = 0.3


def _get_title(json_data: dict, tag: str) -> str:
    overlap = json_data.get("settings", {}).get("overlap")
    return tag if overlap is None else f"{overlap:.0%} overlap"


def _get_site_palette(sites: list[str]) -> dict:
    if len(sites) <= len(SITE_COLORS):
        return dict(zip(sites, SITE_COLORS, strict=False))
    return get_categorical_palette(sites)


def _load_subject_spans(dpath: Path, json_data: dict):
    col_subject = json_data["cols"]["col_subject"]
    df = load_simulated_sites(dpath, json_data)
    df_subjects = (
        df.groupby(["site", col_subject], sort=False)
        .agg(t0=("t0", "first"), t_end=("disease_time", "max"))
        .reset_index()
        .sort_values("t0")
        .reset_index(drop=True)
    )
    df_subjects["y"] = df_subjects.index + 1  # subjects numbered from 1
    return df_subjects


def plot_conditions(dpath_data: Path, tags: list[str], fpath_out: Path):
    sns.set_theme(context="talk", style="ticks")
    dfs, titles = {}, {}
    for tag in tags:
        dpath = dpath_data / tag
        json_data = load_json(dpath / f"{tag}.json")
        dfs[tag] = _load_subject_spans(dpath, json_data)
        titles[tag] = _get_title(json_data, tag)
    sites = sorted({site for df in dfs.values() for site in df["site"]}, key=int)
    palette = _get_site_palette(sites)
    n_subjects = max(len(df) for df in dfs.values())
    t_max = max(df["t_end"].max() for df in dfs.values())

    fig, axes = plt.subplots(
        1,
        len(tags),
        figsize=FIGSIZE,
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    axes = np.atleast_1d(axes)
    for ax, tag in zip(axes, tags, strict=True):
        df = dfs[tag]
        ax.hlines(
            df["y"],
            df["t0"],
            df["t_end"],
            colors=[palette[site] for site in df["site"]],
            linewidths=LINE_WIDTH,
        )
        ax.set_title(titles[tag], fontsize="medium")
        ax.set_xticks([])
        sns.despine(ax=ax)

    ax = axes[0]
    ax.set_xlim(0, t_max * 1.02)
    ax.set_ylim(1, n_subjects)
    ax.set_yticks([1, n_subjects])
    ax.set_ylabel("Subject", labelpad=-20)
    # on the middle panel (not fig.supxlabel) so that the gap to the x-axis matches
    # the gap between the titles and the axes (labelpad tuned on the rendered figure)
    axes[len(axes) // 2].set_xlabel("Disease time", fontsize="medium", labelpad=5)
    ax.legend(
        handles=[Line2D([], [], color=palette[site], lw=3) for site in sites],
        labels=sites,
        title="Site",
        loc="upper left",
        frameon=False,
        fontsize="x-small",
        title_fontsize="x-small",
        handlelength=0.8,
        handletextpad=0.4,
        labelspacing=0.2,
        borderaxespad=0.1,
    )

    fpath_out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fpath = fpath_out.with_suffix(f".{ext}")
        fig.savefig(fpath, dpi=DPI)
        print(f"Saved {fpath}")
    plt.close(fig)


@click.command(context_settings=CLICK_CONTEXT_SETTINGS)
@click.option(
    "--tag",
    "tags",
    type=str,
    multiple=True,
    required=True,
    help="Tag of a simulate_data.py output.",
)
@click.option(
    "--data-dir",
    "dpath_data",
    type=click.Path(path_type=Path, file_okay=False, dir_okay=True),
    help="Dated data directory. Default: latest dated directory in data/.",
)
@click.option(
    "--output",
    "fpath_out",
    type=click.Path(path_type=Path),
    help="Output path without extension (.png and .pdf are written). "
    "Default: <data-dir>/figures/conditions-<first tag>.",
)
def main(tags: tuple[str, ...], dpath_data: Path | None, fpath_out: Path | None):
    if dpath_data is None:
        dpath_data = get_dpath_latest(DEFAULT_DPATH_DATA)
    if fpath_out is None:
        fpath_out = dpath_data / "figures" / f"conditions-{tags[0]}"
    plot_conditions(dpath_data, list(tags), fpath_out)


if __name__ == "__main__":
    main()
