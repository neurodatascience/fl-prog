#!/usr/bin/env python
"""Plot a schematic of the ADNI data splits.

Run on the output of ``get_adni_data.py``.
"""

from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch

from fl_prog.plotting import get_group_order, get_group_palette
from fl_prog.utils.constants import CLICK_CONTEXT_SETTINGS, SITE_COLORS
from fl_prog.utils.io import DEFAULT_DPATH_DATA, get_dpath_latest, load_json

DEFAULT_SEED = 1
# row label -> condition in the get_adni_data.py tag (see run_adni_seeds.sh)
CONDITIONS = {
    "Random\ngrouping": "iid",
    "Grouped by\noriginal site": "noniid_site",
    "Grouped by\ndiagnosis": "noniid_diag",
}
FIGSIZE = (5.12, 2.5)
DPI = 300
BAR_HEIGHT = 0.85
GAP = 0.4  # extra space between attribute bars and condition bars
FONTSIZE = 9
N_ATTRIBUTE_ROWS = 2


def _get_tag(condition: str, seed: int) -> str:
    # same as base_tag in run_adni_seeds.sh
    return f"adni_harmonized_{condition}_s{seed}"


def _normalize_rid(value) -> str:
    return str(value).strip().removesuffix(".0")


def _load_splits(
    dpath_data: Path, fpath_adni_merge: Path, seed: int
) -> tuple[pd.DataFrame, list[str]]:
    # subject -> node, for each condition
    node_by_condition = {}
    for label, condition in CONDITIONS.items():
        tag = _get_tag(condition, seed)
        subjects_by_node = load_json(dpath_data / tag / f"{tag}.json")[
            "subjects_by_node"
        ]
        node_by_condition[label] = pd.Series(
            {
                _normalize_rid(subject): int(node)
                for node, subjects in subjects_by_node.items()
                for subject in subjects
            }
        )
    df = pd.DataFrame(node_by_condition).dropna().astype(int)

    # subject -> ADNI site and baseline diagnosis
    df_adnimerge = pd.read_csv(fpath_adni_merge, dtype=str, low_memory=False)
    df_adnimerge["RID"] = df_adnimerge["RID"].map(_normalize_rid)
    df_adnimerge = df_adnimerge.drop_duplicates("RID").set_index("RID")
    df["Original site"] = df_adnimerge["SITE"].reindex(df.index)
    df["Diagnosis"] = df_adnimerge["DX_bl"].reindex(df.index)
    if df[["Original site", "Diagnosis"]].isna().any().any():
        raise ValueError(f"Subjects without site or diagnosis in {fpath_adni_merge}")

    group_order = get_group_order(df["Diagnosis"])
    df["_dx_rank"] = df["Diagnosis"].map(group_order.index)
    df["_site_num"] = df["Original site"].astype(int)
    df = df.sort_values(["_dx_rank", "_site_num"])
    # relabel nodes in order of first appearance (left to right), so that each split
    # starts with Site 1 (node IDs are arbitrary)
    for label in CONDITIONS:
        df[label] = df[label].map(
            {node: i for i, node in enumerate(df[label].unique(), start=1)}
        )
    print(f"{len(df)} subjects")
    for label in CONDITIONS:
        print(f"\n{label}")
        print(pd.crosstab(df[label], df["Diagnosis"])[group_order])
    return df, group_order


def _get_rows(df: pd.DataFrame, group_order: list[str]) -> tuple[list, dict]:
    node_colors = {i_node: color for i_node, color in enumerate(SITE_COLORS, start=1)}
    diagnosis_palette = get_group_palette(group_order)
    # ADNI site: one distinct categorical color per site (tab20 + tab20b + tab20c =
    # 60 colors), shuffled so that similar shades are unlikely to end up on
    # neighbouring sites
    sites = sorted(df["Original site"].unique(), key=int)
    site_colors = [
        color
        for cmap in ("tab20", "tab20b", "tab20c")
        for color in plt.get_cmap(cmap).colors
    ]
    if len(sites) > len(site_colors):
        raise ValueError(f"More sites ({len(sites)}) than colors ({len(site_colors)})")
    rng = np.random.default_rng(0)
    site_palette = {
        site: site_colors[i]
        for site, i in zip(sites, rng.permutation(len(site_colors)), strict=False)
    }

    rows = [
        ("Original site", [to_rgb(site_palette[site]) for site in df["Original site"]]),
        ("Diagnosis", [to_rgb(diagnosis_palette[dx]) for dx in df["Diagnosis"]]),
    ] + [
        (label, [to_rgb(node_colors[node]) for node in df[label]])
        for label in CONDITIONS
    ]
    return rows, node_colors


def plot_adni_data_split(
    dpath_data: Path, fpath_adni_merge: Path, seed: int, fpath_out: Path
):
    plt.rcParams["svg.fonttype"] = "none"
    df, group_order = _load_splits(dpath_data, fpath_adni_merge, seed)
    rows, node_colors = _get_rows(df, group_order)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    n_subjects = len(df)
    y_positions = []
    for i_row, (_, colors) in enumerate(rows):
        y = i_row + (GAP if i_row >= N_ATTRIBUTE_ROWS else 0)
        y_positions.append(y)
        ax.imshow(
            np.array(colors)[np.newaxis],
            extent=(0, n_subjects, y + BAR_HEIGHT / 2, y - BAR_HEIGHT / 2),
            aspect="auto",
            interpolation="nearest",
        )

    ax.set_xlim(0, n_subjects)
    ax.set_ylim(y_positions[-1] + 0.5, y_positions[0] - 0.5)
    ax.set_yticks(
        y_positions, [label for label, _ in rows], fontsize=FONTSIZE, linespacing=1.0
    )
    ax.set_xticks([])
    ax.tick_params(axis="y", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # label diagnosis segments directly (subjects are sorted by diagnosis)
    y_diagnosis = y_positions[[label for label, _ in rows].index("Diagnosis")]
    x_start = 0
    for dx, n_dx in df["Diagnosis"].value_counts()[group_order].items():
        ax.text(
            x_start + n_dx / 2,
            y_diagnosis,
            dx,
            ha="center",
            va="center",
            fontsize=FONTSIZE - 1,
            color="#262626",
        )
        x_start += n_dx

    ax.legend(
        handles=[
            Patch(color=color, label=f"Site {node}")
            for node, color in node_colors.items()
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.04),
        frameon=False,
        fontsize=FONTSIZE - 1,
        handlelength=1,
        handleheight=1,
        borderaxespad=0,
        ncols=5,
        columnspacing=1,
        handletextpad=0.4,
    )

    fpath_out.parent.mkdir(parents=True, exist_ok=True)
    for ext, dpi in (("svg", "figure"), ("png", DPI)):
        fpath = fpath_out.with_suffix(f".{ext}")
        fig.savefig(fpath, bbox_inches="tight", transparent=True, dpi=dpi)
        print(f"Saved {fpath}")
    plt.close(fig)


@click.command(context_settings=CLICK_CONTEXT_SETTINGS)
@click.option(
    "--data-dir",
    "dpath_data",
    type=click.Path(path_type=Path, file_okay=False, dir_okay=True),
    help="Dated data directory with the get_adni_data.py outputs. "
    "Default: latest dated directory in data/.",
)
@click.option(
    "--adni-merge",
    "fpath_adni_merge",
    type=click.Path(path_type=Path, file_okay=True, dir_okay=False, exists=True),
    required=True,
    envvar="ADNI_MERGE_FILE",
    help="ADNIMERGE.csv (for the original site and baseline diagnosis).",
)
@click.option(
    "--seed",
    type=int,
    default=DEFAULT_SEED,
    show_default=True,
    help="Seed of the splits to show (adni_harmonized_<condition>_s<seed>).",
)
@click.option(
    "--output",
    "fpath_out",
    type=click.Path(path_type=Path),
    help="Output path without extension (.svg and .png are written). "
    "Default: <data-dir>/figures/data_split-s<seed>.",
)
def main(
    dpath_data: Path | None, fpath_adni_merge: Path, seed: int, fpath_out: Path | None
):
    if dpath_data is None:
        dpath_data = get_dpath_latest(DEFAULT_DPATH_DATA)
    if fpath_out is None:
        fpath_out = dpath_data / "figures" / f"data_split-s{seed}"
    plot_adni_data_split(dpath_data, fpath_adni_merge, seed, fpath_out)


if __name__ == "__main__":
    main()
