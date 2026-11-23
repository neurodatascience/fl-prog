#!/usr/bin/env python
"""Plot ground-truth (noiseless) trajectories of simulated subjects by site.

Trajectories are shown on the disease-time axis, before the per-subject time shift
is removed (disease time = t0 + acceleration_factor * observed time). Run on the
output of ``simulate_data.py`` (held-out test subjects, if any, are not shown).
"""

import math
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.collections import LineCollection

from fl_prog.utils.constants import CLICK_CONTEXT_SETTINGS
from fl_prog.utils.io import (
    DEFAULT_DPATH_DATA,
    get_dpath_latest,
    load_json,
    load_simulated_sites,
)
from fl_prog.utils.math import multivariate_logistic

N_COLS = 3
PANEL_SIZE = 3
DPI = 200
LINE_WIDTH = 0.6
N_GRID = 200

sns.set_theme(context="paper", style="ticks")


def plot_ground_truth(dpath: Path, tag: str) -> Path:
    json_data = load_json(dpath / f"{tag}.json")
    params = json_data["params"]
    col_subject = json_data["cols"]["col_subject"]
    k_values = np.asarray(params["k_values"])
    x0_values = np.asarray(params["x0_values"])
    scaling_factors = np.asarray(params["scaling_factors"])
    n_biomarkers = len(k_values)
    biomarker_names = json_data.get(
        "biomarker_names", [f"biomarker_{i}" for i in range(n_biomarkers)]
    )

    df = load_simulated_sites(dpath, json_data)
    df = df.sort_values(["site", col_subject, "disease_time"])
    y_true = multivariate_logistic(
        df["disease_time"].to_numpy(), k_values, x0_values, scaling_factors
    )
    sites = sorted(df["site"].unique(), key=int)
    palette = dict(zip(sites, sns.color_palette("viridis", len(sites)), strict=True))
    n_subjects_site = df.groupby("site")[col_subject].nunique()
    alpha = float(np.clip(0.5 * np.sqrt(50 / n_subjects_site.max()), 0.05, 1))

    t_grid = np.linspace(0, df["disease_time"].max() * 1.05, N_GRID)
    y_grid = multivariate_logistic(t_grid, k_values, x0_values, scaling_factors)

    n_panels = n_biomarkers + 3
    n_rows = math.ceil(n_panels / N_COLS)
    fig, axes = plt.subplots(
        n_rows,
        N_COLS,
        figsize=(PANEL_SIZE * N_COLS, PANEL_SIZE * n_rows),
        squeeze=False,
        layout="constrained",
    )
    axes = axes.ravel()

    # one line per subject (subjects with a single visit are drawn as points)
    groups = list(df.groupby(["site", col_subject], sort=False).indices.values())
    for i_biomarker in range(n_biomarkers):
        ax = axes[i_biomarker]
        for site in sites:
            segments, points = [], []
            for idx in groups:
                if df["site"].iat[idx[0]] != site:
                    continue
                xy = np.column_stack(
                    (df["disease_time"].to_numpy()[idx], y_true[idx, i_biomarker])
                )
                (segments if len(idx) > 1 else points).append(xy)
            ax.add_collection(
                LineCollection(
                    segments,
                    colors=[palette[site]],
                    alpha=alpha,
                    linewidths=LINE_WIDTH,
                    zorder=2,
                )
            )
            if points:
                points = np.concatenate(points)
                ax.scatter(
                    points[:, 0],
                    points[:, 1],
                    color=palette[site],
                    alpha=alpha,
                    s=2,
                    zorder=2,
                )
        ax.plot(
            t_grid, y_grid[:, i_biomarker], color="black", lw=2.5, alpha=0.3, zorder=1
        )
        ax.axvline(x0_values[i_biomarker], color="grey", lw=0.5, ls=":")
        ax.set_title(
            f"{biomarker_names[i_biomarker]}\n"
            f"k={k_values[i_biomarker]:.2f}, x0={x0_values[i_biomarker]:.2f}, "
            f"s={scaling_factors[i_biomarker]:.2f}",
            fontsize="small",
        )
        ax.set_xlabel("Disease time (t0 + accel × t)")
        ax.set_xlim(t_grid[0], t_grid[-1])
        ax.autoscale(axis="y")

    # t0 and disease-time coverage by site
    df_subjects = df.groupby(["site", col_subject], sort=False).agg(
        t0=("t0", "first"),
        acceleration_factor=("acceleration_factor", "first"),
        t_end=("disease_time", "max"),
    )
    df_subjects = df_subjects.reset_index()
    ax = axes[n_biomarkers]
    sns.histplot(
        df_subjects,
        x="t0",
        hue="site",
        hue_order=sites,
        palette=palette,
        element="step",
        bins=40,
        ax=ax,
    )
    ax.set_title("True t0 (time shift) by site", fontsize="small")
    ax = axes[n_biomarkers + 1]
    sns.histplot(
        df_subjects,
        x="acceleration_factor",
        hue="site",
        hue_order=sites,
        palette=palette,
        element="step",
        bins=40,
        ax=ax,
        legend=False,
    )
    ax.set_title("True acceleration factor by site", fontsize="small")
    # disease-time span of each subject, sorted by t0
    ax = axes[n_biomarkers + 2]
    df_sorted = df_subjects.sort_values("t0").reset_index(drop=True)
    ax.hlines(
        df_sorted.index,
        df_sorted["t0"],
        df_sorted["t_end"],
        colors=[palette[site] for site in df_sorted["site"]],
        linewidths=LINE_WIDTH,
    )
    ax.scatter(
        df_sorted["t0"],
        df_sorted.index,
        c=[palette[site] for site in df_sorted["site"]],
        s=1,
    )
    ax.set_xlim(t_grid[0], t_grid[-1])
    ax.set_xlabel("Disease time (t0 + accel × t)")
    ax.set_ylabel("Subject (sorted by t0)")
    ax.set_title("Disease-time span of each subject", fontsize="small")

    for ax in axes[n_panels:]:
        ax.set_visible(False)

    sites = json_data["settings"]["sites"]
    fig.suptitle(
        f"{tag}\n{len(df_subjects)} subjects, {len(df)} visits, "
        f"sites={' '.join(sites) if sites else 'independent'}",
        fontsize="medium",
    )
    fpath_out = dpath / f"{tag}-ground_truth.png"
    fig.savefig(fpath_out, dpi=DPI)
    plt.close(fig)

    print(
        df_subjects.groupby("site")
        .agg(
            n_subjects=("t0", "size"),
            t0_min=("t0", "min"),
            t0_max=("t0", "max"),
            t_end_max=("t_end", "max"),
        )
        .round(2)
        .to_string()
    )
    print(f"Saved ground-truth plot to {fpath_out}")
    return fpath_out


@click.command(context_settings=CLICK_CONTEXT_SETTINGS)
@click.option("--tag", type=str, required=True)
@click.option(
    "--data-dir",
    "dpath_data",
    type=click.Path(path_type=Path, file_okay=False, dir_okay=True),
    default=DEFAULT_DPATH_DATA,
)
def main(tag: str, dpath_data: Path):
    plot_ground_truth(get_dpath_latest(dpath_data) / tag, tag)


if __name__ == "__main__":
    main()
