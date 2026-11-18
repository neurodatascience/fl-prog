#!/usr/bin/env python
"""Schematic figure: fitting a disease model per site vs. on all sites together.

Three simulated sites (one colour each), each limited in a different way:
- small: few subjects
- short follow-up: longitudinal, but visits close together in a narrow disease range
- early disease only

Panel 1: a sigmoid model fitted separately to each site (line in the site's colour).
Panel 2: a single model fitted on all sites together (per-site fits faded for reference).
Both panels show the ground-truth trajectory (dashed) in the background.
"""

import warnings
from pathlib import Path

import click
import numpy as np
import seaborn as sns
from scipy.optimize import OptimizeWarning, curve_fit

from fl_prog.plotting import (
    get_axes_size_for_labels,
    make_fixed_size_axes,
    save_fig_formats,
)
from fl_prog.utils.constants import CLICK_CONTEXT_SETTINGS
from fl_prog.utils.io import DEFAULT_DPATH_RESULTS

DEFAULT_DPATH_OUT = DEFAULT_DPATH_RESULTS / "figures" / "motivation"

# ground truth trajectory
T_MIN, T_MAX = 0.0, 10.0
TRUE_PARAMS = (0.0, 1.0, 1.5, 5.0)  # lo, hi, steepness, midpoint
NOISE_SD = 0.18
VISIT_INTERVAL = 0.8  # default time between visits

# model fitting
P0 = (0.0, 1.0, 1.0, 5.0)
BOUNDS = ([-0.3, 0.3, 0.1, -5.0], [0.3, 2.5, 3.5, 20.0])

# style
SITE_COLORS = ["#F8D75B", "#EF9853", "#58BB5E"]  # yellow, orange, green
COLOR_POOLED = "#6793E9"  # blue
Y_LIM = (-0.5, 1.5)
MARKER_SIZE = 45
LW_DATA = 1.8
LW_FIT = 4.5
COLOR_TRUTH = "#222222"
LW_TRUTH = 3.5
ALPHA_SITE_FIT_REF = 0.3  # per-site fits shown for reference in the pooled panel
DPI = 600

sns.set_theme(context="talk", style="ticks")


def sigmoid(t, lo, hi, k, t0):
    return lo + (hi - lo) / (1 + np.exp(-k * (t - t0)))


def simulate_longitudinal(
    rng,
    n_subjects,
    n_visits,
    t_min=T_MIN,
    t_max=T_MAX,
    visit_interval=VISIT_INTERVAL,
    stratify: bool = False,
):
    """List of (t, y) arrays, one per subject, in disease time.

    If stratify is True, the possible start times are split into n_subjects equal bins
    and one subject starts in each bin, so the subjects spread across the whole range.
    """
    span = (n_visits - 1) * visit_interval
    if stratify:
        bin_width = (t_max - span - t_min) / n_subjects
        t_starts = t_min + bin_width * (
            np.arange(n_subjects) + rng.uniform(size=n_subjects)
        )
    else:
        t_starts = rng.uniform(t_min, t_max - span, n_subjects)
    subjects = []
    for t_start in t_starts:
        t = t_start + visit_interval * np.arange(n_visits)
        y = sigmoid(t, *TRUE_PARAMS) + rng.normal(0, NOISE_SD, n_visits)
        subjects.append((t, y))
    return subjects


def fit_sigmoid(subjects):
    t = np.concatenate([t for t, _ in subjects])
    y = np.concatenate([y for _, y in subjects])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OptimizeWarning)
        params, _ = curve_fit(sigmoid, t, y, p0=P0, bounds=BOUNDS, maxfev=10000)
    return params


def plot_panel(ax, sites, pooled: bool, xlabel, ylabel=None):
    """Plot site data, then either one fit per site or a single fit on all sites."""
    t_grid = np.linspace(T_MIN, T_MAX, 300)

    # ground truth, behind everything else
    ax.plot(
        t_grid,
        sigmoid(t_grid, *TRUE_PARAMS),
        color=COLOR_TRUTH,
        lw=LW_TRUTH,
        ls="--",
        zorder=0,
    )

    for subjects, color in zip(sites, SITE_COLORS):
        for t, y in subjects:
            if len(t) > 1:
                ax.plot(t, y, color=color, lw=LW_DATA, alpha=0.5, zorder=1)
            ax.scatter(t, y, s=MARKER_SIZE, color=color, alpha=0.7, lw=0, zorder=1)
        # per-site fit (faded for reference when showing the pooled fit)
        params = fit_sigmoid(subjects)
        ax.plot(
            t_grid,
            sigmoid(t_grid, *params),
            color=color,
            lw=LW_FIT,
            alpha=ALPHA_SITE_FIT_REF if pooled else 1.0,
            zorder=2,
        )

    if pooled:
        params = fit_sigmoid([subject for subjects in sites for subject in subjects])
        ax.plot(
            t_grid, sigmoid(t_grid, *params), color=COLOR_POOLED, lw=LW_FIT, zorder=3
        )

    ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.set_xlim(T_MIN, T_MAX)
    ax.set_ylim(*Y_LIM)
    ax.set_xticks([])
    ax.set_yticks([])
    sns.despine(ax=ax)


@click.command(context_settings=CLICK_CONTEXT_SETTINGS)
@click.option(
    "--output-dir",
    "dpath_out",
    type=click.Path(path_type=Path, file_okay=False),
    default=DEFAULT_DPATH_OUT,
    show_default=True,
    help="Output directory (PNG and SVG saved for the combined figure and each panel).",
)
@click.option("--seed", type=int, default=8, show_default=True)
@click.option("--transparent", is_flag=True, help="Transparent background.")
def plot_presentation_motivation(dpath_out: Path, seed: int, transparent: bool):
    rng = np.random.default_rng(seed)
    ylabel = "Biomarker"
    xlabel = "Disease time"

    sites = [
        # small: few subjects, spread over the disease course
        simulate_longitudinal(rng, n_subjects=2, n_visits=3, stratify=True),
        # short follow-up: visits close together, narrow disease range
        simulate_longitudinal(
            rng, n_subjects=4, n_visits=4, t_min=5.0, t_max=7.5, visit_interval=0.3
        ),
        # early disease only
        simulate_longitudinal(rng, n_subjects=4, n_visits=4, t_min=0.0, t_max=3.5),
    ]

    panels = {
        "per_site": {"sites": sites, "pooled": False, "xlabel": xlabel},
        "pooled": {"sites": sites, "pooled": True, "xlabel": xlabel},
    }

    # axes just big enough for the axis labels to (almost) span them
    ax_width, ax_height = get_axes_size_for_labels([xlabel], ylabel)

    def save(fig, fname):
        save_fig_formats(fig, dpath_out / fname, dpi=DPI, transparent=transparent)

    # combined figure
    fig, axes = make_fixed_size_axes(len(panels), ax_width, ax_height)
    for i_panel, (ax, panel) in enumerate(zip(axes, panels.values())):
        plot_panel(ax, **panel, ylabel=ylabel if i_panel == 0 else None)
    save(fig, "motivation_all")

    # one figure per panel
    for i_panel, (name, panel) in enumerate(panels.items(), start=1):
        fig, (ax,) = make_fixed_size_axes(1, ax_width, ax_height)
        plot_panel(ax, **panel, ylabel=ylabel)
        save(fig, f"motivation_{i_panel}_{name}")


if __name__ == "__main__":
    plot_presentation_motivation()
