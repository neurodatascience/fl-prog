#!/usr/bin/env python
"""Schematic figure: individual parameters map each subject's data into disease time.

Mirrors ``LogisticRegressionModelWithShift.forward`` in ``fl_prog/model.py``::

    b_hat_ij(t) = s_j * sigmoid(k_j * (phi_i(t) - m_j)),  phi_i(t) = alpha_i * t + tau_i

with t the time since the subject's baseline visit. Toy values, one biomarker and three
subjects: progressing slower than average (alpha < 1), a bit faster and faster, each with
their own visit times.

Panel 1 (subject space): each subject's data, in time since baseline.
Panel 2 (time reparametrization): curved arrow for the acceleration (t -> alpha * t),
         then straight arrow for the time shift (alpha * t -> alpha * t + tau), drawn
         at each subject's last visit.
         Earlier positions of the data are faded.
Panel 3 (model space): data in disease time, with the population trajectory.
"""

from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from fl_prog.plotting import (
    CATEGORICAL_COLORS,
    PARAM_COLORS,
    PARAM_SYMBOLS,
    get_axes_size_for_labels,
    make_fixed_size_axes,
    save_fig_formats,
)
from fl_prog.utils.constants import CLICK_CONTEXT_SETTINGS
from fl_prog.utils.io import DEFAULT_DPATH_RESULTS
from fl_prog.utils.math import multivariate_logistic

DEFAULT_DPATH_OUT = DEFAULT_DPATH_RESULTS / "figures" / "individual_params"

# population parameters (one biomarker)
K = 1.5  # steepness
M = 5.0  # midpoint
S = 1.0  # scaling

# per subject: acceleration, time shift and visit times (years since baseline)
SUBJECTS = [
    (0.6, 2.5, [0.0, 1.3, 2.9, 4.4]),  # slower, earlier in the disease
    (1.35, 2.7, [0.0, 0.6, 1.4, 2.3]),  # a bit faster, in between the other two
    (1.5, 4.0, [0.0, 0.9, 2.0, 3.2]),  # faster, later in the disease
]
NOISE_SD = 0.01

# axes (same for all panels)
T_LIM = (-0.5, 9.7)
Y_LIM = (-0.1, 1.1)
XLABELS = ["Time since\nbaseline", "", "Disease time"]  # middle panel mixes time scales
TITLES = ["Subject space", "", "Model space"]
FNAMES = ["1_subject_space", "2_reparametrization", "3_model_space"]
YLABEL = "Biomarker"

# style
SUBJECT_COLORS = [CATEGORICAL_COLORS[i] for i in (0, 3, 2)]  # blue, purple, green
COLOR_POPULATION = "#c4c4c4"  # light grey: stays in the background
COLOR_ALPHA = PARAM_COLORS["alpha"]
COLOR_TAU = PARAM_COLORS["tau"]
ALPHA_ORIGINAL = 0.2  # data in time since baseline, in the reparametrization panel
ALPHA_ACCELERATED = 0.45  # data after acceleration only
ALPHA_ARROW = 0.85
FS_PARAM = 20
MARKER_SIZE = 45
LW_DATA = 1.8
LW_POPULATION = 3.5
DPI = 600

sns.set_theme(context="talk", style="ticks")


def population(phi):
    """Population trajectory of the single biomarker."""
    return multivariate_logistic(phi, [K], [M], [S])[:, 0]


def simulate_subjects(rng):
    """List of (t, alpha * t, alpha * t + tau, y) arrays, one per subject."""
    subjects = []
    for alpha, tau, visits in SUBJECTS:
        t = np.asarray(visits)
        phi = alpha * t + tau
        y = population(phi) + rng.normal(0, NOISE_SD, len(t))
        subjects.append((t, alpha * t, phi, y))
    return subjects


def plot_data(ax, t, y, color, alpha=1.0, line=True):
    if line:
        ax.plot(t, y, color=color, lw=LW_DATA, alpha=0.5 * alpha, zorder=3)
    ax.scatter(t, y, s=MARKER_SIZE, color=color, lw=0, alpha=alpha, zorder=3)


def arrow(ax, x_from, x_to, y, color, rad=0.0):
    if np.isclose(x_from, x_to):
        return
    ax.annotate(
        "",
        xy=(x_to, y),
        xytext=(x_from, y),
        arrowprops={
            "arrowstyle": "-|>,head_length=0.25,head_width=0.12",
            "color": color,
            "alpha": ALPHA_ARROW,
            "lw": 1.5,
            "shrinkA": 3,
            "shrinkB": 3,
            "connectionstyle": f"arc3,rad={rad}",
        },
        zorder=4,
    )


def plot_panel(ax, step: int, subjects, ylabel=True):
    """step 0: subject space; 1: time reparametrization; 2: model space."""
    for (t, t_accelerated, phi, y), color in zip(subjects, SUBJECT_COLORS):
        if step == 0:
            plot_data(ax, t, y, color)
            continue
        plot_data(ax, phi, y, color)
        if step == 2:
            continue
        plot_data(ax, t, y, color, alpha=ALPHA_ORIGINAL)
        plot_data(ax, t_accelerated, y, color, alpha=ALPHA_ACCELERATED)
        # arc above the data: arc3 bends to the left of the direction of travel
        rad = -1.0 * np.sign(t_accelerated[-1] - t[-1])
        arrow(ax, t[-1], t_accelerated[-1], y[-1], COLOR_ALPHA, rad=rad)
        arrow(ax, t_accelerated[-1], phi[-1], y[-1], COLOR_TAU)

    if step == 1:
        for i, (label, color) in enumerate(
            (
                (rf"$\times\,{PARAM_SYMBOLS['alpha']}$", COLOR_ALPHA),
                (rf"$+\,{PARAM_SYMBOLS['tau']}$", COLOR_TAU),
            )
        ):
            ax.text(
                0.7, 0.32 - 0.17 * i, label, transform=ax.transAxes, fontsize=FS_PARAM,
                color=color, ha="left", va="center",
            )  # fmt: skip

    if step == 2:
        phi = np.linspace(*T_LIM, 300)
        ax.plot(
            phi, population(phi), color=COLOR_POPULATION, lw=LW_POPULATION, zorder=1
        )

    ax.set_title(TITLES[step], va="bottom")
    ax.set_xlabel(
        XLABELS[step], labelpad=1 if step == 0 else None
    )  # close to the 0 tick
    if ylabel:
        ax.set_ylabel(YLABEL)
    ax.set_xlim(*T_LIM)
    ax.set_ylim(*Y_LIM)
    ax.set_xticks([0] if step == 0 else [])  # baseline visit in subject space
    ax.tick_params(axis="x", length=4, pad=1)
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
@click.option("--seed", type=int, default=3, show_default=True)
@click.option("--transparent", is_flag=True, help="Transparent background.")
def plot_presentation_individual(dpath_out: Path, seed: int, transparent: bool):
    plt.rcParams["mathtext.fontset"] = "cm"
    subjects = simulate_subjects(np.random.default_rng(seed))

    # axes just big enough for the axis labels to (almost) span them
    ax_width, ax_height = get_axes_size_for_labels(XLABELS, YLABEL)

    def save(fig, fname):
        save_fig_formats(fig, dpath_out / fname, dpi=DPI, transparent=transparent)

    # combined figure
    fig, axes = make_fixed_size_axes(len(TITLES), ax_width, ax_height)
    for step, ax in enumerate(axes):
        plot_panel(ax, step, subjects, ylabel=step == 0)
    # the first panel's x label sits below its 0 tick: put the last one at the same height
    fig.canvas.draw()
    label_top = axes[0].xaxis.label.get_window_extent().y1
    xlabel_y = axes[0].transAxes.inverted().transform((0, label_top))[1]
    axes[-1].xaxis.set_label_coords(0.5, xlabel_y)
    save(fig, "individual_params_all")

    # one figure per panel (same label height, so panels line up when placed side by side)
    for step, fname in enumerate(FNAMES):
        fig, (ax,) = make_fixed_size_axes(1, ax_width, ax_height)
        plot_panel(ax, step, subjects)
        if step == len(FNAMES) - 1:
            ax.xaxis.set_label_coords(0.5, xlabel_y)
        save(fig, f"individual_params_{fname}")


if __name__ == "__main__":
    plot_presentation_individual()
