"""Shared plotting helpers."""

from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

# ADNI baseline diagnoses, in disease order
DIAGNOSIS_ORDER = ["CN", "SMC", "EMCI", "LMCI", "AD"]

# colorblind-checked categorical palette (fixed order, never cycled)
CATEGORICAL_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#e87ba4"]
CATEGORICAL_PALETTE_FALLBACK = "husl"  # for more categories than CATEGORICAL_COLORS
DIAGNOSIS_PALETTE = "Set2"
COLOR_OTHER = "#8a8a85"

PARAM_COLORS = {
    "k": "#1971c2",  # steepness
    "m": "#0c8599",  # midpoint
    "s": "#7048e8",  # scaling
    "sigma": "#2f9e44",  # noise standard deviation
    "tau": "#e8590c",  # time shift
    "alpha": "#d6336c",  # acceleration
}
PARAM_SYMBOLS = {
    "k": r"\boldsymbol{k}_{\boldsymbol{j}}",
    "m": r"\boldsymbol{m}_{\boldsymbol{j}}",
    "s": r"\boldsymbol{s}_{\boldsymbol{j}}",
    "sigma": r"\boldsymbol{\sigma}_{\boldsymbol{j}}",
    "tau": r"\boldsymbol{\tau}_{\boldsymbol{i}}",
    "alpha": r"\boldsymbol{\alpha}_{\boldsymbol{i}}",
}


def get_group_order(groups: Iterable) -> list[str]:
    """Known diagnoses in disease order, followed by any other labels."""
    groups = {str(group) for group in groups}
    return [group for group in DIAGNOSIS_ORDER if group in groups] + sorted(
        groups - set(DIAGNOSIS_ORDER)
    )


def get_group_palette(group_order: list[str]) -> dict[str, tuple]:
    """Fixed color per known diagnosis; other labels take the remaining colors."""
    colors = sns.color_palette(DIAGNOSIS_PALETTE)
    other_colors = iter(colors[len(DIAGNOSIS_ORDER) :])
    return {
        group: (
            colors[DIAGNOSIS_ORDER.index(group)]
            if group in DIAGNOSIS_ORDER
            else next(other_colors, COLOR_OTHER)
        )
        for group in group_order
    }


def get_biomarker_palette(biomarker_order: list[str]) -> dict[str, tuple]:
    """Seaborn's default palette, in biomarker order."""
    return dict(zip(biomarker_order, sns.color_palette(n_colors=len(biomarker_order))))


def get_categorical_palette(labels: list[str]) -> dict[str, str | tuple]:
    """Colorblind-checked colors if there are few enough labels.

    Otherwise, evenly spaced hues (one distinct color per label).
    """
    if len(labels) > len(CATEGORICAL_COLORS):
        colors = sns.color_palette(CATEGORICAL_PALETTE_FALLBACK, n_colors=len(labels))
    else:
        colors = CATEGORICAL_COLORS
    return dict(zip(labels, colors))


def get_axes_size_for_labels(
    xlabels: Iterable[str], ylabel: str, coverage: float = 0.92
) -> tuple[float, float]:
    """Axes width and height (inches) so the axis labels (almost) span them.

    The width fits the longest x-axis label, the height fits the y-axis label.
    ``coverage`` is the fraction of the axis length covered by its label.
    """
    fig, ax = plt.subplots()
    renderer = fig.canvas.get_renderer()
    x_length = 0.0
    for xlabel in xlabels:
        ax.set_xlabel(xlabel)
        x_length = max(
            x_length, ax.xaxis.label.get_window_extent(renderer).width / fig.dpi
        )
    ax.set_ylabel(ylabel)
    y_length = ax.yaxis.label.get_window_extent(renderer).height / fig.dpi
    plt.close(fig)
    return x_length / coverage, y_length / coverage


def make_fixed_size_axes(
    n_panels: int,
    ax_width: float,
    ax_height: float,
    margin: float = 1.0,
    panel_gap: float = 0.35,
) -> tuple[plt.Figure, list[plt.Axes]]:
    """Figure with axes of an exact size (inches), in a single row.

    ``margin`` is the space around the axes for labels (trimmed when saving with
    ``bbox_inches="tight"``), ``panel_gap`` the horizontal space between panels.
    """
    fig_width = 2 * margin + n_panels * ax_width + (n_panels - 1) * panel_gap
    fig_height = 2 * margin + ax_height
    fig = plt.figure(figsize=(fig_width, fig_height))
    axes = [
        fig.add_axes(
            [
                (margin + i * (ax_width + panel_gap)) / fig_width,
                margin / fig_height,
                ax_width / fig_width,
                ax_height / fig_height,
            ]
        )
        for i in range(n_panels)
    ]
    return fig, axes


def save_fig(fig: sns.FacetGrid | sns.PairGrid | plt.Figure, fpath: Path, **kwargs):
    kwargs_default = {"bbox_inches": "tight", "dpi": 300}
    kwargs_default.update(kwargs)
    fpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fpath, **kwargs_default)
    print(f"Saved figure to {fpath}")


def save_fig_formats(
    fig: plt.Figure,
    fpath_stem: Path,
    formats: Iterable[str] = ("png", "svg"),
    close: bool = True,
    **kwargs,
):
    """Save a figure once per format (fpath_stem + extension), then close it."""
    for fmt in formats:
        save_fig(fig, fpath_stem.parent / f"{fpath_stem.name}.{fmt}", **kwargs)
    if close:
        plt.close(fig)
