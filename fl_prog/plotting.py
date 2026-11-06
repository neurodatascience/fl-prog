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


def save_fig(fig: sns.FacetGrid | sns.PairGrid | plt.Figure, fpath: Path, **kwargs):
    kwargs_default = {"bbox_inches": "tight", "dpi": 300}
    kwargs_default.update(kwargs)
    fpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fpath, **kwargs_default)
    print(f"Saved figure to {fpath}")
