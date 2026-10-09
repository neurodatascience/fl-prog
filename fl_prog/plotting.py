"""Shared plotting helpers."""

from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

# ADNI baseline diagnoses, in disease order
DIAGNOSIS_ORDER = ["CN", "SMC", "EMCI", "LMCI", "AD"]

# colorblind-checked categorical palette (fixed order, never cycled)
CATEGORICAL_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#e87ba4"]
DIAGNOSIS_COLORS = ["#2a78d6", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]
COLOR_OTHER = "#8a8a85"
LINESTYLES = ["-", "--", ":"]


def get_group_order(groups: Iterable) -> list[str]:
    """Known diagnoses in disease order, followed by any other labels."""
    groups = {str(group) for group in groups}
    return [group for group in DIAGNOSIS_ORDER if group in groups] + sorted(
        groups - set(DIAGNOSIS_ORDER)
    )


def get_group_palette(group_order: list[str]) -> dict[str, str]:
    return {
        group: (
            DIAGNOSIS_COLORS[DIAGNOSIS_ORDER.index(group)]
            if group in DIAGNOSIS_ORDER
            else COLOR_OTHER
        )
        for group in group_order
    }


def get_categorical_palette(labels: list[str]) -> dict[str, str]:
    if len(labels) > len(CATEGORICAL_COLORS):
        raise ValueError(
            f"Too many categories for a categorical palette ({len(labels)} > "
            f"{len(CATEGORICAL_COLORS)}), use get_composite_styles instead"
        )
    return dict(zip(labels, CATEGORICAL_COLORS))


def get_composite_styles(labels: list[str]) -> dict[str, tuple[str, str]]:
    """``(color, linestyle)`` per label, for more labels than palette colors."""
    n_colors = len(CATEGORICAL_COLORS)
    if len(labels) > n_colors * len(LINESTYLES):
        raise ValueError(f"Too many categories for composite styles: {len(labels)}")
    return {
        label: (CATEGORICAL_COLORS[i % n_colors], LINESTYLES[i // n_colors])
        for i, label in enumerate(labels)
    }


def save_fig(fig: sns.FacetGrid | sns.PairGrid | plt.Figure, fpath: Path, **kwargs):
    kwargs_default = {"bbox_inches": "tight", "dpi": 300}
    kwargs_default.update(kwargs)
    fpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fpath, **kwargs_default)
    print(f"Saved figure to {fpath}")
