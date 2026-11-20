#!/usr/bin/env python
"""Single-slide results figure for the multi-seed ADNI run (``run_adni_seeds.sh``).

Top row: group trajectories of each federated condition vs the centralized fit
(see ``plot_fedbiomed_comparison_seeds.py``). Bottom row: model fit (R² and MAE,
train and test subjects) and estimated time shifts of one seed by condition and
diagnosis group. Train subjects are shown lighter than test subjects.

Expects the outputs of ``collect_seed_metrics.py`` in ``<results-dir>/figures``.
"""

import dataclasses
import re
from pathlib import Path

import click
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

# scripts directory (on sys.path when running this script)
from plot_fedbiomed_comparison_seeds import (
    DATASETS,
    DNAME_FIGURES,
    LEGEND_HEADERS,
    draw_group_trajectories,
    draw_time_shifts,
    find_seed_files,
    find_tag_dirs,
    get_group_trajectories_legend,
    load_fitted_model,
    prepare_group_trajectories,
)
from plot_synthetic_results_figure import (
    CONTEXT,
    FIGSIZE,
    FIT_METRICS,
    FONT_SCALE,
    LABEL_TEST,
    LABEL_TRAIN,
    N_YTICKS,
    SET_NAMES,
    _lighten,
    _plot_fit,
    _thin_legend_patches,
)
from scipy.stats import pearsonr

from fl_prog.metrics import average_over_biomarkers
from fl_prog.plotting import get_categorical_palette, save_fig
from fl_prog.results import COL_SET_NAME, COL_SUBJECT, COL_TIME_SHIFT
from fl_prog.utils.constants import CLICK_CONTEXT_SETTINGS
from fl_prog.utils.io import DEFAULT_DPATH_RESULTS

DATASET = "adni"
DEFAULT_DPATH_RESULTS_ADNI = (
    Path(DEFAULT_DPATH_RESULTS).resolve().parents[1] / "fl-prog/results/2026_11_08"
)
DEFAULT_SEED_TIME_SHIFTS = 1
FNAME_FIT = "fit_metrics-seeds-all.tsv"
FNAME_OUT = "results_figure-adni"

# model label (collect_seed_metrics.py) -> short label, in plotting order
# (DPMoSt/Leaspy are not shown)
MODELS = {
    "Fed-BioMed centralized": "Centralized",
    "Fed-BioMed IID": "Random grouping",
    "Fed-BioMed non-IID (site)": "Grouped by original site",
    "Fed-BioMed non-IID (diag.)": "Grouped by diagnosis",
}
# federated condition -> label (data split across sites)
CONDITION_LABELS = {
    "iid-federated": "Random grouping",
    "noniid-site": "Grouped by original site",
    "noniid-diag": "Grouped by diagnosis",
}
# time shift x-axis labels (wrapped)
CONDITION_LABELS_WRAPPED = {
    "iid-centralized": "Centralized",
    "iid-federated": "Random\ngrouping",
    "noniid-site": "Grouped by\noriginal site",
    "noniid-diag": "Grouped by\ndiagnosis",
}
# biomarker column -> label (FreeSurfer names to plain names)
HEMISPHERES = {"Left": "Left", "Right": "Right", "lh": "Left", "rh": "Right"}
PATTERN_VOLUME = re.compile(r"^(?P<hemi>Left|Right)-Hippocampus$")
PATTERN_THICKNESS = re.compile(r"^(?P<hemi>lh|rh)_(?P<region>.+)_thickness$")
TIME_SHIFTS_LIGHT_SET = "train"
TIME_SHIFTS_TITLE = r"Estimated time shifts ($\boldsymbol{\tau}_\mathbf{i}$)"
# Pearson correlation of each federated model's time shifts with the centralized
# model's (same seed), separately for train and test subjects
LABEL_CORRELATION = "Pearson r vs\ncentralized:"
FORMAT_CORRELATION = r"$r_\mathrm{{{set_name}}}$ = {r:.2f}"
YLIM_TOP_FACTOR = 1.2  # room above the boxes for the correlations


def get_biomarker_label(col: str) -> str:
    """e.g. Left-Hippocampus -> Left hippocampal volume, lh_entorhinal_thickness ->
    Left entorhinal cortex thickness; other names are unchanged."""
    if match := PATTERN_VOLUME.match(col):
        return f"{HEMISPHERES[match['hemi']]} hippocampal volume"
    if match := PATTERN_THICKNESS.match(col):
        return f"{HEMISPHERES[match['hemi']]} {match['region']} cortex thickness"
    return col


def get_time_shift_correlations(
    df_subject: pd.DataFrame, spec
) -> dict[str, dict[str, float]]:
    """Pearson correlation between each federated model's time shifts and the
    centralized model's: federated condition -> set name (train/test) -> r."""
    correlations = {condition: {} for condition in spec.conditions_federated}
    for set_name in SET_NAMES:
        df_wide = (
            df_subject.query(f"{COL_SET_NAME} == @set_name")
            .pivot(index=COL_SUBJECT, columns="condition", values=COL_TIME_SHIFT)
            .dropna()
        )
        for condition in spec.conditions_federated:
            correlations[condition][set_name] = pearsonr(
                df_wide[spec.condition_centralized], df_wide[condition]
            ).statistic
    return correlations


def _annotate_correlations(ax, df_subject: pd.DataFrame, spec):
    """Correlations with the centralized model (train and test subjects) above each
    federated condition."""
    correlations = get_time_shift_correlations(df_subject, spec)
    for condition, rs in correlations.items():
        print(
            f"Time shift Pearson r ({condition} vs centralized): "
            + ", ".join(f"{set_name} {r:.4f}" for set_name, r in rs.items())
        )
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom, bottom + (top - bottom) * YLIM_TOP_FACTOR)
    conditions = list(spec.condition_labels)
    kwargs = {
        "transform": ax.get_xaxis_transform(),
        "ha": "center",
        "va": "top",
        "fontsize": "small",
    }
    ax.text(
        conditions.index(spec.condition_centralized), 0.98, LABEL_CORRELATION, **kwargs
    )
    for condition, rs in correlations.items():
        text = "\n".join(
            FORMAT_CORRELATION.format(set_name=set_name, r=r)
            for set_name, r in rs.items()
        )
        ax.text(conditions.index(condition), 0.98, text, **kwargs)


def plot_adni_results_figure(
    dpath_results: Path,
    dpath_out: Path | None,
    seed_time_shifts: int,
    extension: str,
    xlim_percentile: float,
    overwrite: bool,
):
    spec = dataclasses.replace(DATASETS[DATASET], conditions_federated=CONDITION_LABELS)
    dpath_figures = dpath_results / DNAME_FIGURES
    dpath_out = dpath_out or dpath_figures
    fpath_out = dpath_out / f"{FNAME_OUT}.{extension}"
    if fpath_out.exists() and not overwrite:
        raise click.ClickException(f"{fpath_out} exists, use --overwrite")

    # data
    dpaths_tag = dict(sorted(find_tag_dirs(dpath_results, spec).items()))
    if seed_time_shifts not in dpaths_tag:
        raise click.ClickException(f"No results for seed {seed_time_shifts}")
    fpaths_json = {
        seed: find_seed_files(dpaths_split, seed, spec)
        for seed, dpaths_split in dpaths_tag.items()
    }
    print(f"Seeds: {list(fpaths_json)}")
    fitted_models = {
        seed: {
            condition: load_fitted_model(
                fpath, spec.condition_setups[condition], time_scale=spec.time_scale
            )
            for condition, fpath in fpaths.items()
        }
        for seed, fpaths in fpaths_json.items()
    }
    # "all": mean of the per-biomarker metrics (also for older pooled results)
    df_fit = average_over_biomarkers(
        pd.read_csv(dpath_figures / FNAME_FIT, sep="\t")
    ).query("col_biomarker == 'all'")
    df_subject = pd.concat(
        [
            fitted_model.df_subject.assign(condition=condition)
            for condition, fitted_model in fitted_models[seed_time_shifts].items()
        ],
        ignore_index=True,
    )

    # figure
    sns.set_theme(context=CONTEXT, style="ticks", font_scale=FONT_SCALE)
    cols_biomarker = next(iter(fitted_models.values()))[
        spec.condition_centralized
    ].cols_biomarker
    trajectories = prepare_group_trajectories(
        fitted_models,
        spec,
        biomarker_labels={col: get_biomarker_label(col) for col in cols_biomarker},
        xlim_percentile=xlim_percentile,
    )
    palette = dict(zip(MODELS, get_categorical_palette(list(MODELS)).values()))
    fig = plt.figure(figsize=FIGSIZE, layout="constrained")
    subfig_top, subfig_bottom = fig.subfigures(2, 1, height_ratios=[1.1, 1])

    axes_top = subfig_top.subplots(1, len(trajectories.panels), sharex=True)
    for ax in axes_top[1:]:
        ax.sharey(axes_top[0])
        ax.tick_params(labelleft=False)
    draw_group_trajectories(
        list(axes_top),
        trajectories,
        linewidth=1,
        linewidth_mean=2.5,
        title_fontsize=plt.rcParams["axes.titlesize"],
    )
    handles, labels = get_group_trajectories_legend(trajectories, linewidth=2)
    legend = subfig_top.legend(
        handles,
        labels,
        loc="outside right center",
        handlelength=1.5,
        labelspacing=0.25,
        fontsize="small",
        frameon=False,
    )
    for text in legend.get_texts():
        if text.get_text() in LEGEND_HEADERS:
            text.set_fontweight("bold")

    n_fit = len(FIT_METRICS)
    axes_bottom = subfig_bottom.subplots(
        1, n_fit + 1, gridspec_kw={"width_ratios": [2.2] * n_fit + [5]}
    )
    _plot_fit(axes_bottom[:n_fit], df_fit, palette, models=MODELS)
    ax_time_shifts = axes_bottom[-1]
    (group_handles, group_labels), _ = draw_time_shifts(
        ax_time_shifts,
        df_subject,
        CONDITION_LABELS_WRAPPED,
        light_set=TIME_SHIFTS_LIGHT_SET,
        title=TIME_SHIFTS_TITLE,
    )
    _annotate_correlations(ax_time_shifts, df_subject, spec)
    legend = ax_time_shifts.legend(
        group_handles,
        group_labels,
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        fontsize="small",
        handlelength=1,
        handletextpad=0.4,
        labelspacing=0.3,
        frameon=False,
    )
    _thin_legend_patches(legend)
    for ax in axes_bottom:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=N_YTICKS))
        ax.title.set_fontsize("medium")

    # shared legend: models, then train/test shading
    handles = [
        Patch(facecolor=palette[model], edgecolor="0.25") for model in MODELS
    ] + [
        Patch(facecolor=_lighten("0.5"), edgecolor="0.25"),
        Patch(facecolor="0.5", edgecolor="0.25"),
    ]
    labels = list(MODELS.values()) + [LABEL_TRAIN, LABEL_TEST]
    legend = subfig_bottom.legend(
        handles,
        labels,
        loc="outside lower center",
        ncols=len(labels),
        fontsize="small",
        handlelength=1,
        columnspacing=1,
        frameon=False,
    )
    _thin_legend_patches(legend)
    sns.despine(fig)

    save_fig(fig, fpath_out, bbox_inches=None)
    plt.close(fig)


@click.command(context_settings=CLICK_CONTEXT_SETTINGS)
@click.option(
    "--results-dir",
    "dpath_results",
    type=click.Path(path_type=Path, file_okay=False, exists=True),
    default=DEFAULT_DPATH_RESULTS_ADNI,
    show_default=True,
    help="Dated results directory of the ADNI multi-seed run.",
)
@click.option(
    "--output-dir",
    "dpath_out",
    type=click.Path(path_type=Path, file_okay=False),
    default=None,
    help="Output directory. Default: <results-dir>/figures.",
)
@click.option(
    "--seed-time-shifts",
    type=int,
    default=DEFAULT_SEED_TIME_SHIFTS,
    show_default=True,
    help="Seed whose estimated time shifts are shown.",
)
@click.option(
    "--format",
    "extension",
    type=click.Choice(["svg", "png", "pdf"]),
    default="svg",
    show_default=True,
)
@click.option(
    "--xlim-percentile",
    type=click.FloatRange(0, 100, min_open=True),
    default=100,
    show_default=True,
    help="Trajectory x-axis ends at this percentile of the centralized models' "
    "shifted visit times.",
)
@click.option("--overwrite", is_flag=True, help="Overwrite an existing figure.")
def main(**params):
    plot_adni_results_figure(**params)


if __name__ == "__main__":
    main()
