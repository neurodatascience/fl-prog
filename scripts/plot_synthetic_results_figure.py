#!/usr/bin/env python
"""Single-slide (16:9) results figure for a multi-seed synthetic run.

Top row: group trajectories (centralized and federated conditions vs ground
truth, see ``plot_fedbiomed_comparison_seeds.py``). Bottom row: model fit (R² and
MAE, train and test), recovery of all model parameters except acceleration
factors, and recovery of acceleration factors by number of visits.

Expects the outputs of ``collect_seed_metrics.py`` in ``<results-dir>/figures``
(with the "_shifted" recovery metrics of ``get_metrics.py``).
"""

from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

# scripts directory (on sys.path when running this script)
from plot_fedbiomed_comparison_seeds import (
    DATASETS,
    DNAME_FIGURES,
    LEGEND_HEADERS,
    draw_group_trajectories,
    find_seed_files,
    find_tag_dirs,
    get_group_trajectories_legend,
    load_fitted_model,
    load_ground_truth,
    prepare_group_trajectories,
)
from scipy.stats import spearmanr

from fl_prog.metrics import average_over_biomarkers
from fl_prog.plotting import get_categorical_palette, save_fig
from fl_prog.results import (
    estimated_params_by_setup,
    get_subjects,
    load_df_data,
    load_json_data,
    true_params_by_subject,
)
from fl_prog.utils.constants import CLICK_CONTEXT_SETTINGS
from fl_prog.utils.io import load_json

DEFAULT_DATASET = "synthetic_fullcov"
DEFAULT_DPATH_RESULTS = Path("results") / "2026_11_11-with-shifted"
FIGSIZE = (14.83, 7.5)  # 16:9 slide (13.33 x 7.5 in) plus 1.5 in of width
CONTEXT = "talk"
FONT_SCALE = 0.8  # talk context, scaled down to fit one slide
N_YTICKS = 3

# model label (collect_seed_metrics.py) -> short label, in plotting order
MODELS = {
    "Fed-BioMed centralized": "Centralized",
    "Fed-BioMed 100% overlap": "100% overlap",
    "Fed-BioMed 50% overlap": "50% overlap",
    "Fed-BioMed 0% overlap": "0% overlap",
}
# DPMoSt/Leaspy are not shown
MODELS_FEDBIOMED = list(MODELS)
FIT_METRICS = {"r2_score": "R²", "mean_absolute_error": "MAE"}
# fit panels of the synthetic and ADNI figures (plot_adni_results_figure.py):
# same y-axis bounds (MAE), or same y-axis span centered on each figure's values
# (R²), covering both datasets' Fed-BioMed results
FIT_YLIMS = {"mean_absolute_error": (0.1, 0.145)}
FIT_YSPANS = {"r2_score": 0.13}
SET_NAMES = ["train", "test"]
LABEL_TRAIN = "Train subjects"
LABEL_TEST = "Test subjects"
# relative errors of the biomarker parameters, in one panel: (set name, metric) ->
# label (x0 after removing the arbitrary time origin; k_j, s_j, sigma_j do not
# depend on it)
RELATIVE_ERROR_METRICS = {
    ("recovery_per_biomarker", "scaling_factors_mean_abs_relative_error"): r"$s_j$",
    ("recovery_per_biomarker", "k_values_mean_abs_relative_error"): r"$k_j$",
    ("recovery_per_biomarker", "sigma_mean_abs_relative_error"): r"$\sigma_j$",
}
RELATIVE_ERROR_TITLE = ", ".join(RELATIVE_ERROR_METRICS.values()) + " recovery"
LABEL_RELATIVE_ERROR = "Relative error"
# errors in months, one panel each (after removing the arbitrary time origin; a
# relative error would depend on it): (set name, metric) -> title. Midpoints m_j
# of all biomarkers, time shifts tau_i of held-out subjects
MONTHS_METRICS = {
    ("recovery_per_biomarker_shifted", "x0_values_mae"): r"$m_j$ recovery",
    ("recovery_per_subject_test_shifted", "time_shifts_mae"): r"$\tau_i$ recovery",
}
MONTHS_PER_MODEL_TIME = 96
LABEL_MONTHS_ERROR = "MAE (months)"
ACCELERATION_TITLE = r"$\alpha_i$ recovery"
# number of timepoints per subject (at most 11; subjects with one timepoint have no
# acceleration estimate). Only ~3-10 training subjects per seed have 11 timepoints
VISIT_BINS = {str(n): (n, n) for n in range(2, 12)}
# Fed-BioMed model label -> (condition, setup) in the dataset spec
CONDITIONS = {
    "Fed-BioMed centralized": "overlap100-centralized",
    "Fed-BioMed 100% overlap": "overlap100-federated",
    "Fed-BioMed 50% overlap": "overlap050-federated",
    "Fed-BioMed 0% overlap": "overlap000-federated",
}


# legend color boxes: outline width relative to the default (patch.linewidth)
LEGEND_PATCH_LINEWIDTH_FACTOR = 0.5


def _thin_legend_patches(legend, factor: float = LEGEND_PATCH_LINEWIDTH_FACTOR):
    """Thinner outlines for the color boxes of a legend."""
    for handle in legend.legend_handles:
        if isinstance(handle, Patch):
            handle.set_linewidth(plt.rcParams["patch.linewidth"] * factor)


def _lighten(color, amount: float = 0.55):
    rgb = np.array(to_rgb(color))
    return tuple(rgb + (1 - rgb) * amount)


def _boxes(ax, groups: list[tuple[float, np.ndarray, tuple]], width: float = 0.7):
    """Box (no outliers) plus jittered points at each position."""
    rng = np.random.default_rng(0)
    for position, values, color in groups:
        ax.boxplot(
            values,
            positions=[position],
            widths=width,
            patch_artist=True,
            showfliers=False,
            boxprops={"facecolor": color, "edgecolor": "0.25", "linewidth": 1},
            medianprops={"color": "0.25", "linewidth": 1.5},
            whiskerprops={"color": "0.25", "linewidth": 1},
            capprops={"color": "0.25", "linewidth": 1},
        )
        ax.scatter(
            position + rng.uniform(-width / 5, width / 5, len(values)),
            values,
            s=6,
            color="0.15",
            alpha=0.6,
            linewidth=0,
            zorder=3,
        )
    ax.set_xticks([])
    ax.set_xlim(-0.6, max(p for p, _, _ in groups) + 0.6)


def _values(df: pd.DataFrame, **conditions) -> np.ndarray:
    mask = np.ones(len(df), dtype=bool)
    for col, value in conditions.items():
        mask &= (df[col] == value).to_numpy()
    return df.loc[mask, "value"].to_numpy()


def _plot_fit(axes, df_fit: pd.DataFrame, palette: dict, models=MODELS):
    """Train (light) and test boxes per model, for each fit metric."""
    for ax, (metric, title) in zip(axes, FIT_METRICS.items()):
        groups = []
        for i_model, model in enumerate(models):
            for i_set, set_name in enumerate(SET_NAMES):
                values = _values(df_fit, model=model, metric=metric, set_name=set_name)
                color = palette[model]
                groups.append(
                    (
                        2.2 * i_model + 0.9 * i_set,
                        values,
                        _lighten(color) if set_name == "train" else color,
                    )
                )
        _boxes(ax, groups, width=0.8)
        ax.set_title(title, fontweight="bold")
        if metric in FIT_YLIMS:
            ax.set_ylim(FIT_YLIMS[metric])
        elif metric in FIT_YSPANS:
            values = np.concatenate([v for _, v, _ in groups])
            middle = (values.min() + values.max()) / 2
            ax.set_ylim(
                middle - FIT_YSPANS[metric] / 2, middle + FIT_YSPANS[metric] / 2
            )


def _plot_recovery(ax_relative, axes_months, df_recovery: pd.DataFrame, palette):
    """Relative errors of the biomarker scale parameters (one group of boxes per
    parameter), then errors in months (midpoints, time shifts), one panel each."""
    n_models = len(MODELS_FEDBIOMED)
    groups = []
    for i_param, (set_name, metric) in enumerate(RELATIVE_ERROR_METRICS):
        for i_model, model in enumerate(MODELS_FEDBIOMED):
            groups.append(
                (
                    i_param * (n_models + 1) + i_model,
                    _values(df_recovery, model=model, set_name=set_name, metric=metric),
                    palette[model],
                )
            )
    _boxes(ax_relative, groups)
    ax_relative.set_xticks(
        [
            i_param * (n_models + 1) + (n_models - 1) / 2
            for i_param in range(len(RELATIVE_ERROR_METRICS))
        ],
        list(RELATIVE_ERROR_METRICS.values()),
    )
    ax_relative.tick_params(axis="x", length=0)
    ax_relative.set_ylim(bottom=0)
    ax_relative.set_ylabel(LABEL_RELATIVE_ERROR)
    ax_relative.set_title(RELATIVE_ERROR_TITLE, fontweight="bold")

    for i_ax, (ax, ((set_name, metric), title)) in enumerate(
        zip(axes_months, MONTHS_METRICS.items())
    ):
        _boxes(
            ax,
            [
                (
                    i_model,
                    _values(df_recovery, model=model, set_name=set_name, metric=metric)
                    * MONTHS_PER_MODEL_TIME,
                    palette[model],
                )
                for i_model, model in enumerate(MODELS_FEDBIOMED)
            ],
        )
        ax.set_title(title, fontweight="bold")
        ax.set_ylim(bottom=0)
        if i_ax == 0:
            ax.set_ylabel(LABEL_MONTHS_ERROR)


def get_df_accel_by_visits(fpaths_json: dict[int, dict[str, Path]]) -> pd.DataFrame:
    """Spearman correlation between true and estimated acceleration factors of the
    training subjects, by number of visits (seed x model x visit bin)."""
    rows = []
    for seed, fpaths in fpaths_json.items():
        for model, condition in CONDITIONS.items():
            json_results = load_json(fpaths[condition])
            json_data = load_json_data(json_results)
            subjects_by_node = json_data["subjects_by_node"]
            setup = condition.split("-")[-1]
            accel_true = true_params_by_subject(
                subjects_by_node, json_data["params"]
            ).acceleration_factors
            accel_est = estimated_params_by_setup(json_results, subjects_by_node)[
                setup
            ].acceleration_factors
            cols = json_data["cols"]
            fpath_data = Path(json_results["settings"]["fpath_config"])
            df_data = load_df_data(
                fpath_data.parent, json_results["settings"]["tag"], cols
            )
            n_visits = (
                df_data.groupby(cols["col_subject"])
                .size()
                .reindex(get_subjects(subjects_by_node))
                .to_numpy()
            )
            for label, (low, high) in VISIT_BINS.items():
                mask = (n_visits >= low) & (n_visits <= high)
                rows.append(
                    {
                        "seed": seed,
                        "model": model,
                        "n_visits": label,
                        "n_subjects": int(mask.sum()),
                        "spearman_r": spearmanr(
                            accel_true[mask], accel_est[mask]
                        ).statistic,
                    }
                )
    return pd.DataFrame(rows)


def _plot_accel_by_visits(ax, df_accel: pd.DataFrame, palette: dict):
    x = np.arange(len(VISIT_BINS))
    offsets = np.linspace(-0.15, 0.15, len(MODELS_FEDBIOMED))
    for offset, model in zip(offsets, MODELS_FEDBIOMED):
        df_summary = (
            df_accel.query("model == @model")
            .groupby("n_visits")["spearman_r"]
            .agg(["mean", "std"])
            .reindex(list(VISIT_BINS))
        )
        ax.errorbar(
            x + offset,
            df_summary["mean"],
            yerr=df_summary["std"],
            color=palette[model],
            marker="o",
            markersize=5,
            linewidth=1.5,
            elinewidth=1,
            capsize=0,
        )
    # tick at every number of timepoints, label every other one (space)
    ax.set_xticks(
        x, [label if i % 2 == 0 else "" for i, label in enumerate(VISIT_BINS)]
    )
    ax.set_xlabel("N timepoints")
    ax.set_ylabel("Spearman \u03c1")
    ax.set_ylim(0, 1)
    ax.set_title(ACCELERATION_TITLE, fontweight="bold")


def plot_synthetic_results_figure(
    dataset: str,
    dpath_results: Path,
    dpath_out: Path | None,
    extension: str,
    xlim_percentile: float,
):
    spec = DATASETS[dataset]
    suffix = f"-{spec.tag_prefix}"
    dpath_figures = dpath_results / DNAME_FIGURES
    dpath_out = dpath_out or dpath_figures

    # data
    dpaths_tag = dict(sorted(find_tag_dirs(dpath_results, spec).items()))
    fpaths_json = {
        seed: find_seed_files(dpaths_split, seed, spec)
        for seed, dpaths_split in dpaths_tag.items()
    }
    print(f"Seeds: {list(fpaths_json)}")
    fitted_models = {
        seed: {
            condition: load_fitted_model(
                fpath,
                spec.condition_setups[condition],
                time_scale=spec.time_scale,
                align_translation=True,
            )
            for condition, fpath in fpaths.items()
        }
        for seed, fpaths in fpaths_json.items()
    }
    json_ground_truth = load_ground_truth(
        next(iter(fpaths_json.values()))[spec.condition_centralized]
    )
    biomarker_labels = {
        col: f"Biomarker {i}"
        for i, col in enumerate(json_ground_truth["cols"]["cols_biomarker"], start=1)
    }
    sns.set_theme(context=CONTEXT, style="ticks", font_scale=FONT_SCALE)
    trajectories = prepare_group_trajectories(
        fitted_models,
        spec,
        json_ground_truth=json_ground_truth,
        biomarker_labels=biomarker_labels,
        xlim_percentile=xlim_percentile,
    )
    # "all": mean of the per-biomarker metrics (also for older pooled results)
    df_fit = average_over_biomarkers(
        pd.read_csv(dpath_figures / f"fit_metrics-seeds{suffix}-all.tsv", sep="\t")
    ).query("col_biomarker == 'all'")
    df_recovery = pd.read_csv(
        dpath_figures / f"param_recovery-seeds{suffix}-all.tsv", sep="\t"
    ).query("col_biomarker == 'all'")
    df_accel = get_df_accel_by_visits(fpaths_json)
    fpath_accel = dpath_out / f"accel_recovery_by_visits{suffix}.tsv"
    fpath_accel.parent.mkdir(parents=True, exist_ok=True)
    df_accel.to_csv(fpath_accel, sep="\t", index=False)
    print(f"Saved acceleration recovery by visits to {fpath_accel}")

    # figure
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
    for ax in axes_top:
        ax.set_xlabel("Months")
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
        1,
        n_fit + 2 + len(MONTHS_METRICS),
        gridspec_kw={
            "width_ratios": [2.2] * n_fit + [2.6] + [1.1] * len(MONTHS_METRICS) + [2.2]
        },
    )
    _plot_fit(axes_bottom[:n_fit], df_fit, palette)
    _plot_recovery(
        axes_bottom[n_fit],
        axes_bottom[n_fit + 1 : n_fit + 1 + len(MONTHS_METRICS)],
        df_recovery,
        palette,
    )
    _plot_accel_by_visits(axes_bottom[-1], df_accel, palette)
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

    fpath_out = dpath_out / f"results_figure{suffix}.{extension}"
    save_fig(fig, fpath_out, bbox_inches=None)
    plt.close(fig)


@click.command(context_settings=CLICK_CONTEXT_SETTINGS)
@click.option(
    "--dataset",
    type=click.Choice([d for d in DATASETS if DATASETS[d].has_ground_truth]),
    default=DEFAULT_DATASET,
    show_default=True,
)
@click.option(
    "--results-dir",
    "dpath_results",
    type=click.Path(path_type=Path, file_okay=False, exists=True),
    default=DEFAULT_DPATH_RESULTS,
    show_default=True,
    help="Dated results directory with the seed summaries in figures/.",
)
@click.option(
    "--output-dir",
    "dpath_out",
    type=click.Path(path_type=Path, file_okay=False),
    default=None,
    help="Output directory. Default: <results-dir>/figures.",
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
    default=99,
    show_default=True,
    help="Trajectory x-axis ends at this percentile of the centralized models' "
    "shifted visit times.",
)
def main(**params):
    plot_synthetic_results_figure(**params)


if __name__ == "__main__":
    main()
