#!/usr/bin/env python
import re
from itertools import combinations
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from scipy.stats import pearsonr, spearmanr

from fl_prog.plotting import (
    get_categorical_palette,
    get_composite_styles,
    get_group_order,
    get_group_palette,
    save_fig,
)
from fl_prog.results import (
    COL_BIOMARKER,
    COL_GROUP,
    COL_MODEL,
    COL_SUBJECT,
    COL_TIME_SHIFT,
    COL_TIME_SHIFTED,
    COL_Y,
    COL_Y_OBS,
    DEFAULT_LEASPY_ALGORITHM_NAME,
    DEFAULT_LEASPY_N_JOBS,
    DEFAULT_LEASPY_SEED,
    MODEL_TYPES,
    FittedModel,
    LeaspyIndividualParametersNotFoundError,
    get_fitted_models,
    iter_result_files,
)
from fl_prog.utils.constants import CLICK_CONTEXT_SETTINGS, Setup
from fl_prog.utils.io import DEFAULT_DPATH_RESULTS, get_dpath_latest

DNAME_FIGURES = "figures"
SETUP_ORDER = [Setup.FEDERATED.value, Setup.CENTRALIZED.value]

LABEL_TIME_SHIFTED = "Disease time (months)"
LABEL_TIME_SHIFT = "Time shift (months)"
LABEL_BIOMARKER = "Biomarker value (scaled)"


def _slugify(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_")


def _sort_key(fitted_model: FittedModel) -> tuple:
    setup = fitted_model.setup
    return (
        MODEL_TYPES.index(fitted_model.model_type),
        SETUP_ORDER.index(setup) if setup in SETUP_ORDER else len(SETUP_ORDER),
        fitted_model.run_tag,
    )


def _make_labels_unique(fitted_models: list[FittedModel]):
    """Disambiguate models with the same label (e.g. runs with different seeds)."""
    labels = [fitted_model.label for fitted_model in fitted_models]
    for fitted_model in fitted_models:
        if labels.count(fitted_model.label) > 1:
            label = f"{fitted_model.label} [{fitted_model.run_tag}]"
            fitted_model.label = label
            for df in (
                fitted_model.df_subject,
                fitted_model.df_long,
                fitted_model.df_trajectory,
            ):
                df[COL_MODEL] = label


def plot_group_trajectories_by_model(
    df_trajectory: pd.DataFrame, model_order: list[str], biomarker_order: list[str]
) -> plt.Figure:
    styles = get_composite_styles(biomarker_order)
    fig, axes = plt.subplots(
        ncols=len(model_order),
        figsize=(4 * len(model_order), 3.5),
        sharey=True,
        squeeze=False,
    )
    for ax, model in zip(axes.flat, model_order):
        df_model = df_trajectory.query(f"{COL_MODEL} == @model")
        for biomarker in biomarker_order:
            df_biomarker = df_model.query(f"{COL_BIOMARKER} == @biomarker")
            color, linestyle = styles[biomarker]
            ax.plot(
                df_biomarker[COL_TIME_SHIFTED],
                df_biomarker[COL_Y],
                color=color,
                linestyle=linestyle,
                linewidth=2,
                label=biomarker,
            )
        ax.set_title(model)
        ax.set_xlabel(LABEL_TIME_SHIFTED)
    axes.flat[0].set_ylabel(LABEL_BIOMARKER)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="Biomarker",
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        frameon=False,
    )
    sns.despine(fig)
    fig.tight_layout()
    return fig


def plot_group_trajectories_by_biomarker(
    fitted_models: list[FittedModel],
    model_order: list[str],
    biomarker_order: list[str],
    n_points: int = 500,
) -> sns.FacetGrid:
    # evaluate all models on a common grid spanning every model's time range
    t_min = min(m.df_trajectory[COL_TIME_SHIFTED].min() for m in fitted_models)
    t_max = max(m.df_trajectory[COL_TIME_SHIFTED].max() for m in fitted_models)
    t_grid = np.linspace(t_min, t_max, n_points)
    df_trajectory = pd.concat(
        [m.get_df_trajectory(t_grid) for m in fitted_models], ignore_index=True
    )

    grid = sns.relplot(
        data=df_trajectory,
        x=COL_TIME_SHIFTED,
        y=COL_Y,
        hue=COL_MODEL,
        hue_order=model_order,
        palette=get_categorical_palette(model_order),
        col=COL_BIOMARKER,
        col_order=biomarker_order,
        col_wrap=3,
        kind="line",
        linewidth=2,
        height=3,
        aspect=1.3,
    )
    grid.set(xlim=(t_min, t_max))
    grid.set_titles("{col_name}")
    grid.set_axis_labels(LABEL_TIME_SHIFTED, LABEL_BIOMARKER)
    return grid


def plot_spaghetti(
    fitted_model: FittedModel,
    biomarker_order: list[str],
    group_order: list[str],
    palette: dict[str, str],
) -> sns.FacetGrid:
    df_long = fitted_model.df_long.dropna(subset=[COL_Y_OBS, COL_TIME_SHIFTED])
    grid = sns.FacetGrid(
        df_long,
        col=COL_BIOMARKER,
        col_order=biomarker_order,
        col_wrap=3,
        height=3,
        aspect=1.3,
    )
    for biomarker, ax in grid.axes_dict.items():
        df_biomarker = df_long.query(f"{COL_BIOMARKER} == @biomarker").sort_values(
            [COL_SUBJECT, COL_TIME_SHIFTED]
        )
        # one line per subject, drawn as a collection for speed
        for group in group_order:
            segments = [
                df_subject[[COL_TIME_SHIFTED, COL_Y_OBS]].to_numpy()
                for _, df_subject in df_biomarker.query(
                    f"{COL_GROUP} == @group"
                ).groupby(COL_SUBJECT)
            ]
            ax.add_collection(
                LineCollection(
                    segments,
                    colors=palette[group],
                    linewidths=0.6,
                    alpha=0.3,
                    rasterized=True,
                )
            )
        df_trajectory = fitted_model.df_trajectory.query(
            f"{COL_BIOMARKER} == @biomarker"
        )
        ax.plot(
            df_trajectory[COL_TIME_SHIFTED],
            df_trajectory[COL_Y],
            color="black",
            linewidth=2.5,
        )
        ax.autoscale_view()
        ax.set_title(biomarker)

    grid.set_axis_labels(LABEL_TIME_SHIFTED, LABEL_BIOMARKER)
    handles = [
        Line2D([], [], color=palette[group], linewidth=2) for group in group_order
    ] + [Line2D([], [], color="black", linewidth=2.5)]
    grid.figure.legend(
        handles,
        group_order + ["Group trajectory"],
        title=fitted_model.label,
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        frameon=False,
    )
    return grid


def get_df_time_shifts_wide(
    df_subject: pd.DataFrame, model_order: list[str]
) -> pd.DataFrame:
    """Subjects x models table of time shifts, with each subject's group."""
    df_wide = df_subject.pivot(
        index=COL_SUBJECT, columns=COL_MODEL, values=COL_TIME_SHIFT
    )[model_order]
    groups = df_subject.groupby(COL_SUBJECT)[COL_GROUP].first()
    return df_wide.assign(**{COL_GROUP: groups.reindex(df_wide.index)})


def get_time_shift_correlations(
    df_wide: pd.DataFrame, model_order: list[str]
) -> pd.DataFrame:
    rows = []
    for model1, model2 in combinations(model_order, 2):
        df_pair = df_wide[[model1, model2]].dropna()
        rows.append(
            {
                "model1": model1,
                "model2": model2,
                "n_subjects": len(df_pair),
                "spearman_r": spearmanr(df_pair[model1], df_pair[model2]).statistic,
                "pearson_r": pearsonr(df_pair[model1], df_pair[model2]).statistic,
            }
        )
    return pd.DataFrame(rows)


def plot_time_shift_comparison(
    df_wide: pd.DataFrame,
    df_correlations: pd.DataFrame,
    model_order: list[str],
    group_order: list[str],
    palette: dict[str, str],
) -> sns.PairGrid:
    grid = sns.PairGrid(
        df_wide.reset_index(),
        vars=model_order,
        hue=COL_GROUP,
        hue_order=group_order,
        palette=palette,
        height=2.5,
        diag_sharey=False,
    )
    grid.map_lower(sns.scatterplot, s=8, alpha=0.5, linewidth=0, rasterized=True)
    grid.map_diag(sns.kdeplot, common_norm=False, linewidth=1.5)

    # correlations over all subjects (not per group) in the upper triangle
    correlations = {
        frozenset((row["model1"], row["model2"])): row
        for _, row in df_correlations.iterrows()
    }
    for i_row, i_col in zip(*np.triu_indices_from(grid.axes, k=1)):
        ax = grid.axes[i_row, i_col]
        row = correlations[frozenset((model_order[i_row], model_order[i_col]))]
        ax.text(
            0.5,
            0.5,
            f"Spearman ρ = {row['spearman_r']:.2f}\n"
            f"Pearson r = {row['pearson_r']:.2f}\n"
            f"n = {row['n_subjects']}",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()

    grid.add_legend(title="Diagnosis")
    grid.figure.suptitle(f"{LABEL_TIME_SHIFT} at baseline", y=1.02)
    return grid


def plot_time_shift_by_group(
    df_subject: pd.DataFrame,
    model_order: list[str],
    group_order: list[str],
    palette: dict[str, str],
) -> sns.FacetGrid:
    grid = sns.catplot(
        data=df_subject,
        x=COL_GROUP,
        y=COL_TIME_SHIFT,
        order=group_order,
        hue=COL_GROUP,
        hue_order=group_order,
        palette=palette,
        col=COL_MODEL,
        col_order=model_order,
        kind="box",
        fliersize=2,
        sharey=False,
        height=3,
        aspect=1.1,
    )
    grid.set_titles("{col_name}")
    grid.set_axis_labels("Diagnosis", LABEL_TIME_SHIFT)
    return grid


def plot_model_fits(
    tag: str,
    dpath_results: Path,
    extension: str = "svg",
    leaspy_algorithm_name: str = DEFAULT_LEASPY_ALGORITHM_NAME,
    leaspy_seed: int = DEFAULT_LEASPY_SEED,
    leaspy_n_jobs: int = DEFAULT_LEASPY_N_JOBS,
):
    dpath_results_tag = get_dpath_latest(dpath_results) / tag
    dpath_figures = dpath_results_tag / DNAME_FIGURES

    fitted_models: list[FittedModel] = []
    for fpath_json_results, json_results in iter_result_files(dpath_results_tag):
        print(f"fpath_json_results: {fpath_json_results}")
        try:
            fitted_models.extend(
                get_fitted_models(
                    fpath_json_results,
                    json_results,
                    tag,
                    leaspy_algorithm_name=leaspy_algorithm_name,
                    leaspy_seed=leaspy_seed,
                    leaspy_n_jobs=leaspy_n_jobs,
                )
            )
        except LeaspyIndividualParametersNotFoundError as exception:
            raise click.ClickException(
                f"{exception} in {fpath_json_results}. Run scripts/get_metrics.py "
                f"--tag {tag} --results-dir {dpath_results} "
                f"--leaspy-algorithm-name {leaspy_algorithm_name} "
                f"--leaspy-seed {leaspy_seed} --leaspy-n-jobs {leaspy_n_jobs} "
                "first to personalize the Leaspy model."
            ) from exception
    if not fitted_models:
        raise click.ClickException(f"No results files found in {dpath_results_tag}")

    fitted_models.sort(key=_sort_key)
    _make_labels_unique(fitted_models)
    model_order = [fitted_model.label for fitted_model in fitted_models]

    df_subject = pd.concat([m.df_subject for m in fitted_models], ignore_index=True)
    df_trajectory = pd.concat(
        [m.df_trajectory for m in fitted_models], ignore_index=True
    )
    biomarker_order = list(df_trajectory[COL_BIOMARKER].unique())
    group_order = get_group_order(df_subject[COL_GROUP].dropna())
    group_palette = get_group_palette(group_order)

    def _save(fig, name: str):
        save_fig(fig, dpath_figures / f"{name}.{extension}")
        plt.close(fig.figure if hasattr(fig, "figure") else fig)

    sns.set_theme(context="notebook", style="ticks")

    # 1) group trajectories
    _save(
        plot_group_trajectories_by_model(df_trajectory, model_order, biomarker_order),
        "group_trajectories-by_model",
    )
    _save(
        plot_group_trajectories_by_biomarker(
            fitted_models, model_order, biomarker_order
        ),
        "group_trajectories-by_biomarker",
    )

    # 2) spaghetti plots
    for fitted_model in fitted_models:
        _save(
            plot_spaghetti(fitted_model, biomarker_order, group_order, group_palette),
            f"spaghetti-{_slugify(fitted_model.label)}",
        )

    # 3) time shift comparison between models
    df_wide = get_df_time_shifts_wide(df_subject, model_order)
    df_correlations = get_time_shift_correlations(df_wide, model_order)
    fpath_correlations = dpath_figures / "time_shift_correlations.tsv"
    df_correlations.to_csv(fpath_correlations, sep="\t", index=False)
    print(f"Saved time shift correlations to {fpath_correlations}")
    print(df_correlations.to_string(index=False))

    if len(model_order) > 1:
        _save(
            plot_time_shift_comparison(
                df_wide, df_correlations, model_order, group_order, group_palette
            ),
            "time_shift_comparison",
        )
    _save(
        plot_time_shift_by_group(df_subject, model_order, group_order, group_palette),
        "time_shift_by_group",
    )


@click.command(context_settings=CLICK_CONTEXT_SETTINGS)
@click.option("--tag", type=str, required=True)
@click.option(
    "--results-dir",
    "dpath_results",
    type=click.Path(path_type=Path, file_okay=False, dir_okay=True),
    default=DEFAULT_DPATH_RESULTS,
    show_envvar=True,
)
@click.option(
    "--format",
    "extension",
    type=click.Choice(["svg", "png", "pdf"]),
    default="svg",
    help="Figure file format.",
)
@click.option(
    "--leaspy-algorithm-name",
    type=str,
    default=DEFAULT_LEASPY_ALGORITHM_NAME,
    help="Algorithm name of the Leaspy personalization to plot.",
    show_envvar=True,
)
@click.option(
    "--leaspy-seed",
    type=int,
    default=DEFAULT_LEASPY_SEED,
    help="Random seed of the Leaspy personalization to plot.",
    envvar="RNG_SEED",
    show_envvar=True,
)
@click.option(
    "--leaspy-n-jobs",
    type=int,
    default=DEFAULT_LEASPY_N_JOBS,
    help="Number of parallel jobs of the Leaspy personalization to plot.",
    envvar="LEASPY_N_JOBS",
    show_envvar=True,
)
def main(**params):
    plot_model_fits(**params)


if __name__ == "__main__":
    main()
