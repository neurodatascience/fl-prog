#!/usr/bin/env python
"""Compare Fed-BioMed runs across data splits, against a centralized Fed-BioMed
run and centralized DPMoSt/Leaspy, showing all seeds of a multi-seed run.

Datasets (``--dataset``):

- ``adni`` (``run_adni_seeds.sh``): splits ``iid`` (Fed-BioMed federated and
  centralized, DPMoSt, Leaspy), ``noniid_site`` and ``noniid_diag`` (Fed-BioMed
  federated).
- ``synthetic`` (``run_synthetic_seeds.sh``): splits
  ``overlap100`` (Fed-BioMed federated and centralized, DPMoSt, Leaspy) and the other
  site splits in ``SYNTHETIC_CONDITION_LABELS`` (Fed-BioMed federated). Ground-truth
  trajectories and time shifts are shown too. ``synthetic_adnilike``: same, with
  ADNI-like disease coverage.

Only default Fed-BioMed runs (FedAvg + Adam) are used, other aggregators/optimizers
in the same directories (e.g. SCAFFOLD) are ignored.

Expected layout::

    <results-dir>/<dataset>_<split>_s<seed>_.../fedbiomed-*-estimated_params.json
    <results-dir>/<dataset>_<split>_s<seed>_.../<model type>-*-fit_quality*.tsv
"""

import dataclasses
import re
from dataclasses import dataclass
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

from fl_prog.metrics import LABEL_TRAIN, get_fit_sets, get_midpoint_offset
from fl_prog.plotting import (
    get_biomarker_palette,
    get_categorical_palette,
    get_group_order,
    get_group_palette,
    save_fig_formats,
)
from fl_prog.results import (
    COL_BIOMARKER,
    COL_GROUP,
    COL_MODEL,
    COL_SET_NAME,
    COL_SUBJECT,
    COL_TIME,
    COL_TIME_SHIFT,
    COL_TIME_SHIFTED,
    COL_Y,
    KEY_INDIVIDUAL_PARAMETERS_TEST,
    MODEL_FEDBIOMED,
    SUFFIX_ESTIMATED_PARAMS,
    FittedModel,
    estimated_params_by_setup,
    filter_result_files_by_setup,
    get_df_true_subject,
    get_fitted_models,
    get_run_tag,
    load_json_data,
    true_params_by_subject,
)
from fl_prog.utils.constants import (
    CLICK_CONTEXT_SETTINGS,
    SYNTHETIC_CONDITION_LABELS,
    Setup,
)
from fl_prog.utils.io import DEFAULT_DPATH_RESULTS, load_json
from fl_prog.utils.math import multivariate_logistic


@dataclass(frozen=True)
class DatasetSpec:
    """Conditions of a multi-seed run (condition: data split + Fed-BioMed setup)."""

    tag_prefix: str
    # condition -> label (trajectory panel titles, time shift x-axis, metrics rows)
    condition_centralized: str
    conditions_federated: dict[str, str]
    # condition -> data split in the tag
    condition_splits: dict[str, str]
    # split with the DPMoSt/Leaspy fits
    split_baselines: str
    default_dpath_results: Path
    # model time -> months (the synthetic data are not rescaled from months)
    time_scale: float = 1.0
    # ground truth (true parameters) available
    has_ground_truth: bool = False

    @property
    def condition_labels(self) -> dict[str, str]:
        return {self.condition_centralized: "Centralized", **self.conditions_federated}

    @property
    def condition_setups(self) -> dict[str, str]:
        return {
            self.condition_centralized: Setup.CENTRALIZED.value,
            **{c: Setup.FEDERATED.value for c in self.conditions_federated},
        }

    @property
    def pattern_tag(self) -> re.Pattern:
        return re.compile(rf"^{self.tag_prefix}_(?P<split>.+?)_s(?P<seed>\d+)_")


DATASETS = {
    "adni": DatasetSpec(
        tag_prefix="adni_harmonized",
        condition_centralized="iid-centralized",
        conditions_federated={
            "iid-federated": "IID",
            "noniid-site": "Non-IID (site)",
            "noniid-diag": "Non-IID (diag.)",
        },
        condition_splits={
            "iid-centralized": "iid",
            "iid-federated": "iid",
            "noniid-site": "noniid_site",
            "noniid-diag": "noniid_diag",
        },
        split_baselines="iid",
        default_dpath_results=Path(DEFAULT_DPATH_RESULTS) / "2026_11_08",
    ),
    # full disease coverage
    "synthetic": DatasetSpec(
        tag_prefix="synth_fullcov",
        condition_centralized="overlap100-centralized",
        conditions_federated={
            f"{condition}-federated": label
            for condition, label in SYNTHETIC_CONDITION_LABELS.items()
        },
        condition_splits={
            "overlap100-centralized": "overlap100",
            **{
                f"{condition}-federated": condition
                for condition in SYNTHETIC_CONDITION_LABELS
            },
        },
        split_baselines="overlap100",
        default_dpath_results=Path(DEFAULT_DPATH_RESULTS) / "2026_11_11",
        # time = months / 96 (see get_simulation_config_from_adni_results.py)
        time_scale=96.0,
        has_ground_truth=True,
    ),
}
# same simulation with ADNI-like disease coverage
DATASETS["synthetic_adnilike"] = dataclasses.replace(
    DATASETS["synthetic"], tag_prefix="synth_adnilike"
)
DEFAULT_DATASET = "adni"
CONDITION_GROUND_TRUTH = "ground-truth"
LABEL_GROUND_TRUTH = "Ground truth"
# default Fed-BioMed run (FedAvg + Adam): ...-fedavg[_adam]-<seed>
PATTERN_RUN_TAG_DEFAULT = re.compile(r"-fedavg(?:_adam)?-(\d+|no_seed)$")
LABEL_DPMOST = "DPMoSt"
LABEL_LEASPY = "Leaspy"
BASELINE_PATTERNS = {
    LABEL_DPMOST: "dpmost-*-fit_quality*.tsv",
    LABEL_LEASPY: "leaspy-*-fit_quality*.tsv",
}
BASELINE_LABELS = tuple(BASELINE_PATTERNS)

# Fed-BioMed personalization key, see get_heldout_personalization_key
PATTERN_PERSONALIZATION = re.compile(r"^(?P<algorithm>[a-z]+)_(?P<seed>\d+)$")

SETUP_LINESTYLES = {Setup.FEDERATED.value: "-", Setup.CENTRALIZED.value: "--"}
LINESTYLE_GROUND_TRUTH = ":"
# group of subjects without a group column (see fl_prog.results._build_fitted_model)
GROUP_UNKNOWN = "unknown"

METRICS = {"mean_absolute_error": "MAE", "r2_score": "R²"}
BIOMARKER_ALL = "all"
# group trajectory plots: clinical scores, then left/right imaging pairs
BIOMARKER_ORDER = [
    "ADAS13",
    "FAQ",
    "MMSE",
    "Left-Hippocampus",
    "Right-Hippocampus",
    "lh_entorhinal_thickness",
    "rh_entorhinal_thickness",
    "lh_parahippocampal_thickness",
    "rh_parahippocampal_thickness",
]
# sets in the time shift plots; held-out subjects' time shifts use all their visits
SET_NAME_ORDER = ["train", "test"]
COL_SEED = "seed"

LABEL_TIME_SHIFTED = "Months"
LABEL_BIOMARKER = "Biomarker values"
LABEL_TIME_SHIFT = "Months"

DNAME_FIGURES = "figures"


# ----- file discovery -----


def _single_match(fpaths: list[Path], description: str) -> Path:
    # dedupe paths reached through e.g. a "latest" symlink
    fpaths = sorted({fpath.resolve() for fpath in fpaths})
    if len(fpaths) != 1:
        raise click.ClickException(
            f"Expected exactly one {description}, found {len(fpaths)}: "
            + ", ".join(str(fpath) for fpath in fpaths)
        )
    return fpaths[0]


def find_tag_dirs(dpath_results: Path, spec: DatasetSpec) -> dict[int, dict[str, Path]]:
    """Seed -> data split -> tag directory."""
    dpaths_tag = {}
    for dpath in sorted(dpath_results.iterdir()):
        match = spec.pattern_tag.search(dpath.name)
        if not dpath.is_dir() or match is None:
            continue
        seed = int(match.group("seed"))
        dpaths_tag.setdefault(seed, {})[match.group("split")] = dpath
    if not dpaths_tag:
        raise click.ClickException(f"No multi-seed tag directories in {dpath_results}")
    return dpaths_tag


def find_seed_files(
    dpaths_split: dict[str, Path], seed: int, spec: DatasetSpec
) -> dict[str, Path]:
    """Fed-BioMed results JSON (default FedAvg + Adam run) per condition, for one
    seed. The file must have results for the condition's setup (e.g. centralized
    and federated fits of the same split can be in one file or in two)."""
    fpaths = {}
    for condition, split in spec.condition_splits.items():
        if split not in dpaths_split:
            raise click.ClickException(f"No {split!r} tag directory for seed {seed}")
        setup = spec.condition_setups[condition]
        fpaths[condition] = _single_match(
            filter_result_files_by_setup(
                [
                    fpath
                    for fpath in dpaths_split[split].glob(
                        f"{MODEL_FEDBIOMED}-*{SUFFIX_ESTIMATED_PARAMS}"
                    )
                    if PATTERN_RUN_TAG_DEFAULT.search(get_run_tag(fpath))
                ],
                setup,
            ),
            f"Fed-BioMed {setup} results file for condition {condition!r}, seed {seed}",
        )
    return fpaths


def _get_metrics_command(fpath_json_results: Path) -> str:
    settings = load_json(fpath_json_results)["settings"]
    return (
        f"scripts/get_metrics.py --tag {settings['tag']} "
        f"--results-dir {settings['dpath_results']}"
    )


# ----- group trajectories -----


def _get_personalization(
    fpath_json_results: Path, json_results: dict
) -> tuple[str, int]:
    """Algorithm and seed used to personalize held-out subjects (as stored by
    ``get_metrics.py``, e.g. ``minimize_3791``)."""
    keys = list(json_results.get(KEY_INDIVIDUAL_PARAMETERS_TEST, {}))
    matches = [m for m in map(PATTERN_PERSONALIZATION.match, keys) if m]
    if len(matches) != 1:
        raise click.ClickException(
            f"Expected one personalization of held-out subjects in "
            f"{fpath_json_results}, found {keys}. "
            f"Run first: {_get_metrics_command(fpath_json_results)}"
        )
    return matches[0].group("algorithm"), int(matches[0].group("seed"))


def _rescale_time(fitted_model: FittedModel, time_scale: float) -> FittedModel:
    """Express times in other units (e.g. months for unscaled synthetic data)."""
    if time_scale == 1:
        return fitted_model
    return dataclasses.replace(
        fitted_model,
        time_factor=fitted_model.time_factor * time_scale,
        df_subject=fitted_model.df_subject.assign(
            **{COL_TIME_SHIFT: fitted_model.df_subject[COL_TIME_SHIFT] * time_scale}
        ),
        df_long=fitted_model.df_long.assign(
            **{
                COL_TIME_SHIFTED: fitted_model.df_long[COL_TIME_SHIFTED] * time_scale,
                COL_TIME: fitted_model.df_long[COL_TIME] * time_scale,
            }
        ),
    )


def _get_time_offset(json_results: dict, setup: str) -> float:
    """Offset (model units) of the estimated disease-time axis relative to the true
    one: mean difference between estimated and true biomarker midpoints (x0), as
    in the "_shifted" recovery metrics of get_metrics.py."""
    json_data = load_json_data(json_results)
    subjects_by_node = json_data["subjects_by_node"]
    return get_midpoint_offset(
        true_params_by_subject(subjects_by_node, json_data["params"]),
        estimated_params_by_setup(json_results, subjects_by_node)[setup],
    )


def _shift_time(fitted_model: FittedModel, offset: float) -> FittedModel:
    """Move the disease-time origin: estimated time t is plotted at t - offset
    (offset in model units; translation only, the time scale is kept)."""
    trajectory_fn = fitted_model.trajectory_fn
    offset_plotted = offset * fitted_model.time_factor
    return dataclasses.replace(
        fitted_model,
        trajectory_fn=lambda t: trajectory_fn(np.asarray(t) + offset),
        df_subject=fitted_model.df_subject.assign(
            **{COL_TIME_SHIFT: fitted_model.df_subject[COL_TIME_SHIFT] - offset_plotted}
        ),
        df_long=fitted_model.df_long.assign(
            **{
                COL_TIME_SHIFTED: fitted_model.df_long[COL_TIME_SHIFTED]
                - offset_plotted
            }
        ),
    )


def load_ground_truth(fpath_json_results: Path) -> dict:
    """Data (with true parameters) of a synthetic run."""
    json_data = load_json(
        Path(load_json(fpath_json_results)["settings"]["fpath_config"])
    )
    if "params" not in json_data:
        raise click.ClickException(
            f"No true parameters in the data of {fpath_json_results}"
        )
    return json_data


def load_fitted_model(
    fpath_json_results: Path,
    expected_setup: str,
    time_scale: float = 1.0,
    align_translation: bool = False,
) -> FittedModel:
    """Fitted model of one setup, with times in plotted units.

    With ``align_translation`` (needs true parameters), the arbitrary origin of the
    estimated disease-time axis is moved so that the estimated biomarker midpoints
    are centered on the true ones (mean difference 0; the time scale is not
    changed)."""
    json_results = load_json(fpath_json_results)
    if not json_results.get("results"):
        raise click.ClickException(
            f"No results in {fpath_json_results} (run still in progress?)"
        )
    tag = json_results["settings"]["tag"]
    algorithm, personalize_seed = _get_personalization(fpath_json_results, json_results)
    fitted_models = [
        fitted_model
        for fitted_model in get_fitted_models(
            fpath_json_results,
            json_results,
            tag,
            fedbiomed_personalize_algorithm=algorithm,
            personalize_seed=personalize_seed,
        )
        if fitted_model.setup == expected_setup
    ]
    if len(fitted_models) != 1:
        raise click.ClickException(
            f"Expected one {expected_setup!r} model in {fpath_json_results}, "
            f"found {len(fitted_models)}"
        )
    fitted_model = _rescale_time(fitted_models[0], time_scale)
    if align_translation:
        fitted_model = _shift_time(
            fitted_model, _get_time_offset(json_results, expected_setup)
        )
    return fitted_model


def _true_trajectory(
    json_data: dict, t_grid: np.ndarray, time_scale: float
) -> pd.DataFrame:
    """Ground-truth group trajectory (long format) at times in plotted units."""
    params = json_data["params"]
    y_true = multivariate_logistic(
        t_grid / time_scale,
        np.asarray(params["k_values"]),
        np.asarray(params["x0_values"]),
        np.asarray(params["scaling_factors"]),
    )
    return (
        pd.DataFrame(y_true, columns=json_data["cols"]["cols_biomarker"])
        .assign(**{COL_TIME_SHIFTED: t_grid})
        .melt(id_vars=[COL_TIME_SHIFTED], var_name=COL_BIOMARKER, value_name=COL_Y)
    )


@dataclass
class GroupTrajectories:
    """Group trajectories to plot: panel title -> setup -> seed -> trajectory."""

    panels: dict[str, dict[str, dict[int, pd.DataFrame]]]
    biomarker_order: list[str]
    biomarker_labels: dict[str, str]
    palette: dict[str, tuple]
    df_true: pd.DataFrame | None
    t_min: float
    t_max: float


def prepare_group_trajectories(
    fitted_models: dict[int, dict[str, FittedModel]],
    spec: DatasetSpec,
    json_ground_truth: dict | None = None,
    biomarker_labels: dict[str, str] | None = None,
    n_points: int = 500,
    xlim_percentile: float = 100,
) -> GroupTrajectories:
    """One panel per federated condition (federated vs centralized fits).

    With a ground truth, a first panel compares the centralized fits to the ground
    truth, and federated panels only show the federated fits.

    The x-axis starts at 0 and ends at the ``xlim_percentile`` percentile of the
    shifted visit times of the centralized models (all seeds pooled; 100: latest
    visit)."""
    models_centralized = [
        models[spec.condition_centralized] for models in fitted_models.values()
    ]
    t_min = 0.0
    t_shifted = np.concatenate(
        [
            # df_long has one row per visit and biomarker: one per visit
            m.df_long.drop_duplicates([COL_SUBJECT, COL_TIME])[COL_TIME_SHIFTED]
            for m in models_centralized
        ]
    )
    t_max = np.nanpercentile(t_shifted, xlim_percentile)
    t_grid = np.linspace(t_min, t_max, n_points)

    cols_biomarker = list(models_centralized[0].cols_biomarker)
    biomarker_labels = biomarker_labels or {}
    # preferred order first, then any other biomarkers
    order_index = {b: i for i, b in enumerate(BIOMARKER_ORDER)}
    biomarker_order = sorted(
        cols_biomarker,
        # column name first, then label (e.g. synthetic names given as labels)
        key=lambda b: order_index.get(
            b, order_index.get(biomarker_labels.get(b, b), len(order_index))
        ),
    )
    dfs_centralized = {
        seed: models[spec.condition_centralized].get_df_trajectory(t_grid)
        for seed, models in fitted_models.items()
    }
    df_true = (
        None
        if json_ground_truth is None
        else _true_trajectory(json_ground_truth, t_grid, spec.time_scale)
    )

    panels = {}
    if df_true is not None:
        panels[spec.condition_labels[spec.condition_centralized]] = {
            Setup.CENTRALIZED.value: dfs_centralized
        }
    for condition, title in spec.conditions_federated.items():
        panels[title] = {
            Setup.FEDERATED.value: {
                seed: models[condition].get_df_trajectory(t_grid)
                for seed, models in fitted_models.items()
            }
        }
        if df_true is None:
            # no centralized panel: compare to the centralized fits in each panel
            panels[title][Setup.CENTRALIZED.value] = dfs_centralized
    return GroupTrajectories(
        panels=panels,
        biomarker_order=biomarker_order,
        biomarker_labels=biomarker_labels,
        palette=get_biomarker_palette(biomarker_order),
        df_true=df_true,
        t_min=t_min,
        t_max=t_max,
    )


def draw_group_trajectories(
    axes,
    trajectories: GroupTrajectories,
    linewidth: float = 1.5,
    alpha: float = 0.25,
    linewidth_mean: float = 4.0,
    alpha_mean: float = 0.8,
    title_fontsize=None,
):
    """Draw each panel into the given axes (sharing x and y): one faint line per
    seed and a thick line for the mean over seeds (federated solid vs centralized
    dashed, ground truth dotted)."""
    palette = trajectories.palette
    for ax, (title, dfs_by_setup) in zip(axes, trajectories.panels.items()):
        for setup, dfs_seeds in dfs_by_setup.items():
            df_seeds = pd.concat(dfs_seeds.values(), ignore_index=True)
            df_mean = (
                df_seeds.groupby([COL_BIOMARKER, COL_TIME_SHIFTED])[COL_Y]
                .mean()
                .reset_index()
            )
            # faint line per seed, then mean over seeds on top
            for df_model, lw, a in [
                *[(df, linewidth, alpha) for df in dfs_seeds.values()],
                (df_mean, linewidth_mean, alpha_mean),
            ]:
                for biomarker in trajectories.biomarker_order:
                    df_biomarker = df_model.query(f"{COL_BIOMARKER} == @biomarker")
                    ax.plot(
                        df_biomarker[COL_TIME_SHIFTED],
                        df_biomarker[COL_Y],
                        color=palette[biomarker],
                        linestyle=SETUP_LINESTYLES[setup],
                        linewidth=lw,
                        alpha=a,
                    )
        if trajectories.df_true is not None:
            for biomarker in trajectories.biomarker_order:
                df_biomarker = trajectories.df_true.query(
                    f"{COL_BIOMARKER} == @biomarker"
                )
                ax.plot(
                    df_biomarker[COL_TIME_SHIFTED],
                    df_biomarker[COL_Y],
                    color=palette[biomarker],
                    linestyle=LINESTYLE_GROUND_TRUTH,
                    linewidth=linewidth_mean,
                    alpha=alpha_mean,
                )
        # same size as axis labels by default
        ax.set_title(
            title,
            fontweight="bold",
            fontsize=title_fontsize or plt.rcParams["axes.labelsize"],
        )
        ax.set_xlabel(LABEL_TIME_SHIFTED)
        # few ticks: large fonts and possibly long time ranges
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
    axes[0].set_xlim(trajectories.t_min, trajectories.t_max)
    axes[0].set_ylabel(LABEL_BIOMARKER)


LEGEND_HEADERS = ("Biomarker", "Setup")


def get_group_trajectories_legend(
    trajectories: GroupTrajectories, linewidth: float = 2.5
) -> tuple[list, list[str]]:
    """Legend handles and labels with section headers (``LEGEND_HEADERS``)."""

    def _header():
        return Line2D([], [], linestyle="none")

    linestyles = {
        setup.capitalize(): linestyle for setup, linestyle in SETUP_LINESTYLES.items()
    }
    if trajectories.df_true is not None:
        linestyles[LABEL_GROUND_TRUTH] = LINESTYLE_GROUND_TRUTH
    handles = (
        [_header()]
        + [
            Line2D([], [], color=trajectories.palette[b], linewidth=linewidth)
            for b in trajectories.biomarker_order
        ]
        + [_header(), _header()]
        + [
            Line2D([], [], color="black", linestyle=linestyle, linewidth=linewidth)
            for linestyle in linestyles.values()
        ]
    )
    labels = (
        [LEGEND_HEADERS[0]]
        + [
            trajectories.biomarker_labels.get(b, b)
            for b in trajectories.biomarker_order
        ]
        + ["", LEGEND_HEADERS[1]]
        + list(linestyles)
    )
    return handles, labels


def plot_group_trajectories_by_condition(
    fitted_models: dict[int, dict[str, FittedModel]],
    spec: DatasetSpec,
    json_ground_truth: dict | None = None,
    biomarker_labels: dict[str, str] | None = None,
    n_points: int = 500,
    # default: 5 x 5 per panel
    figsize: tuple[float, float] | None = None,
    context: str = "poster",
    font_scale: float = 1.0,
    # legend keeps the previous ("talk" context) text size
    legend_fontsize: float = sns.plotting_context("talk", font_scale=0.9)[
        "legend.fontsize"
    ],
    linewidth: float = 1.5,
    alpha: float = 0.25,
    linewidth_mean: float = 4.0,
    alpha_mean: float = 0.8,
    xlim_percentile: float = 100,
) -> plt.Figure:
    """Group trajectories in a single row of panels (see
    ``prepare_group_trajectories``), with the legend on the right."""
    trajectories = prepare_group_trajectories(
        fitted_models,
        spec,
        json_ground_truth=json_ground_truth,
        biomarker_labels=biomarker_labels,
        n_points=n_points,
        xlim_percentile=xlim_percentile,
    )
    if figsize is None:
        figsize = (5 * len(trajectories.panels), 5)

    with sns.plotting_context(context, font_scale=font_scale):
        fig, axes = plt.subplots(
            ncols=len(trajectories.panels),
            figsize=figsize,
            sharex=True,
            sharey=True,
            squeeze=False,
            layout="constrained",
        )
        draw_group_trajectories(
            list(axes.flat),
            trajectories,
            linewidth=linewidth,
            alpha=alpha,
            linewidth_mean=linewidth_mean,
            alpha_mean=alpha_mean,
        )
        sns.despine(fig)

        # single legend with section headers, inside the figure bounds
        handles, labels = get_group_trajectories_legend(trajectories)
        legend = fig.legend(
            handles,
            labels,
            loc="outside right center",
            handlelength=2,
            labelspacing=0.3,
            fontsize=legend_fontsize,
            frameon=False,
        )
        for text in legend.get_texts():
            if text.get_text() in LEGEND_HEADERS:
                text.set_fontweight("bold")
    return fig


# ----- fit metrics -----


def load_fit_quality(fpath_tsv: Path, model_label: str, setup: str) -> pd.DataFrame:
    df = pd.read_csv(fpath_tsv, sep="\t")
    if setup in set(df["setup"]):
        df = df.query("setup == @setup")
    return df.assign(**{COL_MODEL: model_label})


def get_df_metrics_seed(
    fpaths_json: dict[str, Path], dpath_baselines: Path, seed: int, spec: DatasetSpec
) -> pd.DataFrame:
    dfs = []
    for condition, fpath_json in fpaths_json.items():
        fpaths_tsv = list(
            fpath_json.parent.glob(f"{get_run_tag(fpath_json)}-fit_quality*.tsv")
        )
        if not fpaths_tsv:
            raise click.ClickException(
                f"Metrics file not found for {fpath_json}. "
                f"Run first: {_get_metrics_command(fpath_json)}"
            )
        fpath_tsv = _single_match(
            fpaths_tsv, f"Fed-BioMed metrics file ({condition}, seed {seed})"
        )
        dfs.append(
            load_fit_quality(
                fpath_tsv,
                spec.condition_labels[condition],
                spec.condition_setups[condition],
            )
        )

    for label, pattern in BASELINE_PATTERNS.items():
        fpath_tsv = _single_match(
            list(dpath_baselines.glob(pattern)), f"{label} metrics file (seed {seed})"
        )
        dfs.append(load_fit_quality(fpath_tsv, label, Setup.CENTRALIZED.value))

    df_metrics = pd.concat(dfs, ignore_index=True).assign(**{COL_SEED: seed})
    return df_metrics.query(
        "metric in @METRICS and col_biomarker == @BIOMARKER_ALL"
    ).reset_index(drop=True)


def _plot_metric_seeds(
    ax: plt.Axes,
    df_metrics: pd.DataFrame,
    metric: str,
    model_order: list[str],
    set_name_order: list[str],
    palette: dict,
    offset_step: float = 0.2,
    jitter: float = 0.04,
    rng_seed: int = 0,
) -> None:
    """Plot one row per model on ``ax``, with the sets offset vertically.

    Each seed is a small dot; the mean is a large marker with a ±SD error bar.
    """
    rng = np.random.default_rng(rng_seed)
    markers = {
        set_name: "o" if set_name == LABEL_TRAIN else "D" for set_name in set_name_order
    }
    y_positions = {model: i for i, model in enumerate(model_order)}
    offset = offset_step * (len(set_name_order) - 1) / 2
    offsets = dict(
        zip(set_name_order, np.linspace(-offset, offset, len(set_name_order)))
    )
    n_ours = sum(model not in BASELINE_LABELS for model in model_order)

    df_metric = df_metrics.query("metric == @metric")
    for set_name in set_name_order:
        df_set = df_metric.query("set_name == @set_name")
        y_seeds = (
            df_set[COL_MODEL].map(y_positions)
            + offsets[set_name]
            + rng.uniform(-jitter, jitter, len(df_set))
        )
        ax.scatter(
            df_set["value"],
            y_seeds,
            s=18,
            color=palette[set_name],
            marker=markers[set_name],
            alpha=0.5,
            linewidth=0,
            zorder=2,
        )
        df_summary = df_set.groupby(COL_MODEL)["value"].agg(["mean", "std"])
        y_summary = df_summary.index.map(y_positions) + offsets[set_name]
        ax.errorbar(
            df_summary["mean"],
            y_summary,
            xerr=df_summary["std"],
            fmt="none",
            ecolor=palette[set_name],
            elinewidth=2,
            capsize=0,
            zorder=3,
        )
        ax.scatter(
            df_summary["mean"],
            y_summary,
            s=90,
            color=palette[set_name],
            marker=markers[set_name],
            edgecolor="white",
            linewidth=1,
            label=set_name,
            zorder=4,
        )
    # separate our model from the baselines (DPMoSt, Leaspy)
    if 0 < n_ours < len(model_order):
        ax.axhline(n_ours - 0.5, color="grey", linewidth=1, ls=":")
    ax.set_title(METRICS[metric], fontweight="bold")
    ax.set_xlabel("")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.tick_params(axis="y", length=0)
    ax.margins(x=0.1)
    ax.set_ylim(len(model_order) - 0.5, -0.5)  # first model on top
    ax.grid(axis="x", color="0.92", zorder=0)
    ax.set_axisbelow(True)
    sns.despine(ax=ax, left=True)


def get_fit_set_palette(set_name_order: list[str]) -> dict:
    """Map train to grey and the other sets to darker and darker shades of one colour."""
    test_sets = [s for s in set_name_order if s != LABEL_TRAIN]
    color_test = next(iter(get_categorical_palette(["test"]).values()))
    shades = sns.light_palette(color_test, n_colors=len(test_sets) + 1)[1:]
    return {LABEL_TRAIN: "0.6", **dict(zip(test_sets, shades, strict=True))}


def plot_metrics(
    df_metrics: pd.DataFrame,
    model_order: list[str],
    set_name_order: list[str],
    width: float = 10,
    height_per_model: float = 0.6,
    context: str = "talk",
    font_scale: float = 0.8,
) -> plt.Figure:
    """Plot MAE and R² for each model and set in ``set_name_order``, over seeds."""
    palette = get_fit_set_palette(set_name_order)
    figsize = (width, max(4, height_per_model * len(model_order)))
    with sns.plotting_context(context, font_scale=font_scale):
        legend_fontsize = plt.rcParams["legend.fontsize"]
        fig, axes = plt.subplots(
            ncols=len(METRICS), figsize=figsize, sharey=True, layout="constrained"
        )
        for ax, metric in zip(axes, METRICS):
            _plot_metric_seeds(
                ax, df_metrics, metric, model_order, set_name_order, palette
            )
        axes[0].set_yticks(range(len(model_order)), model_order)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            [f"{label}\n(mean ± SD)" for label in labels],
            loc="outside right center",
            fontsize=legend_fontsize,
            handletextpad=0.2,
            labelspacing=1.0,
            frameon=False,
        )
    return fig


# ----- time shifts -----


def _lighten(color, amount: float = 0.55):
    """Blend a color with white."""
    rgb = np.array(to_rgb(color))
    return tuple(rgb + (1 - rgb) * amount)


def draw_time_shifts(
    ax,
    df_subject: pd.DataFrame,
    condition_labels: dict[str, str],
    light_set: str = "test",
    title: str | None = "Estimated time shifts",
) -> tuple[tuple[list, list[str]], tuple[list, list[str]]]:
    """Box plot of time shifts for each Fed-BioMed condition (x-axis), colored
    by diagnosis group, with train and held-out test subjects side by side
    (``light_set`` lighter).

    Returns legend (handles, labels) for the groups (empty if there are no groups)
    and for the train/test shading."""
    group_order = get_group_order(df_subject[COL_GROUP].dropna())
    group_palette = get_group_palette(group_order)
    set_names = [s for s in SET_NAME_ORDER if s in set(df_subject[COL_SET_NAME])]
    conditions = [c for c in condition_labels if c in set(df_subject["condition"])]

    # one box per (group, set), train then test within each group
    col_hue = "group_set"
    df_plot = df_subject.assign(
        **{col_hue: df_subject[COL_GROUP] + "|" + df_subject[COL_SET_NAME]}
    )
    hue_order = [f"{g}|{s}" for g in group_order for s in set_names]
    palette = {
        f"{g}|{s}": (_lighten(group_palette[g]) if s == light_set else group_palette[g])
        for g in group_order
        for s in set_names
    }
    sns.boxplot(
        data=df_plot,
        x="condition",
        y=COL_TIME_SHIFT,
        order=conditions,
        hue=col_hue,
        hue_order=hue_order,
        palette=palette,
        fliersize=1.5,
        linewidth=0.8,
        linecolor="#3f3f3f",
        gap=0.1,
        width=0.95,  # little space between conditions
        legend=False,
        ax=ax,
    )
    ax.set_xticks(
        range(len(conditions)),
        [condition_labels[c].replace(" (", "\n(") for c in conditions],
    )
    ax.set_xlabel("")
    ax.set_ylabel(LABEL_TIME_SHIFT)
    if title is not None:
        ax.set_title(title, fontweight="bold")

    # no group legend if there are no groups (e.g. synthetic data)
    groups_legend = [] if group_order == [GROUP_UNKNOWN] else group_order
    group_handles = [
        Patch(facecolor=group_palette[group], edgecolor="#3f3f3f")
        for group in groups_legend
    ]
    set_handles = [
        Patch(
            facecolor=_lighten("grey") if s == light_set else "grey",
            edgecolor="#3f3f3f",
        )
        for s in set_names
    ]
    return (group_handles, groups_legend), (set_handles, set_names)


def plot_time_shifts(
    df_subject: pd.DataFrame,
    condition_labels: dict[str, str],
    title: str = "Estimated time shifts",
    figsize: tuple[float, float] = (9, 4.5),
    context: str = "talk",
    font_scale: float = 0.8,
) -> plt.Figure:
    """Time shifts by condition (see ``draw_time_shifts``, test lighter), with the
    legend below."""
    with sns.plotting_context(context, font_scale=font_scale):
        legend_fontsize = plt.rcParams["legend.fontsize"]
        fig, ax = plt.subplots(figsize=figsize, layout="constrained")
        (group_handles, group_labels), (set_handles, set_names) = draw_time_shifts(
            ax, df_subject, condition_labels, title=title
        )
        # groups, then train/test shading (grey)
        labels = group_labels + [s.capitalize() for s in set_names]
        fig.legend(
            group_handles + set_handles,
            labels,
            loc="outside lower center",
            ncols=len(labels),
            fontsize=legend_fontsize,
            handlelength=1,
            handletextpad=0.4,
            columnspacing=0.8,
            frameon=False,
        )
        sns.despine(ax=ax)
    return fig


# ----- main -----


def plot_fedbiomed_comparison_seeds(
    dataset: str,
    dpath_results: Path | None,
    seeds: tuple[int, ...],
    dpath_out: Path | None,
    xlim_percentile: float = 100,
    skip_trajectories: bool = False,
    skip_metrics: bool = False,
    skip_time_shifts: bool = False,
    align_translation: bool = True,
    forecast_after: int | None = None,
):
    spec = DATASETS[dataset]
    dpath_results = dpath_results or spec.default_dpath_results
    if not dpath_results.is_dir():
        raise click.ClickException(f"Results directory not found: {dpath_results}")
    dpath_out = dpath_out or dpath_results / DNAME_FIGURES
    # dataset in the output names (keep original names for ADNI), as in
    # collect_seed_metrics.py
    suffix = "" if dataset == DEFAULT_DATASET else f"-{spec.tag_prefix}"

    dpaths_tag = find_tag_dirs(dpath_results, spec)
    if seeds:
        missing = sorted(set(seeds) - set(dpaths_tag))
        if missing:
            raise click.ClickException(f"No results for seeds {missing}")
        dpaths_tag = {seed: dpaths_tag[seed] for seed in seeds}
    dpaths_tag = dict(sorted(dpaths_tag.items()))
    fpaths_json = {
        seed: find_seed_files(dpaths_split, seed, spec)
        for seed, dpaths_split in dpaths_tag.items()
    }
    print(f"Seeds: {list(fpaths_json)}")

    sns.set_theme(context="notebook", style="ticks")

    if not skip_metrics:
        try:
            df_metrics, set_name_order = get_fit_sets(
                pd.concat(
                    [
                        get_df_metrics_seed(
                            fpaths_json[seed],
                            dpaths_tag[seed][spec.split_baselines],
                            seed,
                            spec,
                        )
                        for seed in fpaths_json
                    ],
                    ignore_index=True,
                ),
                forecast_after=forecast_after,
            )
        except ValueError as exception:
            raise click.ClickException(str(exception)) from exception
        model_order = [*spec.condition_labels.values(), *BASELINE_LABELS]
        fpath_tsv = dpath_out / f"fit_metrics-comparison-seeds{suffix}.tsv"
        fpath_tsv.parent.mkdir(parents=True, exist_ok=True)
        df_metrics.to_csv(fpath_tsv, sep="\t", index=False)
        print(f"Saved metrics to {fpath_tsv}")
        print(
            df_metrics.pivot_table(
                index=COL_MODEL,
                columns=["set_name", "metric"],
                values="value",
                aggfunc=["mean", "std", "count"],
            )
            .reindex(model_order)
            .round(4)
            .to_string()
        )
        fig = plot_metrics(df_metrics, model_order, set_name_order)
        save_fig_formats(
            fig,
            dpath_out / f"fit_metrics-comparison-seeds{suffix}",
            bbox_inches=None,
        )
        plt.close(fig)

    if skip_trajectories and skip_time_shifts:
        return

    fitted_models = {
        seed: {
            condition: load_fitted_model(
                fpath,
                spec.condition_setups[condition],
                time_scale=spec.time_scale,
                align_translation=align_translation and spec.has_ground_truth,
            )
            for condition, fpath in fpaths.items()
        }
        for seed, fpaths in fpaths_json.items()
    }

    # ground truth: same true parameters for all seeds/conditions, but true time
    # shifts differ by seed
    json_ground_truth = {}
    if spec.has_ground_truth:
        json_ground_truth = {
            seed: load_ground_truth(fpaths[spec.condition_centralized])
            for seed, fpaths in fpaths_json.items()
        }
    json_ground_truth_first = next(iter(json_ground_truth.values()), None)
    # synthetic data: generic labels (Biomarker 1, 2, ...) in column order
    biomarker_labels = None
    if json_ground_truth_first is not None:
        biomarker_labels = {
            col: f"Biomarker {i}"
            for i, col in enumerate(
                json_ground_truth_first["cols"]["cols_biomarker"], start=1
            )
        }

    if not skip_trajectories:
        fig = plot_group_trajectories_by_condition(
            fitted_models,
            spec,
            json_ground_truth=json_ground_truth_first,
            biomarker_labels=biomarker_labels,
            xlim_percentile=xlim_percentile,
        )
        # constrained layout keeps everything inside the figure: keep exact size
        save_fig_formats(
            fig,
            dpath_out / f"group_trajectories-by_condition-seeds{suffix}",
            bbox_inches=None,
        )
        plt.close(fig)

    if not skip_time_shifts:
        for seed, models in fitted_models.items():
            dfs_subject = [
                fitted_model.df_subject.assign(condition=condition)
                for condition, fitted_model in models.items()
            ]
            condition_labels = spec.condition_labels
            if seed in json_ground_truth:
                df_true = get_df_true_subject(json_ground_truth[seed])
                group = dfs_subject[0].drop_duplicates(COL_SUBJECT)
                group = group.set_index(COL_SUBJECT)[COL_GROUP]
                dfs_subject.insert(
                    0,
                    df_true.assign(
                        **{
                            COL_TIME_SHIFT: df_true[COL_TIME_SHIFT] * spec.time_scale,
                            COL_GROUP: df_true[COL_SUBJECT].astype(str).map(group),
                            "condition": CONDITION_GROUND_TRUTH,
                        }
                    ),
                )
                condition_labels = {
                    CONDITION_GROUND_TRUTH: LABEL_GROUND_TRUTH,
                    **condition_labels,
                }
            df_subject = pd.concat(dfs_subject, ignore_index=True)
            fig = plot_time_shifts(df_subject, condition_labels)
            save_fig_formats(
                fig,
                dpath_out / f"time_shifts-by_condition-seed{seed}{suffix}",
                bbox_inches=None,
            )
            plt.close(fig)


@click.command(context_settings=CLICK_CONTEXT_SETTINGS)
@click.option(
    "--dataset",
    type=click.Choice(list(DATASETS)),
    default=DEFAULT_DATASET,
    show_default=True,
    help="Multi-seed run to plot (sets the conditions to compare).",
)
@click.option(
    "--results-dir",
    "dpath_results",
    type=click.Path(path_type=Path, file_okay=False, exists=True),
    default=None,
    help="Dated results directory of a multi-seed run (one subdirectory per "
    "data split and seed). Default: results/2026_11_08 (adni), "
    "results/2026_11_11 (synthetic, synthetic_adnilike).",
)
@click.option(
    "--seed",
    "seeds",
    type=int,
    multiple=True,
    help="Seed to include (can be repeated). Default: all seeds found.",
)
@click.option(
    "--output-dir",
    "dpath_out",
    type=click.Path(path_type=Path, file_okay=False),
    default=None,
    help="Output directory for figures. Default: <results-dir>/figures.",
)
@click.option(
    "--xlim-percentile",
    type=click.FloatRange(0, 100, min_open=True),
    default=100,
    show_default=True,
    help="Group trajectories x-axis ends at this percentile of the shifted visit "
    "times of the centralized models (all seeds pooled). 100: latest visit.",
)
@click.option(
    "--align-translation/--no-align-translation",
    default=True,
    show_default=True,
    help="With a ground truth (synthetic data): shift each fitted model's "
    "disease-time axis so that its biomarker midpoints are centered on the true "
    "ones (mean difference 0; time scale unchanged).",
)
@click.option(
    "--forecast-after",
    type=click.IntRange(min=1),
    default=None,
    help="Plot test scores of forecasts after the first K visits. "
    "Default: the largest K in the results.",
)
@click.option("--skip-trajectories", is_flag=True, help="Skip group trajectories.")
@click.option("--skip-metrics", is_flag=True, help="Skip fit metrics.")
@click.option("--skip-time-shifts", is_flag=True, help="Skip per-seed time shifts.")
def main(**params):
    plot_fedbiomed_comparison_seeds(**params)


if __name__ == "__main__":
    main()
