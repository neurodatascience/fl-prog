#!/usr/bin/env python
"""Validate the per-subject acceleration factors of the multi-seed runs (#40).

Regularization diagnostics: is the penalty on log acceleration factors
(``fl_prog.model``) acting mostly as a prior on subjects with little data, and, on
synthetic data, how well are the true acceleration factors recovered?

For each seed, loads:

- the Fed-BioMed fits (centralized and/or each federated condition, see
  ``plot_fedbiomed_comparison_seeds.DATASETS``),
- the Leaspy fit (reference, Gaussian prior on xi = log alpha),
- optionally, centralized fits of penalty variants (``run_alpha_penalty_sweep.sh``,
  one subdirectory per variant in ``--variants-dir``),
- on synthetic data, the true parameters,

joins the number of visits and follow-up span of each subject, and writes
per-subject tables, summary tables and figures:

- distribution of log alpha, and fraction of subjects "pinned" at alpha = 1
- log alpha vs number of visits / follow-up span
- (ADNI) stability of each subject's log alpha across seeds (train subjects)
- (synthetic) recovery of the true log alpha
- federated vs centralized log alpha

Expects held-out subjects to have been personalized (``get_metrics.py``).
"""

import re
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# scripts directory (on sys.path when running this script)
from plot_fedbiomed_comparison_seeds import (
    DATASETS,
    DatasetSpec,
    find_seed_files,
    find_tag_dirs,
    load_fitted_model,
)
from scipy.stats import pearsonr, spearmanr

from fl_prog.plotting import CATEGORICAL_COLORS, COLOR_OTHER, save_fig_formats
from fl_prog.results import (
    COL_ACCELERATION_FACTOR,
    COL_GROUP,
    COL_MODEL,
    COL_SET_NAME,
    COL_SUBJECT,
    COL_TIME,
    COL_TIME_SHIFT,
    COL_Y_OBS,
    KEY_LEASPY_INDIVIDUAL_PARAMETERS,
    MODEL_FEDBIOMED,
    MODEL_LEASPY,
    SUFFIX_ESTIMATED_PARAMS,
    get_df_true_subject,
    get_fitted_models,
    load_json_data,
)
from fl_prog.utils.constants import CLICK_CONTEXT_SETTINGS, Setup
from fl_prog.utils.io import DEFAULT_DPATH_RESULTS, load_json

DPATH_RESULTS_MAIN = (
    Path(DEFAULT_DPATH_RESULTS).resolve().parents[1] / "fl-prog/results"
)
DEFAULT_DPATH_OUT_PARENT = DPATH_RESULTS_MAIN / "2026_11_23-acceleration_factors"
DEFAULT_DATASET = "adni"

LABEL_TRUTH = "Ground truth"
LABEL_LEASPY = "Leaspy"
PREFIX_FEDBIOMED = "Fed-BioMed"
PREFIX_VARIANT = "Local"
COLOR_TRUTH = "black"
# variant family (penalty type) -> sequential palette (light to dark = weak to strong)
VARIANT_PALETTES = {"l1": "Blues", "l2": "Oranges"}
COLOR_VARIANT_NONE = "#4a3aa7"
VARIANT_NONE = "none"
OTHER_VARIANTS_PALETTE = "Dark2"
PATTERN_VARIANT = re.compile(r"^(?P<penalty>l[12])_(?P<lambda>[0-9.e-]+)$")

COL_SEED = "seed"
COL_LOG_ALPHA = "log_alpha"
COL_LOG_ALPHA_TRUE = "log_alpha_true"
COL_N_VISITS = "n_visits"
COL_FOLLOWUP = "followup_months"
COL_N_VISITS_BIN = "n_visits_bin"
COL_SUBSET = "subjects"
SET_NAMES = ("train", "test")
ANALYSIS_REGULARIZATION = "regularization"
ANALYSIS_LEASPY = "leaspy"
ANALYSES = (ANALYSIS_REGULARIZATION, ANALYSIS_LEASPY)
CONDITION_NONE = "none"

# |log alpha| below this counts as pinned at alpha = 1 (1% speed difference)
PINNED_TOL = 0.01
# visits >= this are binned together
MAX_N_VISITS_BIN = 8
# subjects need at least this many train fits to assess stability across seeds
MIN_SEEDS_STABILITY = 3
# subject subsets for recovery metrics (the simulation sets alpha = 1 for subjects
# with 1 visit, which favours pinning them)
SUBSETS = {"all": 1, "2+ visits": 2}
SUBSET_MAIN = "2+ visits"
N_COLS_MAX = 5

PATTERN_LEASPY_KEY = re.compile(r"^(?P<algorithm>.+)-(?P<seed>\d+)-(?P<n_jobs>\d+)$")


# ----- loading -----


def _single_file(dpath: Path, pattern: str) -> Path:
    fpaths = sorted(dpath.glob(pattern))
    if len(fpaths) != 1:
        raise click.ClickException(
            f"Expected one {pattern} file in {dpath}, found {len(fpaths)}"
        )
    return fpaths[0]


def _load_leaspy_model(fpath: Path):
    """Centralized Leaspy fit, with the personalization stored in the file."""
    json_results = load_json(fpath)
    keys = list(json_results.get(KEY_LEASPY_INDIVIDUAL_PARAMETERS, {}))
    matches = [m for m in map(PATTERN_LEASPY_KEY.match, keys) if m]
    if len(matches) != 1:
        raise click.ClickException(
            f"Expected one Leaspy personalization in {fpath}, found {keys}"
        )
    match = matches[0]
    fitted_models = [
        fitted_model
        for fitted_model in get_fitted_models(
            fpath,
            json_results,
            json_results["settings"]["tag"],
            leaspy_algorithm_name=match.group("algorithm"),
            leaspy_seed=int(match.group("seed")),
            leaspy_n_jobs=int(match.group("n_jobs")),
        )
        if fitted_model.setup == Setup.CENTRALIZED.value
    ]
    if len(fitted_models) != 1:
        raise click.ClickException(f"Expected one centralized Leaspy fit in {fpath}")
    return fitted_models[0]


def _get_df_visits(fitted_model, time_scale: float) -> pd.DataFrame:
    """Number of visits (timepoints with any observed biomarker) and follow-up
    span (months) of each subject."""
    df_long = fitted_model.df_long
    df_obs = df_long.loc[df_long[COL_Y_OBS].notna()]
    times = df_obs.drop_duplicates([COL_SUBJECT, COL_TIME]).groupby(COL_SUBJECT)[
        COL_TIME
    ]
    return pd.DataFrame(
        {
            COL_N_VISITS: times.size(),
            COL_FOLLOWUP: (times.max() - times.min()) * time_scale,
        }
    )


def _to_df_subject(df_subject: pd.DataFrame, label: str, seed: int) -> pd.DataFrame:
    df_subject = df_subject[
        [COL_SUBJECT, COL_SET_NAME, COL_TIME_SHIFT, COL_ACCELERATION_FACTOR]
        + ([COL_GROUP] if COL_GROUP in df_subject.columns else [])
    ].copy()
    df_subject[COL_SUBJECT] = df_subject[COL_SUBJECT].astype(str)
    df_subject.insert(0, COL_MODEL, label)
    df_subject.insert(0, COL_SEED, seed)
    df_subject[COL_LOG_ALPHA] = np.log(df_subject[COL_ACCELERATION_FACTOR])
    return df_subject


def _find_variant_dirs(
    dpath_variants: Path | None, names: list[str] | None = None
) -> dict[str, Path]:
    """Variant name -> results directory (with "latest" inside). With ``names``,
    only these variants, in this order."""
    if dpath_variants is None:
        return {}
    dpaths = {
        dpath.name: dpath / "latest"
        for dpath in sorted(dpath_variants.iterdir())
        if (dpath / "latest").is_dir()
    }
    if not dpaths:
        raise click.ClickException(f"No variant results in {dpath_variants}")
    if names:
        missing = set(names) - set(dpaths)
        if missing:
            raise click.ClickException(
                f"Variants {sorted(missing)} not in {dpath_variants}: {sorted(dpaths)}"
            )
        return {name: dpaths[name] for name in names}

    def _sort_key(name: str):
        match = PATTERN_VARIANT.match(name)
        if match is None:
            return ("", 0.0)
        return (match.group("penalty"), float(match.group("lambda")))

    return dict(sorted(dpaths.items(), key=lambda item: _sort_key(item[0])))


def _variant_label(name: str) -> str:
    return f"{PREFIX_VARIANT} {name}"


def get_model_colors(models: list[str], variant_names: list[str]) -> dict:
    colors = {LABEL_TRUTH: COLOR_TRUTH, LABEL_LEASPY: COLOR_OTHER}
    fedbiomed = [m for m in models if m.startswith(PREFIX_FEDBIOMED)]
    colors.update(dict(zip(fedbiomed, CATEGORICAL_COLORS)))
    families: dict[str, list[str]] = {}
    others = []
    for name in variant_names:
        match = PATTERN_VARIANT.match(name)
        if name == VARIANT_NONE:
            colors[_variant_label(name)] = COLOR_VARIANT_NONE
        elif match is None:
            others.append(name)
        else:
            families.setdefault(match.group("penalty"), []).append(name)
    # other variants (e.g. experimental options): distinct hues
    for name, color in zip(
        others, sns.color_palette(OTHER_VARIANTS_PALETTE, n_colors=max(len(others), 3))
    ):
        colors[_variant_label(name)] = color
    for penalty, names in families.items():
        palette = sns.color_palette(VARIANT_PALETTES[penalty], n_colors=len(names) + 1)
        for name, color in zip(names, palette[1:]):
            colors[_variant_label(name)] = color
    return colors


def load_df_subject(
    spec: DatasetSpec,
    dpath_results: Path,
    seeds: list[int] | None,
    conditions: list[str],
    variant_dirs: dict[str, Path],
    dpath_baselines: Path | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Per-subject parameters of all models and seeds, and model labels (in
    plotting order). Leaspy fits are loaded from ``dpath_baselines`` (default:
    ``dpath_results``)."""
    dpaths_tag = find_tag_dirs(dpath_results, spec)
    dpaths_tag_baselines = (
        dpaths_tag if dpath_baselines is None else find_tag_dirs(dpath_baselines, spec)
    )
    dpaths_tag_variants = {
        name: find_tag_dirs(dpath.resolve(), spec)
        for name, dpath in variant_dirs.items()
    }
    models = []
    dfs = []
    for seed in sorted(dpaths_tag):
        if seeds and seed not in seeds:
            continue
        print(f"Loading seed {seed}")
        dpaths_split = dpaths_tag[seed]
        fpaths = find_seed_files(dpaths_split, seed, spec) if conditions else {}
        fpath_leaspy = _single_file(
            dpaths_tag_baselines[seed][spec.split_baselines],
            f"{MODEL_LEASPY}-*{SUFFIX_ESTIMATED_PARAMS}",
        )
        # (fitted model, label, factor to rescale time shifts to plotted units);
        # Leaspy first: its times are in model units (for the visit counts)
        loaded = [
            (_load_leaspy_model(fpath_leaspy), LABEL_LEASPY, spec.time_scale),
            *[
                (
                    load_fitted_model(
                        fpaths[condition],
                        spec.condition_setups[condition],
                        time_scale=spec.time_scale,
                    ),
                    f"{PREFIX_FEDBIOMED} {spec.condition_labels[condition]}",
                    1.0,
                )
                for condition in conditions
            ],
            *[
                (
                    load_fitted_model(
                        _single_file(
                            dpaths_tag_variant[seed][spec.split_baselines],
                            f"{MODEL_FEDBIOMED}-*{SUFFIX_ESTIMATED_PARAMS}",
                        ),
                        Setup.CENTRALIZED.value,
                        time_scale=spec.time_scale,
                    ),
                    _variant_label(name),
                    1.0,
                )
                for name, dpaths_tag_variant in dpaths_tag_variants.items()
            ],
        ]
        df_visits = _get_df_visits(loaded[0][0], spec.time_scale)

        dfs_seed = []
        if spec.has_ground_truth:
            df_true = get_df_true_subject(load_json_data(load_json(fpath_leaspy)))
            df_true[COL_TIME_SHIFT] *= spec.time_scale
            dfs_seed.append(_to_df_subject(df_true, LABEL_TRUTH, seed))
            if LABEL_TRUTH not in models:
                models.append(LABEL_TRUTH)
        for fitted_model, label, time_scale in loaded:
            df_subject = fitted_model.df_subject.copy()
            df_subject[COL_TIME_SHIFT] *= time_scale
            dfs_seed.append(_to_df_subject(df_subject, label, seed))
            if label not in models:
                models.append(label)

        df_seed = pd.concat(dfs_seed, ignore_index=True)
        dfs.append(df_seed.join(df_visits, on=COL_SUBJECT, validate="m:1"))

    df = pd.concat(dfs, ignore_index=True)
    if df[COL_N_VISITS].isna().any():
        raise RuntimeError("Some subjects have no visits")
    df[COL_N_VISITS_BIN] = df[COL_N_VISITS].clip(upper=MAX_N_VISITS_BIN)
    if spec.has_ground_truth:
        df_true = df.loc[df[COL_MODEL] == LABEL_TRUTH]
        df = df.merge(
            df_true[[COL_SEED, COL_SUBJECT, COL_LOG_ALPHA]].rename(
                columns={COL_LOG_ALPHA: COL_LOG_ALPHA_TRUE}
            ),
            on=[COL_SEED, COL_SUBJECT],
            how="left",
            validate="m:1",
        )
    return df, models


# ----- summaries -----


def _spearman(x: pd.Series, y: pd.Series) -> float:
    if len(x) < 3 or x.nunique() < 2 or y.nunique() < 2:
        return np.nan
    return spearmanr(x, y).statistic


def _pearson(x: pd.Series, y: pd.Series) -> float:
    if len(x) < 3 or x.nunique() < 2 or y.nunique() < 2:
        return np.nan
    return pearsonr(x, y).statistic


def _summarize(df: pd.DataFrame) -> pd.Series:
    log_alpha = df[COL_LOG_ALPHA]
    return pd.Series(
        {
            "n_subjects": len(df),
            "mean_log_alpha": log_alpha.mean(),
            "sd_log_alpha": log_alpha.std(),
            "frac_pinned": (log_alpha.abs() < PINNED_TOL).mean(),
            "spearman_log_alpha_n_visits": _spearman(log_alpha, df[COL_N_VISITS]),
            "spearman_abs_log_alpha_n_visits": _spearman(
                log_alpha.abs(), df[COL_N_VISITS]
            ),
            "spearman_abs_log_alpha_followup": _spearman(
                log_alpha.abs(), df[COL_FOLLOWUP]
            ),
        }
    )


def get_df_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per model, set and seed."""
    return (
        df.groupby([COL_MODEL, COL_SET_NAME, COL_SEED], sort=False)
        .apply(_summarize, include_groups=False)
        .reset_index()
    )


def get_df_by_visits(df: pd.DataFrame) -> pd.DataFrame:
    """Per model, set and number of visits (pooled over seeds)."""
    return (
        df.groupby([COL_MODEL, COL_SET_NAME, COL_N_VISITS_BIN], sort=False)
        .apply(_summarize, include_groups=False)
        .drop(columns=["spearman_log_alpha_n_visits"])
        .reset_index()
        .sort_values([COL_MODEL, COL_SET_NAME, COL_N_VISITS_BIN])
    )


def get_df_stability(df: pd.DataFrame) -> pd.DataFrame:
    """SD across seeds of each train subject's log alpha (same subjects across
    seeds, i.e. real data only)."""
    df_train = df.loc[df[COL_SET_NAME] == "train"]
    df_stability = (
        df_train.groupby([COL_MODEL, COL_SUBJECT], sort=False)
        .agg(
            n_seeds=(COL_SEED, "nunique"),
            sd_log_alpha_across_seeds=(COL_LOG_ALPHA, "std"),
            mean_log_alpha=(COL_LOG_ALPHA, "mean"),
            n_visits=(COL_N_VISITS, "median"),
        )
        .reset_index()
    )
    df_stability = df_stability.loc[df_stability["n_seeds"] >= MIN_SEEDS_STABILITY]
    df_stability[COL_N_VISITS_BIN] = df_stability[COL_N_VISITS].clip(
        upper=MAX_N_VISITS_BIN
    )
    return df_stability


def get_df_recovery(df: pd.DataFrame) -> pd.DataFrame:
    """Recovery of the true log alpha, per model, seed, set and subject subset."""
    rows = []
    df_est = df.loc[df[COL_MODEL] != LABEL_TRUTH]
    for subset, min_n_visits in SUBSETS.items():
        df_subset = df_est.loc[df_est[COL_N_VISITS] >= min_n_visits]
        for (model, seed, set_name), df_group in df_subset.groupby(
            [COL_MODEL, COL_SEED, COL_SET_NAME], sort=False
        ):
            est, true = df_group[COL_LOG_ALPHA], df_group[COL_LOG_ALPHA_TRUE]
            rows.append(
                {
                    COL_MODEL: model,
                    COL_SEED: seed,
                    COL_SET_NAME: set_name,
                    COL_SUBSET: subset,
                    "n_subjects": len(df_group),
                    "pearson": _pearson(est, true),
                    "spearman": _spearman(est, true),
                    "mae": (est - true).abs().mean(),
                    "bias": (est - true).mean(),
                    # MAE after removing the mean offset (scale of alpha vs k is
                    # only identified through the prior)
                    "mae_centered": ((est - est.mean()) - (true - true.mean()))
                    .abs()
                    .mean(),
                    "sd_ratio": est.std() / true.std(),
                    "frac_pinned": (est.abs() < PINNED_TOL).mean(),
                }
            )
    return pd.DataFrame(rows)


PARAMS_AGREEMENT = {
    COL_LOG_ALPHA: "log acceleration factor",
    COL_TIME_SHIFT: "time shift",
}


def get_df_agreement(df: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """Spearman correlation of each model's log acceleration factors and time
    shifts with Leaspy's (and, on synthetic data, of every model with the truth),
    per seed and set (same subjects)."""
    keys = [COL_SEED, COL_SET_NAME, COL_SUBJECT]
    references = [LABEL_LEASPY] + ([LABEL_TRUTH] if LABEL_TRUTH in models else [])
    rows = []
    for reference in references:
        df_ref = df.loc[df[COL_MODEL] == reference, keys + list(PARAMS_AGREEMENT)]
        for model in models:
            if model in (reference, LABEL_TRUTH):
                continue
            df_merged = df.loc[df[COL_MODEL] == model].merge(
                df_ref, on=keys, suffixes=("", "_ref")
            )
            for (seed, set_name), df_group in df_merged.groupby(
                [COL_SEED, COL_SET_NAME]
            ):
                for param in PARAMS_AGREEMENT:
                    rows.append(
                        {
                            COL_MODEL: model,
                            "reference": reference,
                            COL_SEED: seed,
                            COL_SET_NAME: set_name,
                            "parameter": param,
                            "n_subjects": len(df_group),
                            "spearman": _spearman(
                                df_group[param], df_group[f"{param}_ref"]
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def get_df_fed_vs_cent(
    df: pd.DataFrame, models: list[str], label_centralized: str
) -> pd.DataFrame:
    """Agreement of log alpha of each model with Fed-BioMed centralized, per seed
    and set (same subjects)."""
    df_cent = df.loc[df[COL_MODEL] == label_centralized]
    keys = [COL_SEED, COL_SET_NAME, COL_SUBJECT]
    rows = []
    for model in models:
        if model == label_centralized:
            continue
        df_merged = df.loc[df[COL_MODEL] == model].merge(
            df_cent[keys + [COL_LOG_ALPHA]], on=keys, suffixes=("", "_cent")
        )
        for (seed, set_name), df_group in df_merged.groupby([COL_SEED, COL_SET_NAME]):
            x, y = df_group[f"{COL_LOG_ALPHA}_cent"], df_group[COL_LOG_ALPHA]
            rows.append(
                {
                    COL_MODEL: model,
                    COL_SEED: seed,
                    COL_SET_NAME: set_name,
                    "n_subjects": len(df_group),
                    "pearson": _pearson(x, y),
                    "spearman": _spearman(x, y),
                    "mean_diff": (y - x).mean(),
                    "sd_diff": (y - x).std(),
                }
            )
    return pd.DataFrame(rows)


# ----- figures -----


def _subplots_wrap(n: int, width: float = 3.0, height: float = 2.8, **kwargs):
    n_cols = min(n, N_COLS_MAX)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(width * n_cols, height * n_rows),
        squeeze=False,
        **kwargs,
    )
    axes = axes.ravel()
    for ax in axes[n:]:
        ax.set_visible(False)
    return fig, list(axes[:n])


def _visit_ticks(visit_bins) -> list[str]:
    return [f"{v}+" if v == MAX_N_VISITS_BIN else str(v) for v in visit_bins]


def plot_histograms(
    df: pd.DataFrame, models: list[str], colors: dict, fpath_stem: Path
):
    """log alpha distribution per model (pooled over seeds), train vs test."""
    fig, axes = _subplots_wrap(len(models), sharex=True)
    bins = np.linspace(*np.nanquantile(df[COL_LOG_ALPHA], [0.005, 0.995]), 50)
    for ax, model in zip(axes, models):
        df_model = df.loc[df[COL_MODEL] == model]
        for set_name, alpha in zip(SET_NAMES, (0.9, 0.4)):
            values = df_model.loc[df_model[COL_SET_NAME] == set_name, COL_LOG_ALPHA]
            ax.hist(
                values,
                bins=bins,
                density=True,
                histtype="stepfilled",
                color=colors[model],
                alpha=alpha,
                label=f"{set_name} ({(values.abs() < PINNED_TOL).mean():.0%} pinned)",
            )
        ax.axvline(0, color="black", linewidth=0.8, linestyle=":")
        ax.set_title(model, fontsize="small")
        ax.set_xlabel("log acceleration factor")
        ax.set_ylabel("Density")
        ax.legend(fontsize="x-small", frameon=False)
    fig.tight_layout()
    save_fig_formats(fig, fpath_stem)


def plot_boxplots_by_visits(
    df: pd.DataFrame, models: list[str], colors: dict, fpath_stem: Path
):
    """log alpha vs number of visits (train, pooled over seeds), per model."""
    df_train = df.loc[df[COL_SET_NAME] == "train"]
    visit_bins = sorted(df_train[COL_N_VISITS_BIN].unique())
    fig, axes = _subplots_wrap(len(models), sharey=True)
    for ax, model in zip(axes, models):
        df_model = df_train.loc[df_train[COL_MODEL] == model]
        ax.boxplot(
            [
                df_model.loc[df_model[COL_N_VISITS_BIN] == v, COL_LOG_ALPHA]
                for v in visit_bins
            ],
            tick_labels=_visit_ticks(visit_bins),
            showfliers=False,
            patch_artist=True,
            boxprops={"facecolor": colors[model], "alpha": 0.6},
            medianprops={"color": "black"},
        )
        ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
        ax.set_title(model, fontsize="small")
        ax.set_xlabel("Number of visits")
        ax.set_ylabel("log acceleration factor")
    fig.tight_layout()
    save_fig_formats(fig, fpath_stem)


def plot_lines_by_visits(
    df: pd.DataFrame,
    df_stability: pd.DataFrame | None,
    models: list[str],
    colors: dict,
    fpath_stem: Path,
):
    """Spread of log alpha, fraction pinned (and across-seed SD) vs visits."""
    df_train = df.loc[df[COL_SET_NAME] == "train"]
    visit_bins = sorted(df_train[COL_N_VISITS_BIN].unique())
    df_by_visits = get_df_by_visits(df_train)
    panels = [
        (df_by_visits, "sd_log_alpha", "SD of log α (across subjects)"),
        (df_by_visits, "frac_pinned", f"Fraction |log α| < {PINNED_TOL}"),
    ]
    if df_stability is not None:
        panels.append(
            (
                df_stability.groupby([COL_MODEL, COL_N_VISITS_BIN])[
                    "sd_log_alpha_across_seeds"
                ]
                .median()
                .reset_index(),
                "sd_log_alpha_across_seeds",
                "Median SD of log α\n(across seeds, same subject)",
            )
        )
    fig, axes = plt.subplots(1, len(panels) + 1, figsize=(4 * len(panels) + 2.5, 3.2))
    for ax, (df_panel, col, ylabel) in zip(axes, panels):
        for model in models:
            df_model = df_panel.loc[df_panel[COL_MODEL] == model]
            if df_model.empty:
                continue
            ax.plot(
                df_model[COL_N_VISITS_BIN],
                df_model[col],
                color=colors[model],
                linewidth=2,
                linestyle="--" if model == LABEL_TRUTH else "-",
                marker="o",
                markersize=4,
                label=model,
            )
        ax.set_xticks(visit_bins, _visit_ticks(visit_bins))
        ax.set_xlabel("Number of visits")
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0)
    handles, labels = axes[0].get_legend_handles_labels()
    axes[-1].axis("off")
    axes[-1].legend(handles, labels, fontsize="small", frameon=False, loc="center left")
    fig.tight_layout()
    save_fig_formats(fig, fpath_stem)


def plot_vs_followup(
    df: pd.DataFrame, models: list[str], colors: dict, seed: int, fpath_stem: Path
):
    """log alpha vs follow-up span, one seed, train subjects."""
    df_seed = df.loc[(df[COL_SEED] == seed) & (df[COL_SET_NAME] == "train")]
    fig, axes = _subplots_wrap(len(models), sharex=True, sharey=True)
    for ax, model in zip(axes, models):
        df_model = df_seed.loc[df_seed[COL_MODEL] == model]
        ax.scatter(
            df_model[COL_FOLLOWUP],
            df_model[COL_LOG_ALPHA],
            s=6,
            alpha=0.5,
            color=colors[model],
            linewidths=0,
        )
        ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
        rho = _spearman(df_model[COL_LOG_ALPHA].abs(), df_model[COL_FOLLOWUP])
        ax.set_title(f"{model}\nρ(|log α|, follow-up) = {rho:.2f}", fontsize="small")
        ax.set_xlabel("Follow-up (months)")
        ax.set_ylabel("log acceleration factor")
    fig.tight_layout()
    save_fig_formats(fig, fpath_stem)


def plot_vs_truth(
    df: pd.DataFrame, models: list[str], colors: dict, seed: int, fpath_stem: Path
):
    """Estimated vs true log alpha, one seed, train subjects with 2+ visits."""
    models = [m for m in models if m != LABEL_TRUTH]
    df_seed = df.loc[
        (df[COL_SEED] == seed)
        & (df[COL_SET_NAME] == "train")
        & (df[COL_N_VISITS] >= SUBSETS[SUBSET_MAIN])
    ]
    fig, axes = _subplots_wrap(len(models), sharex=True, sharey=True)
    lims = np.nanquantile(
        pd.concat([df_seed[COL_LOG_ALPHA], df_seed[COL_LOG_ALPHA_TRUE]]), [0.002, 0.998]
    )
    for ax, model in zip(axes, models):
        df_model = df_seed.loc[df_seed[COL_MODEL] == model]
        x, y = df_model[COL_LOG_ALPHA_TRUE], df_model[COL_LOG_ALPHA]
        ax.scatter(x, y, s=6, alpha=0.5, color=colors[model], linewidths=0)
        ax.plot(lims, lims, color="black", linewidth=0.8, linestyle=":")
        ax.set_xlim(*lims)
        ax.set_ylim(*lims)
        ax.set_title(
            f"{model}\nρ = {_spearman(x, y):.2f}, MAE = {(y - x).abs().mean():.2f}",
            fontsize="small",
        )
        ax.set_xlabel("True log α")
        ax.set_ylabel("Estimated log α")
    fig.tight_layout()
    save_fig_formats(fig, fpath_stem)


def plot_recovery(
    df_recovery: pd.DataFrame, models: list[str], colors: dict, fpath_stem: Path
):
    """Recovery metrics per model (one dot per seed, mean as a bar), subjects with
    2+ visits; train (filled) and test (open)."""
    models = [m for m in models if m != LABEL_TRUTH]
    df_main = df_recovery.loc[df_recovery[COL_SUBSET] == SUBSET_MAIN]
    metrics = {
        "spearman": "Spearman ρ (est. vs true)",
        "mae_centered": "MAE of log α (offset removed)",
        "sd_ratio": "SD ratio (est. / true)",
        "frac_pinned": f"Fraction |log α| < {PINNED_TOL}",
    }
    fig, axes = plt.subplots(
        len(metrics),
        1,
        figsize=(max(6, 0.75 * len(models) + 2), 2.4 * len(metrics)),
        sharex=True,
    )
    x = np.arange(len(models))
    offsets = {"train": -0.15, "test": 0.15}
    for ax, (metric, ylabel) in zip(axes, metrics.items()):
        for set_name, offset in offsets.items():
            df_set = df_main.loc[df_main[COL_SET_NAME] == set_name]
            for i, model in enumerate(models):
                values = df_set.loc[df_set[COL_MODEL] == model, metric]
                filled = set_name == "train"
                ax.scatter(
                    np.full(len(values), x[i] + offset),
                    values,
                    s=12,
                    facecolors=colors[model] if filled else "none",
                    edgecolors=colors[model],
                    linewidths=1,
                    alpha=0.8,
                    label=set_name if i == 0 else None,
                )
                ax.hlines(
                    values.mean(),
                    x[i] + offset - 0.12,
                    x[i] + offset + 0.12,
                    color="black",
                    linewidth=1.5,
                )
        if metric == "sd_ratio":
            ax.axhline(1, color="black", linewidth=0.8, linestyle=":")
        ax.set_ylabel(ylabel, fontsize="small")
        ax.grid(axis="y", linewidth=0.4, alpha=0.5)
    axes[0].legend(fontsize="x-small", frameon=False)
    axes[-1].set_xticks(x, models, rotation=40, ha="right", fontsize="small")
    fig.tight_layout()
    save_fig_formats(fig, fpath_stem)


def plot_fed_vs_cent(
    df: pd.DataFrame,
    models: list[str],
    colors: dict,
    label_centralized: str,
    seed: int,
    fpath_stem: Path,
):
    """Each model vs Fed-BioMed centralized, one seed, train subjects: scatter
    (top) and Bland-Altman (bottom)."""
    models = [m for m in models if m not in (label_centralized, LABEL_TRUTH)]
    df_seed = df.loc[(df[COL_SEED] == seed) & (df[COL_SET_NAME] == "train")]
    df_cent = df_seed.loc[
        df_seed[COL_MODEL] == label_centralized, [COL_SUBJECT, COL_LOG_ALPHA]
    ]
    fig, axes = plt.subplots(
        2, len(models), figsize=(3 * len(models), 5.6), squeeze=False
    )
    for i_col, model in enumerate(models):
        df_merged = df_seed.loc[df_seed[COL_MODEL] == model].merge(
            df_cent, on=COL_SUBJECT, suffixes=("", "_cent")
        )
        x, y = df_merged[f"{COL_LOG_ALPHA}_cent"], df_merged[COL_LOG_ALPHA]
        style = {"s": 6, "alpha": 0.5, "color": colors[model], "linewidths": 0}

        ax = axes[0, i_col]
        ax.scatter(x, y, **style)
        lims = [min(x.min(), y.min()), max(x.max(), y.max())]
        ax.plot(lims, lims, color="black", linewidth=0.8, linestyle=":")
        ax.set_title(
            f"{model}\nr = {_pearson(x, y):.2f}, ρ = {_spearman(x, y):.2f}",
            fontsize="small",
        )
        ax.set_xlabel(f"log α ({label_centralized})", fontsize="small")
        ax.set_ylabel("log α")

        ax = axes[1, i_col]
        mean, diff = (x + y) / 2, y - x
        ax.scatter(mean, diff, **style)
        for value, linestyle in [
            (diff.mean(), "-"),
            (diff.mean() + 1.96 * diff.std(), "--"),
            (diff.mean() - 1.96 * diff.std(), "--"),
        ]:
            ax.axhline(value, color="black", linewidth=0.8, linestyle=linestyle)
        ax.set_xlabel("Mean log α")
        ax.set_ylabel("Difference (model − centralized)")
    fig.tight_layout()
    save_fig_formats(fig, fpath_stem)


def plot_agreement_leaspy(
    df: pd.DataFrame, models: list[str], colors: dict, seed: int, fpath_stem: Path
):
    """Each model vs Leaspy, one seed, train subjects: log acceleration factors
    (top) and time shifts (bottom)."""
    models = [m for m in models if m not in (LABEL_LEASPY, LABEL_TRUTH)]
    df_seed = df.loc[(df[COL_SEED] == seed) & (df[COL_SET_NAME] == "train")]
    df_ref = df_seed.loc[
        df_seed[COL_MODEL] == LABEL_LEASPY, [COL_SUBJECT, *PARAMS_AGREEMENT]
    ]
    fig, axes = plt.subplots(
        len(PARAMS_AGREEMENT),
        len(models),
        figsize=(3 * len(models), 2.9 * len(PARAMS_AGREEMENT)),
        squeeze=False,
    )
    for i_col, model in enumerate(models):
        df_merged = df_seed.loc[df_seed[COL_MODEL] == model].merge(
            df_ref, on=COL_SUBJECT, suffixes=("", "_ref")
        )
        for i_row, (param, label) in enumerate(PARAMS_AGREEMENT.items()):
            ax = axes[i_row, i_col]
            x, y = df_merged[f"{param}_ref"], df_merged[param]
            ax.scatter(x, y, s=6, alpha=0.5, color=colors[model], linewidths=0)
            ax.set_title(f"{model}\nρ = {_spearman(x, y):.2f}", fontsize="small")
            ax.set_xlabel(f"{label} (Leaspy)", fontsize="small")
            ax.set_ylabel(label, fontsize="small")
    fig.tight_layout()
    save_fig_formats(fig, fpath_stem)


# ----- main -----


def _save_tsv(df: pd.DataFrame, fpath: Path):
    fpath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(fpath, sep="\t", index=False)
    print(f"Saved {fpath}")


def _print_overview(
    df_summary: pd.DataFrame,
    df_fed_vs_cent: pd.DataFrame | None,
    df_recovery: pd.DataFrame | None,
):
    cols = [
        "n_subjects",
        "sd_log_alpha",
        "frac_pinned",
        "spearman_abs_log_alpha_n_visits",
        "spearman_abs_log_alpha_followup",
    ]
    with pd.option_context("display.width", 250, "display.precision", 3):
        print("\nMean over seeds:")
        print(
            df_summary.groupby([COL_MODEL, COL_SET_NAME], sort=False)[cols]
            .mean()
            .to_string()
        )
        if df_fed_vs_cent is not None and not df_fed_vs_cent.empty:
            print("\nAgreement with Fed-BioMed centralized (mean over seeds):")
            print(
                df_fed_vs_cent.groupby([COL_MODEL, COL_SET_NAME], sort=False)[
                    ["pearson", "spearman", "mean_diff", "sd_diff"]
                ]
                .mean()
                .to_string()
            )
        if df_recovery is not None:
            print(f"\nRecovery of true log alpha ({SUBSET_MAIN}, mean over seeds):")
            print(
                df_recovery.loc[df_recovery[COL_SUBSET] == SUBSET_MAIN]
                .groupby([COL_MODEL, COL_SET_NAME], sort=False)[
                    [
                        "pearson",
                        "spearman",
                        "mae",
                        "mae_centered",
                        "bias",
                        "sd_ratio",
                        "frac_pinned",
                    ]
                ]
                .mean()
                .to_string()
            )


def _default_results_dir(spec: DatasetSpec) -> Path:
    # default_dpath_results is relative to this checkout, data live in the main one
    return DPATH_RESULTS_MAIN / spec.default_dpath_results.name


@click.command(context_settings=CLICK_CONTEXT_SETTINGS)
@click.option(
    "--dataset",
    type=click.Choice(list(DATASETS)),
    default=DEFAULT_DATASET,
    show_default=True,
)
@click.option(
    "--results-dir",
    "dpath_results",
    type=click.Path(path_type=Path, exists=True, file_okay=False, dir_okay=True),
    help="Multi-seed results (default: the dataset's default results directory).",
)
@click.option(
    "--baselines-dir",
    "dpath_baselines",
    type=click.Path(path_type=Path, exists=True, file_okay=False, dir_okay=True),
    help="Results with the Leaspy fits (default: --results-dir).",
)
@click.option(
    "--analysis",
    "analyses",
    type=click.Choice(ANALYSES),
    multiple=True,
    help="Analyses to run (default: all).",
)
@click.option(
    "--variants-dir",
    "dpath_variants",
    type=click.Path(path_type=Path, exists=True, file_okay=False, dir_okay=True),
    help="Centralized fits of penalty variants (run_alpha_penalty_sweep.sh).",
)
@click.option(
    "--variant",
    "variant_names",
    multiple=True,
    help="Variants (subdirectories of --variants-dir) to include, in this order "
    "(default: all).",
)
@click.option(
    "--condition",
    "conditions",
    multiple=True,
    help=f"Fed-BioMed conditions to include (default: all; {CONDITION_NONE!r}: "
    "none, e.g. with --variants-dir).",
)
@click.option(
    "--out-dir",
    "dpath_out",
    type=click.Path(path_type=Path, file_okay=False, dir_okay=True),
    help=f"Default: {DEFAULT_DPATH_OUT_PARENT}/<dataset>[-<variants dir name>].",
)
@click.option(
    "--seed",
    "seeds",
    type=int,
    multiple=True,
    help="Seeds (data splits) to include (default: all).",
)
@click.option(
    "--plot-seed",
    type=int,
    default=1,
    show_default=True,
    help="Seed for the single-seed scatter plots.",
)
def main(
    dataset: str,
    dpath_results: Path | None,
    dpath_baselines: Path | None,
    analyses: tuple[str],
    dpath_variants: Path | None,
    variant_names: tuple[str],
    conditions: tuple[str],
    dpath_out: Path | None,
    seeds: tuple[int],
    plot_seed: int,
):
    spec = DATASETS[dataset]
    if dpath_results is None:
        dpath_results = _default_results_dir(spec)
    analyses = set(analyses) or set(ANALYSES)
    conditions = list(conditions) or list(spec.condition_labels)
    if conditions == [CONDITION_NONE]:
        conditions = []
    unknown = set(conditions) - set(spec.condition_labels)
    if unknown:
        raise click.BadParameter(
            f"Unknown conditions {sorted(unknown)}, choose from {list(spec.condition_labels)}"
        )
    if dpath_out is None:
        dpath_out = DEFAULT_DPATH_OUT_PARENT / (
            dataset if dpath_variants is None else f"{dataset}-{dpath_variants.name}"
        )
    variant_dirs = _find_variant_dirs(dpath_variants, list(variant_names))

    df, models = load_df_subject(
        spec, dpath_results, list(seeds), conditions, variant_dirs, dpath_baselines
    )
    colors = get_model_colors(models, list(variant_dirs))
    if plot_seed not in set(df[COL_SEED]):
        plot_seed = int(df[COL_SEED].min())

    label_centralized = (
        f"{PREFIX_FEDBIOMED} {spec.condition_labels[spec.condition_centralized]}"
    )
    prefix = "acceleration_factors"
    columns_subject = [
        c
        for c in [
            COL_SEED,
            COL_MODEL,
            COL_SUBJECT,
            COL_GROUP,
            COL_SET_NAME,
            COL_TIME_SHIFT,
            COL_ACCELERATION_FACTOR,
            COL_LOG_ALPHA,
            COL_LOG_ALPHA_TRUE,
            COL_N_VISITS,
            COL_FOLLOWUP,
        ]
        if c in df.columns
    ]
    _save_tsv(df[columns_subject], dpath_out / f"{prefix}-subjects.tsv")

    if ANALYSIS_REGULARIZATION in analyses:
        _run_regularization_analysis(
            df, spec, models, colors, label_centralized, plot_seed, dpath_out, prefix
        )
    if ANALYSIS_LEASPY in analyses:
        df_agreement = get_df_agreement(df, models)
        _save_tsv(df_agreement, dpath_out / f"{prefix}-agreement.tsv")
        plot_agreement_leaspy(
            df,
            models,
            colors,
            plot_seed,
            dpath_out / f"{prefix}-agreement_leaspy-s{plot_seed}",
        )
        with pd.option_context("display.width", 250, "display.precision", 3):
            print("\nSpearman correlation with the reference (mean over seeds):")
            print(
                df_agreement.pivot_table(
                    index=["reference", COL_MODEL],
                    columns=["parameter", COL_SET_NAME],
                    values="spearman",
                    aggfunc="mean",
                    sort=False,
                ).to_string()
            )


def _run_regularization_analysis(
    df: pd.DataFrame,
    spec: DatasetSpec,
    models: list[str],
    colors: dict,
    label_centralized: str,
    plot_seed: int,
    dpath_out: Path,
    prefix: str,
):
    df_summary = get_df_summary(df)
    df_by_visits = get_df_by_visits(df)
    df_fed_vs_cent = None
    if label_centralized in models:
        df_fed_vs_cent = get_df_fed_vs_cent(df, models, label_centralized)
    # synthetic data: each seed is a new population (no stability across seeds)
    df_stability = None if spec.has_ground_truth else get_df_stability(df)
    df_recovery = get_df_recovery(df) if spec.has_ground_truth else None

    _save_tsv(df_summary, dpath_out / f"{prefix}-summary.tsv")
    _save_tsv(df_by_visits, dpath_out / f"{prefix}-by_visits.tsv")
    if df_stability is not None:
        _save_tsv(df_stability, dpath_out / f"{prefix}-stability.tsv")
    if df_fed_vs_cent is not None:
        _save_tsv(df_fed_vs_cent, dpath_out / f"{prefix}-fed_vs_cent.tsv")
    if df_recovery is not None:
        _save_tsv(df_recovery, dpath_out / f"{prefix}-recovery.tsv")

    plot_histograms(df, models, colors, dpath_out / f"{prefix}-histograms")
    plot_boxplots_by_visits(
        df, models, colors, dpath_out / f"{prefix}-boxplots_by_visits"
    )
    plot_lines_by_visits(
        df, df_stability, models, colors, dpath_out / f"{prefix}-by_visits"
    )
    plot_vs_followup(
        df, models, colors, plot_seed, dpath_out / f"{prefix}-vs_followup-s{plot_seed}"
    )
    if df_recovery is not None:
        plot_vs_truth(
            df, models, colors, plot_seed, dpath_out / f"{prefix}-vs_truth-s{plot_seed}"
        )
        plot_recovery(df_recovery, models, colors, dpath_out / f"{prefix}-recovery")
    if df_fed_vs_cent is not None and len(models) > 1:
        plot_fed_vs_cent(
            df,
            models,
            colors,
            label_centralized,
            plot_seed,
            dpath_out / f"{prefix}-fed_vs_cent-s{plot_seed}",
        )

    _print_overview(df_summary, df_fed_vs_cent, df_recovery)


if __name__ == "__main__":
    main()
