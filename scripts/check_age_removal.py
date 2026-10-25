#!/usr/bin/env python

"""
Check that normative-model z-scores (output of apply_normative_models.py) no
longer depend on age.

Tests are run on cognitively normal rows that were not used for model adaptation
(by default), for both the raw biomarker values and the z-scores:

- Cross-sectional: Pearson correlation between biomarker and age (one visit per subject)
- Non-linear: R^2 of a quadratic fit (age + age^2), one visit per subject
- Spread: Pearson correlation between |biomarker - mean| and age, one visit per subject
- Longitudinal: mean within-subject slope vs age (per year), one-sample t-test
- Moments: mean and SD of z-scores (expected ~0 and ~1)

A biomarker fails if the cross-sectional or longitudinal z-score test is
significant after FDR correction and the effect is non-negligible.
"""

import sys
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from apply_normative_models import (
    COL_AGE_ADNIMERGE,
    COL_GROUP_ADNIMERGE,
    DF_TRANSFORM_PER_MODEL,
    FNAME_HARMONIZED,
    FNAME_SETTINGS,
    _get_merged_df,
)
from scipy import stats

from fl_prog.plotting import save_fig
from fl_prog.utils.constants import CLICK_CONTEXT_SETTINGS, COL_ADAPTATION
from fl_prog.utils.io import load_json

FNAME_SUMMARY = "age_check.tsv"
FNAME_FIG_CORRS = "age_check-correlations.png"
FNAME_FIG_SCATTER = "age_check-scatter.png"

DEFAULT_ALPHA = 0.05
DEFAULT_MIN_ABS_R = 0.1  # cross-sectional correlation below this is negligible
DEFAULT_MIN_ABS_SLOPE = 0.02  # z-score change per year below this is negligible
DEFAULT_SUBSET = "held-out"

N_COLS = 3
PANEL_SIZE = 2.5

sns.set_theme(context="paper", style="ticks")


def _get_z_cols(df_raw: pd.DataFrame, settings: dict) -> list[str]:
    """Map pcntoolkit response variable names back to column names."""
    z_cols = []
    for model_name, response_vars in settings["harmonized_vars_map"].items():
        _, col_map = DF_TRANSFORM_PER_MODEL.get(model_name, lambda df: (df, None))(
            df_raw
        )
        col_map = col_map or {}
        z_cols.extend(col_map.get(var, var) for var in response_vars)
    return list(dict.fromkeys(z_cols))


def _select_rows(
    df: pd.DataFrame, subset: str, adaptation_groups: list[str]
) -> pd.DataFrame:
    df = df.loc[df[COL_GROUP_ADNIMERGE].isin(adaptation_groups)]
    if subset == "held-out":
        return df.loc[~df[COL_ADAPTATION].astype(bool)]
    elif subset == "adaptation":
        return df.loc[df[COL_ADAPTATION].astype(bool)]
    return df


def _first_visits(df: pd.DataFrame, col_subject: str) -> pd.DataFrame:
    return df.sort_values(COL_AGE_ADNIMERGE).groupby(col_subject).head(1)


def _quadratic_r2(x: np.ndarray, y: np.ndarray) -> float:
    y_pred = np.polyval(np.polyfit(x, y, deg=2), x)
    return 1 - np.sum((y - y_pred) ** 2) / np.sum((y - y.mean()) ** 2)


def _cross_sectional_stats(age: pd.Series, y: pd.Series) -> dict:
    mask = y.notna() & age.notna()
    age, y = age[mask].to_numpy(), y[mask].to_numpy()
    r, p = stats.pearsonr(age, y)
    r_spread, p_spread = stats.pearsonr(age, np.abs(y - y.mean()))
    return {
        "r": r,
        "p": p,
        "r2_quadratic": _quadratic_r2(age, y),
        "r_spread": r_spread,
        "p_spread": p_spread,
    }


def _longitudinal_stats(df: pd.DataFrame, col: str, col_subject: str) -> dict:
    slopes = []
    for _, df_subject in (
        df[[col_subject, COL_AGE_ADNIMERGE, col]].dropna().groupby(col_subject)
    ):
        if df_subject[COL_AGE_ADNIMERGE].nunique() < 2:
            continue
        slopes.append(
            np.polyfit(df_subject[COL_AGE_ADNIMERGE], df_subject[col], deg=1)[0]
        )
    slopes = np.asarray(slopes)
    if len(slopes) < 2:
        return {"slope": np.nan, "p_slope": np.nan, "n_subjects_slope": len(slopes)}
    return {
        "slope": slopes.mean(),
        "p_slope": stats.ttest_1samp(slopes, 0).pvalue,
        "n_subjects_slope": len(slopes),
    }


def _get_stats(
    df: pd.DataFrame, cols: list[str], col_subject: str, prefix: str
) -> pd.DataFrame:
    df_first = _first_visits(df, col_subject)
    rows = {}
    for col in cols:
        rows[col] = {
            **_cross_sectional_stats(df_first[COL_AGE_ADNIMERGE], df_first[col]),
            **_longitudinal_stats(df, col, col_subject),
            "mean": df[col].mean(),
            "sd": df[col].std(),
        }
    df_stats = pd.DataFrame.from_dict(rows, orient="index")
    for col_p in ("p", "p_spread", "p_slope"):
        p_values = df_stats[col_p]
        df_stats[f"{col_p}_fdr"] = np.nan
        df_stats.loc[p_values.notna(), f"{col_p}_fdr"] = stats.false_discovery_control(
            p_values.dropna()
        )
    return df_stats.add_prefix(prefix)


def _plot_correlations(df_summary: pd.DataFrame, fpath: Path):
    df_plot = pd.concat(
        [
            df_summary[["raw_r"]].rename(columns={"raw_r": "r"}).assign(data="raw"),
            df_summary[["z_r"]].rename(columns={"z_r": "r"}).assign(data="z-score"),
        ]
    )
    fig, ax = plt.subplots(figsize=(PANEL_SIZE * 1.5, PANEL_SIZE))
    sns.histplot(data=df_plot, x="r", hue="data", bins=30, ax=ax)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Pearson r with age (first visit)")
    sns.despine(fig)
    save_fig(fig, fpath)
    plt.close(fig)


def _plot_scatter(
    df_raw: pd.DataFrame, df_z: pd.DataFrame, cols: list[str], fpath: Path
):
    n_rows = len(cols)
    fig, axes = plt.subplots(
        n_rows,
        2,
        figsize=(PANEL_SIZE * 2, PANEL_SIZE * n_rows),
        squeeze=False,
        sharex=True,
    )
    for i_row, col in enumerate(cols):
        for ax, df, label in zip(axes[i_row], (df_raw, df_z), ("raw", "z-score")):
            sns.regplot(
                data=df,
                x=COL_AGE_ADNIMERGE,
                y=col,
                lowess=True,
                scatter_kws={"s": 3, "alpha": 0.3},
                line_kws={"color": "red"},
                ax=ax,
            )
            ax.set_title(f"{col} ({label})", fontsize="small")
            ax.set_ylabel("")
    sns.despine(fig)
    fig.tight_layout()
    save_fig(fig, fpath)
    plt.close(fig)


@click.command(context_settings=CLICK_CONTEXT_SETTINGS)
@click.argument(
    "dpath_harmonized",
    type=click.Path(path_type=Path, exists=True, file_okay=False, dir_okay=True),
)
@click.option(
    "--subset",
    type=click.Choice(["held-out", "adaptation", "all"]),
    default=DEFAULT_SUBSET,
    show_default=True,
    help="Which rows of the adaptation group(s) to test.",
)
@click.option("--alpha", type=float, default=DEFAULT_ALPHA, show_default=True)
@click.option(
    "--min-abs-r",
    type=float,
    default=DEFAULT_MIN_ABS_R,
    show_default=True,
    help="Smallest cross-sectional |r| with age counted as a failure.",
)
@click.option(
    "--min-abs-slope",
    type=float,
    default=DEFAULT_MIN_ABS_SLOPE,
    show_default=True,
    help="Smallest within-subject |slope| (z-score per year) counted as a failure.",
)
@click.option(
    "--dpath-out",
    type=click.Path(path_type=Path, file_okay=False, dir_okay=True),
    default=None,
    help="Output directory for summary and figures (default: DPATH_HARMONIZED).",
)
def check_age_removal(
    dpath_harmonized: Path,
    subset: str,
    alpha: float,
    min_abs_r: float,
    min_abs_slope: float,
    dpath_out: Path | None,
):
    dpath_out = dpath_out or dpath_harmonized
    settings = load_json(dpath_harmonized / FNAME_SETTINGS)
    config = load_json(Path(settings["fpath_config"]))
    col_subject = config["col_subject_original"]

    df_z = pd.read_csv(dpath_harmonized / FNAME_HARMONIZED, index_col=0)
    df_raw = _get_merged_df(
        Path(settings["fpath_idps"]), Path(settings["fpath_adni_merge"]), config
    )
    df_raw = df_raw.loc[df_z.index]

    z_cols = [col for col in _get_z_cols(df_raw, settings) if col in df_z.columns]
    click.secho(f"N z-scored biomarkers: {len(z_cols)}")

    # use age at visit and metadata from raw data for both
    df_z[COL_AGE_ADNIMERGE] = df_raw[COL_AGE_ADNIMERGE]
    df_raw[COL_ADAPTATION] = df_z[COL_ADAPTATION]

    adaptation_groups = settings["adaptation_groups"]
    df_raw = _select_rows(df_raw, subset, adaptation_groups)
    df_z = _select_rows(df_z, subset, adaptation_groups)
    click.secho(
        f"Testing {len(df_z)} rows ({df_z[col_subject].nunique()} subjects)"
        f" from {adaptation_groups}, subset={subset}"
    )

    df_summary = pd.concat(
        [
            _get_stats(df_raw, z_cols, col_subject, prefix="raw_"),
            _get_stats(df_z, z_cols, col_subject, prefix="z_"),
        ],
        axis="columns",
    )
    df_summary["fail_cross_sectional"] = (df_summary["z_p_fdr"] < alpha) & (
        df_summary["z_r"].abs() >= min_abs_r
    )
    df_summary["fail_longitudinal"] = (df_summary["z_p_slope_fdr"] < alpha) & (
        df_summary["z_slope"].abs() >= min_abs_slope
    )
    df_summary["fail"] = (
        df_summary["fail_cross_sectional"] | df_summary["fail_longitudinal"]
    )

    fpath_summary = dpath_out / FNAME_SUMMARY
    df_summary.to_csv(fpath_summary, sep="\t")
    click.secho(f"Summary saved to {fpath_summary}.", fg="green")

    _plot_correlations(df_summary, dpath_out / FNAME_FIG_CORRS)
    cols_scatter = [col for col in config.get("measures", []) if col in z_cols]
    if cols_scatter:
        _plot_scatter(df_raw, df_z, cols_scatter, dpath_out / FNAME_FIG_SCATTER)

    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(
            df_summary[
                ["raw_r", "z_r", "raw_r2_quadratic", "z_r2_quadratic"]
                + ["raw_slope", "z_slope", "z_mean", "z_sd", "z_r_spread"]
            ]
            .describe()
            .loc[["mean", "std", "min", "50%", "max"]]
            .round(3)
        )

    n_fail = df_summary["fail"].sum()
    for col_fail in ("fail_cross_sectional", "fail_longitudinal"):
        click.secho(f"{col_fail}: {df_summary[col_fail].sum()}/{len(df_summary)}")
    if n_fail > 0:
        print(
            df_summary.loc[
                df_summary["fail"], ["z_r", "z_p_fdr", "z_slope", "z_p_slope_fdr"]
            ].round(4)
        )
        click.secho(
            f"Age effect remains in {n_fail}/{len(df_summary)} biomarkers.",
            fg="red",
            bold=True,
        )
        sys.exit(1)
    click.secho("No remaining age effect found.", fg="green", bold=True)


if __name__ == "__main__":
    check_age_removal()
