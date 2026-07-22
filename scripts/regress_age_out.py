#!/usr/bin/env python

"""
Remove biological AGE effects from ADNI biomarker columns by regression.

For each biomarker, fit:

    y = beta_0 + beta_age * AGE + beta_time * t + epsilon

Then output (new biomarker values):

    y_adj = y - beta_age * AGE

This removes the fitted age effect while preserving the observed
longitudinal time effect for downstream disease-progression fitting.

If --min-max is used, y_adj is globally (all sites) min-max scaled per biomarker after age
adjustment, which is useful for compatibility with the current sigmoid downstream model.
"""

from pathlib import Path
import copy
import json

import click
import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor

from fl_prog.utils.constants import CLICK_CONTEXT_SETTINGS
from fl_prog.utils.io import DEFAULT_DPATH_DATA, get_dpath_latest, save_json


DEFAULT_AGE_COL = "AGE"


def _replace_tag_prefix(fname: str, tag: str, out_tag: str) -> str:
    """Rename adni_iid-1.tsv to adni_iid_age_adjusted-1.tsv."""
    if not fname.startswith(tag):
        raise ValueError(f"Expected filename {fname!r} to start with tag {tag!r}")
    return f"{out_tag}{fname[len(tag) :]}"


def _get_input_files(dpath_in: Path, tag: str, mode: str) -> list[Path]:
    """Select TSV files to process."""
    fname_merged = f"{tag}-merged.tsv"

    if (
        mode == "merged"
    ):  # if just want to regress on the merged file (centralized), if it exists
        fpath_merged = dpath_in / fname_merged
        if not fpath_merged.exists():
            raise FileNotFoundError(
                f"Could not find {fpath_merged}. Run merge_data.py first."
            )
        return [fpath_merged]

    # Exclude merged file to avoid duplicate rows.
    fpaths = [
        fpath
        for fpath in sorted(dpath_in.glob(f"{tag}-*.tsv"))
        if fpath.name != fname_merged
    ]

    if not fpaths:
        raise FileNotFoundError(
            f"No site TSV files found for tag {tag!r} in {dpath_in}"
        )

    return fpaths


def _fit_age_model(
    df: pd.DataFrame,
    biomarker_col: str,
    age_col: str,
    time_col: str,
    huber: bool,
) -> dict[str, float]:
    """
    Fit:

        y = beta_0 + beta_age * AGE + beta_time * t + epsilon

    OLS is used by default. If huber=True, Huber regression is used to reduce
    sensitivity to outliers. If Huber fails, OLS is used as fallback.
    """
    cols = [biomarker_col, age_col, time_col]
    valid = df[cols].apply(pd.to_numeric, errors="coerce").dropna()

    if len(valid) < 3:
        raise ValueError(
            f"Not enough complete rows to regress {biomarker_col!r} "
            f"on {age_col!r} and {time_col!r}."
        )

    x = valid[[age_col, time_col]].to_numpy(dtype=float)
    y = valid[biomarker_col].to_numpy(dtype=float)

    if huber and len(valid) >= 10:
        try:
            model = HuberRegressor(alpha=0.0, fit_intercept=True)
            model.fit(x, y)

            return {
                "beta_0": float(model.intercept_),
                "beta_age": float(model.coef_[0]),
                "beta_time": float(model.coef_[1]),
                "method": "huber",
                "n_rows": int(len(valid)),
            }
        except Exception as err:
            print(
                f"Warning: Huber regression failed for {biomarker_col!r}; "
                f"falling back to OLS. Error: {err}"
            )

    # OLS solves y ~= beta_0 + beta_age * age + beta_time * time.
    # The design matrix columns are [1, age, time], and the least-squares
    # coefficients use the pseudo-inverse.
    design = np.column_stack([np.ones(len(valid)), x])
    beta_0, beta_age, beta_time = np.linalg.pinv(design) @ y

    return {
        "beta_0": float(beta_0),
        "beta_age": float(beta_age),
        "beta_time": float(beta_time),
        "method": "ols",
        "n_rows": int(len(valid)),
    }


def _fit_models(
    df: pd.DataFrame,
    biomarker_cols: list[str],
    age_col: str,
    time_col: str,
    huber: bool,
) -> dict[str, dict[str, float]]:
    """Fit one age/time model per biomarker."""
    return {
        biomarker_col: _fit_age_model(
            df=df,
            biomarker_col=biomarker_col,
            age_col=age_col,
            time_col=time_col,
            huber=huber,
        )
        for biomarker_col in biomarker_cols
    }


def _apply_age_adjustment(
    df: pd.DataFrame,
    models: dict[str, dict[str, float]],
    biomarker_cols: list[str],
    age_col: str,
) -> pd.DataFrame:
    """
    Apply:

        y_adj = y - beta_age * AGE

    The fitted beta_time * t term is intentionally not subtracted.
    """
    df = df.copy()
    age = pd.to_numeric(df[age_col], errors="coerce")

    for biomarker_col in biomarker_cols:
        beta_age = models[biomarker_col]["beta_age"]
        y = pd.to_numeric(df[biomarker_col], errors="coerce")
        df[biomarker_col] = y - beta_age * age

    return df


def _fit_global_min_max(
    dfs: list[pd.DataFrame],
    biomarker_cols: list[str],
) -> dict[str, tuple[float, float]]:
    """Fit one global min/max per biomarker across all processed files."""
    df_all = pd.concat(dfs, ignore_index=True)

    min_max = {}
    for col in biomarker_cols:
        values = pd.to_numeric(df_all[col], errors="coerce")
        min_value = float(values.min())
        max_value = float(values.max())

        if not np.isfinite(min_value) or not np.isfinite(max_value):
            raise ValueError(f"Cannot min-max scale {col!r}; values are not finite.")

        if min_value == max_value:
            raise ValueError(f"Cannot min-max scale {col!r}; min equals max.")

        min_max[col] = (min_value, max_value)

    return min_max


def _apply_min_max(
    df: pd.DataFrame,
    min_max: dict[str, tuple[float, float]],
) -> pd.DataFrame:
    """Apply global min-max scaling to biomarker columns."""
    df = df.copy()

    for col, (min_value, max_value) in min_max.items():
        df[col] = (pd.to_numeric(df[col], errors="coerce") - min_value) / (
            max_value - min_value
        )

    return df


@click.command(context_settings=CLICK_CONTEXT_SETTINGS)
@click.option("--tag", type=str, required=True)
@click.option("--out-tag", type=str, default=None)
@click.option(
    "--mode",
    type=click.Choice(["site", "pooled-sites", "merged"]),
    default="pooled-sites",
    show_default=True,
    help=(
        "'site': fit one model per site TSV. "
        "'pooled-sites': fit one shared model across site TSVs. "
        "'merged': fit one model on the merged TSV."
    ),
)
@click.option("--min-max/--no-min-max", default=False, show_default=True)
@click.option(
    "--huber/--ols",
    default=False,
    show_default=True,
    help="Use Huber robust regression instead of ordinary least squares (default).",
)
@click.option(
    "--data-dir",
    "dpath_data",
    type=click.Path(path_type=Path, file_okay=False, dir_okay=True),
    default=DEFAULT_DPATH_DATA,
)
@click.option("--age-col", type=str, default=None)
@click.option("--keep-age/--drop-age", default=True, show_default=True)
<<<<<<< HEAD


=======
>>>>>>> 0f41859478a9c80d141c71860dac5fc8dd75e6ff
def regress_age_out(
    tag: str,
    out_tag: str | None,
    mode: str,
    min_max: bool,
    huber: bool,
    dpath_data: Path,
    age_col: str | None,
    keep_age: bool,
):
    if out_tag is None:
        suffix = "age_adjusted_minmax" if min_max else "age_adjusted"
        out_tag = f"{tag}_{suffix}"

    dpath_latest = get_dpath_latest(dpath_data)
    dpath_in = dpath_latest / tag
    dpath_out = dpath_latest / out_tag
    dpath_out.mkdir(parents=True, exist_ok=True)

    fpath_json = dpath_in / f"{tag}.json"
    json_data = json.loads(fpath_json.read_text())

    col_subject = json_data["cols"]["col_subject"]
    time_col = json_data["cols"]["col_timepoint"]
    age_col = age_col or json_data["cols"].get("col_age", DEFAULT_AGE_COL)
    biomarker_cols = json_data["cols"]["cols_biomarker"]

    fpaths_in = _get_input_files(dpath_in, tag, mode)

    dfs_by_path = {
        fpath: pd.read_csv(fpath, sep="\t", dtype={col_subject: str})
        for fpath in fpaths_in
    }

    for fpath, df in dfs_by_path.items():
        missing_cols = [
            col for col in [age_col, time_col, *biomarker_cols] if col not in df.columns
        ]
        if missing_cols:
            raise ValueError(f"{fpath} is missing columns: {missing_cols}")

    if mode == "pooled-sites":
        df_fit = pd.concat(dfs_by_path.values(), ignore_index=True)
        pooled_models = _fit_models(
            df=df_fit,
            biomarker_cols=biomarker_cols,
            age_col=age_col,
            time_col=time_col,
            huber=huber,
        )

    transformed_by_path = {}
    models_by_file = {}

    for fpath, df in dfs_by_path.items():
        if mode == "pooled-sites":
            models = pooled_models
        else:
            models = _fit_models(
                df=df,
                biomarker_cols=biomarker_cols,
                age_col=age_col,
                time_col=time_col,
                huber=huber,
            )

        transformed_by_path[fpath] = _apply_age_adjustment(
            df=df,
            models=models,
            biomarker_cols=biomarker_cols,
            age_col=age_col,
        )
        models_by_file[fpath.name] = models

    min_max_by_col = None
    if min_max:
        min_max_by_col = _fit_global_min_max(
            dfs=list(transformed_by_path.values()),
            biomarker_cols=biomarker_cols,
        )
        transformed_by_path = {
            fpath: _apply_min_max(df, min_max_by_col)
            for fpath, df in transformed_by_path.items()
        }

    out_json = copy.deepcopy(json_data)
    out_json["settings"]["tag"] = out_tag
    out_json["settings"]["dpath_out"] = str(dpath_out)
    out_json["settings"]["age_regression"] = {
        "source_tag": tag,
        "mode": mode,
        "age_col": age_col,
        "time_col": time_col,
        "model": "biomarker ~ 1 + AGE + time",
        "adjustment": "biomarker_adjusted = biomarker - beta_age * AGE",
        "huber": huber,
        "min_max": min_max,
        "min_max_by_col": min_max_by_col,
        "models_by_file": models_by_file,
    }

    out_json["node_id_map"] = {}
    out_json["subjects_by_node"] = {}

    for fpath_in, df_out in transformed_by_path.items():
        if not keep_age:
            df_out = df_out.drop(columns=[age_col])

        fname_out = _replace_tag_prefix(fpath_in.name, tag, out_tag)
        fpath_out = dpath_out / fname_out
        df_out.to_csv(fpath_out, sep="\t", index=False)

        node_id = json_data["node_id_map"][fpath_in.name]
        out_json["node_id_map"][fname_out] = node_id

        if node_id in json_data.get("subjects_by_node", {}):
<<<<<<< HEAD
            out_json["subjects_by_node"][node_id] = json_data["subjects_by_node"][node_id]
=======
            out_json["subjects_by_node"][node_id] = json_data["subjects_by_node"][
                node_id
            ]
>>>>>>> 0f41859478a9c80d141c71860dac5fc8dd75e6ff

        print(f"Saved age-adjusted data to {fpath_out}")

    if keep_age:
        out_json["cols"]["col_age"] = age_col
    else:
        out_json["cols"].pop("col_age", None)

    fpath_json_out = dpath_out / f"{out_tag}.json"
    save_json(fpath_json_out, out_json)
    print(f"Saved metadata to {fpath_json_out}")

    if mode in {"site", "pooled-sites"}:
        print(f"Next: ./scripts/merge_data.py --tag {out_tag}")


if __name__ == "__main__":
    regress_age_out()
<<<<<<< HEAD




=======
>>>>>>> 0f41859478a9c80d141c71860dac5fc8dd75e6ff
