#!/usr/bin/env python

from pathlib import Path

import click
import pandas as pd

from fl_prog.utils.constants import CLICK_CONTEXT_SETTINGS
from fl_prog.utils.io import (
    DEFAULT_DPATH_RESULTS,
    get_dpath_latest,
    load_json,
)


def _get_fpath_json_results(tag: str, dpath_results: Path) -> Path:
    dpath_results_latest = get_dpath_latest(dpath_results)
    return dpath_results_latest / tag / f"{tag}-estimated_params.json"


def _load_json_results(fpath_json_results: Path) -> dict:
    return load_json(fpath_json_results)


def _get_dpath_data(json_results: dict) -> Path:
    return Path(json_results["settings"]["dpath_data"])


def _get_fpath_json_data(json_results: dict) -> Path:
    return Path(json_results["settings"]["fpath_config"])


def _load_json_data(json_results: dict) -> dict:
    return load_json(_get_fpath_json_data(json_results))


def _get_cols(json_data: dict) -> dict:
    return json_data["cols"]


def _get_subjects_by_node(json_data: dict) -> dict:
    return json_data["subjects_by_node"]


def _get_true_params(json_data: dict) -> dict:
    return json_data["params"]


def _load_df_data(json_data: dict, dpath_data: Path, tag: str) -> pd.DataFrame:
    cols = _get_cols(json_data)
    fpath_data = dpath_data / f"{tag}-merged.tsv"
    return pd.read_csv(fpath_data, sep="\t", dtype={cols["col_subject"]: str})


def _save_tsv(df_metrics: pd.DataFrame, fpath_out: Path):
    fpath_out.parent.mkdir(parents=True, exist_ok=True)
    df_metrics.to_csv(fpath_out, sep="\t", index=False)
    print(f"Saved metrics to {fpath_out}")


def assess_fit(
    tag: str,
    dpath_results: Path,
):
    fpath_json_results = _get_fpath_json_results(tag, dpath_results)
    print(f"fpath_json_results: {fpath_json_results}")

    json_results = _load_json_results(fpath_json_results)

    dpath_data = _get_dpath_data(json_results)
    print(f"dpath_data: {dpath_data}")

    json_data = _load_json_data(json_results)

    cols = _get_cols(json_data)
    subjects_by_node = _get_subjects_by_node(json_data)
    true_params = _get_true_params(json_data)

    df_data = _load_df_data(json_data, dpath_data, tag)

    col_subject = cols["col_subject"]
    n_biomarkers = len(cols["cols_biomarker"])
    n_subjects = df_data[col_subject].nunique()
    print(f"n_biomarkers: {n_biomarkers}")
    print(f"n_subjects: {n_subjects}")
    print(
        "n_subjects_by_node: "
        + ", ".join(
            f"{node_id}: {len(subjects)}"
            for node_id, subjects in subjects_by_node.items()
        )
    )

    assert set(cols["cols_biomarker"]).issubset(df_data.columns)
    subjects_in_nodes = sorted(
        {
            str(subject)
            for node_id, subjects in subjects_by_node.items()
            if node_id != "centralized"
            for subject in subjects
        },
    )
    assert subjects_in_nodes == sorted(df_data[col_subject].unique()), (
        "subjects_by_node does not match subjects in merged data"
    )

    df_metrics = pd.DataFrame(
        columns=["setup", "set_name", "col_biomarker", "metric", "value"]
    )

    fpath_out = fpath_json_results.with_name(f"{tag}-fit_quality.tsv")
    _save_tsv(df_metrics, fpath_out)

    return df_metrics


@click.command(context_settings=CLICK_CONTEXT_SETTINGS)
@click.option("--tag", type=str, required=True)
@click.option(
    "--results-dir",
    "dpath_results",
    type=click.Path(path_type=Path, file_okay=False, dir_okay=True),
    default=DEFAULT_DPATH_RESULTS,
)
def main(**params):
    assess_fit(**params)


if __name__ == "__main__":
    main()
