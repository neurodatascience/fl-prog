#!/usr/bin/env python

from dataclasses import dataclass
from pathlib import Path

import click
import numpy as np
import pandas as pd

from fl_prog.utils.constants import CLICK_CONTEXT_SETTINGS, NODE_PREFIX
from fl_prog.utils.io import (
    DEFAULT_DPATH_RESULTS,
    get_dpath_latest,
    load_json,
)


@dataclass
class ModelParams:
    """Parameters aligned to a canonical subject order.

    Per-feature arrays have shape (n_features,); per-subject arrays have
    shape (n_subjects,), aligned to the merged data's subject order.
    """

    k_values: np.ndarray
    x0_values: np.ndarray
    vertical_shifts: np.ndarray
    scaling_factors: np.ndarray
    sigma: np.ndarray
    time_shifts: np.ndarray
    acceleration_factors: np.ndarray


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


def _true_params_by_subject(subjects_by_node: dict, params: dict) -> ModelParams:
    """Align true simulation parameters to the canonical subject order.

    True per-subject parameters are stored as lists per site in file order.
    Site ``i`` (0-indexed) corresponds to node ``str(i + 1)``, whose subjects
    are listed in ``subjects_by_node`` in the same order as in the site TSV.
    """
    per_feature = {
        "k_values": np.asarray(params["k_values"], dtype=float),
        "x0_values": np.asarray(params["x0_values"], dtype=float),
        "vertical_shifts": np.asarray(params["vertical_shifts"], dtype=float),
        "scaling_factors": np.asarray(params["scaling_factors"], dtype=float),
        "sigma": np.asarray(params["sigmas"], dtype=float),
    }

    time_shifts = []
    acceleration_factors = []
    for i_site, (time_shifts_site, acc_site) in enumerate(
        zip(
            params["time_shifts"],
            params["acceleration_factors"],
            strict=True,
        )
    ):
        node_id = str(i_site + 1)
        subjects = subjects_by_node[node_id]
        if len(time_shifts_site) != len(subjects):
            raise ValueError(
                f"true time_shifts for node {node_id} have length "
                f"{len(time_shifts_site)} but {len(subjects)} subjects"
            )
        if len(acc_site) != len(subjects):
            raise ValueError(
                f"true acceleration_factors for node {node_id} have length "
                f"{len(acc_site)} but {len(subjects)} subjects"
            )
        time_shifts.append(np.asarray(time_shifts_site, dtype=float))
        acceleration_factors.append(np.asarray(acc_site, dtype=float))

    return ModelParams(
        **per_feature,
        time_shifts=np.concatenate(time_shifts),
        acceleration_factors=np.concatenate(acceleration_factors),
    )


def _estimated_params_by_setup(
    json_results: dict, subjects_by_node: dict
) -> dict[str, ModelParams]:
    """Align estimated parameters to the canonical subject order, per setup.

    Estimated per-subject parameters are keyed by node (``node_1``, ...,
    ``node_centralized``); subjects are listed in ``subjects_by_node`` in the
    same order as in the estimated arrays. Node keys are processed in JSON
    order, which matches the merged data's subject order.
    """
    params_by_setup = {}
    for setup, result in json_results["results"].items():
        per_feature = {
            "k_values": np.asarray(result["estimated_k_values"], dtype=float),
            "x0_values": np.asarray(result["estimated_x0_values"], dtype=float),
            "vertical_shifts": np.asarray(
                result["estimated_vertical_shifts"], dtype=float
            ),
            "scaling_factors": np.asarray(
                result["estimated_scaling_factors"], dtype=float
            ),
            "sigma": np.asarray(result["estimated_sigma"], dtype=float),
        }

        estimated_time_shifts = result["estimated_time_shifts"]
        estimated_acceleration_factors = result["estimated_acceleration_factors"]

        time_shifts = []
        acceleration_factors = []
        for node_id, shifts in estimated_time_shifts.items():
            subjects = subjects_by_node[node_id.removeprefix(NODE_PREFIX)]
            if len(shifts) != len(subjects):
                raise ValueError(
                    f"estimated time_shifts for {node_id} have length "
                    f"{len(shifts)} but {len(subjects)} subjects"
                )
            acceleration_factors_node = estimated_acceleration_factors[node_id]
            if len(acceleration_factors_node) != len(subjects):
                raise ValueError(
                    f"estimated acceleration_factors for {node_id} have length "
                    f"{len(acceleration_factors_node)} but {len(subjects)} subjects"
                )
            time_shifts.append(np.asarray(shifts, dtype=float))
            acceleration_factors.append(
                np.asarray(acceleration_factors_node, dtype=float)
            )

        params_by_setup[setup] = ModelParams(
            **per_feature,
            time_shifts=np.concatenate(time_shifts),
            acceleration_factors=np.concatenate(acceleration_factors),
        )
    return params_by_setup


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

    true_params_by_subject = _true_params_by_subject(subjects_by_node, true_params)
    assert true_params_by_subject.time_shifts.shape == (n_subjects,)
    assert true_params_by_subject.acceleration_factors.shape == (n_subjects,)
    print(
        f"true time_shifts range: [{true_params_by_subject.time_shifts.min():.3f}, "
        f"{true_params_by_subject.time_shifts.max():.3f}]"
    )
    print(
        f"true acceleration_factors range: [{true_params_by_subject.acceleration_factors.min():.3f}, "
        f"{true_params_by_subject.acceleration_factors.max():.3f}]"
    )

    estimated_by_setup = _estimated_params_by_setup(json_results, subjects_by_node)
    for setup, estimated_params in estimated_by_setup.items():
        assert estimated_params.time_shifts.shape == (n_subjects,)
        assert estimated_params.acceleration_factors.shape == (n_subjects,)
        print(
            f"{setup}: estimated time_shifts range: "
            f"[{estimated_params.time_shifts.min():.3f}, "
            f"{estimated_params.time_shifts.max():.3f}], "
            f"estimated acceleration_factors range: "
            f"[{estimated_params.acceleration_factors.min():.3f}, "
            f"{estimated_params.acceleration_factors.max():.3f}]"
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
