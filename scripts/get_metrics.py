#!/usr/bin/env python
from pathlib import Path

import click
import numpy as np
import pandas as pd

from fl_prog.metrics import (
    compute_per_subject_recovery,
    compute_predictive_metrics,
    compute_recovery_metrics,
)
from fl_prog.results import (
    DEFAULT_LEASPY_ALGORITHM_NAME,
    DEFAULT_LEASPY_N_JOBS,
    DEFAULT_LEASPY_SEED,
    MODEL_LEASPY,
    LeaspyIndividualParametersNotFoundError,
    PredictFn,
    estimated_params_by_setup,
    get_cols,
    get_dpath_data,
    get_model_type,
    get_predict_fn,
    get_run_tag,
    get_subject_ids_from_frame,
    iter_result_files,
    load_df_data,
    load_df_data_leaspy,
    load_json_data,
    load_leaspy_individual_parameters,
    load_leaspy_model,
    personalize_leaspy,
    predict_leaspy,
    save_leaspy_individual_parameters,
    true_params_by_subject,
    validate_subject_ids,
)
from fl_prog.utils.constants import CLICK_CONTEXT_SETTINGS, NODE_ID_CENTRALIZED
from fl_prog.utils.io import DEFAULT_DPATH_RESULTS, get_dpath_latest


def _save_tsv(df_metrics: pd.DataFrame, fpath_out: Path):
    fpath_out.parent.mkdir(parents=True, exist_ok=True)
    df_metrics.to_csv(fpath_out, sep="\t", index=False)
    click.secho(f"\tSaved metrics to {fpath_out}", fg="green")


def get_metrics_single_run(
    json_results: dict,
    tag: str,
    predict_fn: PredictFn,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    print(f"\tUsing predict_fn: {predict_fn.__name__}")

    dpath_data = get_dpath_data(json_results)
    print(f"\tdpath_data: {dpath_data}")

    json_data = load_json_data(json_results)

    cols = get_cols(json_data)
    subjects_by_node = json_data["subjects_by_node"]
    try:
        true_params = json_data["params"]
    except KeyError:
        true_params = None
        click.secho(
            f"\tTrue parameters not found in {json_results['settings']['fpath_config']}."
            " Skipping recovery metrics.",
            fg="yellow",
        )

    df_data_train = load_df_data(dpath_data, tag, cols, test=False)
    try:
        df_data_test = load_df_data(dpath_data, tag, cols, test=True)
    except FileNotFoundError:
        df_data_test = None

    # same for train/test data
    col_subject = cols["col_subject"]
    n_subjects = df_data_train[col_subject].nunique()

    if not set(cols["cols_biomarker"]).issubset(df_data_train.columns):
        raise ValueError(
            "cols_biomarker contains columns not present in the merged data"
        )
    subjects_in_nodes = sorted(
        {
            str(subject)
            for node_id, subjects in subjects_by_node.items()
            if node_id != NODE_ID_CENTRALIZED
            for subject in subjects
        },
    )
    if subjects_in_nodes != sorted(df_data_train[col_subject].unique()):
        raise ValueError("subjects_by_node does not match subjects in merged data")

    if true_params is not None:
        true_params_aligned = true_params_by_subject(subjects_by_node, true_params)
        if true_params_aligned.time_shifts.shape != (n_subjects,):
            raise ValueError(
                f"true time_shifts have shape {true_params_aligned.time_shifts.shape} "
                f"but expected ({n_subjects},)"
            )
        if true_params_aligned.acceleration_factors.shape != (n_subjects,):
            raise ValueError(
                f"true acceleration_factors have shape "
                f"{true_params_aligned.acceleration_factors.shape} "
                f"but expected ({n_subjects},)"
            )
        print(
            f"true time_shifts range: [{true_params_aligned.time_shifts.min():.3f}, "
            f"{true_params_aligned.time_shifts.max():.3f}]"
        )
        print(
            f"true acceleration_factors range: [{true_params_aligned.acceleration_factors.min():.3f}, "
            f"{true_params_aligned.acceleration_factors.max():.3f}]"
        )

    estimated_by_setup = estimated_params_by_setup(json_results, subjects_by_node)
    for setup, estimated_params in estimated_by_setup.items():
        if estimated_params.time_shifts.shape != (n_subjects,):
            raise ValueError(
                f"{setup}: estimated time_shifts have shape "
                f"{estimated_params.time_shifts.shape} but expected ({n_subjects},)"
            )
        if estimated_params.acceleration_factors.shape != (n_subjects,):
            raise ValueError(
                f"{setup}: estimated acceleration_factors have shape "
                f"{estimated_params.acceleration_factors.shape} but expected "
                f"({n_subjects},)"
            )

    cols_biomarker = cols["cols_biomarker"]
    t_train = df_data_train[cols["col_timepoint"]].to_numpy(dtype=float)
    subject_ids_train = df_data_train[cols["col_subject_index"]].to_numpy(dtype=int)
    y_true_train = df_data_train[cols_biomarker].to_numpy(dtype=float)

    validate_subject_ids(subject_ids_train, n_subjects, "train")

    subject_ids_test = np.array([], dtype=int)
    if df_data_test is not None:
        t_test = df_data_test[cols["col_timepoint"]].to_numpy(dtype=float)
        subject_ids_test = get_subject_ids_from_frame(df_data_test, cols, df_data_train)
        y_true_test = df_data_test[cols_biomarker].to_numpy(dtype=float)
        validate_subject_ids(subject_ids_test, n_subjects, "test")

    rows: list[dict[str, str | float]] = []
    rows_recovery: list[dict[str, str | float]] = []
    for setup, estimated_params in estimated_by_setup.items():
        y_pred_train = predict_fn(estimated_params, t_train, subject_ids_train)
        rows.extend(
            compute_predictive_metrics(
                y_true_train,
                y_pred_train,
                cols_biomarker,
                setup=setup,
                set_name="train",
            )
        )
        if df_data_test is not None:
            y_pred_test = predict_fn(estimated_params, t_test, subject_ids_test)
            rows.extend(
                compute_predictive_metrics(
                    y_true_test,
                    y_pred_test,
                    cols_biomarker,
                    setup=setup,
                    set_name="test",
                )
            )
        if true_params is not None:
            rows_recovery.extend(
                compute_recovery_metrics(
                    true_params_aligned,
                    estimated_params,
                    cols_biomarker,
                    setup=setup,
                )
            )
            rows_recovery.extend(
                compute_per_subject_recovery(
                    true_params_aligned, estimated_params, setup=setup
                )
            )

    df_metrics = pd.DataFrame(
        rows, columns=["setup", "set_name", "col_biomarker", "metric", "value"]
    )

    df_recovery = pd.DataFrame(
        rows_recovery,
        columns=["setup", "set_name", "col_biomarker", "metric", "value"],
    )

    return df_metrics, df_recovery


def _get_leaspy_individual_parameters(
    fpath_json_results: Path,
    json_results: dict,
    setup: str,
    model,
    df_leaspy: pd.DataFrame,
    algorithm_name=DEFAULT_LEASPY_ALGORITHM_NAME,
    seed=DEFAULT_LEASPY_SEED,
    n_jobs=DEFAULT_LEASPY_N_JOBS,
) -> pd.DataFrame:
    """Load stored personalized parameters, or personalize and store them."""
    personalization_kwargs = {
        "algorithm_name": algorithm_name,
        "seed": seed,
        "n_jobs": n_jobs,
    }
    try:
        df_individual_params = load_leaspy_individual_parameters(
            json_results, setup, **personalization_kwargs
        )
    except LeaspyIndividualParametersNotFoundError:
        pass
    else:
        print(f"\tUsing stored Leaspy individual parameters ({setup})")
        return df_individual_params

    print(f"\tPersonalizing Leaspy model ({setup})")
    df_individual_params = personalize_leaspy(
        model, df_leaspy, **personalization_kwargs
    )
    save_leaspy_individual_parameters(
        fpath_json_results,
        json_results,
        setup,
        df_individual_params,
        **personalization_kwargs,
    )
    return df_individual_params


def get_metrics_single_run_leaspy(
    fpath_json_results: Path,
    json_results: dict,
    tag: str,
    algorithm_name=DEFAULT_LEASPY_ALGORITHM_NAME,
    seed=DEFAULT_LEASPY_SEED,
    n_jobs=DEFAULT_LEASPY_N_JOBS,
) -> pd.DataFrame:
    dpath_data = get_dpath_data(json_results)
    print(f"dpath_data: {dpath_data}")

    json_data = load_json_data(json_results)

    cols = get_cols(json_data)
    cols_biomarker = cols["cols_biomarker"]

    df_data_train = load_df_data_leaspy(dpath_data, tag, cols, test=False)
    try:
        df_data_test = load_df_data_leaspy(dpath_data, tag, cols, test=True)
    except FileNotFoundError:
        df_data_test = None

    rows: list[dict[str, str | float]] = []
    for setup, model_params in json_results["results"].items():
        model = load_leaspy_model(model_params)

        # get the individual parameters from the train data
        # (stored in the results file after the first run)
        df_individual_params = _get_leaspy_individual_parameters(
            fpath_json_results,
            json_results,
            setup,
            model,
            df_data_train,
            algorithm_name=algorithm_name,
            seed=seed,
            n_jobs=n_jobs,
        )

        for df, set_name in [(df_data_train, "train"), (df_data_test, "test")]:
            if df is None:
                continue

            rows.extend(
                compute_predictive_metrics(
                    y_true=df[cols_biomarker].to_numpy(dtype=float),
                    y_pred=predict_leaspy(model, df_individual_params, df),
                    cols_biomarker=cols_biomarker,
                    setup=setup,
                    set_name=set_name,
                )
            )

    return pd.DataFrame(
        rows, columns=["setup", "set_name", "col_biomarker", "metric", "value"]
    )


def get_metrics(
    tag: str,
    dpath_results: Path,
    leaspy_algorithm_name=DEFAULT_LEASPY_ALGORITHM_NAME,
    leaspy_seed=DEFAULT_LEASPY_SEED,
    leaspy_n_jobs=DEFAULT_LEASPY_N_JOBS,
):
    dpath_results_latest = get_dpath_latest(dpath_results)
    for fpath_json_results, json_results in iter_result_files(
        dpath_results_latest / tag
    ):
        run_tag = get_run_tag(fpath_json_results)
        model_type = get_model_type(fpath_json_results)
        print(f"fpath_json_results: {fpath_json_results}")

        if model_type == MODEL_LEASPY:
            df_metrics = get_metrics_single_run_leaspy(
                fpath_json_results,
                json_results,
                tag,
                algorithm_name=leaspy_algorithm_name,
                seed=leaspy_seed,
                n_jobs=leaspy_n_jobs,
            )
            df_recovery = pd.DataFrame()
            suffix = f"-{leaspy_algorithm_name}_{leaspy_seed}"
        else:
            df_metrics, df_recovery = get_metrics_single_run(
                json_results, tag, predict_fn=get_predict_fn(model_type)
            )
            suffix = ""

        fpath_metrics_out = fpath_json_results.with_name(
            f"{run_tag}-fit_quality{suffix}.tsv"
        )
        _save_tsv(df_metrics, fpath_metrics_out)

        if not df_recovery.empty:
            fpath_recovery_out = fpath_json_results.with_name(
                f"{run_tag}-param_recovery{suffix}.tsv"
            )
            _save_tsv(df_recovery, fpath_recovery_out)


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
    "--leaspy-algorithm-name",
    type=str,
    default=DEFAULT_LEASPY_ALGORITHM_NAME,
    help="Algorithm name for Leaspy model personalization.",
    show_envvar=True,
)
@click.option(
    "--leaspy-seed",
    type=int,
    default=DEFAULT_LEASPY_SEED,
    help="Random seed for Leaspy model personalization.",
    envvar="RNG_SEED",
    show_envvar=True,
)
@click.option(
    "--leaspy-n-jobs",
    type=int,
    default=DEFAULT_LEASPY_N_JOBS,
    help="Number of parallel jobs for Leaspy model personalization.",
    envvar="LEASPY_N_JOBS",
    show_envvar=True,
)
def main(**params):
    get_metrics(**params)


if __name__ == "__main__":
    main()
