#!/usr/bin/env python
import dataclasses
import fnmatch
import functools
from collections.abc import Callable, Iterable
from pathlib import Path

import click
import numpy as np
import pandas as pd

from fl_prog.metrics import (
    COL_FORECAST_AFTER,
    COL_N_TIMEPOINTS,
    SET_NAME_TEST_FIT,
    SET_NAME_TEST_FORECAST,
    SUFFIX_SHIFTED,
    compute_per_subject_recovery,
    compute_predictive_metrics,
    compute_recovery_metrics,
    compute_time_axis_metrics,
    get_midpoint_offset,
    shift_time_axis,
)
from fl_prog.model import ModelParams
from fl_prog.personalization import (
    DEFAULT_DPMOST_N_OUTER_ITER,
    DEFAULT_PERSONALIZE_ALGORITHM,
    DEFAULT_PERSONALIZE_SEED,
    PERSONALIZE_ALGORITHMS,
    get_first_timepoints_mask,
    get_penalty_weight,
    personalize_dpmost,
    personalize_fl_prog,
)
from fl_prog.results import (
    DEFAULT_LEASPY_ALGORITHM_NAME,
    DEFAULT_LEASPY_N_JOBS,
    DEFAULT_LEASPY_SEED,
    MODEL_DPMOST,
    MODEL_LEASPY,
    LeaspyIndividualParametersNotFoundError,
    PredictFn,
    estimated_params_by_setup,
    get_cols,
    get_dpath_data,
    get_heldout_personalization_key,
    get_leaspy_heldout_setup,
    get_model_type,
    get_predict_fn,
    get_run_tag,
    get_subject_ids_from_frame,
    get_subjects,
    has_heldout_subjects,
    iter_result_files,
    load_df_data,
    load_df_data_leaspy,
    load_dpmost_normalisation,
    load_json_data,
    load_leaspy_individual_parameters,
    load_leaspy_model,
    personalize_leaspy,
    predict_leaspy,
    save_leaspy_individual_parameters,
    save_personalized_params,
    true_params_by_subject,
    validate_subject_ids,
)
from fl_prog.utils.constants import (
    CLICK_CONTEXT_SETTINGS,
    LEASPY_COL_SUBJECT,
    LEASPY_COL_TIMEPOINT,
)
from fl_prog.utils.io import DEFAULT_DPATH_RESULTS, get_dpath_latest

# held-out subjects are also personalized on their first N visits only, for each N
DEFAULT_PERSONALIZE_N_TIMEPOINTS = (1, 2, 3)
# recovery set names of held-out subjects (the only ones that depend on N)
PREFIX_SET_NAME_RECOVERY_TEST = "recovery_per_subject_test"


def _save_tsv(df_metrics: pd.DataFrame, fpath_out: Path):
    fpath_out.parent.mkdir(parents=True, exist_ok=True)
    df_metrics.to_csv(fpath_out, sep="\t", index=False)
    click.secho(f"\tSaved metrics to {fpath_out}", fg="green")


def _get_test_row_sets(
    subjects: np.ndarray | pd.Series | pd.Index,
    timepoints: np.ndarray | pd.Series | pd.Index,
    heldout: bool,
    n_timepoints: int | None,
    forecast_after: int | None = None,
) -> tuple[np.ndarray, list[tuple[str, np.ndarray]]]:
    """Rows of the test data used for personalization, and the (set name, rows)
    to compute metrics on.

    Without ``n_timepoints``, held-out subjects are personalized and scored on
    all their visits ("test"). Otherwise, they are personalized on their first
    ``n_timepoints`` visits, and scored separately on these ("test_fit") and on
    their visits after the first ``forecast_after`` ("test_forecast") timepoints.
    Only subjects with more than ``forecast_after`` visits are scored.
    """
    rows_all = np.ones(len(subjects), dtype=bool)
    if n_timepoints is None:
        if forecast_after is not None:
            raise click.UsageError(
                "--forecast-after requires --personalize-n-timepoints"
            )
        return rows_all, [("test", rows_all)]
    if not heldout:
        raise click.UsageError(
            "--personalize-n-timepoints only applies to held-out subjects "
            "(split_train_test.py --mode subject)"
        )
    if forecast_after is None:
        forecast_after = n_timepoints
    elif forecast_after < n_timepoints:
        raise click.UsageError(
            f"--forecast-after ({forecast_after}) must be at least "
            f"--personalize-n-timepoints ({n_timepoints})"
        )
    subjects = pd.Series(np.asarray(subjects))
    rows_fit = get_first_timepoints_mask(subjects, timepoints, n_timepoints)
    rows_before_forecast = get_first_timepoints_mask(
        subjects, timepoints, forecast_after
    )
    n_visits = subjects.map(subjects.value_counts())
    rows_scored = (n_visits > forecast_after).to_numpy()
    n_excluded = subjects[~rows_scored].nunique()
    print(
        f"\tPersonalizing held-out subjects on their first {n_timepoints} visit(s) "
        f"and forecasting visits after the first {forecast_after}; excluding "
        f"{n_excluded} of {subjects.nunique()} subjects with no later visits "
        "from scoring"
    )
    return rows_fit, [
        ("test_fit", rows_fit & rows_scored),
        ("test_forecast", ~rows_before_forecast & rows_scored),
    ]


def _get_personalize_fn(
    model_type: str,
    fpath_json_results: Path,
    json_results: dict,
    cols: dict,
    df_data_train: pd.DataFrame,
    df_data_test: pd.DataFrame,
    subject_ids_test: np.ndarray,
    subjects_test: list[str],
    algorithm: str = DEFAULT_PERSONALIZE_ALGORITHM,
    seed: int = DEFAULT_PERSONALIZE_SEED,
    dpmost_n_outer_iter: int = DEFAULT_DPMOST_N_OUTER_ITER,
) -> Callable[[ModelParams], ModelParams]:
    """Function estimating held-out test subjects' parameters from fitted ones.

    The returned parameters are indexed by ``subject_ids_test``.
    """
    cols_biomarker = cols["cols_biomarker"]
    match model_type:
        case "fedbiomed":
            print(f"\tPersonalization algorithm: {algorithm} (seed {seed})")
            model_args = json_results["settings"]["model_args"]["lr_with_shift"]
            penalty_weight = get_penalty_weight(
                n_entries=len(df_data_train),
                n_features=len(cols_biomarker),
                n_subjects=df_data_train[cols["col_subject"]].nunique(),
            )
            t_test = df_data_test[cols["col_timepoint"]].to_numpy(dtype=float)
            y_test = df_data_test[cols_biomarker].to_numpy(dtype=float)

            def _personalize(params: ModelParams) -> ModelParams:
                return personalize_fl_prog(
                    params,
                    model_args,
                    t_test,
                    subject_ids_test,
                    y_test,
                    penalty_weight=penalty_weight,
                    algorithm=algorithm,
                    seed=seed,
                )

        case "dpmost":
            fpath_model = fpath_json_results.with_name(
                f"{get_run_tag(fpath_json_results)}-model.pkl"
            )
            optimize_args = json_results["settings"]["optimize_args"]

            def _personalize(params: ModelParams) -> ModelParams:
                time_shifts = personalize_dpmost(
                    fpath_model,
                    df_data_test,
                    cols,
                    init_time_shift=params.time_shifts.mean(),
                    optimize_args=optimize_args,
                    n_outer_iter=dpmost_n_outer_iter,
                )
                return dataclasses.replace(
                    params,
                    time_shifts=time_shifts.loc[subjects_test].to_numpy(dtype=float),
                    acceleration_factors=np.ones(len(subjects_test)),
                )

        case _:
            raise ValueError(f"Cannot personalize model type {model_type!r}")

    return _personalize


def get_metrics_single_run(
    json_results: dict,
    tag: str,
    predict_fn: PredictFn,
    skip_recovery: bool = False,
    fpath_json_results: Path | None = None,
    model_type: str | None = None,
    personalize_algorithm: str = DEFAULT_PERSONALIZE_ALGORITHM,
    personalize_seed: int = DEFAULT_PERSONALIZE_SEED,
    dpmost_n_outer_iter: int = DEFAULT_DPMOST_N_OUTER_ITER,
    personalize_n_timepoints: int | None = None,
    forecast_after: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Predictive and recovery metrics for a fl_prog (Fed-BioMed) or DPMoSt run."""
    print(f"\tUsing predict_fn: {getattr(predict_fn, 'func', predict_fn).__name__}")

    dpath_data = get_dpath_data(json_results)
    print(f"\tdpath_data: {dpath_data}")

    json_data = load_json_data(json_results)

    cols = get_cols(json_data)
    subjects_by_node = json_data["subjects_by_node"]
    heldout = has_heldout_subjects(subjects_by_node)
    if skip_recovery:
        true_params = None
        click.secho(
            "\tEstimated parameters are in normalized units. Skipping recovery metrics.",
            fg="yellow",
        )
    elif "params" in json_data:
        true_params = json_data["params"]
    else:
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
    if sorted(get_subjects(subjects_by_node)) != sorted(
        df_data_train[col_subject].unique()
    ):
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
        if heldout and df_data_test is not None:
            true_params_test = true_params_by_subject(
                subjects_by_node, true_params, test=True
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
    personalize_fn = None
    if df_data_test is not None:
        t_test = df_data_test[cols["col_timepoint"]].to_numpy(dtype=float)
        y_true_test = df_data_test[cols_biomarker].to_numpy(dtype=float)
        rows_personalize, test_row_sets = _get_test_row_sets(
            df_data_test[col_subject],
            t_test,
            heldout=heldout,
            n_timepoints=personalize_n_timepoints,
            forecast_after=forecast_after,
        )
        if heldout:
            # held-out subjects are indexed from 0, in subjects_by_node order
            subject_ids_test, subjects_test = pd.factorize(df_data_test[col_subject])
            subjects_test = subjects_test.tolist()
            if subjects_test != get_subjects(subjects_by_node, test=True):
                raise ValueError(
                    "Held-out subjects in the test data do not match subjects_by_node"
                )
            personalize_fn = _get_personalize_fn(
                model_type,
                fpath_json_results,
                json_results,
                cols,
                df_data_train,
                df_data_test[rows_personalize],
                subject_ids_test[rows_personalize],
                subjects_test,
                algorithm=personalize_algorithm,
                seed=personalize_seed,
                dpmost_n_outer_iter=dpmost_n_outer_iter,
            )
        else:
            subject_ids_test = get_subject_ids_from_frame(
                df_data_test, cols, df_data_train
            )
            validate_subject_ids(subject_ids_test, n_subjects, "test")

    rows: list[dict[str, str | float]] = []
    rows_recovery: list[dict[str, str | float]] = []
    for setup, estimated_params in estimated_by_setup.items():
        # the origin of the disease-time axis is arbitrary: also report recovery
        # after removing it (midpoint offset, "_shifted"). The same offset is used
        # for held-out subjects.
        if true_params is not None:
            midpoint_offset = get_midpoint_offset(true_params_aligned, estimated_params)
            rows_recovery.extend(
                compute_time_axis_metrics(midpoint_offset, setup=setup)
            )
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
            params_test = estimated_params
            if personalize_fn is not None:
                print(f"\tPersonalizing held-out test subjects ({setup})")
                params_test = personalize_fn(estimated_params)
                save_personalized_params(
                    fpath_json_results,
                    json_results,
                    get_heldout_personalization_key(
                        model_type,
                        fedbiomed_personalize_algorithm=personalize_algorithm,
                        personalize_seed=personalize_seed,
                        dpmost_personalize_n_outer_iter=dpmost_n_outer_iter,
                        personalize_n_timepoints=personalize_n_timepoints,
                    ),
                    setup,
                    subjects_test,
                    params_test,
                )
                if true_params is not None:
                    rows_recovery.extend(
                        compute_per_subject_recovery(
                            true_params_test,
                            params_test,
                            setup=setup,
                            set_name="recovery_per_subject_test",
                        )
                    )
                    rows_recovery.extend(
                        compute_per_subject_recovery(
                            true_params_test,
                            shift_time_axis(params_test, midpoint_offset),
                            setup=setup,
                            set_name=f"recovery_per_subject_test{SUFFIX_SHIFTED}",
                        )
                    )
            y_pred_test = predict_fn(params_test, t_test, subject_ids_test)
            for set_name, rows_set in test_row_sets:
                rows.extend(
                    compute_predictive_metrics(
                        y_true_test[rows_set],
                        y_pred_test[rows_set],
                        cols_biomarker,
                        setup=setup,
                        set_name=set_name,
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
            estimated_params_shifted = shift_time_axis(
                estimated_params, midpoint_offset
            )
            rows_recovery.extend(
                compute_recovery_metrics(
                    true_params_aligned,
                    estimated_params_shifted,
                    cols_biomarker,
                    setup=setup,
                    set_name=f"recovery_per_biomarker{SUFFIX_SHIFTED}",
                )
            )
            rows_recovery.extend(
                compute_per_subject_recovery(
                    true_params_aligned,
                    estimated_params_shifted,
                    setup=setup,
                    set_name=f"recovery_per_subject{SUFFIX_SHIFTED}",
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
    personalize_n_timepoints: int | None = None,
    forecast_after: int | None = None,
) -> pd.DataFrame:
    dpath_data = get_dpath_data(json_results)
    print(f"dpath_data: {dpath_data}")

    json_data = load_json_data(json_results)

    cols = get_cols(json_data)
    cols_biomarker = cols["cols_biomarker"]
    heldout = has_heldout_subjects(json_data["subjects_by_node"])

    df_data_train = load_df_data_leaspy(dpath_data, tag, cols, test=False)
    try:
        df_data_test = load_df_data_leaspy(dpath_data, tag, cols, test=True)
    except FileNotFoundError:
        df_data_test = None

    rows_personalize, test_row_sets = None, []
    if df_data_test is not None:
        rows_personalize, test_row_sets = _get_test_row_sets(
            df_data_test.index.get_level_values(LEASPY_COL_SUBJECT),
            df_data_test.index.get_level_values(LEASPY_COL_TIMEPOINT),
            heldout=heldout,
            n_timepoints=personalize_n_timepoints,
            forecast_after=forecast_after,
        )

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

        # held-out test subjects need their own individual parameters
        df_individual_params_test = df_individual_params
        if heldout and df_data_test is not None:
            df_individual_params_test = _get_leaspy_individual_parameters(
                fpath_json_results,
                json_results,
                get_leaspy_heldout_setup(setup, personalize_n_timepoints),
                model,
                df_data_test[rows_personalize],
                algorithm_name=algorithm_name,
                seed=seed,
                n_jobs=n_jobs,
            )

        datasets = [(df_data_train, df_individual_params, "train")] + [
            (df_data_test[rows_set], df_individual_params_test, set_name)
            for set_name, rows_set in test_row_sets
        ]
        for df, df_ip, set_name in datasets:
            rows.extend(
                compute_predictive_metrics(
                    y_true=df[cols_biomarker].to_numpy(dtype=float),
                    y_pred=predict_leaspy(model, df_ip, df),
                    cols_biomarker=cols_biomarker,
                    setup=setup,
                    set_name=set_name,
                )
            )

    return pd.DataFrame(
        rows, columns=["setup", "set_name", "col_biomarker", "metric", "value"]
    )


def _get_metrics_single_run_leaspy_with_recovery(
    *args, **kwargs
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Call ``get_metrics_single_run_leaspy`` and return its metrics with an empty
    recovery table."""
    return get_metrics_single_run_leaspy(*args, **kwargs), pd.DataFrame()


def _with_n_timepoints(
    df: pd.DataFrame, n_timepoints: int | None, forecast_after: int | None
) -> pd.DataFrame:
    """Add ``n_timepoints`` and ``forecast_after`` columns to ``df`` (empty if
    None)."""
    return df.assign(
        **{
            COL_N_TIMEPOINTS: pd.array([n_timepoints] * len(df), dtype="Int64"),
            COL_FORECAST_AFTER: pd.array([forecast_after] * len(df), dtype="Int64"),
        }
    )


def get_metrics(
    tag: str,
    dpath_results: Path,
    leaspy_algorithm_name=DEFAULT_LEASPY_ALGORITHM_NAME,
    leaspy_seed=DEFAULT_LEASPY_SEED,
    leaspy_n_jobs=DEFAULT_LEASPY_N_JOBS,
    fedbiomed_personalize_algorithm=DEFAULT_PERSONALIZE_ALGORITHM,
    personalize_seed=DEFAULT_PERSONALIZE_SEED,
    dpmost_personalize_n_outer_iter=DEFAULT_DPMOST_N_OUTER_ITER,
    personalize_n_timepoints: Iterable[int] = DEFAULT_PERSONALIZE_N_TIMEPOINTS,
    forecast_after: int | None = None,
    run_tag_pattern: str = "*",
):
    """Compute the metrics of each run of ``tag`` and save them to one metrics file
    and one recovery file per run.

    Held-out subjects are personalized on all their visits, then on their first N
    visits for each N in ``personalize_n_timepoints``. ``forecast_after`` defaults
    to the largest N.
    """
    n_values = sorted(set(personalize_n_timepoints))
    if n_values and forecast_after is None:
        forecast_after = n_values[-1]
    if not n_values and forecast_after is not None:
        raise click.UsageError("--forecast-after requires --personalize-n-timepoints")

    dpath_results_latest = get_dpath_latest(dpath_results)
    for fpath_json_results, json_results in iter_result_files(
        dpath_results_latest / tag
    ):
        run_tag = get_run_tag(fpath_json_results)
        if not fnmatch.fnmatchcase(run_tag, run_tag_pattern):
            continue
        model_type = get_model_type(fpath_json_results)
        print(f"fpath_json_results: {fpath_json_results}")
        heldout = has_heldout_subjects(load_json_data(json_results)["subjects_by_node"])
        # personalization on the first N visits only applies to held-out subjects
        n_values_run = n_values if heldout else []
        if n_values and not heldout:
            click.secho(
                "\tNo held-out subjects: skipping personalization on the first visits",
                fg="yellow",
            )

        # metrics (and recovery) of this run for one personalization setting
        # (personalize_n_timepoints, forecast_after)
        if model_type == MODEL_LEASPY:
            get_metrics_run = functools.partial(
                _get_metrics_single_run_leaspy_with_recovery,
                fpath_json_results,
                json_results,
                tag,
                algorithm_name=leaspy_algorithm_name,
                seed=leaspy_seed,
                n_jobs=leaspy_n_jobs,
            )
            suffix = f"-{leaspy_algorithm_name}_{leaspy_seed}"
        else:
            normalisation = None
            if model_type == MODEL_DPMOST:
                normalisation = load_dpmost_normalisation(
                    fpath_json_results,
                    json_results,
                    get_cols(load_json_data(json_results))["cols_biomarker"],
                )
                if normalisation is not None:
                    print("\tApplying DPMoSt data normalization")
            get_metrics_run = functools.partial(
                get_metrics_single_run,
                json_results,
                tag,
                predict_fn=get_predict_fn(model_type, normalisation=normalisation),
                skip_recovery=normalisation is not None,
                fpath_json_results=fpath_json_results,
                model_type=model_type,
                personalize_algorithm=fedbiomed_personalize_algorithm,
                personalize_seed=personalize_seed,
                dpmost_n_outer_iter=dpmost_personalize_n_outer_iter,
            )
            suffix = ""
            if heldout:
                personalization_key = get_heldout_personalization_key(
                    model_type,
                    fedbiomed_personalize_algorithm=fedbiomed_personalize_algorithm,
                    personalize_seed=personalize_seed,
                    dpmost_personalize_n_outer_iter=dpmost_personalize_n_outer_iter,
                )
                suffix = f"-{personalization_key}"

        # personalization on all visits
        df_metrics, df_recovery = get_metrics_run(
            personalize_n_timepoints=None, forecast_after=None
        )
        dfs_metrics = [_with_n_timepoints(df_metrics, None, None)]
        dfs_recovery = [_with_n_timepoints(df_recovery, None, None)]
        # personalization on the first N visits: only the held-out subjects' rows
        # depend on N (train rows are the same as above)
        for n_timepoints in n_values_run:
            df_metrics_n, df_recovery_n = get_metrics_run(
                personalize_n_timepoints=n_timepoints, forecast_after=forecast_after
            )
            dfs_metrics.append(
                _with_n_timepoints(
                    df_metrics_n.loc[
                        df_metrics_n["set_name"].isin(
                            [SET_NAME_TEST_FIT, SET_NAME_TEST_FORECAST]
                        )
                    ],
                    n_timepoints,
                    forecast_after,
                )
            )
            if not df_recovery_n.empty:
                dfs_recovery.append(
                    _with_n_timepoints(
                        df_recovery_n.loc[
                            df_recovery_n["set_name"].str.startswith(
                                PREFIX_SET_NAME_RECOVERY_TEST
                            )
                        ],
                        n_timepoints,
                        forecast_after,
                    )
                )
        df_metrics = pd.concat(dfs_metrics, ignore_index=True)
        dfs_recovery = [df for df in dfs_recovery if not df.empty]
        df_recovery = (
            pd.concat(dfs_recovery, ignore_index=True)
            if dfs_recovery
            else pd.DataFrame()
        )

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
@click.option(
    "--fedbiomed-personalize-algorithm",
    type=click.Choice(PERSONALIZE_ALGORITHMS),
    default=DEFAULT_PERSONALIZE_ALGORITHM,
    help="Algorithm for estimating held-out subjects' parameters (Fed-BioMed).",
)
@click.option(
    "--personalize-seed",
    type=int,
    default=DEFAULT_PERSONALIZE_SEED,
    help="Random seed for estimating held-out subjects' parameters (Fed-BioMed).",
    envvar="RNG_SEED",
    show_envvar=True,
)
@click.option(
    "--dpmost-personalize-n-outer-iter",
    type=click.IntRange(min=1),
    default=DEFAULT_DPMOST_N_OUTER_ITER,
    help="Number of outer iterations for estimating held-out subjects' time shifts (DPMoSt).",
)
@click.option(
    "--personalize-n-timepoints",
    type=click.IntRange(min=1),
    multiple=True,
    default=DEFAULT_PERSONALIZE_N_TIMEPOINTS,
    help="Also estimate held-out subjects' parameters from their first N visits "
    "only (can be repeated), and score these (test_fit) and later visits "
    "(test_forecast, see --forecast-after) separately. Subjects without enough "
    "visits are not scored. Results for all N are in the same file, in the "
    f"{COL_N_TIMEPOINTS}/{COL_FORECAST_AFTER} columns (empty for personalization "
    "on all visits).",
)
@click.option(
    "--forecast-after",
    type=click.IntRange(min=1),
    default=None,
    help="Score forecasts on visits after the first K (>= every N) only, the same "
    "visits for every N. Subjects with K visits or fewer are not scored. Default: "
    "largest N.",
)
@click.option(
    "--run-tag",
    "run_tag_pattern",
    type=str,
    default="*",
    show_default=True,
    help=(
        "Only process results files whose run tag (file name without "
        "-estimated_params.json) matches this glob pattern."
    ),
)
def main(**params):
    get_metrics(**params)


if __name__ == "__main__":
    main()
