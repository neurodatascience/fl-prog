"""Load model fitting results and apply fitted parameters to subject data."""

import sys
import tempfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from fl_prog.metrics import PER_SUBJECT_FIELDS
from fl_prog.model import LogisticRegressionModelWithShift, ModelParams
from fl_prog.utils.constants import (
    LEASPY_COL_SUBJECT,
    LEASPY_COL_TIMEPOINT,
    NODE_PREFIX,
)
from fl_prog.utils.io import (
    DPATH_PROJECT,
    format_df_for_leaspy,
    load_json,
    save_json,
)

TRUE_FEATURE_KEYS = {
    "k_values": "k_values",
    "x0_values": "x0_values",
    "scaling_factors": "scaling_factors",
    "sigma": "sigmas",
}

ESTIMATED_FEATURE_KEYS = {
    "k_values": "estimated_k_values",
    "x0_values": "estimated_x0_values",
    "scaling_factors": "estimated_scaling_factors",
    "sigma": "estimated_sigma",
}

MODEL_FEDBIOMED = "fedbiomed"
MODEL_DPMOST = "dpmost"
MODEL_LEASPY = "leaspy"
MODEL_TYPES = (MODEL_FEDBIOMED, MODEL_DPMOST, MODEL_LEASPY)

SUFFIX_ESTIMATED_PARAMS = "-estimated_params.json"

# top-level key in Leaspy results JSON files for personalized parameters
KEY_LEASPY_INDIVIDUAL_PARAMETERS = "individual_parameters"

DEFAULT_LEASPY_ALGORITHM_NAME = "scipy_minimize"
DEFAULT_LEASPY_SEED = 0
DEFAULT_LEASPY_N_JOBS = 1

DPATH_DPMOST = DPATH_PROJECT / "vendored" / "dpmost" / "models" / "dpmost"

# column names for fitted data tables
COL_MODEL = "model"
COL_SUBJECT = "subject"
COL_GROUP = "group"
COL_SET_NAME = "set_name"
COL_BIOMARKER = "biomarker"
COL_TIME = "t"
COL_TIME_SHIFTED = "t_shifted"
COL_Y_OBS = "y_obs"
COL_Y_PRED = "y_pred"
COL_Y = "y"
COL_TIME_SHIFT = "time_shift"
COL_ACCELERATION_FACTOR = "acceleration_factor"

PredictFn = Callable[[ModelParams, np.ndarray, np.ndarray], np.ndarray]


# ----- results/data loading -----


def get_dpath_data(json_results: dict) -> Path:
    return Path(json_results["settings"]["dpath_data"])


def load_json_data(json_results: dict) -> dict:
    fpath_json_data = Path(json_results["settings"]["fpath_config"])
    return load_json(fpath_json_data)


def get_cols(json_data: dict) -> dict[str, Any]:
    return json_data["cols"]


def get_max_time(json_data: dict) -> float | None:
    """Factor converting scaled timepoints back to months, if available."""
    return json_data.get("settings", {}).get("config", {}).get("max_time")


def get_model_type(fpath_json_results: Path) -> str:
    model_type = fpath_json_results.name.split("-")[0]
    if model_type not in MODEL_TYPES:
        raise ValueError(
            f"Unknown model type {model_type!r} for {fpath_json_results}."
            f" Expected one of {MODEL_TYPES}"
        )
    return model_type


def get_run_tag(fpath_json_results: Path) -> str:
    return fpath_json_results.name.removesuffix(SUFFIX_ESTIMATED_PARAMS)


def iter_result_files(dpath_results_tag: Path) -> Iterator[tuple[Path, dict]]:
    """Yield ``(fpath, json_results)`` for each results file, sorted by name."""
    for fpath_json_results in sorted(
        dpath_results_tag.glob(f"*{SUFFIX_ESTIMATED_PARAMS}")
    ):
        yield fpath_json_results, load_json(fpath_json_results)


def load_df_data(
    dpath_data: Path, tag: str, cols: dict, test: bool = False
) -> pd.DataFrame:
    if test:
        suffix = "-test"
    else:
        suffix = "-merged"
    fpath_data = dpath_data / f"{tag}{suffix}.tsv"
    return pd.read_csv(fpath_data, sep="\t", dtype={cols["col_subject"]: str})


def load_df_data_leaspy(
    dpath_data: Path, tag: str, cols: dict, test: bool = False
) -> pd.DataFrame:
    df = load_df_data(dpath_data, tag, cols, test=test)
    return to_df_leaspy(df, cols)


def to_df_leaspy(df: pd.DataFrame, cols: dict) -> pd.DataFrame:
    return format_df_for_leaspy(
        df=df.copy(),
        col_subject=cols["col_subject"],
        col_timepoint=cols["col_timepoint"],
        cols_biomarker=cols["cols_biomarker"],
    )


# ----- parameter alignment -----


def build_model_params(
    feature_arrays: dict[str, np.ndarray],
    per_node: dict[str, dict[str, list[float]]],
    subjects_by_node: dict[str, list[str]],
) -> ModelParams:
    """Assemble a ModelParams from per-feature arrays and per-node lists.

    ``per_node`` maps node ids (without the ``node_`` prefix) to dicts with
    ``time_shifts`` and ``acceleration_factors`` lists, each aligned to
    ``subjects_by_node[node_id]``. Nodes must be given in merged-data subject
    order (site order); their lists are validated and concatenated in that
    order.
    """
    time_shifts = []
    acceleration_factors = []
    for node_id, node_params in per_node.items():
        subjects = subjects_by_node[node_id]
        for param_name in PER_SUBJECT_FIELDS:
            if len(node_params[param_name]) != len(subjects):
                raise ValueError(
                    f"{param_name} for node {node_id} have length "
                    f"{len(node_params[param_name])} but {len(subjects)} subjects"
                )
        time_shifts.append(np.asarray(node_params["time_shifts"], dtype=float))
        acceleration_factors.append(
            np.asarray(node_params["acceleration_factors"], dtype=float)
        )

    return ModelParams(
        **feature_arrays,
        time_shifts=np.concatenate(time_shifts),
        acceleration_factors=np.concatenate(acceleration_factors),
    )


def true_params_by_subject(
    subjects_by_node: dict[str, list[str]], params: dict[str, Any]
) -> ModelParams:
    """Align true simulation parameters to the canonical subject order.

    True per-subject parameters are stored as lists per site in file order.
    Site ``i`` (0-indexed) corresponds to node ``str(i + 1)``, whose subjects
    are listed in ``subjects_by_node`` in the same order as in the site TSV.
    """
    feature_arrays = {
        field: np.asarray(params[source_key], dtype=float)
        for field, source_key in TRUE_FEATURE_KEYS.items()
    }

    per_node = {
        str(i_site + 1): {
            "time_shifts": time_shifts_site,
            "acceleration_factors": acc_site,
        }
        for i_site, (time_shifts_site, acc_site) in enumerate(
            zip(params["time_shifts"], params["acceleration_factors"], strict=True)
        )
    }

    return build_model_params(feature_arrays, per_node, subjects_by_node)


def estimated_params_by_setup(
    json_results: dict, subjects_by_node: dict[str, list[str]]
) -> dict[str, ModelParams]:
    """Align estimated parameters to the canonical subject order, per setup.

    Estimated per-subject parameters are keyed by node (``node_1``, ...,
    ``node_centralized``); subjects are listed in ``subjects_by_node`` in the
    same order as in the estimated arrays. Node keys are processed in JSON
    order, which matches the merged data's subject order.
    """
    params_by_setup: dict[str, ModelParams] = {}
    for setup, result in json_results["results"].items():
        feature_arrays = {
            field: np.asarray(result[source_key], dtype=float)
            for field, source_key in ESTIMATED_FEATURE_KEYS.items()
        }

        estimated_time_shifts = result["estimated_time_shifts"]
        estimated_acceleration_factors = result["estimated_acceleration_factors"]
        per_node = {
            node_id.removeprefix(NODE_PREFIX): {
                "time_shifts": shifts,
                "acceleration_factors": estimated_acceleration_factors[node_id],
            }
            for node_id, shifts in estimated_time_shifts.items()
        }

        params_by_setup[setup] = build_model_params(
            feature_arrays, per_node, subjects_by_node
        )
    return params_by_setup


def subjects_by_setup(
    json_results: dict, subjects_by_node: dict[str, list[str]]
) -> dict[str, list[str]]:
    """Subject IDs in the same order as ``estimated_params_by_setup`` arrays."""
    return {
        setup: [
            str(subject)
            for node_id in result["estimated_time_shifts"]
            for subject in subjects_by_node[node_id.removeprefix(NODE_PREFIX)]
        ]
        for setup, result in json_results["results"].items()
    }


def validate_subject_ids(subject_ids: np.ndarray, n_subjects: int, set_name: str):
    """Ensure subject indices point into the merged data's subject range."""
    if subject_ids.size and (subject_ids.min() < 0 or subject_ids.max() >= n_subjects):
        raise ValueError(
            f"{set_name}: subject indices fall outside the merged data's subject "
            f"range [0, {n_subjects})"
        )


def get_subject_ids_from_frame(
    df_frame: pd.DataFrame, cols: dict, df_data_train: pd.DataFrame
) -> np.ndarray:
    """Map subjects to their global index from the train data."""
    col_subject = cols["col_subject"]
    col_subject_index = cols["col_subject_index"]
    subject_index_map = dict(
        zip(df_data_train[col_subject], df_data_train[col_subject_index])
    )

    subject_ids = df_frame[col_subject].map(subject_index_map)
    if subject_ids.isna().any():
        missing_subjects = df_frame.loc[subject_ids.isna(), col_subject].tolist()
        raise ValueError(
            "test set contains subjects not present in the merged train data: "
            f"{missing_subjects[:5]}"
        )
    return subject_ids.to_numpy(dtype=int)


# ----- prediction -----


def predict_fl_prog(
    params: ModelParams, t: np.ndarray, subject_ids: np.ndarray
) -> np.ndarray:
    """Model forward pass.

    Returns predictions of shape (n_entries, n_features)
    """
    model = LogisticRegressionModelWithShift(
        n_participants=len(params.time_shifts),
        n_features=len(params.k_values),
    )

    t = torch.tensor(t, dtype=torch.float)
    subject_ids = torch.tensor(subject_ids, dtype=torch.long)

    return model.forward(t, subject_ids, params=params).detach().numpy()


def predict_dpmost(
    params: ModelParams, t: np.ndarray, subject_ids: np.ndarray
) -> np.ndarray:
    """Predict individual datapoints for DPMoSt.

    Returns predictions of shape (n_entries, n_features)
    """
    if str(DPATH_DPMOST) not in sys.path:
        sys.path.append(str(DPATH_DPMOST))

    from utility import sigmoid_eval

    t = torch.tensor(t + params.time_shifts[subject_ids], dtype=torch.float).reshape(
        -1, 1
    )
    theta = torch.tensor(
        np.hstack(
            [
                params.x0_values[:, None],
                params.k_values[:, None],
                params.scaling_factors[:, None],
            ],
        ),
        dtype=torch.float,
    )

    return sigmoid_eval(t, theta).detach().numpy()


def get_predict_fn(model_type: str) -> PredictFn:
    match model_type:
        case "dpmost":
            return predict_dpmost
        case "fedbiomed":
            return predict_fl_prog
        case _:
            raise ValueError(f"No predict function for model type {model_type!r}")


def get_shifted_time(
    params: ModelParams, t: np.ndarray, subject_ids: np.ndarray, model_type: str
) -> np.ndarray:
    """Reparametrized (disease) time of each entry, matching the predict functions."""
    shifts = params.time_shifts[subject_ids]
    if model_type == MODEL_DPMOST:
        # DPMoSt has no acceleration factors
        return t + shifts
    return t * params.acceleration_factors[subject_ids] + shifts


def predict_group_trajectory(
    params: ModelParams, t_shifted: np.ndarray, predict_fn: PredictFn
) -> np.ndarray:
    """Evaluate the group trajectory at reparametrized timepoints.

    Uses a single virtual subject with no time shift and no acceleration so that
    the model's own predict function (and sigmoid parametrization) is reused.
    """
    params_group = ModelParams(
        k_values=params.k_values,
        x0_values=params.x0_values,
        scaling_factors=params.scaling_factors,
        sigma=params.sigma,
        time_shifts=np.zeros(1),
        acceleration_factors=np.ones(1),
    )
    return predict_fn(params_group, t_shifted, np.zeros(len(t_shifted), dtype=int))


# ----- Leaspy -----


def get_leaspy_personalization_key(
    algorithm_name: str = DEFAULT_LEASPY_ALGORITHM_NAME,
    seed: int = DEFAULT_LEASPY_SEED,
    n_jobs: int = DEFAULT_LEASPY_N_JOBS,
) -> str:
    return f"{algorithm_name}-{seed}-{n_jobs}"


def load_leaspy_model(model_params: dict):
    from leaspy.models import BaseModel

    with tempfile.NamedTemporaryFile(mode="rt", suffix=".json", delete=True) as tmp:
        save_json(Path(tmp.name), model_params)
        tmp.flush()
        return BaseModel.load(tmp.name)


def personalize_leaspy(
    model,
    df_leaspy: pd.DataFrame,
    algorithm_name: str = DEFAULT_LEASPY_ALGORITHM_NAME,
    seed: int = DEFAULT_LEASPY_SEED,
    n_jobs: int = DEFAULT_LEASPY_N_JOBS,
) -> pd.DataFrame:
    """Estimate individual parameters. Returns a dataframe indexed by subject."""
    from leaspy.algo import AlgorithmSettings, algorithm_factory
    from leaspy.io.data import Data, Dataset

    personalize_settings = AlgorithmSettings(algorithm_name, seed=seed, n_jobs=n_jobs)
    algorithm = algorithm_factory(personalize_settings)
    individual_parameters = algorithm.run(
        model, Dataset(Data.from_dataframe(df_leaspy, data_type="visit"))
    )
    return individual_parameters.to_dataframe()


def get_leaspy_individual_parameters(
    fpath_json_results: Path,
    json_results: dict,
    setup: str,
    model,
    df_leaspy: pd.DataFrame,
    algorithm_name: str = DEFAULT_LEASPY_ALGORITHM_NAME,
    seed: int = DEFAULT_LEASPY_SEED,
    n_jobs: int = DEFAULT_LEASPY_N_JOBS,
) -> pd.DataFrame:
    """Load personalized Leaspy parameters from the results file, or compute them.

    Computed parameters are written back to ``fpath_json_results`` under
    ``individual_parameters -> <algorithm_name>-<seed>-<n_jobs> -> <setup>``.
    """
    key = get_leaspy_personalization_key(algorithm_name, seed, n_jobs)
    stored = (
        json_results.get(KEY_LEASPY_INDIVIDUAL_PARAMETERS, {}).get(key, {}).get(setup)
    )
    if stored is not None:
        print(f"\tUsing stored Leaspy individual parameters ({key}, {setup})")
        return pd.DataFrame(stored).set_index(LEASPY_COL_SUBJECT)

    print(f"\tPersonalizing Leaspy model ({key}, {setup})")
    df_ip = personalize_leaspy(
        model, df_leaspy, algorithm_name=algorithm_name, seed=seed, n_jobs=n_jobs
    )
    json_results.setdefault(KEY_LEASPY_INDIVIDUAL_PARAMETERS, {}).setdefault(key, {})[
        setup
    ] = df_ip.reset_index().to_dict(orient="list")
    save_json(fpath_json_results, json_results)
    print(f"\tSaved Leaspy individual parameters to {fpath_json_results}")
    return df_ip


def predict_leaspy(model, df_ip: pd.DataFrame, df_leaspy: pd.DataFrame) -> np.ndarray:
    """Predict at the visits of ``df_leaspy``, in row order.

    Returns predictions of shape (n_entries, n_features)
    """
    from leaspy.io.outputs import IndividualParameters

    cols_biomarker = list(df_leaspy.columns)
    individual_parameters = IndividualParameters.from_dataframe(df_ip)
    times_by_subject = df_leaspy.index.get_level_values(LEASPY_COL_TIMEPOINT).groupby(
        df_leaspy.index.get_level_values(LEASPY_COL_SUBJECT)
    )
    predictions = model.estimate(times_by_subject, individual_parameters)
    df_y_pred = pd.DataFrame(index=df_leaspy.index, columns=cols_biomarker)
    for subject, times_sub in times_by_subject.items():
        df_y_pred.loc[(subject, list(times_sub)), cols_biomarker] = predictions[subject]
    return df_y_pred.to_numpy(dtype=float)


def predict_group_trajectory_leaspy(model, t_shifted: np.ndarray) -> np.ndarray:
    """Evaluate the Leaspy trajectory of a subject with ``tau = xi = 0``.

    Leaspy reparametrizes time as ``exp(xi) * (t - tau)``, so this subject's
    trajectory is the model evaluated directly at reparametrized time.
    """
    from leaspy.io.outputs import IndividualParameters

    individual_parameters = IndividualParameters()
    individual_parameters.add_individual_parameters("group", {"tau": 0.0, "xi": 0.0})
    return np.asarray(
        model.estimate({"group": list(t_shifted)}, individual_parameters)["group"],
        dtype=float,
    )


def leaspy_to_shift_acc(df_ip: pd.DataFrame) -> pd.DataFrame:
    """Convert Leaspy ``tau``/``xi`` to fl_prog-style time shift and acceleration.

    ``exp(xi) * (t - tau) = acc * t + shift`` with ``acc = exp(xi)`` and
    ``shift = -exp(xi) * tau``.
    """
    acc = np.exp(df_ip["xi"].astype(float))
    return pd.DataFrame(
        {
            COL_TIME_SHIFT: -acc * df_ip["tau"].astype(float),
            COL_ACCELERATION_FACTOR: acc,
        },
        index=df_ip.index.astype(str),
    )


# ----- fitted data for plotting -----


@dataclass
class FittedModel:
    """Fitted parameters applied to subject data, in months if possible.

    - ``df_subject``: one row per subject (subject, group, time_shift,
      acceleration_factor)
    - ``df_long``: one row per visit and biomarker (subject, group, set_name,
      biomarker, t, t_shifted, y_obs, y_pred)
    - ``df_trajectory``: group trajectory (biomarker, t_shifted, y)
    """

    label: str
    model_type: str
    setup: str
    run_tag: str
    df_subject: pd.DataFrame
    df_long: pd.DataFrame
    df_trajectory: pd.DataFrame


def get_model_label(model_type: str, setup: str, n_setups: int) -> str:
    if n_setups == 1:
        return model_type
    return f"{model_type} ({setup})"


def _load_df_data_all(
    dpath_data: Path, tag: str, cols: dict
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(df_train, df_all)`` with ``df_all`` = train + test (if any)."""
    df_train = load_df_data(dpath_data, tag, cols, test=False)
    dfs = [df_train.assign(**{COL_SET_NAME: "train"})]
    try:
        df_test = load_df_data(dpath_data, tag, cols, test=True)
    except FileNotFoundError:
        pass
    else:
        df_test = df_test.copy()
        df_test[cols["col_subject_index"]] = get_subject_ids_from_frame(
            df_test, cols, df_train
        )
        dfs.append(df_test.assign(**{COL_SET_NAME: "test"}))
    return df_train, pd.concat(dfs, ignore_index=True)


def _get_trajectory_grid(t_shifted: np.ndarray, n_points: int = 200) -> np.ndarray:
    t_min, t_max = np.nanmin(t_shifted), np.nanmax(t_shifted)
    padding = 0.05 * (t_max - t_min)
    return np.linspace(t_min - padding, t_max + padding, n_points)


def _build_fitted_model(
    label: str,
    model_type: str,
    setup: str,
    run_tag: str,
    df_all: pd.DataFrame,
    cols: dict,
    df_shift_acc: pd.DataFrame,
    t_shifted: np.ndarray,
    y_pred: np.ndarray,
    t_grid: np.ndarray,
    y_grid: np.ndarray,
    time_factor: float,
) -> FittedModel:
    col_subject = cols["col_subject"]
    col_group = cols.get("col_group")
    cols_biomarker = cols["cols_biomarker"]

    if col_group is not None and col_group in df_all.columns:
        group_by_subject = df_all.groupby(col_subject)[col_group].first()
    else:
        group_by_subject = pd.Series("unknown", index=df_all[col_subject].unique())
    group_by_subject.index = group_by_subject.index.astype(str)

    df_subject = df_shift_acc.copy()
    df_subject[COL_TIME_SHIFT] *= time_factor
    df_subject.index.name = COL_SUBJECT
    df_subject = df_subject.reset_index()
    df_subject.insert(1, COL_GROUP, df_subject[COL_SUBJECT].map(group_by_subject))
    df_subject.insert(0, COL_MODEL, label)

    df_wide = pd.DataFrame(
        {
            COL_SUBJECT: df_all[col_subject].astype(str).to_numpy(),
            COL_SET_NAME: df_all[COL_SET_NAME].to_numpy(),
            COL_TIME: df_all[cols["col_timepoint"]].to_numpy(dtype=float) * time_factor,
            COL_TIME_SHIFTED: t_shifted * time_factor,
        }
    )
    df_obs = pd.concat(
        [df_wide, df_all[cols_biomarker].reset_index(drop=True)], axis="columns"
    ).melt(id_vars=list(df_wide.columns), var_name=COL_BIOMARKER, value_name=COL_Y_OBS)
    df_pred = pd.concat(
        [df_wide, pd.DataFrame(y_pred, columns=cols_biomarker)], axis="columns"
    ).melt(id_vars=list(df_wide.columns), var_name=COL_BIOMARKER, value_name=COL_Y_PRED)
    df_long = df_obs.assign(**{COL_Y_PRED: df_pred[COL_Y_PRED].to_numpy()})
    df_long.insert(1, COL_GROUP, df_long[COL_SUBJECT].map(group_by_subject))
    df_long.insert(0, COL_MODEL, label)

    df_trajectory = (
        pd.DataFrame(y_grid, columns=cols_biomarker)
        .assign(**{COL_TIME_SHIFTED: t_grid * time_factor})
        .melt(id_vars=[COL_TIME_SHIFTED], var_name=COL_BIOMARKER, value_name=COL_Y)
    )
    df_trajectory.insert(0, COL_MODEL, label)

    return FittedModel(
        label=label,
        model_type=model_type,
        setup=setup,
        run_tag=run_tag,
        df_subject=df_subject,
        df_long=df_long,
        df_trajectory=df_trajectory,
    )


def get_fitted_models(
    fpath_json_results: Path,
    json_results: dict,
    tag: str,
    leaspy_algorithm_name: str = DEFAULT_LEASPY_ALGORITHM_NAME,
    leaspy_seed: int = DEFAULT_LEASPY_SEED,
    leaspy_n_jobs: int = DEFAULT_LEASPY_N_JOBS,
) -> list[FittedModel]:
    """Apply the fitted parameters of each setup in a results file to the data."""
    model_type = get_model_type(fpath_json_results)
    run_tag = get_run_tag(fpath_json_results)

    json_data = load_json_data(json_results)
    cols = get_cols(json_data)
    max_time = get_max_time(json_data)
    time_factor = 1.0 if max_time is None else float(max_time)

    df_train, df_all = _load_df_data_all(get_dpath_data(json_results), tag, cols)
    t_all = df_all[cols["col_timepoint"]].to_numpy(dtype=float)
    n_setups = len(json_results["results"])

    fitted_models = []
    if model_type == MODEL_LEASPY:
        df_leaspy_train = to_df_leaspy(df_train, cols)
        df_leaspy_all = to_df_leaspy(df_all, cols)
        for setup, model_params in json_results["results"].items():
            model = load_leaspy_model(model_params)
            df_ip = get_leaspy_individual_parameters(
                fpath_json_results,
                json_results,
                setup,
                model,
                df_leaspy_train,
                algorithm_name=leaspy_algorithm_name,
                seed=leaspy_seed,
                n_jobs=leaspy_n_jobs,
            )
            df_shift_acc = leaspy_to_shift_acc(df_ip)
            subjects = df_all[cols["col_subject"]].astype(str)
            t_shifted = (
                subjects.map(df_shift_acc[COL_ACCELERATION_FACTOR]).to_numpy() * t_all
                + subjects.map(df_shift_acc[COL_TIME_SHIFT]).to_numpy()
            )
            t_grid = _get_trajectory_grid(t_shifted)
            fitted_models.append(
                _build_fitted_model(
                    label=get_model_label(model_type, setup, n_setups),
                    model_type=model_type,
                    setup=setup,
                    run_tag=run_tag,
                    df_all=df_all,
                    cols=cols,
                    df_shift_acc=df_shift_acc,
                    t_shifted=t_shifted,
                    y_pred=predict_leaspy(model, df_ip, df_leaspy_all),
                    t_grid=t_grid,
                    y_grid=predict_group_trajectory_leaspy(model, t_grid),
                    time_factor=time_factor,
                )
            )
        return fitted_models

    predict_fn = get_predict_fn(model_type)
    subjects_by_node = json_data["subjects_by_node"]
    params_by_setup = estimated_params_by_setup(json_results, subjects_by_node)
    subject_order_by_setup = subjects_by_setup(json_results, subjects_by_node)

    # subject order in parameter arrays must match the data's subject index
    index_to_subject = (
        df_train.drop_duplicates(cols["col_subject_index"])
        .set_index(cols["col_subject_index"])[cols["col_subject"]]
        .astype(str)
        .sort_index()
    )
    subject_ids_all = df_all[cols["col_subject_index"]].to_numpy(dtype=int)

    for setup, params in params_by_setup.items():
        subject_order = subject_order_by_setup[setup]
        if (
            list(index_to_subject.index) != list(range(len(subject_order)))
            or list(index_to_subject) != subject_order
        ):
            raise ValueError(
                f"{setup}: subject order in estimated parameters does not match "
                f"{cols['col_subject_index']} in the data"
            )
        validate_subject_ids(subject_ids_all, len(subject_order), setup)

        df_shift_acc = pd.DataFrame(
            {
                COL_TIME_SHIFT: params.time_shifts,
                COL_ACCELERATION_FACTOR: (
                    np.ones_like(params.acceleration_factors)
                    if model_type == MODEL_DPMOST
                    else params.acceleration_factors
                ),
            },
            index=pd.Index(subject_order),
        )
        t_shifted = get_shifted_time(params, t_all, subject_ids_all, model_type)
        t_grid = _get_trajectory_grid(t_shifted)
        fitted_models.append(
            _build_fitted_model(
                label=get_model_label(model_type, setup, n_setups),
                model_type=model_type,
                setup=setup,
                run_tag=run_tag,
                df_all=df_all,
                cols=cols,
                df_shift_acc=df_shift_acc,
                t_shifted=t_shifted,
                y_pred=predict_fn(params, t_all, subject_ids_all),
                t_grid=t_grid,
                y_grid=predict_group_trajectory(params, t_grid, predict_fn),
                time_factor=time_factor,
            )
        )
    return fitted_models
