"""Load model fitting results and apply fitted parameters to subject data."""

import pickle
import tempfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from fl_prog.metrics import PER_SUBJECT_FIELDS
from fl_prog.model import LogisticRegressionModelWithShift, ModelParams
from fl_prog.personalization import (
    DEFAULT_DPMOST_N_OUTER_ITER,
    DEFAULT_PERSONALIZE_ALGORITHM,
    DEFAULT_PERSONALIZE_SEED,
    get_n_timepoints_suffix,
)
from fl_prog.utils.constants import (
    LEASPY_COL_SUBJECT,
    LEASPY_COL_TIMEPOINT,
    NODE_ID_CENTRALIZED,
    NODE_ID_TEST,
    NODE_PREFIX,
)
from fl_prog.utils.io import (
    add_dpmost_to_path,
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

# top-level key in Fed-BioMed/DPMoSt results JSON files for personalized parameters
# of held-out test subjects
KEY_INDIVIDUAL_PARAMETERS_TEST = "individual_parameters_test"

DEFAULT_LEASPY_ALGORITHM_NAME = "scipy_minimize"
DEFAULT_LEASPY_SEED = 0
DEFAULT_LEASPY_N_JOBS = 1


# key in DPMoSt results JSON settings for the target ranges of the normalisation
KEY_DPMOST_NORMALISATION_RANGES = "normalisation_ranges"

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


def get_time_factor(json_data: dict) -> float:
    """Factor converting model time to months (1 if not available)."""
    max_time = get_max_time(json_data)
    return 1.0 if max_time is None else float(max_time)


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


def get_short_labels(run_tags: list[str]) -> dict[str, str]:
    """Only keep the run tag fields that differ between runs."""
    fields = [run_tag.split("-") for run_tag in run_tags]
    if len({len(f) for f in fields}) != 1:
        return {run_tag: run_tag for run_tag in run_tags}
    i_differ = [i for i, values in enumerate(zip(*fields)) if len(set(values)) > 1]
    return {
        run_tag: "-".join(f[i] for i in i_differ) or run_tag
        for run_tag, f in zip(run_tags, fields)
    }


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


def has_heldout_subjects(subjects_by_node: dict[str, list[str]]) -> bool:
    """Whether the test set consists of subjects held out from training."""
    return NODE_ID_TEST in subjects_by_node


def _get_node_ids(subjects_by_node: dict[str, list[str]], test: bool) -> list[str]:
    """Nodes with the training subjects (sites) or the held-out test subjects."""
    if test:
        return [NODE_ID_TEST] if NODE_ID_TEST in subjects_by_node else []
    return [
        node_id
        for node_id in subjects_by_node
        if node_id not in (NODE_ID_CENTRALIZED, NODE_ID_TEST)
    ]


def get_subjects(
    subjects_by_node: dict[str, list[str]], test: bool = False
) -> list[str]:
    """Training (or held-out test) subject IDs, in the order of their parameters.

    The centralized node is excluded since it duplicates the site subjects.
    """
    return [
        str(subject)
        for node_id in _get_node_ids(subjects_by_node, test)
        for subject in subjects_by_node[node_id]
    ]


def true_params_by_subject(
    subjects_by_node: dict[str, list[str]],
    params: dict[str, Any],
    test: bool = False,
) -> ModelParams:
    """Align true simulation parameters to the canonical subject order.

    True per-subject parameters are stored as lists per site in file order,
    aligned with the entries of ``subjects_by_node`` (in order, excluding the
    centralized node). Each site's subjects are listed in ``subjects_by_node`` in
    the same order as in the site TSV. Held-out test subjects (if any) are
    excluded from the returned parameters, unless ``test`` is True, in which case
    only the held-out test subjects are returned.
    """
    node_ids = [
        node_id for node_id in subjects_by_node if node_id != NODE_ID_CENTRALIZED
    ]
    node_ids_kept = _get_node_ids(subjects_by_node, test)
    feature_arrays = {
        field: np.asarray(params[source_key], dtype=float)
        for field, source_key in TRUE_FEATURE_KEYS.items()
    }

    per_node = {
        node_id: {
            "time_shifts": time_shifts_site,
            "acceleration_factors": acc_site,
        }
        for node_id, time_shifts_site, acc_site in zip(
            node_ids,
            params["time_shifts"],
            params["acceleration_factors"],
            strict=True,
        )
        if node_id in node_ids_kept
    }
    if test and not per_node:
        raise ValueError("No held-out test subjects in subjects_by_node")

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


@dataclass
class DPMoStNormalisation:
    """Min/max values used by DPMoSt to normalise the data and time."""

    y_min: np.ndarray
    y_max: np.ndarray
    t_min: float | None
    t_max: float | None
    range_data: tuple[float, float]
    range_time: tuple[float, float]

    def normalise_time(self, t: np.ndarray) -> np.ndarray:
        # DPMoSt does not normalise time if it has zero variance
        if self.t_min is None or self.t_max is None:
            return t
        new_min, new_max = self.range_time
        return (t - self.t_min) / (self.t_max - self.t_min) * (
            new_max - new_min
        ) + new_min

    def unnormalise_time_shifts(self, time_shifts: np.ndarray) -> np.ndarray:
        """Convert time shifts from normalised to original time units."""
        if self.t_min is None or self.t_max is None:
            return time_shifts
        new_min, new_max = self.range_time
        return time_shifts * (self.t_max - self.t_min) / (new_max - new_min)

    def unnormalise_y(self, y: np.ndarray) -> np.ndarray:
        new_min, new_max = self.range_data
        return (y - new_min) / (new_max - new_min) * (
            self.y_max - self.y_min
        ) + self.y_min


def load_dpmost_normalisation(
    fpath_json_results: Path, json_results: dict, cols_biomarker: list[str]
) -> DPMoStNormalisation | None:
    """Get the normalisation applied by DPMoSt.

    Min/max values come from the fitted model pickle and target ranges from the
    results JSON settings. Returns None if the model was fit without
    normalisation.
    """
    fpath_model = fpath_json_results.with_name(
        f"{get_run_tag(fpath_json_results)}-model.pkl"
    )
    if not fpath_model.exists():
        if json_results["settings"]["model_args"].get("do_normalisation", False):
            raise FileNotFoundError(
                f"DPMoSt model was fit with normalisation but {fpath_model} "
                "does not exist"
            )
        return None

    add_dpmost_to_path()
    with fpath_model.open("rb") as file_model:
        model = pickle.load(file_model)

    if not model.do_normalisation:
        return None

    def _to_float(value) -> float | None:
        return None if value is None else float(value)

    try:
        ranges = json_results["settings"][KEY_DPMOST_NORMALISATION_RANGES]
    except KeyError:
        raise KeyError(
            f"DPMoSt model was fit with normalisation but the results file for "
            f"{fpath_model} does not record {KEY_DPMOST_NORMALISATION_RANGES!r}"
        )
    data_min = model.data_normalisation_min
    data_max = model.data_normalisation_max
    return DPMoStNormalisation(
        y_min=np.array([_to_float(data_min[col]) for col in cols_biomarker]),
        y_max=np.array([_to_float(data_max[col]) for col in cols_biomarker]),
        t_min=_to_float(data_min["time"]),
        t_max=_to_float(data_max["time"]),
        range_data=tuple(ranges["data"]),
        range_time=tuple(ranges["time"]),
    )


def predict_dpmost(
    params: ModelParams,
    t: np.ndarray,
    subject_ids: np.ndarray,
    normalisation: DPMoStNormalisation | None = None,
) -> np.ndarray:
    """Predict individual datapoints for DPMoSt.

    If ``normalisation`` is given, ``t`` is normalised before prediction and
    the predictions are converted back to the original data units.

    Returns predictions of shape (n_entries, n_features)
    """
    add_dpmost_to_path()

    from utility import sigmoid_eval

    if normalisation is not None:
        t = normalisation.normalise_time(t)

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

    y_pred = sigmoid_eval(t, theta).detach().numpy()
    if normalisation is not None:
        y_pred = normalisation.unnormalise_y(y_pred)
    return y_pred


def get_predict_fn(
    model_type: str, normalisation: DPMoStNormalisation | None = None
) -> PredictFn:
    match model_type:
        case "dpmost":
            if normalisation is not None:
                return partial(predict_dpmost, normalisation=normalisation)
            return predict_dpmost
        case "fedbiomed":
            return predict_fl_prog
        case _:
            raise ValueError(f"No predict function for model type {model_type!r}")


def get_time_shifts(
    params: ModelParams, normalisation: DPMoStNormalisation | None = None
) -> np.ndarray:
    """Per-subject time shifts in the same time units as the data."""
    if normalisation is None:
        return params.time_shifts
    return normalisation.unnormalise_time_shifts(params.time_shifts)


def get_shifted_time(
    params: ModelParams,
    t: np.ndarray,
    subject_ids: np.ndarray,
    model_type: str,
    normalisation: DPMoStNormalisation | None = None,
) -> np.ndarray:
    """Reparametrized (disease) time of each entry, matching the predict functions.

    Returned in the same time units as ``t``.
    """
    shifts = get_time_shifts(params, normalisation)[subject_ids]
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


# ----- personalized parameters of held-out subjects (Fed-BioMed/DPMoSt) -----


def get_heldout_personalization_key(
    model_type: str,
    fedbiomed_personalize_algorithm: str = DEFAULT_PERSONALIZE_ALGORITHM,
    personalize_seed: int = DEFAULT_PERSONALIZE_SEED,
    dpmost_personalize_n_outer_iter: int = DEFAULT_DPMOST_N_OUTER_ITER,
    personalize_n_timepoints: int | None = None,
) -> str:
    """Key identifying how held-out subjects were personalized (also used in
    metrics file names)."""
    match model_type:
        case "fedbiomed":
            key = f"{fedbiomed_personalize_algorithm}_{personalize_seed}"
        case "dpmost":
            key = f"{dpmost_personalize_n_outer_iter}"
        case _:
            raise ValueError(f"No held-out personalization for {model_type!r}")
    return f"{key}{get_n_timepoints_suffix(personalize_n_timepoints)}"


def get_leaspy_heldout_setup(setup: str, n_timepoints: int | None = None) -> str:
    """Setup name under which held-out subjects' Leaspy parameters are stored."""
    return f"{setup}-test{get_n_timepoints_suffix(n_timepoints)}"


class PersonalizedParametersNotFoundError(LookupError):
    """Personalized parameters of held-out subjects are missing from a results
    file."""


def save_personalized_params(
    fpath_json_results: Path,
    json_results: dict,
    personalization_key: str,
    setup: str,
    subjects: list[str],
    params: ModelParams,
):
    """Store held-out subjects' per-subject parameters in a results file.

    They are stored (in model units, like the training subjects' estimates) under
    ``individual_parameters_test -> <personalization_key> -> <setup>``.
    """
    json_results.setdefault(KEY_INDIVIDUAL_PARAMETERS_TEST, {}).setdefault(
        personalization_key, {}
    )[setup] = {
        COL_SUBJECT: list(subjects),
        "time_shifts": params.time_shifts,
        "acceleration_factors": params.acceleration_factors,
    }
    save_json(fpath_json_results, json_results)
    print(f"\tSaved personalized parameters of held-out subjects ({setup})")


def load_personalized_params(
    json_results: dict, personalization_key: str, setup: str
) -> pd.DataFrame:
    """Held-out subjects' per-subject parameters, indexed by subject."""
    stored = (
        json_results.get(KEY_INDIVIDUAL_PARAMETERS_TEST, {})
        .get(personalization_key, {})
        .get(setup)
    )
    if stored is None:
        raise PersonalizedParametersNotFoundError(
            f"No personalized parameters of held-out subjects for "
            f"{personalization_key!r} and setup {setup!r}"
        )
    return pd.DataFrame(stored).set_index(COL_SUBJECT)


def get_df_true_subject(json_data: dict) -> pd.DataFrame | None:
    """True per-subject parameters, in months if possible.

    One row per subject (subject, set_name, time_shift, acceleration_factor),
    with held-out test subjects (if any) as ``test``. Returns None if the data
    have no true parameters.
    """
    params = json_data.get("params")
    if params is None or not set(PER_SUBJECT_FIELDS) <= set(params):
        return None

    subjects_by_node = json_data["subjects_by_node"]
    time_factor = get_time_factor(json_data)
    dfs = []
    for set_name, test in [("train", False), ("test", True)]:
        if test and not has_heldout_subjects(subjects_by_node):
            continue
        true_params = true_params_by_subject(subjects_by_node, params, test=test)
        dfs.append(
            pd.DataFrame(
                {
                    COL_SUBJECT: get_subjects(subjects_by_node, test=test),
                    COL_SET_NAME: set_name,
                    COL_TIME_SHIFT: true_params.time_shifts * time_factor,
                    COL_ACCELERATION_FACTOR: true_params.acceleration_factors,
                }
            )
        )
    return pd.concat(dfs, ignore_index=True)


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


class LeaspyIndividualParametersNotFoundError(LookupError):
    """Personalized Leaspy parameters are missing from a results file."""


def load_leaspy_individual_parameters(
    json_results: dict,
    setup: str,
    algorithm_name: str = DEFAULT_LEASPY_ALGORITHM_NAME,
    seed: int = DEFAULT_LEASPY_SEED,
    n_jobs: int = DEFAULT_LEASPY_N_JOBS,
) -> pd.DataFrame:
    """Load personalized Leaspy parameters stored in a results file.

    They are stored under
    ``individual_parameters -> <algorithm_name>-<seed>-<n_jobs> -> <setup>``.
    """
    key = get_leaspy_personalization_key(algorithm_name, seed, n_jobs)
    stored = (
        json_results.get(KEY_LEASPY_INDIVIDUAL_PARAMETERS, {}).get(key, {}).get(setup)
    )
    if stored is None:
        raise LeaspyIndividualParametersNotFoundError(
            f"No Leaspy individual parameters for personalization {key!r} and setup "
            f"{setup!r}"
        )
    return pd.DataFrame(stored).set_index(LEASPY_COL_SUBJECT)


def save_leaspy_individual_parameters(
    fpath_json_results: Path,
    json_results: dict,
    setup: str,
    df_individual_params: pd.DataFrame,
    algorithm_name: str = DEFAULT_LEASPY_ALGORITHM_NAME,
    seed: int = DEFAULT_LEASPY_SEED,
    n_jobs: int = DEFAULT_LEASPY_N_JOBS,
):
    key = get_leaspy_personalization_key(algorithm_name, seed, n_jobs)
    json_results.setdefault(KEY_LEASPY_INDIVIDUAL_PARAMETERS, {}).setdefault(key, {})[
        setup
    ] = df_individual_params.reset_index().to_dict(orient="list")
    save_json(fpath_json_results, json_results)
    print(f"\tSaved Leaspy individual parameters to {fpath_json_results}")


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

    - ``df_subject``: one row per subject (subject, group, set_name, time_shift,
      acceleration_factor), with held-out test subjects (if any) as ``test``
    - ``df_long``: one row per visit and biomarker (subject, group, set_name,
      biomarker, t, t_shifted, y_obs, y_pred)
    - ``df_trajectory``: group trajectory (biomarker, t_shifted, y) over the
      range of the model's subject data; use ``get_df_trajectory`` for other
      timepoints
    """

    label: str
    model_type: str
    setup: str
    run_tag: str
    df_subject: pd.DataFrame
    df_long: pd.DataFrame
    df_trajectory: pd.DataFrame
    cols_biomarker: list[str]
    # group trajectory at reparametrized timepoints (in unscaled model time)
    trajectory_fn: Callable[[np.ndarray], np.ndarray]
    time_factor: float

    def get_df_trajectory(self, t_shifted: np.ndarray) -> pd.DataFrame:
        """Group trajectory at reparametrized timepoints given in months."""
        t_shifted = np.asarray(t_shifted, dtype=float)
        df_trajectory = (
            pd.DataFrame(
                self.trajectory_fn(t_shifted / self.time_factor),
                columns=self.cols_biomarker,
            )
            .assign(**{COL_TIME_SHIFTED: t_shifted})
            .melt(id_vars=[COL_TIME_SHIFTED], var_name=COL_BIOMARKER, value_name=COL_Y)
        )
        df_trajectory.insert(0, COL_MODEL, self.label)
        return df_trajectory


def get_model_label(model_type: str, setup: str, n_setups: int) -> str:
    if n_setups == 1:
        return model_type
    return f"{model_type} ({setup})"


def _load_df_data_train_test(
    dpath_data: Path, tag: str, cols: dict
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Load train and test data.

    test is None if there is no test file.
    """
    df_train = load_df_data(dpath_data, tag, cols, test=False)
    try:
        df_test = load_df_data(dpath_data, tag, cols, test=True)
    except FileNotFoundError:
        df_test = None
    return df_train, df_test


def _concat_train_test(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame | None,
    cols: dict,
    heldout_subjects: list[str] | None = None,
) -> pd.DataFrame:
    """Stack train and test (if any) rows, with a set name column.

    Test subjects are either train subjects (e.g. timepoint split), indexed by
    their index in the (merged) train data, or held-out subjects (given in
    ``heldout_subjects``), indexed after the train subjects in that order.
    """
    dfs = [df_train.assign(**{COL_SET_NAME: "train"})]
    if df_test is not None:
        df_test = df_test.copy()
        if heldout_subjects is None:
            subject_ids = get_subject_ids_from_frame(df_test, cols, df_train)
        else:
            n_subjects_train = df_train[cols["col_subject_index"]].nunique()
            subject_ids = (
                df_test[cols["col_subject"]]
                .astype(str)
                .map({s: n_subjects_train + i for i, s in enumerate(heldout_subjects)})
            )
            if subject_ids.isna().any():
                raise ValueError("Test subjects are not all in heldout_subjects")
        df_test[cols["col_subject_index"]] = subject_ids
        dfs.append(df_test.assign(**{COL_SET_NAME: "test"}))
    return pd.concat(dfs, ignore_index=True)


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
    trajectory_fn: Callable[[np.ndarray], np.ndarray],
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

    # subjects with only test rows are held-out subjects
    is_train_by_subject = df_all.groupby(col_subject)[COL_SET_NAME].agg(
        lambda set_names: (set_names == "train").any()
    )
    is_train_by_subject.index = is_train_by_subject.index.astype(str)

    df_subject = df_shift_acc.copy()
    df_subject[COL_TIME_SHIFT] *= time_factor
    df_subject.index.name = COL_SUBJECT
    df_subject = df_subject.reset_index()
    df_subject.insert(1, COL_GROUP, df_subject[COL_SUBJECT].map(group_by_subject))
    df_subject.insert(
        2,
        COL_SET_NAME,
        np.where(df_subject[COL_SUBJECT].map(is_train_by_subject), "train", "test"),
    )
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

    fitted_model = FittedModel(
        label=label,
        model_type=model_type,
        setup=setup,
        run_tag=run_tag,
        df_subject=df_subject,
        df_long=df_long,
        df_trajectory=pd.DataFrame(),
        cols_biomarker=cols_biomarker,
        trajectory_fn=trajectory_fn,
        time_factor=time_factor,
    )
    fitted_model.df_trajectory = fitted_model.get_df_trajectory(
        _get_trajectory_grid(t_shifted) * time_factor
    )
    return fitted_model


def get_fitted_models(
    fpath_json_results: Path,
    json_results: dict,
    tag: str,
    leaspy_algorithm_name: str = DEFAULT_LEASPY_ALGORITHM_NAME,
    leaspy_seed: int = DEFAULT_LEASPY_SEED,
    leaspy_n_jobs: int = DEFAULT_LEASPY_N_JOBS,
    fedbiomed_personalize_algorithm: str = DEFAULT_PERSONALIZE_ALGORITHM,
    personalize_seed: int = DEFAULT_PERSONALIZE_SEED,
    dpmost_personalize_n_outer_iter: int = DEFAULT_DPMOST_N_OUTER_ITER,
    personalize_n_timepoints: int | None = None,
) -> list[FittedModel]:
    """Apply the fitted parameters of each setup in a results file to the data.

    Held-out test subjects (if any) use their personalized parameters.

    Raises ``LeaspyIndividualParametersNotFoundError`` if a Leaspy model has not
    been personalized yet, or ``PersonalizedParametersNotFoundError`` if
    held-out subjects of a Fed-BioMed/DPMoSt model have not been personalized
    yet (see ``scripts/get_metrics.py``).
    """
    model_type = get_model_type(fpath_json_results)
    run_tag = get_run_tag(fpath_json_results)

    json_data = load_json_data(json_results)
    cols = get_cols(json_data)
    time_factor = get_time_factor(json_data)
    subjects_by_node = json_data["subjects_by_node"]

    df_train, df_test = _load_df_data_train_test(
        get_dpath_data(json_results), tag, cols
    )
    heldout_subjects = None
    if df_test is not None and has_heldout_subjects(subjects_by_node):
        heldout_subjects = get_subjects(subjects_by_node, test=True)
    df_all = _concat_train_test(df_train, df_test, cols, heldout_subjects)
    t_all = df_all[cols["col_timepoint"]].to_numpy(dtype=float)
    n_setups = len(json_results["results"])

    fitted_models = []
    if model_type == MODEL_LEASPY:
        df_leaspy_all = to_df_leaspy(df_all, cols)
        for setup, model_params in json_results["results"].items():
            model = load_leaspy_model(model_params)
            personalization_kwargs = {
                "algorithm_name": leaspy_algorithm_name,
                "seed": leaspy_seed,
                "n_jobs": leaspy_n_jobs,
            }
            df_ip = load_leaspy_individual_parameters(
                json_results, setup, **personalization_kwargs
            )
            if heldout_subjects is not None:
                df_ip = pd.concat(
                    [
                        df_ip,
                        load_leaspy_individual_parameters(
                            json_results,
                            get_leaspy_heldout_setup(setup, personalize_n_timepoints),
                            **personalization_kwargs,
                        ),
                    ]
                )
            df_shift_acc = leaspy_to_shift_acc(df_ip)
            subjects = df_all[cols["col_subject"]].astype(str)
            t_shifted = (
                subjects.map(df_shift_acc[COL_ACCELERATION_FACTOR]).to_numpy() * t_all
                + subjects.map(df_shift_acc[COL_TIME_SHIFT]).to_numpy()
            )
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
                    trajectory_fn=partial(predict_group_trajectory_leaspy, model),
                    time_factor=time_factor,
                )
            )
        return fitted_models

    normalisation = None
    if model_type == MODEL_DPMOST:
        normalisation = load_dpmost_normalisation(
            fpath_json_results, json_results, cols["cols_biomarker"]
        )
    predict_fn = get_predict_fn(model_type, normalisation=normalisation)
    personalization_key = get_heldout_personalization_key(
        model_type,
        fedbiomed_personalize_algorithm=fedbiomed_personalize_algorithm,
        personalize_seed=personalize_seed,
        dpmost_personalize_n_outer_iter=dpmost_personalize_n_outer_iter,
        personalize_n_timepoints=personalize_n_timepoints,
    )
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
        if heldout_subjects is not None:
            # held-out subjects are indexed after the train subjects
            df_params_test = load_personalized_params(
                json_results, personalization_key, setup
            ).loc[heldout_subjects]
            params = replace(
                params,
                time_shifts=np.concatenate(
                    [params.time_shifts, df_params_test["time_shifts"]]
                ),
                acceleration_factors=np.concatenate(
                    [
                        params.acceleration_factors,
                        df_params_test["acceleration_factors"],
                    ]
                ),
            )
            subject_order = subject_order + heldout_subjects
        validate_subject_ids(subject_ids_all, len(subject_order), setup)

        df_shift_acc = pd.DataFrame(
            {
                COL_TIME_SHIFT: get_time_shifts(params, normalisation),
                COL_ACCELERATION_FACTOR: (
                    np.ones_like(params.acceleration_factors)
                    if model_type == MODEL_DPMOST
                    else params.acceleration_factors
                ),
            },
            index=pd.Index(subject_order),
        )
        t_shifted = get_shifted_time(
            params, t_all, subject_ids_all, model_type, normalisation=normalisation
        )
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
                trajectory_fn=partial(
                    predict_group_trajectory, params, predict_fn=predict_fn
                ),
                time_factor=time_factor,
            )
        )
    return fitted_models
