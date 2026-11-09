"""Estimate individual parameters of new (held-out) subjects with a fitted model."""

import dataclasses
import pickle
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from fl_prog.model import LogisticRegressionModelWithShift, ModelParams
from fl_prog.utils.io import add_dpmost_to_path, format_df_for_dpmost

PERSONALIZE_MINIMIZE = "minimize"
PERSONALIZE_MODE_POSTERIOR = "mode_posterior"
PERSONALIZE_MEAN_POSTERIOR = "mean_posterior"
PERSONALIZE_ALGORITHMS = (
    PERSONALIZE_MINIMIZE,
    PERSONALIZE_MODE_POSTERIOR,
    PERSONALIZE_MEAN_POSTERIOR,
)

DEFAULT_PERSONALIZE_ALGORITHM = PERSONALIZE_MINIMIZE
DEFAULT_PERSONALIZE_SEED = 0
DEFAULT_MINIMIZE_MAX_ITER = 1000
DEFAULT_DPMOST_N_OUTER_ITER = 200

# Leaspy defaults for mode_posterior/mean_posterior
# (leaspy/algo/data/default_{mode,mean}_posterior.json and
# leaspy.samplers.IndividualGibbsSampler)
DEFAULT_N_ITER = 1000
DEFAULT_BURN_IN_FRAC = 0.5
DEFAULT_ACCEPTANCE_HISTORY_LENGTH = 25
DEFAULT_ACCEPTANCE_RATE_BOUNDS = (0.2, 0.4)
DEFAULT_ADAPTIVE_STD_FACTOR = 0.1
STD_SCALE_FACTOR = 0.5

# names of the sampled per-subject variables
VAR_TIME_SHIFTS = "time_shifts"
VAR_LOG_ACCELERATION_FACTORS = "log_acceleration_factors"

EnergyFn = Callable[[dict[str, torch.Tensor]], torch.Tensor]


# ----- personalization on a subset of visits -----


def get_first_timepoints_mask(
    subjects: np.ndarray | pd.Series | pd.Index,
    timepoints: np.ndarray | pd.Series | pd.Index,
    n_timepoints: int | None,
) -> np.ndarray:
    """True for each subject's first ``n_timepoints`` visits (by time), in row
    order. All True if ``n_timepoints`` is None."""
    subjects = np.asarray(subjects)
    if n_timepoints is None:
        return np.ones(len(subjects), dtype=bool)
    ranks = (
        pd.Series(np.asarray(timepoints, dtype=float))
        .groupby(subjects)
        .rank(method="first")
    )
    return (ranks <= n_timepoints).to_numpy()


def get_n_timepoints_suffix(n_timepoints: int | None) -> str:
    """Suffix identifying personalization on each subject's first visits only."""
    return "" if n_timepoints is None else f"-first{n_timepoints}"


# ----- fl_prog (Fed-BioMed) model -----


def get_penalty_weight(n_entries: int, n_features: int, n_subjects: int) -> float:
    """Weight of the per-subject penalties relative to the summed data term."""
    return n_entries * n_features / n_subjects


def _sample(
    energy: EnergyFn,
    init: dict[str, torch.Tensor],
    init_std: dict[str, float],
    algorithm: str,
    seed: int = DEFAULT_PERSONALIZE_SEED,
    n_iter: int = DEFAULT_N_ITER,
    burn_in_frac: float = DEFAULT_BURN_IN_FRAC,
    history_length: int = DEFAULT_ACCEPTANCE_HISTORY_LENGTH,
    acceptance_rate_bounds: tuple[float, float] = DEFAULT_ACCEPTANCE_RATE_BOUNDS,
    adaptive_std_factor: float = DEFAULT_ADAPTIVE_STD_FACTOR,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Sampler-based estimate (analogous to Leaspy mode_posterior/mean_posterior).

    Metropolis-within-Gibbs over the per-subject variables, with one random-walk
    proposal per variable and subject, accepted independently per subject. The
    proposal std is adapted every ``history_length`` iterations to keep each
    subject's acceptance rate within ``acceptance_rate_bounds``. Samples after
    burn-in give either the lowest-energy sample per subject (``mode_posterior``)
    or the mean (``mean_posterior``).

    Returns the estimates and the final acceptance rate of each variable.
    """
    generator = torch.Generator().manual_seed(seed)
    names = list(init)
    n_subjects = len(init[names[0]])
    n_burn_in = int(n_iter * burn_in_frac)
    rate_min, rate_max = acceptance_rate_bounds

    values = {name: value.clone() for name, value in init.items()}
    stds = {name: torch.full((n_subjects,), init_std[name]) for name in names}
    accepted_history = {name: torch.zeros(history_length, n_subjects) for name in names}

    samples = {name: [] for name in names}
    energies = []
    with torch.no_grad():
        current_energy = energy(values)
        for i_iter in range(n_iter):
            for i_name in torch.randperm(len(names), generator=generator):
                name = names[i_name]
                noise = torch.randn(n_subjects, generator=generator)
                proposal = {**values, name: values[name] + stds[name] * noise}
                proposed_energy = energy(proposal)
                accepted = torch.rand(n_subjects, generator=generator) < torch.exp(
                    current_energy - proposed_energy
                )
                if name == VAR_TIME_SHIFTS:
                    # time shifts are constrained to be positive in the model
                    accepted &= proposal[name] > 0

                values[name] = torch.where(accepted, proposal[name], values[name])
                current_energy = torch.where(accepted, proposed_energy, current_energy)

                accepted_history[name][i_iter % history_length] = accepted.float()
                if (i_iter + 1) % history_length == 0:
                    acceptance_rate = accepted_history[name].mean(dim=0)
                    stds[name][acceptance_rate < rate_min] *= 1 - adaptive_std_factor
                    stds[name][acceptance_rate > rate_max] *= 1 + adaptive_std_factor

            if i_iter >= n_burn_in:
                for name in names:
                    samples[name].append(values[name].clone())
                energies.append(current_energy.clone())

    samples = {name: torch.stack(values) for name, values in samples.items()}
    if algorithm == PERSONALIZE_MODE_POSTERIOR:
        i_best = torch.argmin(torch.stack(energies), dim=0)
        estimates = {
            name: values[i_best, torch.arange(n_subjects)]
            for name, values in samples.items()
        }
    elif algorithm == PERSONALIZE_MEAN_POSTERIOR:
        estimates = {name: values.mean(dim=0) for name, values in samples.items()}
    else:
        raise ValueError(f"Unknown sampler-based algorithm {algorithm!r}")

    acceptance_rates = {
        name: history.mean(dim=0) for name, history in accepted_history.items()
    }
    return estimates, acceptance_rates


def _get_init_std(values: np.ndarray) -> float:
    """Initial proposal std: half the spread of the training estimates."""
    std = np.std(values)
    if not np.isfinite(std) or std <= 0:
        std = 1.0
    return STD_SCALE_FACTOR * float(std)


def personalize_fl_prog(
    params: ModelParams,
    model_args: dict,
    t: np.ndarray,
    subject_ids: np.ndarray,
    y: np.ndarray,
    penalty_weight: float,
    algorithm: str = DEFAULT_PERSONALIZE_ALGORITHM,
    seed: int = DEFAULT_PERSONALIZE_SEED,
) -> ModelParams:
    """Estimate time shifts (and acceleration factors) of new subjects.

    ``params`` are the fitted parameters, whose per-subject values (from the
    training subjects) are only used for initialization. ``subject_ids`` index
    the new subjects from 0. ``model_args`` are the ``lr_with_shift`` model
    arguments used for training. Returns parameters with the fitted population
    values and the new subjects' per-subject values.
    """
    if algorithm not in PERSONALIZE_ALGORITHMS:
        raise ValueError(
            f"Unknown algorithm {algorithm!r}. Expected one of {PERSONALIZE_ALGORITHMS}"
        )

    n_subjects = int(subject_ids.max()) + 1
    model = LogisticRegressionModelWithShift.get_personalization_model(
        params, n_subjects, **model_args
    )
    t = torch.tensor(t, dtype=torch.float)
    subject_ids = torch.tensor(subject_ids, dtype=torch.long)
    y = torch.tensor(y, dtype=torch.float)

    if algorithm == PERSONALIZE_MINIMIZE:
        model.personalize(t, subject_ids, y, penalty_weight)
        return model.to_params()

    population_params = model.to_params()

    def energy(variables: dict[str, torch.Tensor]) -> torch.Tensor:
        time_shifts = variables[VAR_TIME_SHIFTS]
        acceleration_factors = torch.exp(
            variables.get(VAR_LOG_ACCELERATION_FACTORS, torch.zeros(n_subjects))
        )
        y_pred = model(
            t,
            subject_ids,
            params=dataclasses.replace(
                population_params,
                time_shifts=time_shifts,
                acceleration_factors=acceleration_factors,
            ),
        )
        return model.get_loss_per_subject(
            y_pred,
            y,
            subject_ids,
            penalty_weight,
            time_shifts=time_shifts,
            acceleration_factors=acceleration_factors,
        )

    # start from the average training subject (as in get_personalization_model)
    init = {VAR_TIME_SHIFTS: torch.as_tensor(population_params.time_shifts).float()}
    init_std = {VAR_TIME_SHIFTS: _get_init_std(params.time_shifts)}
    if model.with_acceleration:
        init[VAR_LOG_ACCELERATION_FACTORS] = torch.zeros(n_subjects)
        init_std[VAR_LOG_ACCELERATION_FACTORS] = _get_init_std(
            np.log(params.acceleration_factors)
        )

    estimates, acceptance_rates = _sample(energy, init, init_std, algorithm, seed=seed)
    for name, rates in acceptance_rates.items():
        print(
            f"\t{name} acceptance rate: mean {rates.mean():.1%}, "
            f"range [{rates.min():.1%}, {rates.max():.1%}]"
        )

    acceleration_factors = np.ones(n_subjects)
    if model.with_acceleration:
        acceleration_factors = np.exp(estimates[VAR_LOG_ACCELERATION_FACTORS].numpy())
    return dataclasses.replace(
        population_params,
        time_shifts=estimates[VAR_TIME_SHIFTS].numpy().astype(float),
        acceleration_factors=acceleration_factors.astype(float),
    )


# ----- DPMoSt -----


def personalize_dpmost(
    fpath_model: Path,
    df: pd.DataFrame,
    cols: dict,
    init_time_shift: float,
    optimize_args: dict,
    n_outer_iter: int = DEFAULT_DPMOST_N_OUTER_ITER,
) -> pd.Series:
    """Estimate time shifts of new subjects with a fitted DPMoSt model.

    A new DPMoSt model is built on ``df`` (normalised with the fitted model's
    normalisation), with all population parameters copied from the fitted model
    and frozen. Only the time shifts (and cluster probabilities, which are trivial
    with one cluster) are optimised with DPMoSt's own objective and optimiser.

    ``optimize_args`` are the arguments of ``DPMoSt.optimise`` used for fitting;
    only the time shift learning rate and number of inner iterations are reused.
    Returns time shifts (in DPMoSt's normalised time units) indexed by subject.
    """
    add_dpmost_to_path()
    from DPMoSt import DPMoSt

    with fpath_model.open("rb") as file_model:
        model: DPMoSt = pickle.load(file_model)

    df_dpmost = format_df_for_dpmost(
        df=df.copy(),
        col_subject=cols["col_subject"],
        col_timepoint=cols["col_timepoint"],
        cols_biomarker=cols["cols_biomarker"],
    ).reset_index(drop=True)
    name_biomarkers = list(model.name_biomarkers)
    if list(df_dpmost.columns[2:]) != name_biomarkers:
        raise ValueError(
            f"Biomarkers {list(df_dpmost.columns[2:])} do not match the fitted "
            f"model's biomarkers {name_biomarkers}"
        )

    model_new = DPMoSt(
        data=df_dpmost,
        n_clusters=model.n_clusters,
        initialise=False,
        time_shift_eval=True,
        theta_eval=False,
        noise_std_eval=False,
        xi_eval=False,
        pi_eval=True,
        do_normalisation=model.do_normalisation,
        # normalise with the training data's min/max (copies, DPMoSt fills them in)
        data_normalisation_min=dict(model.data_normalisation_min),
        data_normalisation_max=dict(model.data_normalisation_max),
        dict_monotonicity=dict(model.dict_monotonicity),
        device=model.device,
        verbose=False,
    )

    # copy (frozen) population parameters
    model_new.modality_scale = model.modality_scale
    model_new.xi = model.xi.detach().clone()
    model_new.log_noise_std = model.log_noise_std.detach().clone()
    model_new.theta = {
        key: [value.detach().clone() for value in values]
        for key, values in model.theta.items()
    }
    model_new.pi = torch.full(
        (model_new.n_subjects, model.n_clusters),
        1.0 / model.n_clusters,
        dtype=torch.float32,
        device=model_new.device,
    )
    model_new.time_shift = torch.full(
        (model_new.n_subjects,),
        float(init_time_shift),
        dtype=torch.float32,
        device=model_new.device,
        requires_grad=True,
    )

    model_new.optimise(
        n_outer_iterations=n_outer_iter,
        n_final_iterations=0,
        n_inner_iterations_time_shift=optimize_args["n_inner_iterations_time_shift"],
        lr_time_shift=optimize_args["lr_time_shift"],
        stopping_criteria=False,
    )
    return pd.Series(
        model_new.time_shift.detach().cpu().numpy().astype(float),
        index=pd.Index(model_new.subjects, name=cols["col_subject"]),
    )
