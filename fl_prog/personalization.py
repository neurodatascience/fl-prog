"""Estimate individual parameters of new (held-out) subjects with a fitted model."""

import pickle
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from fl_prog.model import LogisticRegressionModelWithShift, ModelParams, Positive
from fl_prog.utils.constants import Penalty
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


# ----- fl_prog (Fed-BioMed) model -----


def get_penalty_weight(n_entries: int, n_features: int, n_subjects: int) -> float:
    """Weight of the per-subject penalties relative to the summed data term."""
    return n_entries * n_features / n_subjects


def _get_fl_prog_energy(
    params: ModelParams,
    model_args: dict,
    t: np.ndarray,
    subject_ids: np.ndarray,
    y: np.ndarray,
    n_subjects: int,
    penalty_weight: float,
) -> EnergyFn:
    """Per-subject negative log-posterior of the fl_prog model (up to a constant).

    Returns a function of the per-subject variables giving one value per subject.
    """
    lambda_time_shifts = model_args["lambda_time_shifts"]
    lambda_acceleration_factors = model_args["lambda_acceleration_factors"]
    penalty_time_shifts = Penalty(model_args["penalty_time_shifts"])
    penalty_acceleration_factors = Penalty(model_args["penalty_acceleration_factors"])
    time_shift_min, time_shift_max = model_args["expected_time_shift_range"]

    model = LogisticRegressionModelWithShift(
        n_participants=n_subjects, n_features=len(params.k_values)
    )
    apply_penalty = LogisticRegressionModelWithShift._apply_penalty

    t = torch.tensor(t, dtype=torch.float)
    subject_ids = torch.tensor(subject_ids, dtype=torch.long)
    y = torch.tensor(y, dtype=torch.float)
    mask = ~torch.isnan(y)
    y = torch.where(mask, y, 0.0)
    sigma_sq = torch.tensor(params.sigma, dtype=torch.float) ** 2

    def energy(variables: dict[str, torch.Tensor]) -> torch.Tensor:
        time_shifts = variables[VAR_TIME_SHIFTS]
        log_acceleration_factors = variables.get(
            VAR_LOG_ACCELERATION_FACTORS, torch.zeros(n_subjects)
        )
        y_pred = model.forward(
            t,
            subject_ids,
            params=ModelParams(
                k_values=params.k_values,
                x0_values=params.x0_values,
                scaling_factors=params.scaling_factors,
                sigma=params.sigma,
                time_shifts=time_shifts,
                acceleration_factors=torch.exp(log_acceleration_factors),
            ),
        )

        # Gaussian negative log-likelihood (missing values do not contribute)
        nll = torch.where(
            mask,
            (y - y_pred) ** 2 / (2 * sigma_sq)
            + 0.5 * torch.log(2 * torch.pi * sigma_sq),
            0.0,
        ).sum(dim=1)
        attachment = torch.zeros(n_subjects).index_add(0, subject_ids, nll)

        regularity = lambda_time_shifts * (
            apply_penalty(torch.relu(time_shift_min - time_shifts), penalty_time_shifts)
            + apply_penalty(
                torch.relu(time_shifts - time_shift_max), penalty_time_shifts
            )
        ) + lambda_acceleration_factors * apply_penalty(
            log_acceleration_factors, penalty_acceleration_factors
        )
        return attachment + penalty_weight * regularity

    return energy


def _minimize(
    energy: EnergyFn,
    init: dict[str, torch.Tensor],
    max_iter: int = DEFAULT_MINIMIZE_MAX_ITER,
) -> dict[str, torch.Tensor]:
    """MAP estimate of the per-subject variables (analogous to Leaspy scipy_minimize).

    Time shifts are optimized through the same softplus parametrization as during
    training to keep them positive. The energy is a sum of independent per-subject
    terms, so all subjects are optimized jointly.
    """
    positive = Positive()
    unconstrained = {
        name: (positive.right_inverse(value) if name == VAR_TIME_SHIFTS else value)
        .clone()
        .requires_grad_(True)
        for name, value in init.items()
    }

    def _constrain(name: str, value: torch.Tensor) -> torch.Tensor:
        return F.softplus(value) if name == VAR_TIME_SHIFTS else value

    optimizer = torch.optim.LBFGS(
        list(unconstrained.values()),
        max_iter=max_iter,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-9,
        tolerance_change=1e-12,
    )

    def closure():
        optimizer.zero_grad()
        loss = energy(
            {name: _constrain(name, value) for name, value in unconstrained.items()}
        ).sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    return {
        name: _constrain(name, value).detach() for name, value in unconstrained.items()
    }


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
    with_acceleration = model_args["with_acceleration"]

    # start from the average training subject
    init = {
        VAR_TIME_SHIFTS: torch.full((n_subjects,), float(np.mean(params.time_shifts)))
    }
    init_std = {VAR_TIME_SHIFTS: _get_init_std(params.time_shifts)}
    if with_acceleration:
        init[VAR_LOG_ACCELERATION_FACTORS] = torch.zeros(n_subjects)
        init_std[VAR_LOG_ACCELERATION_FACTORS] = _get_init_std(
            np.log(params.acceleration_factors)
        )

    energy = _get_fl_prog_energy(
        params, model_args, t, subject_ids, y, n_subjects, penalty_weight
    )
    if algorithm == PERSONALIZE_MINIMIZE:
        estimates = _minimize(energy, init)
    else:
        estimates, acceptance_rates = _sample(
            energy, init, init_std, algorithm, seed=seed
        )
        for name, rates in acceptance_rates.items():
            print(
                f"\t{name} acceptance rate: mean {rates.mean():.1%}, "
                f"range [{rates.min():.1%}, {rates.max():.1%}]"
            )

    if with_acceleration:
        acceleration_factors = torch.exp(estimates[VAR_LOG_ACCELERATION_FACTORS])
    else:
        acceleration_factors = torch.ones(n_subjects)
    return ModelParams(
        k_values=params.k_values,
        x0_values=params.x0_values,
        scaling_factors=params.scaling_factors,
        sigma=params.sigma,
        time_shifts=estimates[VAR_TIME_SHIFTS].numpy().astype(float),
        acceleration_factors=acceleration_factors.numpy().astype(float),
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
