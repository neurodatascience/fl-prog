#!/usr/bin/env python
"""Fit the fl-prog model on the pooled (centralized) data without Fed-BioMed.

Mimics a centralized ``run_fedbiomed.py`` run (same model, loss and Adam schedule:
the optimizer is re-created every round, as on a Fed-BioMed node, and every update
uses the full data) so that model variants, e.g. other penalties on the acceleration
factors, can be compared quickly without Fed-BioMed instances.

Experimental options (issue #40):

- ``--center-time``: fit ``alpha_i * (t - mean_i(t)) + shift'_i`` instead of
  ``alpha_i * t + shift_i`` (the time shift prior then applies to the disease time
  at the middle of follow-up). Shifts are converted back before saving.
- ``--learn-acceleration-prior``: Gaussian prior log(alpha_i) ~ N(0, tau^2) with tau
  estimated between rounds (one EM step per round, Laplace/Gauss-Newton posterior
  variances), instead of a fixed lambda. Implemented as an L2 penalty with
  lambda = n_subjects / (2 * numel * tau^2), i.e. the same per-subject weight as
  in the (mean-normalized) training loss.
- ``--oracle-population``: fix the population parameters (k, x0, scaling, sigma) to
  their true values (synthetic data) and fit only the per-subject parameters, to
  get an upper bound on how well the acceleration factors can be recovered.

The results file has the same layout as a Fed-BioMed results file (setup
``centralized``, node ``node_centralized``), so ``get_metrics.py`` (personalization
of held-out subjects, metrics) and ``fl_prog.results.get_fitted_models`` work on it.
With a learned prior, the final lambda is saved in the model arguments (used to
personalize held-out subjects). Run tags start with ``fedbiomed-local``.
"""

import json
from pathlib import Path

import click
import numpy as np
import pandas as pd
import torch

from fl_prog.model import LogisticRegressionModelWithShift
from fl_prog.results import true_params_by_subject
from fl_prog.utils.constants import (
    CLICK_CONTEXT_SETTINGS,
    NODE_ID_CENTRALIZED,
    NODE_PREFIX,
    Penalty,
    Setup,
)
from fl_prog.utils.io import (
    DEFAULT_DPATH_DATA,
    DEFAULT_DPATH_RESULTS,
    get_dpath_latest,
    load_json,
    save_json,
)

DEFAULT_N_ROUNDS = 30
DEFAULT_N_UPDATES = 25
DEFAULT_LEARNING_RATE = 0.05
DEFAULT_LAMBDA_TIME_SHIFTS = 0.01
DEFAULT_PENALTY_TIME_SHIFTS = Penalty.L2
DEFAULT_LAMBDA_ACCELERATION_FACTORS = 0.1
DEFAULT_PENALTY_ACCELERATION_FACTORS = Penalty.L2
DEFAULT_LAMBDA_K_VALUES = 0.01
DEFAULT_PENALTY_K_VALUES = Penalty.L2
DEFAULT_RANDOM_SEED = 3791
DEFAULT_PRIOR_SD_INIT = 1.0
DEFAULT_EM_WARMUP_ROUNDS = 3
# batch size recorded in the run tag (as in run_fedbiomed.py): all data
BATCH_SIZE = 100000


def get_run_tag(
    n_rounds: int,
    n_updates: int,
    learning_rate: float,
    penalty_time_shifts: Penalty,
    lambda_time_shifts: float,
    penalty_acceleration_factors: Penalty,
    lambda_acceleration_factors: float,
    penalty_k_values: Penalty,
    lambda_k_values: float,
    random_seed: int,
    center_time: bool = False,
    learn_acceleration_prior: bool = False,
    oracle_population: bool = False,
) -> str:
    """Same fields as run_fedbiomed.py run tags (always with acceleration and
    scaling), with ``fedbiomed-local`` as prefix."""
    accel = (
        "accel_em"
        if learn_acceleration_prior
        else f"accel_{penalty_acceleration_factors.value}_{lambda_acceleration_factors}"
    )
    return "-".join(
        [
            "fedbiomed",
            "local",
            f"{n_rounds}",
            f"{n_updates}",
            f"{BATCH_SIZE}",
            f"lr{learning_rate}",
            f"timeshifts_{penalty_time_shifts.value}_{lambda_time_shifts}",
            accel,
            f"k_{penalty_k_values.value}_{lambda_k_values}",
            "scaling",
            *(["centered"] if center_time else []),
            *(["oracle"] if oracle_population else []),
            f"{random_seed}",
        ]
    )


class AccelerationPrior:
    """Gaussian prior on log acceleration factors with an estimated SD (tau).

    One EM step per round: given the current (MAP) log alpha_i and their
    Gauss-Newton posterior variances, tau^2 <- mean_i(log(alpha_i)^2 + var_i).
    """

    def __init__(self, model, t, subject_ids, y, sd_init: float, warmup_rounds: int):
        self.model = model
        self.t = t
        self.subject_ids = subject_ids
        self.y = y
        self.tau2 = sd_init**2
        self.warmup_rounds = warmup_rounds
        self.n_subjects = model.n_participants
        self.numel = y.numel()
        self.history = []
        model.penalty_acceleration_factors = Penalty.L2.value
        self._set_lambda()

    @property
    def lambda_(self) -> float:
        # loss = sum(NLL) / numel + lambda * mean_i(u_i^2); times numel:
        # sum(NLL) + lambda * numel / n_subjects * sum_i(u_i^2), to match
        # sum(NLL) + sum_i(u_i^2) / (2 tau^2)
        return self.n_subjects / (2 * self.numel * self.tau2)

    def _set_lambda(self):
        self.model.lambda_acceleration_factors = self.lambda_

    @torch.no_grad()
    def posterior_variances(self) -> torch.Tensor:
        """Gauss-Newton approximation of Var(log alpha_i | data)."""
        model = self.model
        alpha = model.acceleration_factors[self.subject_ids]
        z = self.t * alpha + model.time_shifts[self.subject_ids]
        sig = torch.sigmoid(model.k_values * (z.view(-1, 1) - model.x0_values))
        # d prediction / d log(alpha_i)
        dpred = (
            model.scaling_factors
            * model.k_values
            * sig
            * (1 - sig)
            * (alpha * self.t).view(-1, 1)
        )
        mask = ~torch.isnan(self.y)
        gn_rows = (dpred**2 / model.sigma**2 * mask).sum(dim=1)
        gn = torch.zeros(self.n_subjects).index_add_(0, self.subject_ids, gn_rows)
        return 1 / (gn + 1 / self.tau2)

    @torch.no_grad()
    def update(self, i_round: int):
        if i_round < self.warmup_rounds:
            return
        u = torch.log(self.model.acceleration_factors)
        self.tau2 = float(torch.mean(u**2 + self.posterior_variances()))
        self._set_lambda()
        self.history.append(self.tau2**0.5)


def train(
    model: LogisticRegressionModelWithShift,
    t: torch.Tensor,
    subject_ids: torch.Tensor,
    y: torch.Tensor,
    n_rounds: int,
    n_updates: int,
    learning_rate: float,
    prior: AccelerationPrior | None = None,
) -> list[float]:
    """Full-batch Adam, with a new optimizer every round (as on a Fed-BioMed node).
    Returns the loss at the end of each round."""
    losses = []
    for i_round in range(n_rounds):
        if prior is not None:
            # values aggregated at the end of the previous round
            prior.update(i_round)
        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad], lr=learning_rate
        )
        for _ in range(n_updates):
            optimizer.zero_grad()
            loss = model.get_loss(model(t, subject_ids), y)
            loss.backward()
            optimizer.step()
        losses.append(loss.item())
    return losses


def _set_true_population_params(model: LogisticRegressionModelWithShift, json_data):
    """Fix k, x0, scaling and sigma to their true values (not trained)."""
    true_params = true_params_by_subject(
        json_data["subjects_by_node"], json_data["params"]
    )
    with torch.no_grad():
        for name, values in [
            ("k_values", true_params.k_values),
            ("scaling_factors", true_params.scaling_factors),
            ("sigma", true_params.sigma),
        ]:
            setattr(model, name, torch.as_tensor(values, dtype=torch.float))
            getattr(model.parametrizations, name).original.requires_grad_(False)
        model.x0_values.copy_(torch.as_tensor(true_params.x0_values))
        model.x0_values.requires_grad_(False)


def fit_local_centralized(
    tag: str,
    dpath_data: Path,
    dpath_results: Path,
    n_rounds: int = DEFAULT_N_ROUNDS,
    n_updates: int = DEFAULT_N_UPDATES,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    lambda_time_shifts: float = DEFAULT_LAMBDA_TIME_SHIFTS,
    penalty_time_shifts: Penalty = DEFAULT_PENALTY_TIME_SHIFTS,
    lambda_acceleration_factors: float = DEFAULT_LAMBDA_ACCELERATION_FACTORS,
    penalty_acceleration_factors: Penalty = DEFAULT_PENALTY_ACCELERATION_FACTORS,
    lambda_k_values: float = DEFAULT_LAMBDA_K_VALUES,
    penalty_k_values: Penalty = DEFAULT_PENALTY_K_VALUES,
    center_time: bool = False,
    learn_acceleration_prior: bool = False,
    prior_sd_init: float = DEFAULT_PRIOR_SD_INIT,
    em_warmup_rounds: int = DEFAULT_EM_WARMUP_ROUNDS,
    oracle_population: bool = False,
    random_seed: int = DEFAULT_RANDOM_SEED,
    overwrite: bool = False,
):
    if learn_acceleration_prior:
        penalty_acceleration_factors = Penalty.L2
    run_tag = get_run_tag(
        n_rounds,
        n_updates,
        learning_rate,
        penalty_time_shifts,
        lambda_time_shifts,
        penalty_acceleration_factors,
        lambda_acceleration_factors,
        penalty_k_values,
        lambda_k_values,
        random_seed,
        center_time=center_time,
        learn_acceleration_prior=learn_acceleration_prior,
        oracle_population=oracle_population,
    )
    dpath_out = get_dpath_latest(dpath_results, use_today=True) / tag
    fpath_out = dpath_out / f"{run_tag}-estimated_params.json"
    if fpath_out.exists() and "results" in load_json(fpath_out) and not overwrite:
        click.secho(
            f"{fpath_out} already exists. Use --overwrite to overwrite.", fg="red"
        )
        print(run_tag)
        return

    # --data-dir can be a dated data directory (e.g. data/2026_11_11) or its parent
    if (dpath_data / tag).is_dir():
        dpath_data = dpath_data.resolve() / tag
    else:
        dpath_data = get_dpath_latest(dpath_data) / tag
    fpath_config = dpath_data / f"{tag}.json"
    config = json.loads(fpath_config.read_text())
    cols = config["cols"]
    col_subject_id = cols["col_subject_index"]
    cols_biomarker = cols["cols_biomarker"]

    df = pd.read_csv(dpath_data / f"{tag}-merged.tsv", sep="\t")
    n_participants = df[col_subject_id].nunique()
    if sorted(df[col_subject_id].unique()) != list(range(n_participants)):
        raise ValueError(f"{col_subject_id} should be 0..{n_participants - 1}")

    model_args = {
        "colnames": {
            "col_subject_id": col_subject_id,
            "col_time": cols["col_timepoint"],
            "cols_biomarker": cols_biomarker,
        },
        "lr_with_shift": {
            "n_features": len(cols_biomarker),
            "lambda_time_shifts": lambda_time_shifts,
            "penalty_time_shifts": penalty_time_shifts.value,
            "with_acceleration": True,
            "lambda_acceleration_factors": lambda_acceleration_factors,
            "penalty_acceleration_factors": penalty_acceleration_factors.value,
            "with_scaling": True,
            "lambda_k_values": lambda_k_values,
            "penalty_k_values": penalty_k_values.value,
        },
    }

    torch.manual_seed(random_seed)
    model = LogisticRegressionModelWithShift(
        n_participants=n_participants, **model_args["lr_with_shift"]
    )
    if oracle_population:
        _set_true_population_params(model, config)

    t = torch.tensor(df[cols["col_timepoint"]].to_numpy(), dtype=torch.float)
    subject_ids = torch.tensor(df[col_subject_id].to_numpy(), dtype=torch.long)
    y = torch.tensor(df[cols_biomarker].to_numpy(), dtype=torch.float)
    t_mean = torch.zeros(n_participants)
    if center_time:
        t_mean = torch.as_tensor(
            df.groupby(col_subject_id)[cols["col_timepoint"]]
            .mean()
            .sort_index()
            .to_numpy(),
            dtype=torch.float,
        )
    t_fit = t - t_mean[subject_ids]

    prior = None
    if learn_acceleration_prior:
        prior = AccelerationPrior(
            model, t_fit, subject_ids, y, prior_sd_init, em_warmup_rounds
        )
    losses = train(
        model, t_fit, subject_ids, y, n_rounds, n_updates, learning_rate, prior
    )
    print(f"{tag} {run_tag}: final loss {losses[-1]:.4f}")

    def _np(tensor: torch.Tensor) -> np.ndarray:
        return tensor.detach().numpy()

    acceleration_factors = model.acceleration_factors.detach()
    # back to alpha * t + shift (disease time at t = 0)
    time_shifts = model.time_shifts.detach() - acceleration_factors * t_mean
    if prior is not None:
        model_args["lr_with_shift"]["lambda_acceleration_factors"] = prior.lambda_

    node = f"{NODE_PREFIX}{NODE_ID_CENTRALIZED}"
    settings = {
        "tag": tag,
        "dpath_data": str(dpath_data),
        "dpath_results": str(dpath_results),
        "fpath_config": str(fpath_config),
        "setups": [Setup.CENTRALIZED.value],
        "n_rounds": n_rounds,
        "n_updates": n_updates,
        "batch_size": BATCH_SIZE,
        "learning_rate": learning_rate,
        "random_seed": random_seed,
        "run_tag": run_tag,
        "model_args": model_args,
        "center_time": center_time,
        "learn_acceleration_prior": learn_acceleration_prior,
        "prior_sd_init": prior_sd_init,
        "em_warmup_rounds": em_warmup_rounds,
        "oracle_population": oracle_population,
        "trainer": "fit_local_centralized.py (no Fed-BioMed)",
    }
    results = {
        "estimated_k_values": _np(model.k_values),
        "estimated_x0_values": _np(model.x0_values),
        "estimated_scaling_factors": _np(model.scaling_factors),
        "estimated_sigma": _np(model.sigma),
        "estimated_time_shifts": {node: _np(time_shifts)},
        "estimated_acceleration_factors": {node: _np(acceleration_factors)},
        "losses": losses,
    }
    if prior is not None:
        results["acceleration_prior_sd"] = prior.history
    dpath_out.mkdir(parents=True, exist_ok=True)
    save_json(
        fpath_out,
        {"settings": settings, "results": {Setup.CENTRALIZED.value: results}},
    )
    print(f"Saved results to {fpath_out}")
    # run tag on the last line (e.g. for get_metrics.py --run-tag)
    print(run_tag)


@click.command(context_settings=CLICK_CONTEXT_SETTINGS)
@click.option("--tag", type=str, required=True)
@click.option(
    "--data-dir",
    "dpath_data",
    type=click.Path(path_type=Path, file_okay=False, dir_okay=True),
    default=DEFAULT_DPATH_DATA,
)
@click.option(
    "--results-dir",
    "dpath_results",
    type=click.Path(path_type=Path, file_okay=False, dir_okay=True),
    default=DEFAULT_DPATH_RESULTS,
)
@click.option("--n-rounds", type=int, default=DEFAULT_N_ROUNDS, show_default=True)
@click.option("--n-updates", type=int, default=DEFAULT_N_UPDATES, show_default=True)
@click.option(
    "--learning-rate", type=float, default=DEFAULT_LEARNING_RATE, show_default=True
)
@click.option(
    "--lambda-time-shifts",
    type=float,
    default=DEFAULT_LAMBDA_TIME_SHIFTS,
    show_default=True,
)
@click.option(
    "--penalty-time-shifts",
    type=click.Choice(Penalty, case_sensitive=False),
    default=DEFAULT_PENALTY_TIME_SHIFTS,
    show_default=True,
)
@click.option(
    "--lambda-acceleration-factors",
    type=float,
    default=DEFAULT_LAMBDA_ACCELERATION_FACTORS,
    show_default=True,
)
@click.option(
    "--penalty-acceleration-factors",
    type=click.Choice(Penalty, case_sensitive=False),
    default=DEFAULT_PENALTY_ACCELERATION_FACTORS,
    show_default=True,
)
@click.option(
    "--lambda-k-values", type=float, default=DEFAULT_LAMBDA_K_VALUES, show_default=True
)
@click.option(
    "--penalty-k-values",
    type=click.Choice(Penalty, case_sensitive=False),
    default=DEFAULT_PENALTY_K_VALUES,
    show_default=True,
)
@click.option(
    "--center-time/--no-center-time",
    default=False,
    help="Center each subject's time at their mean visit time.",
)
@click.option(
    "--learn-acceleration-prior/--fixed-acceleration-prior",
    default=False,
    help="L2 prior on log alpha with an SD estimated between rounds (EM).",
)
@click.option(
    "--prior-sd-init", type=float, default=DEFAULT_PRIOR_SD_INIT, show_default=True
)
@click.option(
    "--em-warmup-rounds",
    type=int,
    default=DEFAULT_EM_WARMUP_ROUNDS,
    show_default=True,
    help="Rounds with the initial prior SD before EM updates.",
)
@click.option(
    "--oracle-population/--no-oracle-population",
    default=False,
    help="Fix population parameters to their true values (synthetic data).",
)
@click.option(
    "--random-seed",
    type=int,
    default=DEFAULT_RANDOM_SEED,
    envvar="RNG_SEED",
    show_default=True,
)
@click.option("--overwrite/--no-overwrite", default=False)
def main(**params):
    fit_local_centralized(**params)


if __name__ == "__main__":
    main()
