from collections.abc import Iterable
from dataclasses import dataclass
from typing import ClassVar, Self

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import parametrize

from fl_prog.utils.constants import Penalty


@dataclass
class ModelParams:
    k_values: np.ndarray
    x0_values: np.ndarray
    scaling_factors: np.ndarray
    sigma: np.ndarray
    time_shifts: np.ndarray
    acceleration_factors: np.ndarray


class Positive(nn.Module):
    """Constrain a parameter to be positive using the softplus function."""

    def forward(self, param):
        return F.softplus(param)

    def right_inverse(self, constrained_param):
        return constrained_param + torch.log(-torch.expm1(-constrained_param))


class LogisticRegressionModelWithShift(nn.Module):
    parametrization_dict: ClassVar = {
        "k_values": (Positive(),),
        "sigma": (Positive(),),
        "time_shifts": (Positive(),),
        "scaling_factors": (Positive(),),
        "acceleration_factors": (Positive(),),
    }

    # fitted at the population level, frozen during personalization
    population_param_names: ClassVar = (
        "k_values",
        "x0_values",
        "sigma",
        "scaling_factors",
    )

    sigmoid_levels = torch.tensor([0.05, 0.95])  # for computing initial k_values
    x0_init_std_fraction = (
        0.1  # std of x0 initialization as a fraction of expected_time_shift_diff
    )

    def __init__(
        self,
        n_participants: int,
        n_features: int,
        expected_time_shift_range: Iterable[float] = (0.0, 0.0),
        lambda_time_shifts: float = 1.0,
        lambda_acceleration_factors: float = 1.0,
        lambda_k_values: float = 0.01,
        with_acceleration=False,
        with_scaling=False,
        penalty_time_shifts: Penalty = "l2",
        penalty_acceleration_factors: Penalty = "l1",
        penalty_k_values: Penalty = "l2",
    ):
        super().__init__()

        if len(expected_time_shift_range) != 2:
            raise ValueError(
                f"expected_time_shift_range must have length 2, got {len(expected_time_shift_range)}"
            )

        if lambda_time_shifts < 0:
            raise ValueError(
                f"lambda_time_shifts must be non-negative, got {lambda_time_shifts}"
            )
        if lambda_acceleration_factors < 0:
            raise ValueError(
                f"lambda_acceleration_factors must be non-negative, got {lambda_acceleration_factors}"
            )
        if lambda_k_values < 0:
            raise ValueError(
                f"lambda_k_values must be non-negative, got {lambda_k_values}"
            )

        expected_time_shift_range = torch.tensor(
            expected_time_shift_range, dtype=torch.float
        )

        # for initializing some parameters
        expected_time_shift_middle = torch.mean(expected_time_shift_range)
        expected_time_shift_diff = (
            expected_time_shift_range[1] - expected_time_shift_range[0]
        )
        if expected_time_shift_diff != 0:
            starting_k_value = (
                2
                * torch.log(self.sigmoid_levels[1] / self.sigmoid_levels[0])
                / expected_time_shift_diff
            )
        else:
            starting_k_value = torch.tensor(0.0)

        self.n_participants = n_participants
        self.n_features = n_features
        self.expected_time_shift_range = expected_time_shift_range
        self.lambda_time_shifts = lambda_time_shifts
        self.lambda_acceleration_factors = lambda_acceleration_factors
        self.lambda_k_values = lambda_k_values
        self.with_acceleration = with_acceleration
        self.with_scaling = with_scaling
        self.penalty_time_shifts = penalty_time_shifts
        self.penalty_acceleration_factors = penalty_acceleration_factors
        self.penalty_k_values = penalty_k_values

        # slopes
        self.k_values = nn.Parameter(torch.rand(self.n_features) + starting_k_value)

        # midpoints
        self.x0_values = nn.Parameter(
            torch.randn(self.n_features)
            * (expected_time_shift_diff * self.x0_init_std_fraction)
            + expected_time_shift_middle
        )

        self.time_shifts = nn.Parameter(
            torch.abs(torch.randn(self.n_participants)) + expected_time_shift_middle
        )

        self.sigma = nn.Parameter(torch.ones(self.n_features) * 0.5)

        self.scaling_factors = torch.ones(self.n_features)
        self.acceleration_factors = torch.ones(self.n_participants)
        if with_scaling:
            self.scaling_factors = nn.Parameter(self.scaling_factors)
        if with_acceleration:
            self.acceleration_factors = nn.Parameter(self.acceleration_factors)

        # constrain some parameters
        for param_name, parametrizations in self.parametrization_dict.items():
            if isinstance(getattr(self, param_name), nn.Parameter):
                for parametrization in parametrizations:
                    parametrize.register_parametrization(
                        self, param_name, parametrization
                    )

    @classmethod
    def _apply_parametrization(
        cls, param_name: str, unparametrized_param: torch.Tensor
    ):
        parametrizations = cls.parametrization_dict[param_name]
        constrained_param = unparametrized_param
        for parametrization in parametrizations:
            constrained_param = parametrization(constrained_param)
        return constrained_param

    @staticmethod
    def _apply_penalty(tensor: torch.Tensor, penalty_type: Penalty) -> torch.Tensor:
        match penalty_type:
            case Penalty.L1:
                return torch.abs(tensor)
            case Penalty.L2:
                return tensor**2
            case _:
                raise ValueError(
                    f"Unknown penalty type: {penalty_type}."
                    f" Valid options are: {[e.value for e in Penalty]}"
                )

    @classmethod
    def get_k_values(cls, unparametrized_k_values: torch.Tensor) -> torch.Tensor:
        return cls._apply_parametrization("k_values", unparametrized_k_values)

    @classmethod
    def get_time_shifts(cls, unparametrized_time_shifts: torch.Tensor) -> torch.Tensor:
        return cls._apply_parametrization("time_shifts", unparametrized_time_shifts)

    @classmethod
    def get_scaling_factors(
        cls, unparametrized_scaling_factors: torch.Tensor
    ) -> torch.Tensor:
        return cls._apply_parametrization(
            "scaling_factors", unparametrized_scaling_factors
        )

    @classmethod
    def get_acceleration_factors(
        cls, unparametrized_acceleration_factors: torch.Tensor
    ) -> torch.Tensor:
        return cls._apply_parametrization(
            "acceleration_factors", unparametrized_acceleration_factors
        )

    @classmethod
    def get_sigma(cls, unparametrized_sigma: torch.Tensor) -> torch.Tensor:
        return cls._apply_parametrization("sigma", unparametrized_sigma)

    def forward(
        self,
        t: torch.Tensor,
        participant_ids: torch.Tensor,
        params: ModelParams | None = None,
    ):
        # training
        if params is None:
            k_values = self.k_values
            x0_values = self.x0_values
            scaling_factors = self.scaling_factors
            time_shifts = self.time_shifts
            acceleration_factors = self.acceleration_factors
        # inference
        else:
            k_values = torch.as_tensor(params.k_values, dtype=torch.float)
            x0_values = torch.as_tensor(params.x0_values, dtype=torch.float)
            scaling_factors = torch.as_tensor(params.scaling_factors, dtype=torch.float)
            time_shifts = torch.as_tensor(params.time_shifts, dtype=torch.float)
            acceleration_factors = torch.as_tensor(
                params.acceleration_factors, dtype=torch.float
            )

        shift = time_shifts[participant_ids.to(torch.long)].squeeze(-1)
        acceleration = acceleration_factors[participant_ids.to(torch.long)].squeeze(-1)
        shifted_t = t.view(-1) * acceleration + shift

        linear_combination = k_values * (shifted_t.view(-1, 1) - x0_values)
        output = torch.sigmoid(linear_combination)
        output = output * scaling_factors
        return output

    def _get_nll(self, predicted: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
        """Negative Gaussian log-likelihood of each entry (summed over features)."""
        sigma_sq = self.sigma**2

        # trick to handle missing data (NAs)
        # replace missing values by the predicted values, so that the loss is 0 for those entries
        actual_mask = ~torch.isnan(actual)
        actual_no_na = torch.where(actual_mask, actual, predicted)

        return (
            (actual_no_na - predicted) ** 2 / (2 * sigma_sq)
            + torch.where(
                actual_mask, 0.5 * torch.log(2 * torch.pi * sigma_sq), torch.tensor(0.0)
            )
        ).sum(dim=1)

    def _get_participant_penalties(
        self,
        time_shifts: torch.Tensor | None = None,
        acceleration_factors: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Penalty on each participant's time shift and acceleration factor."""
        if time_shifts is None:
            time_shifts = self.time_shifts
        if acceleration_factors is None:
            acceleration_factors = self.acceleration_factors

        # penalize time shifts that are outside the expected range
        # equivalent to normal L1/L2 regularization if expected_time_shift_range is (0, 0)
        penalties = self.lambda_time_shifts * (
            self._apply_penalty(
                torch.relu(self.expected_time_shift_range[0] - time_shifts),
                self.penalty_time_shifts,
            )
            + self._apply_penalty(
                torch.relu(time_shifts - self.expected_time_shift_range[1]),
                self.penalty_time_shifts,
            )
        )

        # also penalize acceleration factors
        penalties += self.lambda_acceleration_factors * self._apply_penalty(
            torch.log(acceleration_factors), self.penalty_acceleration_factors
        )
        return penalties

    def get_loss(self, predicted: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
        # subject-dependent component
        loss = (
            self._get_nll(predicted, actual).sum() / actual.numel()
            + self._get_participant_penalties().mean()
        )

        # penalize slopes too, otherwise the sigmoids collapse to step functions
        loss += self.lambda_k_values * torch.mean(
            self._apply_penalty(torch.log(self.k_values), self.penalty_k_values)
        )

        return loss

    @classmethod
    def get_personalization_model(
        cls, params: ModelParams, n_participants: int, /, **kwargs
    ) -> Self:
        """Model with fitted (frozen) population parameters, for new participants."""
        kwargs.pop("n_participants", None)
        n_features = kwargs.pop("n_features", len(params.k_values))
        if n_features != len(params.k_values):
            raise ValueError(
                f"n_features={n_features} but params have {len(params.k_values)} "
                "features"
            )
        model = cls(n_participants=n_participants, n_features=n_features, **kwargs)

        # set and freeze biomarker-specific params
        with torch.no_grad():
            for name in cls.population_param_names:
                value = torch.as_tensor(getattr(params, name), dtype=torch.float)
                if parametrize.is_parametrized(model, name):
                    # assignment goes through the parametrization's right_inverse
                    setattr(model, name, value)
                    model.parametrizations[name].original.requires_grad_(False)
                elif isinstance(getattr(model, name), nn.Parameter):
                    getattr(model, name).copy_(value)
                    getattr(model, name).requires_grad_(False)
                else:
                    # not a parameter (e.g. scaling factors if with_scaling=False)
                    setattr(model, name, value)
            model.time_shifts = torch.full(
                (n_participants,), float(np.mean(params.time_shifts))
            )
            if model.with_acceleration:
                model.acceleration_factors = torch.ones(n_participants)
        return model

    def get_loss_per_subject(
        self,
        predicted: torch.Tensor,
        actual: torch.Tensor,
        participant_ids: torch.Tensor,
        penalty_weight: float,
        time_shifts: torch.Tensor | None = None,
        acceleration_factors: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Per-participant loss: summed negative log-likelihood + weighted penalties."""
        penalties = self._get_participant_penalties(time_shifts, acceleration_factors)
        nll_per_participant = torch.zeros(len(penalties)).index_add(
            0, participant_ids.to(torch.long), self._get_nll(predicted, actual)
        )
        return nll_per_participant + penalty_weight * penalties

    def personalize(
        self,
        t: torch.Tensor,
        participant_ids: torch.Tensor,
        actual: torch.Tensor,
        penalty_weight: float,
        max_iter: int = 1000,
    ):
        """Fit the per-participant parameters, with population parameters frozen.

        Minimizes ``get_loss_per_subject`` with L-BFGS (the participants'
        losses are independent, so they are fit jointly). Use
        ``get_personalization_model`` to create the model.
        """
        params = [param for param in self.parameters() if param.requires_grad]
        optimizer = torch.optim.LBFGS(
            params,
            max_iter=max_iter,
            line_search_fn="strong_wolfe",
            tolerance_grad=1e-9,
            tolerance_change=1e-12,
        )

        def closure():
            optimizer.zero_grad()
            loss = self.get_loss_per_subject(
                self(t, participant_ids), actual, participant_ids, penalty_weight
            ).sum()
            loss.backward()
            return loss

        optimizer.step(closure)

    def to_params(self) -> ModelParams:
        with torch.no_grad():
            return ModelParams(
                **{
                    name: torch.as_tensor(getattr(self, name)).numpy().astype(float)
                    for name in (
                        *self.population_param_names,
                        "time_shifts",
                        "acceleration_factors",
                    )
                }
            )
