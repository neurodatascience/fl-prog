from collections.abc import Iterable
from typing import ClassVar

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import parametrize

from fl_prog.utils.constants import Penalty


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
        with_acceleration=False,
        with_scaling=False,
        penalty_time_shifts: Penalty = "l2",
        penalty_acceleration_factors: Penalty = "l1",
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
        self.with_acceleration = with_acceleration
        self.with_scaling = with_scaling
        self.penalty_time_shifts = penalty_time_shifts
        self.penalty_acceleration_factors = penalty_acceleration_factors

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

    def forward(self, t: torch.Tensor, participant_ids: torch.Tensor):
        shift = self.time_shifts[participant_ids.to(torch.long)].squeeze(-1)
        acceleration = self.acceleration_factors[
            participant_ids.to(torch.long)
        ].squeeze(-1)
        shifted_t = t.view(-1) * acceleration + shift

        linear_combination = self.k_values * (shifted_t.view(-1, 1) - self.x0_values)
        output = torch.sigmoid(linear_combination)
        if self.with_scaling:
            scaling_factors = self.scaling_factors
            output = scaling_factors * output
        return output

    def get_loss(self, predicted: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:

        sigma_sq = self.sigma**2

        # trick to handle missing data (NAs)
        # replace missing values by the predicted values, so that the loss is 0 for those entries
        actual_mask = ~torch.isnan(actual)
        actual_no_na = torch.where(actual_mask, actual, predicted)

        # negative Gaussian log-likelihood
        loss = torch.mean(
            (actual_no_na - predicted) ** 2 / (2 * sigma_sq)
            + torch.where(
                actual_mask, 0.5 * torch.log(2 * torch.pi * sigma_sq), torch.tensor(0.0)
            )
        )

        # penalize time shifts that are outside the expected range
        # equivalent to normal L1/L2 regularization if expected_time_shift_range is (0, 0)
        loss += self.lambda_time_shifts * torch.mean(
            self._apply_penalty(
                torch.relu(self.expected_time_shift_range[0] - self.time_shifts),
                self.penalty_time_shifts,
            )
            + self._apply_penalty(
                torch.relu(self.time_shifts - self.expected_time_shift_range[1]),
                self.penalty_time_shifts,
            )
        )

        # also penalize acceleration factors
        loss += self.lambda_acceleration_factors * torch.mean(
            self._apply_penalty(
                torch.log(self.acceleration_factors),
                self.penalty_acceleration_factors,
            )
        )

        return loss
