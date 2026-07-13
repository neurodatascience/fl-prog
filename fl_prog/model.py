from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize


class Positive(nn.Module):
    """Constrain a parameter to be positive using the softplus function."""

    def forward(self, param):
        return F.softplus(param)

    def right_inverse(self, constrained_param):
        return constrained_param + torch.log(-torch.expm1(-constrained_param))


class LogisticRegressionModelWithShift(nn.Module):

    parametrization_dict = {
        "k_values": Positive(),
        "sigma": Positive(),
        "scaling_factors": Positive(),
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
        lambda_: float = 1.0,
        with_shift=False,
        with_scaling=False,
    ):
        super().__init__()

        if lambda_ < 0:
            raise ValueError(f"lambda_ must be non-negative, got {lambda_}")

        if len(expected_time_shift_range) != 2:
            raise ValueError(
                f"expected_time_shift_range must have length 2, got {len(expected_time_shift_range)}"
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
        self.lambda_ = lambda_
        self.with_shift = with_shift
        self.with_scaling = with_scaling

        # slopes
        self.k_values = nn.Parameter(torch.rand(self.n_features) + starting_k_value)

        # midpoints
        self.x0_values = nn.Parameter(
            torch.randn(self.n_features)
            * (expected_time_shift_diff * self.x0_init_std_fraction)
            + expected_time_shift_middle
        )

        self.time_shifts = nn.Parameter(
            torch.randn(self.n_participants) + expected_time_shift_middle
        )

        self.sigma = nn.Parameter(torch.ones(self.n_features) * 0.5)

        # does not work well
        self.vertical_shifts = torch.zeros(self.n_features)
        self.scaling_factors = torch.ones(self.n_features)
        if with_shift:
            self.vertical_shifts = nn.Parameter(self.vertical_shifts)
        if with_scaling:
            self.scaling_factors = nn.Parameter(self.scaling_factors)

        # constrain some parameters
        for param_name, parametrization in self.parametrization_dict.items():
            if isinstance(getattr(self, param_name), nn.Parameter):
                parametrize.register_parametrization(self, param_name, parametrization)

    @classmethod
    def get_k_values(cls, unparametrized_k_values: torch.Tensor) -> torch.Tensor:
        return cls.parametrization_dict["k_values"](unparametrized_k_values)

    @classmethod
    def get_scaling_factors(
        cls, unparametrized_scaling_factors: torch.Tensor
    ) -> torch.Tensor:
        return cls.parametrization_dict["scaling_factors"](
            unparametrized_scaling_factors
        )

    @classmethod
    def get_sigma(cls, unparametrized_sigma: torch.Tensor) -> torch.Tensor:
        return cls.parametrization_dict["sigma"](unparametrized_sigma)

    def forward(self, t: torch.Tensor, participant_ids: torch.Tensor):
        shift = self.time_shifts[participant_ids.to(torch.long)].squeeze(-1)
        shifted_t = t.view(-1) + shift

        linear_combination = self.k_values * (shifted_t.view(-1, 1) - self.x0_values)
        output = torch.sigmoid(linear_combination)
        if self.with_scaling:
            scaling_factors = self.scaling_factors
            output = scaling_factors * output
        if self.with_shift:
            output = output + self.vertical_shifts
        return output

    def get_loss(self, predicted, actual):
        sigma_sq = self.sigma**2

        # negative Gaussian log-likelihood
        loss = torch.sum(
            (actual - predicted) ** 2 / (2 * sigma_sq)
            + 0.5 * torch.log(2 * torch.pi * sigma_sq)
        )

        # penalize time shifts that are outside the expected range
        # equivalent to L2 regularization if expected_time_shift_range is (0, 0)
        loss += self.lambda_ * torch.sum(
            (torch.relu(self.expected_time_shift_range[0] - self.time_shifts) ** 2)
            + (torch.relu(self.time_shifts - self.expected_time_shift_range[1]) ** 2)
        )

        return loss
