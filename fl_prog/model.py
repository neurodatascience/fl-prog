import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize


class Positive(nn.Module):
    """Constrain a parameter to be positive using the softplus function."""

    # def forward(self, param):
    #     return F.softplus(param)

    # def right_inverse(self, constrained_param):
    #     return constrained_param + torch.log(-torch.expm1(-constrained_param))

    def forward(self, param):
        return torch.exp(param)

    def right_inverse(self, constrained_param):
        return torch.log(constrained_param)


class LogisticRegressionModelWithShift(nn.Module):

    parametrization_dict = {
        "k_values": Positive(),
        "sigma": Positive(),
        "scaling_factors": Positive(),
    }

    def __init__(
        self,
        n_participants: int,
        n_features: int,
        lambda_: float = 0,
        with_shift=False,
        with_scaling=False,
    ):
        super().__init__()

        self.n_participants = n_participants
        self.n_features = n_features
        self.lambda_ = lambda_
        self.with_shift = with_shift
        self.with_scaling = with_scaling

        # slopes
        self.k_values = nn.Parameter(self.get_k_values(torch.randn(self.n_features)))
        # midpoints
        self.x0_values = nn.Parameter(torch.randn(self.n_features))  # standard normal

        self.time_shifts = nn.Parameter(torch.zeros(self.n_participants))
        self.sigma = nn.Parameter(torch.tensor(1.0))

        # does not work well
        if with_shift:
            self.vertical_shifts = nn.Parameter(torch.zeros(self.n_features))
        if with_scaling:
            self.log_scaling_factors = nn.Parameter(torch.zeros(self.n_features))

        for param_name, parametrization in self.parametrization_dict.items():
            if hasattr(self, param_name):
                parametrize.register_parametrization(self, param_name, parametrization)

    def get_k_values(self, unparametrized_k_values: torch.Tensor) -> torch.Tensor:
        return self.parametrization_dict["k_values"](unparametrized_k_values)

    def get_scaling_factors(
        self, unparametrized_scaling_factors: torch.Tensor
    ) -> torch.Tensor:
        return self.parametrization_dict["scaling_factors"](
            unparametrized_scaling_factors
        )

    def get_sigma(self, unparametrized_sigma: torch.Tensor) -> torch.Tensor:
        return self.parametrization_dict["sigma"](unparametrized_sigma)

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
        loss = torch.sum(
            (actual - predicted) ** 2 / (2 * sigma_sq)
            + 0.5 * torch.log(2 * torch.pi * sigma_sq)
        )
        loss += self.lambda_ * torch.sum(self.time_shifts**2)
        return loss
