import torch
import torch.nn as nn


class LogisticRegressionModelWithShift(nn.Module):
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

        self.log_k_values = nn.Parameter(torch.randn(self.n_features))  # k: slope
        self.x0_values = nn.Parameter(torch.randn(self.n_features))  # x0: midpoint
        self.time_shifts = nn.Parameter(torch.zeros(self.n_participants))
        self.log_sigma_sq = nn.Parameter(torch.tensor(0.0))

        if with_shift:
            self.vertical_shifts = nn.Parameter(torch.zeros(self.n_features))
        if with_scaling:
            self.log_scaling_factors = nn.Parameter(torch.zeros(self.n_features))

    def get_k_values(self, log_k_values: torch.Tensor) -> torch.Tensor:
        # constrain k_values to be positive
        # log_k_values (parameters to optimize) can be any real number
        return torch.exp(log_k_values)

    def get_scaling_factors(self, log_scaling_factors: torch.Tensor) -> torch.Tensor:
        # constrain scaling factors to be positive
        return torch.exp(log_scaling_factors)

    def get_sigma(self, log_sigma_sq: torch.Tensor) -> torch.Tensor:
        sigma_sq = torch.exp(log_sigma_sq)
        return torch.sqrt(sigma_sq)

    def forward(self, t: torch.Tensor, participant_ids: torch.Tensor):
        shift = self.time_shifts[participant_ids.to(torch.long)].squeeze(-1)
        shifted_t = t.view(-1) + shift

        linear_combination = self.get_k_values(self.log_k_values) * (
            shifted_t.view(-1, 1) - self.x0_values
        )
        output = torch.sigmoid(linear_combination)
        if self.with_scaling:
            scaling_factors = self.get_scaling_factors(self.log_scaling_factors)
            output = scaling_factors * output
        if self.with_shift:
            output = output + self.vertical_shifts
        return output

    def get_loss(self, predicted, actual):
        sigma_sq = torch.exp(self.log_sigma_sq)
        loss = torch.sum(
            (actual - predicted) ** 2 / (2 * sigma_sq)
            + 0.5 * torch.log(2 * torch.pi * sigma_sq)
        )
        loss += self.lambda_ * torch.sum(self.time_shifts**2)
        return loss
