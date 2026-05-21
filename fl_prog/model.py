import torch
import torch.nn as nn


class LogisticRegressionModelWithShift(nn.Module):
    def __init__(self, n_participants: int, n_features: int):
        super().__init__()

        self.n_participants = n_participants
        self.n_features = n_features

        self.log_k_values = nn.Parameter(torch.randn(self.n_features))
        self.x0_values = nn.Parameter(torch.randn(self.n_features))
        self.time_shifts = nn.Parameter(torch.zeros(self.n_participants))
        self.log_sigma_sq = nn.Parameter(torch.tensor(0.0))

    def get_k_values(self, log_k_values: torch.Tensor) -> torch.Tensor:
        # constrain k_values to be positive
        # log_k_values (parameters to optimize) can be any real number
        return torch.exp(log_k_values)

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
        return output

    def get_loss(self, predicted, actual):
        sigma_sq = torch.exp(self.log_sigma_sq)
        loss = torch.mean(
            (actual - predicted) ** 2 / (2 * sigma_sq)
            + 0.5 * torch.log(2 * torch.pi * sigma_sq)
        )
        loss += torch.mean(torch.abs(self.time_shifts))
        return loss
