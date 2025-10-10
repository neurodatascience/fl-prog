from typing import Iterable
import pandas as pd

import torch
import torch.nn as nn

from pathlib import Path


class LogisticRegressionModelWithShift(nn.Module):
    def __init__(self, n_participants: int, n_features: int):
        super().__init__()

        self.n_participants = n_participants
        self.n_features = n_features

        self.unique_participant_ids = None

        self.log_k_values = nn.Parameter(torch.randn(self.n_features))
        self.x0_values = nn.Parameter(torch.randn(self.n_features))
        self.time_shifts = nn.Parameter(torch.zeros(self.n_participants))
        self.log_sigma_sq = nn.Parameter(torch.tensor(0.0))

    def get_k_values(self, log_k_values: torch.Tensor) -> torch.Tensor:
        # constrain k_values to be positive
        # log_k_values (parameters to optimize) can be any real number
        return torch.exp(log_k_values)

    def _check_participant_ids(self, participant_ids):
        if isinstance(participant_ids, torch.Tensor):
            participant_ids = participant_ids.tolist()
        if self.unique_participant_ids is not None:
            if len(self.unique_participant_ids) != len(self.time_shifts):
                raise ValueError(
                    "Number of unique participants IDs "
                    f"({len(self.unique_participant_ids)}) does not match"
                    f" number of time shifts ({len(self.time_shifts)})"
                )
            if unknown_participants := set(participant_ids) - set(
                self.unique_participant_ids
            ):
                raise ValueError(
                    f"participant_ids contain unknown participant IDs:"
                    f" {unknown_participants}. Make sure every call to forward() uses "
                    "the entire dataset. Known participant IDs are: "
                    f"{self.unique_participant_ids}"
                )

    def _set_unique_participant_ids(self, participant_ids: Iterable):
        if self.unique_participant_ids is None:
            self.unique_participant_ids = (
                pd.Series(participant_ids).sort_values().drop_duplicates().tolist()
            )

    def _get_time_shift_idx(self, participant_ids):
        self._set_unique_participant_ids(participant_ids)
        self._check_participant_ids(participant_ids)
        return [self.unique_participant_ids.index(p) for p in participant_ids]

    def forward(self, t: torch.Tensor, participant_ids):
        shifted_t = (
            t.squeeze() + self.time_shifts[self._get_time_shift_idx(participant_ids)]
        )
        linear_combination = self.get_k_values(self.log_k_values) * (
            shifted_t.view(-1, 1) - self.x0_values
        )
        output = torch.sigmoid(linear_combination)
        return output

    def get_loss(self, predicted, actual):
        sigma_sq = torch.exp(self.log_sigma_sq)
        loss = torch.sum(
            (actual - predicted) ** 2 / (2 * sigma_sq)
            + 0.5 * torch.log(2 * torch.pi * sigma_sq)
        )
        loss += torch.mean(torch.abs(self.time_shifts))
        return loss
