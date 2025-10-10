import copy
from typing import Optional

import torch
from fedbiomed.researcher.aggregators.fedavg import FedAverage


class SelectiveFedAverage(FedAverage):
    def __init__(self, ignore_list: Optional[list[str]] = None):
        super().__init__()
        self.ignore_list = ignore_list if ignore_list else []

    def aggregate(self, model_params, weights, *args, **kwargs):
        params_copy = copy.deepcopy(model_params)
        for node in model_params.keys():
            for item in self.ignore_list:
                if item in params_copy[node]:
                    params_copy[node].pop(item)
        params = super().aggregate(params_copy, weights, *args, **kwargs)
        for item in self.ignore_list:
            params[item] = torch.tensor([])
        return params
