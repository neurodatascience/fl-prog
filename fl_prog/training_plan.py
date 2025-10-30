import json
from functools import wraps
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
import torch.optim as optim
from fedbiomed.common.data import DataManager
from fedbiomed.common.models import TorchModel
from fedbiomed.common.training_plans import TorchTrainingPlan


from fl_prog.model import LogisticRegressionModelWithShift
from fl_prog.utils.constants import FNAME_NODE_CONFIG, FNAME_WEIGHTS


class FLProgTrainingPlan(TorchTrainingPlan):
    col_subject_id: str
    col_time: str
    cols_biomarker: list[str]

    config_path = Path(FNAME_NODE_CONFIG)
    default_config = {"n_participants": 0}

    class LocalParamsTorchModel(TorchModel):
        weights_path = Path(FNAME_WEIGHTS)

        def __init__(self, model, param_names: Optional[list[str]] = None):
            super().__init__(model)
            self.param_names = param_names if param_names else []

        def get_weights(
            self, only_trainable: bool = False, exclude_buffers: bool = True
        ):
            w = super().get_weights(only_trainable, exclude_buffers)
            to_save = {
                param_name: w[param_name]
                for param_name in self.param_names
                if param_name in w
            }
            torch.save(to_save, self.weights_path)
            return w

        def set_weights(self, weights):
            if self.weights_path.exists():
                w_dict: dict = torch.load(self.weights_path)
                for key, val in w_dict.items():
                    weights[key] = val
                self.weights_path.unlink()
            super().set_weights(weights)

    def _configure_model_and_optimizer(self, initialize_optimizer: bool = True):
        super()._configure_model_and_optimizer(initialize_optimizer)
        self._model = FLProgTrainingPlan.LocalParamsTorchModel(
            self._model.model,
            ["time_shifts"],
        )
        if not self._model.weights_path.exists():
            self._model.get_weights()

    @staticmethod
    def set_colnames(func):
        @wraps(func)
        def wrapper(self: "FLProgTrainingPlan", *args, **kwargs):
            colnames = self.model_args()["colnames"]
            self.col_subject_id = colnames["col_subject_id"]
            self.col_time = colnames["col_time"]
            self.cols_biomarker = colnames["cols_biomarker"]
            return func(self, *args, **kwargs)

        return wrapper

    def _load_data(self, dataset_path) -> pd.DataFrame:
        df = pd.read_csv(dataset_path, sep="\t")
        return df

    def model(self) -> LogisticRegressionModelWithShift:
        # for type annotation only
        return super().model()

    @set_colnames
    def init_model(self):
        model_args: dict = self.model_args()
        tag = model_args["tag"]
        kwargs: dict = model_args["lr_with_shift"]
        if self.config_path.exists():
            with open(self.config_path) as f:
                config = json.load(f)
            kwargs.update(config[tag])
        else:
            kwargs.update(self.default_config)
        model = LogisticRegressionModelWithShift(**kwargs)
        return model

    def init_dependencies(self):
        deps = [
            "import json",
            "from functools import wraps",
            "from pathlib import Path",
            "from typing import Optional",
            "import pandas as pd",
            "import torch",
            "import torch.optim as optim",
            "from fedbiomed.common.data import DataManager",
            "from fedbiomed.common.models import TorchModel",
            "from fl_prog.model import LogisticRegressionModelWithShift",
            "from fl_prog.utils.constants import FNAME_NODE_CONFIG, FNAME_WEIGHTS",
        ]
        return deps

    def init_optimizer(self, optimizer_args: dict) -> optim.Optimizer:
        optimizer = optim.Adam(
            list(self.model().parameters()), lr=optimizer_args.get("lr", 0.01)
        )
        return optimizer

    @set_colnames
    def training_data(self):
        df = self._load_data(self.dataset_path)
        time_and_subject_ids = df.loc[:, [self.col_time, self.col_subject_id]]
        biomarkers = df.loc[:, self.cols_biomarker]
        return DataManager(dataset=time_and_subject_ids, target=biomarkers)

    def training_step(self, data, target):
        output = self.model()(data[:, 0], data[:, 1])
        loss = self.model().get_loss(output, target)
        return loss
