import json
from functools import wraps
from pathlib import Path

import torch.optim as optim
from fedbiomed.common.datamanager import DataManager
from fedbiomed.common.dataset import TabularDataset
from fedbiomed.common.optimizers.optimizer import Optimizer
from fedbiomed.common.optimizers.declearn import (
    AdamModule,
    RidgeRegularizer,
    ScaffoldClientModule,
)
from fedbiomed.common.training_plans import TorchTrainingPlan


from fl_prog.model import LogisticRegressionModelWithShift
from fl_prog.utils.constants import FNAME_NODE_CONFIG, FNAME_WEIGHTS


class FLProgTrainingPlan(TorchTrainingPlan):
    col_subject_id: str
    col_time: str
    cols_biomarker: list[str]

    config_path = Path(FNAME_NODE_CONFIG)
    default_config = {"n_participants": 0}

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

    def tag_parameters(self, name):
        tags = set()
        if name == "time_shifts":
            tags.update({"local", "persistent"})
        return tags

    def init_dependencies(self):
        deps = [
            "import json",
            "from functools import wraps",
            "from pathlib import Path",
            "import torch.optim as optim",
            "from fedbiomed.common.datamanager import DataManager",
            "from fedbiomed.common.dataset import TabularDataset",
            "from fedbiomed.common.optimizers.optimizer import Optimizer",
            "from fedbiomed.common.optimizers.declearn import AdamModule, RidgeRegularizer, ScaffoldClientModule",
            "from fl_prog.model import LogisticRegressionModelWithShift",
            "from fl_prog.utils.constants import FNAME_NODE_CONFIG, FNAME_WEIGHTS",
        ]
        return deps

    def init_optimizer(self, optimizer_args: dict) -> optim.Optimizer:
        if optimizer_args.get("aggregator_name") == "scaffold":
            optimizer = Optimizer(
                lr=optimizer_args.get("lr", 0.01),
                modules=[AdamModule(), ScaffoldClientModule()],
                regularizers=[RidgeRegularizer()],
            )
        else:
            optimizer = optim.Adam(
                list(self.model().parameters()),
                lr=optimizer_args.get("lr", 0.01),
            )
        return optimizer

    @set_colnames
    def training_data(self):
        return DataManager(
            dataset=TabularDataset(
                input_columns=[self.col_time, self.col_subject_id],
                target_columns=self.cols_biomarker,
            )
        )

    def training_step(self, data, target):
        output = self.model()(data[:, 0], data[:, 1])
        loss = self.model().get_loss(output, target)
        return loss
