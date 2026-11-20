from functools import wraps
from typing import ClassVar

from fedbiomed.common.datamanager import DataManager
from fedbiomed.common.dataset import TabularDataset
from fedbiomed.common.optimizers.declearn import AdamModule
from fedbiomed.common.optimizers.optimizer import Optimizer
from fedbiomed.common.training_plans import TorchTrainingPlan
from torch import optim

from fl_prog.model import LogisticRegressionModelWithShift
from fl_prog.scaffold import LocalAwareScaffoldClientModule


class FLProgTrainingPlan(TorchTrainingPlan):
    col_subject_id: str
    col_time: str
    cols_biomarker: list[str]

    default_config: ClassVar = {"n_participants": 0}

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
        kwargs: dict = model_args["lr_with_shift"]
        kwargs["n_participants"] = model_args["node_specific_args"][
            "n_participants"
        ].get(self.node_id(), 0)
        model = LogisticRegressionModelWithShift(**kwargs)
        return model

    def tag_parameters(self, name):
        tags = set()
        if name in (
            "parametrizations.time_shifts.original",
            "parametrizations.acceleration_factors.original",
        ):
            tags.update({"local", "persistent"})
        return tags

    def init_dependencies(self):
        deps = [
            "from functools import wraps",
            "from typing import ClassVar",
            "import torch.optim as optim",
            "from fedbiomed.common.datamanager import DataManager",
            "from fedbiomed.common.dataset import TabularDataset",
            "from fedbiomed.common.optimizers.declearn import AdamModule",
            "from fedbiomed.common.optimizers.optimizer import Optimizer",
            "from fl_prog.model import LogisticRegressionModelWithShift",
            "from fl_prog.scaffold import LocalAwareScaffoldClientModule",
        ]
        return deps

    def init_optimizer(self, optimizer_args: dict) -> optim.Optimizer:
        learning_rate = optimizer_args.get("lr", 0.01)
        if optimizer_args.get("aggregator_name") == "scaffold_adam":
            local_names = [
                name
                for name, _ in self.model().named_parameters()
                if "local" in self.tag_parameters(name)
            ]
            optimizer = Optimizer(
                lr=learning_rate,
                # module order matters
                modules=[LocalAwareScaffoldClientModule(local_names), AdamModule()],
            )
        elif optimizer_args.get("optimizer_name") == "sgd":
            # plain SGD is required by Fed-BioMed's native Scaffold aggregator
            model = self.model()
            local_lr_multiplier = optimizer_args.get("local_lr_multiplier")
            if local_lr_multiplier is None:
                param_groups = [{"params": list(model.parameters())}]
            else:
                # use a separate learning rate for local params since their
                # gradients are ~1/n_participants smaller than the shared ones
                is_local = {
                    name: "local" in self.tag_parameters(name)
                    for name, _ in model.named_parameters()
                }
                param_groups = [
                    {
                        "params": [
                            p for n, p in model.named_parameters() if not is_local[n]
                        ]
                    },
                    {
                        "params": [
                            p for n, p in model.named_parameters() if is_local[n]
                        ],
                        "lr": learning_rate
                        * max(model.n_participants, 1)
                        * float(local_lr_multiplier),
                    },
                ]
            optimizer = optim.SGD(param_groups, lr=learning_rate)
        else:
            optimizer = optim.Adam(
                list(self.model().parameters()),
                lr=learning_rate,
            )
        return optimizer

    def update_optimizer_args(self) -> dict:
        # With native Scaffold, Fed-BioMed reports one learning rate per model
        # parameter, but the researcher-side aggregator expects only the
        # non-local (aggregated) ones -> drop local params (they have no
        # control variates, so their learning rate is not needed)
        optimizer_args = super().update_optimizer_args()
        lr = optimizer_args.get("lr")
        if isinstance(lr, dict):
            optimizer_args["lr"] = {
                name: value
                for name, value in lr.items()
                if "local" not in self.tag_parameters(name)
            }
        return optimizer_args

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

        # FedProx proximal term restricted to shared (non-local) parameters
        # (Fed-BioMed's built-in FedProx also penalizes local per-subject params)
        mu = (self._optimizer_args or {}).get("fedprox_shared_mu")
        if mu:
            init_params = self._model.init_params  # snapshot at round start
            prox = sum(
                ((param - init_params[name]) ** 2).sum()
                for name, param in self.model().named_parameters()
                if param.requires_grad and "local" not in self.tag_parameters(name)
            )
            loss = loss + float(mu) / 2 * prox
        return loss
