# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: fl-prog (3.10.18)
#     language: python
#     name: python3
# ---

# %%
import json
import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from fl_prog.utils.constants import DNAME_LATEST

np.set_printoptions(precision=3, suppress=True)

N_ROUNDS = 10
N_UPDATES = 70
BATCH_SIZE = 100000
LEARNING_RATE = 0.01

data_dir = Path(os.environ["DPATH_DATA"]) / DNAME_LATEST

json_data = json.loads((data_dir / "simulated_data.json").read_text())

n_biomarkers = json_data["settings"]["n_biomarkers"]
params = json_data["params"]
k_values = np.array(params["k_values"])
x0_values = np.array(params["x0_values"])
time_shifts = np.array(params["time_shifts"])

df = pd.read_csv(data_dir / "simulated_data-merged.tsv", sep="\t")
display(df)

fig, axes = plt.subplots(ncols=2)
for i_ax, ax in enumerate(axes):
    ax: plt.Axes

    for i_subject, (subject, df_subject) in enumerate(df.groupby("subject")):
        for i_biomarker, biomarker in enumerate(
            [f"biomarker_{i}" for i in range(n_biomarkers)]
        ):
            ax.plot(
                (
                    df_subject["timepoint"] + time_shifts[i_subject]
                    if i_ax == 1
                    else df_subject["timepoint"]
                ),
                df_subject[biomarker],
                marker="oXxp*"[i_subject % 5],
                color=f"C{i_biomarker}",
                alpha=0.1,
            )

    for i_biomarker, (k, x0) in enumerate(zip(k_values, x0_values)):
        t = np.linspace(0, 1, 100)
        y = 1 / (1 + np.exp(-k * (t - x0)))
        ax.plot(t, y, color=f"C{i_biomarker}", linestyle="--", alpha=1)


def check_model_fit(
    estimated_k_values, estimated_x0_values, estimated_time_shifts, align_x=False
):
    _, ax = plt.subplots()
    ax: plt.Axes

    x_value_differences = x0_values - estimated_x0_values

    for i_biomarker, (k, x0) in enumerate(zip(k_values, x0_values)):
        t = np.linspace(-1, 2, 100)
        y = 1 / (1 + np.exp(-k * (t - x0)))
        ax.plot(t, y, color=f"C{i_biomarker}", linestyle="--", alpha=0.8)

        y_pred = torch.sigmoid(
            torch.tensor(
                estimated_k_values[i_biomarker] * (t - estimated_x0_values[i_biomarker])
            )
        ).numpy()
        ax.plot(
            t + x_value_differences[i_biomarker] if align_x else t,
            y_pred,
            color=f"C{i_biomarker}",
            linestyle="-",
            alpha=0.8,
        )

    print("===== k values =====")
    print(k_values)
    print(estimated_k_values)
    print("===== x0 values =====")
    print(x0_values)
    print(estimated_x0_values)
    print("===== x value differences =====")
    print(x0_values - estimated_x0_values)
    print("===== time shifts =====")
    print(time_shifts)
    print(estimated_time_shifts)
    if len(estimated_time_shifts) > 0:
        print("===== time shift offset difference =====")
        if len(time_shifts) < len(estimated_time_shifts):
            warnings.warn(
                "Number of estimated time shifts is larger than the number of true time "
                "shifts. Clipping estimated time shifts."
            )
            estimated_time_shifts = estimated_time_shifts[: len(time_shifts)]

        print((time_shifts - estimated_time_shifts).std())


# %%
import importlib

from fl_prog.aggregator import SelectiveFedAverage

import fl_prog.training_plan
from fl_prog.training_plan import FLProgTrainingPlan
from fl_prog.utils.io import working_directory

importlib.reload(fl_prog.training_plan)


fedbiomed_dir = "../fedbiomed"
with working_directory(fedbiomed_dir):
    from fedbiomed.researcher.federated_workflows import Experiment

    experiment = Experiment(
        nodes=["NODE_CENTRALIZED"],
        tags=["centralized"],
        training_plan_class=FLProgTrainingPlan,
        model_args={
            "colnames": {
                "col_subject_id": "subject",
                "col_time": "timepoint",
                "cols_biomarker": [f"biomarker_{i}" for i in range(n_biomarkers)],
            },
            "lr_with_shift": {
                "n_features": n_biomarkers,
            },
        },
        round_limit=N_ROUNDS,
        training_args={
            "num_updates": N_UPDATES,
            "loader_args": {"batch_size": BATCH_SIZE, "shuffle": False},
            "optimizer_args": {"lr": LEARNING_RATE},
        },
        aggregator=SelectiveFedAverage(["time_shifts"]),
        node_selection_strategy=None,
    )
    experiment.run()

# %%
fbm_model = experiment.training_plan().model()
final_params = experiment.aggregated_params()[N_ROUNDS - 1]["params"]

check_model_fit(
    fbm_model.get_k_values(final_params["log_k_values"]).data.numpy(),
    final_params["x0_values"].data.numpy(),
    final_params["time_shifts"].data.numpy(),
    align_x=True,
)

# %%
from torch import optim

from fl_prog.model import LogisticRegressionModelWithShift

model = LogisticRegressionModelWithShift(
    len(set(df["subject"])), n_features=n_biomarkers
)

# Define the optimizer
optimizer = optim.Adam(list(model.parameters()), lr=LEARNING_RATE)

# Optimization loop
num_epochs = (
    N_UPDATES * N_ROUNDS
)  # Increased epochs for potentially harder optimization
for epoch in range(num_epochs):
    # Forward pass
    # Pass both timestamps and sample indices to the model
    model_predictions = model(
        torch.tensor(df["timepoint"]), torch.tensor(df["subject"])
    )
    loss = model.get_loss(
        model_predictions,
        torch.tensor(
            df[[f"biomarker_{i}" for i in range(n_biomarkers)]].values,
        ),
    )

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print loss periodically
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

check_model_fit(
    model.get_k_values(model.log_k_values).data.numpy(),
    model.x0_values.data.numpy(),
    model.time_shifts.data.numpy(),
    align_x=True,
)
