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
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from fl_prog.utils.constants import DNAME_LATEST
from fl_prog.utils.io import DEFAULT_DPATH_DATA, DEFAULT_DPATH_RESULTS, DPATH_PROJECT

# ===== Select results to plot =====
# change config file instead of editing this notebook directly
fpath_notebook_config = DPATH_PROJECT / "notebooks" / "model_fits_synthetic.json"
if fpath_notebook_config.exists():
    notebook_config = json.loads(fpath_notebook_config.read_text())
else:
    notebook_config = {}
TAG = notebook_config.get("tag", "iid_with_acceleration")
dname_data_date = notebook_config.get("dname_data_date", DNAME_LATEST)
dname_results_date = notebook_config.get("dname_results_date", DNAME_LATEST)

THEME = "paper"  # paper, talk

np.set_printoptions(precision=3, suppress=True)
plt.rcParams["svg.fonttype"] = "none"
sns.set_theme(THEME, style="ticks")

TIME_LABEL = "Time"
BIOMARKER_LABEL = "Biomarker value"
N_SUBJECTS = 1  # number of subjects per site to plot

data_dir = DEFAULT_DPATH_DATA / dname_data_date / TAG
results_dir = DEFAULT_DPATH_RESULTS / dname_results_date / TAG

fpath_json_data = data_dir / f"{TAG}.json"
json_data = json.loads(fpath_json_data.read_text())
print(f"fpath_json_data: {fpath_json_data}")

try:
    fpath_json_results = results_dir / f"{TAG}-estimated_params.json"
    results_dict = json.loads(fpath_json_results.read_text())
    print(f"fpath_json_results: {fpath_json_results}")
except FileNotFoundError:
    print("Results not available")

n_biomarkers = json_data["settings"]["n_biomarkers"]

params = json_data["params"]
k_values = np.array(params["k_values"])
x0_values = np.array(params["x0_values"])
vertical_shifts = np.array(params["vertical_shifts"])
scaling_factors = np.array(params["scaling_factors"])
time_shifts = np.concatenate(params["time_shifts"])
acceleration_factors = np.concatenate(params["acceleration_factors"])
sigmas = params["sigmas"]

n_sites = len(json_data["settings"]["n_subjects_all"])

df = pd.concat(
    {
        i + 1: pd.read_csv(data_dir / f"{TAG}-{i + 1}.tsv", sep="\t")
        for i in range(n_sites)
    },
    axis="index",
    names=["site", "tmp"],
)
df = df.reset_index(level="tmp", drop=True).reset_index()
# df = pd.read_csv(data_dir / f"{TAG}-merged.tsv", sep="\t")
display(df)


def save_fig(fig: sns.FacetGrid | plt.Figure, fname, extension="svg", **kwargs):
    kwargs_default = {"bbox_inches": "tight", "dpi": 300}
    kwargs_default.update(kwargs)

    fpath: Path = (results_dir / fname).with_suffix(f".{extension}")
    fpath.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(fpath, **kwargs_default)
    print(f"Saved figure to {fpath}")


# %%
# # horizontal
# fig_data, axes = plt.subplots(
#     ncols=2, figsize={"paper": (6, 2.5), "talk": (8, 4)}[THEME]
# )

# vertical
fig_data, axes = plt.subplots(
    nrows=2, figsize={"paper": (2.5, 4), "talk": (4, 8)}[THEME]
)


for i_ax, ax in enumerate(axes):
    ax: plt.Axes

    n_subjects_per_site = {}
    for i_subject, (subject, df_subject) in enumerate(df.groupby("subject")):
        site = df_subject["site"].unique().item()
        if site not in n_subjects_per_site:
            n_subjects_per_site[site] = 0

        if n_subjects_per_site[site] >= N_SUBJECTS:
            continue

        for i_biomarker, biomarker in enumerate(json_data["cols"]["cols_biomarker"]):
            ax.plot(
                (
                    (
                        df_subject[json_data["cols"]["col_timepoint"]]
                        + time_shifts[i_subject]
                    )
                    * acceleration_factors[i_subject]
                    if i_ax == 0
                    else df_subject[json_data["cols"]["col_timepoint"]]
                ),
                df_subject[biomarker],
                marker="oXxp*"[site - 1],
                color=f"C{i_biomarker}",
                # alpha=0.1,
                alpha=0.3,
            )
        n_subjects_per_site[site] += 1

    for i_biomarker, (k, x0, vertical_shift, scaling_factor) in enumerate(
        zip(k_values, x0_values, vertical_shifts, scaling_factors)
    ):
        t = np.linspace(0, 1.5, 100)
        y = 1 / (1 + np.exp(-k * (t - x0))) * scaling_factor + vertical_shift
        ax.plot(t, y, color=f"C{i_biomarker}", linestyle="--", alpha=1)

    ax.set_xlabel(TIME_LABEL)
    ax.set_ylabel(BIOMARKER_LABEL)

axes[0].set_title("Before time shift and acceleration")
axes[1].set_title("After time shift and acceleration")
fig_data.tight_layout()

XLIM = axes[0].get_xlim()
YLIM = axes[0].get_ylim()

# %%
save_fig(fig_data, f"{TAG}-data-{THEME}-{N_SUBJECTS}subjects", extension="svg")


# %%
def check_model_fit(
    estimated_k_values,
    estimated_x0_values,
    estimated_vertical_shifts,
    estimated_scaling_factors,
    estimated_time_shifts,
    estimated_acceleration_factors,
    estimated_sigma,
    align_x=False,
    ax: plt.Axes | None = None,
    **kwargs,
):
    if ax is None:
        _, ax = plt.subplots()

    estimated_k_values = np.array(estimated_k_values)
    estimated_x0_values = np.array(estimated_x0_values)
    estimated_vertical_shifts = np.array(estimated_vertical_shifts)
    estimated_scaling_factors = np.array(estimated_scaling_factors)
    estimated_sigma = np.array(estimated_sigma)

    estimated_time_shifts = np.hstack(list(estimated_time_shifts.values()))
    estimated_acceleration_factors = np.hstack(
        list(estimated_acceleration_factors.values())
    )

    mean_x_value_difference = (x0_values - estimated_x0_values).mean()

    for i_biomarker, (k, x0, vertical_shift, scaling_factor) in enumerate(
        zip(k_values, x0_values, vertical_shifts, scaling_factors)
    ):
        # ground truth
        t = np.linspace(-1, 2, 100)
        y = 1 / (1 + np.exp(-k * (t - x0))) * scaling_factor + vertical_shift
        ax.plot(t, y, color=f"C{i_biomarker}", linestyle="--", alpha=0.8)

        if align_x:
            offset = mean_x_value_difference
        else:
            offset = 0.0

        # simulations
        y_pred = (
            1
            / (
                1
                + np.exp(
                    -estimated_k_values[i_biomarker]
                    * (t - offset - estimated_x0_values[i_biomarker])
                )
            )
            * estimated_scaling_factors[i_biomarker]
            + estimated_vertical_shifts[i_biomarker]
        )
        ax.plot(
            t,
            y_pred,
            color=f"C{i_biomarker}",
            linestyle="-",
            alpha=0.8,
        )

        ax.set_xlabel(TIME_LABEL)
        ax.set_ylabel(BIOMARKER_LABEL)

        try:
            ax.set_xlim(XLIM)
            ax.set_ylim(YLIM)
        except NameError:
            pass

    print("===== k values =====")
    print(k_values)
    print(estimated_k_values)
    print("===== x0 values =====")
    print(x0_values)
    print(estimated_x0_values)
    print("===== x value offsets =====")
    print(x0_values - estimated_x0_values)
    print("===== x value offset std =====")
    print((x0_values - estimated_x0_values).std())
    print("===== vertical shifts =====")
    print(vertical_shifts)
    print(estimated_vertical_shifts)
    print("===== scaling factors =====")
    print(scaling_factors)
    print(estimated_scaling_factors)
    print("===== time shift correlation =====")
    print(np.corrcoef(time_shifts, estimated_time_shifts)[0, 1])
    print("===== acceleration factor correlation =====")
    print(np.corrcoef(acceleration_factors, estimated_acceleration_factors)[0, 1])
    print("===== sigma =====")
    print(sigmas)
    print(estimated_sigma)


# # horizontal
# fig_model_fit, axes = plt.subplots(
#     figsize={"paper": (6, 2.5), "talk": (8, 4)}[THEME], ncols=2
# )

# vertical
fig_model_fit, axes = plt.subplots(
    nrows=len(results_dict["results"]),
    figsize={"paper": (2.5, 4), "talk": (4, 8)}[THEME],
    squeeze=False,
)

for ax, setup in zip(axes.flatten(), results_dict["results"].keys()):
    check_model_fit(
        **results_dict["results"][setup],
        align_x=True,
        ax=ax,
    )
    ax.set_title(setup.capitalize())

fig_model_fit.tight_layout()

# %%
save_fig(fig_model_fit, f"{TAG}-model_fit-{THEME}", extension="svg")

# %%
N_SUBJECTS_FITS = 2

# vertical
fig_fit_data, axes = plt.subplots(ncols=3, figsize=(8, 2))

for i_ax, ax in enumerate(axes):
    ax: plt.Axes

    match i_ax:
        case 0:
            time_shifts_to_plot = time_shifts
            acceleration_factors_to_plot = acceleration_factors
        case 1:
            time_shifts_to_plot = np.hstack(
                list(
                    results_dict["results"]["centralized"][
                        "estimated_time_shifts"
                    ].values()
                )
            )
            acceleration_factors_to_plot = np.hstack(
                list(
                    results_dict["results"]["centralized"][
                        "estimated_acceleration_factors"
                    ].values()
                )
            )
        case 2:
            time_shifts_to_plot = np.hstack(
                list(
                    results_dict["results"]["federated"][
                        "estimated_time_shifts"
                    ].values()
                )
            )
            acceleration_factors_to_plot = np.hstack(
                list(
                    results_dict["results"]["federated"][
                        "estimated_acceleration_factors"
                    ].values()
                )
            )

    n_subjects = 0
    for i_subject, (subject, df_subject) in enumerate(df.groupby("subject")):
        site = df_subject["site"].unique().item()

        if n_subjects >= N_SUBJECTS_FITS:
            continue

        print(subject)

        for i_biomarker, biomarker in enumerate(json_data["cols"]["cols_biomarker"]):
            ax.plot(
                (
                    (
                        df_subject[json_data["cols"]["col_timepoint"]]
                        + time_shifts_to_plot[i_subject]
                    )
                    * acceleration_factors_to_plot[i_subject]
                ),
                df_subject[biomarker],
                marker="oXxp*"[site - 1],
                color=f"C{i_biomarker}",
                # alpha=0.1,
                alpha=0.2,
            )
        n_subjects += 1

    for i_biomarker, (k, x0, vertical_shift, scaling_factor) in enumerate(
        zip(k_values, x0_values, vertical_shifts, scaling_factors)
    ):
        t = np.linspace(-0.5, 1.5, 100)
        y = 1 / (1 + np.exp(-k * (t - x0))) * scaling_factor + vertical_shift
        ax.plot(t, y, color=f"C{i_biomarker}", linestyle="--", alpha=1)

    ax.set_xlabel(TIME_LABEL)
    ax.set_ylabel(BIOMARKER_LABEL)

axes[0].set_title("Ground truth")
axes[1].set_title("Centralized")
axes[2].set_title("Federated")
fig_fit_data.tight_layout()

for ax in axes:
    ax.set_xlim(-0, 0.5)
