# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: fl-prog (3.10.18.final.0)
#     language: python
#     name: python3
# ---

# %%
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from fl_prog.utils.constants import DNAME_LATEST
from fl_prog.utils.io import DEFAULT_DPATH_RESULTS, DPATH_PROJECT

# ===== Select results to plot =====
# change config file instead of editing this notebook directly
fpath_notebook_config = DPATH_PROJECT / "notebooks" / "model_fits_adni.json"
if fpath_notebook_config.exists():
    notebook_config = json.loads(fpath_notebook_config.read_text())
else:
    notebook_config = {}

TAG = notebook_config.get("tag", "adni_iid")
dname_results_date = notebook_config.get("dname_results_date", DNAME_LATEST)
fpath_adni_merge = Path(os.environ.get("ADNI_MERGE_FILE"))

dpath_results = DEFAULT_DPATH_RESULTS / dname_results_date / TAG
fpath_json = dpath_results / f"{TAG}-estimated_params.json"
print(fpath_json)

json_content = json.loads(fpath_json.read_text())
col_subject = json_content["settings"]["config"]["cols"]["col_subject"]
cols_biomarker = json_content["settings"]["config"]["cols"]["cols_biomarker"]
col_months = "months"  # scaled back to original months

subjects_by_node = json_content["settings"]["config"]["subjects_by_node"]
time_scaling_factor = json_content["settings"]["config"]["settings"]["config"][
    "max_time"
]
min_max_by_measure = json_content["settings"]["config"]["settings"]["config"][
    "min_max_by_measure"
]
flipped = json_content["settings"]["config"]["settings"]["config"].get("flip", True)
results = json_content["results"]
# results


def save_fig(fig: sns.FacetGrid | plt.Figure, fname, extension="svg", **kwargs):
    kwargs_default = {"bbox_inches": "tight", "dpi": 300}
    kwargs_default.update(kwargs)

    fpath: Path = (dpath_results / fname).with_suffix(f".{extension}")
    fpath.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(fpath, **kwargs_default)
    print(f"Saved figure to {fpath}")


# %%
import pandas as pd

from fl_prog.utils.constants import NODE_PREFIX


def get_time_shift_by_subject(
    estimated_time_shifts, subjects_by_node, time_scaling_factor=1.0
):

    dfs_time_shifts = []
    for node_id, time_shifts in estimated_time_shifts.items():
        df_time_shifts = pd.DataFrame(
            {
                "participant_id": [
                    str(subject)
                    for subject in subjects_by_node[node_id.removeprefix(NODE_PREFIX)]
                ],
                # "participant_id": subjects_by_node[node_id.removeprefix(NODE_PREFIX)],
                "estimated_time_shift": time_shifts,
            }
        )
        df_time_shifts["estimated_time_shift"] = (
            df_time_shifts["estimated_time_shift"].astype(float) * time_scaling_factor
        )
        dfs_time_shifts.append(df_time_shifts)

    time_shift_by_subject = (
        pd.concat(dfs_time_shifts, ignore_index=True)
        .set_index("participant_id")
        .squeeze()
        .to_dict()
    )
    return time_shift_by_subject


time_shift_by_subject = get_time_shift_by_subject(
    results["federated"]["estimated_time_shifts"],
    subjects_by_node,
    time_scaling_factor=time_scaling_factor,
)
# time_shift_by_subject

# print(min(time_shift_by_subject.values()))
# print(max(time_shift_by_subject.values()))
# print(max(time_shift_by_subject.values()) - min(time_shift_by_subject.values()))

# %%
import numpy as np

data_fitted_models = []

for setup in results:
    for biomarker, k, x0, vertical_shift, scaling_factor in zip(
        cols_biomarker,
        results[setup]["estimated_k_values"],
        results[setup]["estimated_x0_values"],
        results[setup]["estimated_vertical_shifts"],
        results[setup]["estimated_scaling_factors"],
    ):
        months = np.linspace(-0.5, 4, 100)
        values = scaling_factor * (
            1 / (1 + np.exp(-k * (months - x0))) + vertical_shift
        )

        biomarker_min, biomarker_max = min_max_by_measure[biomarker]

        data_fitted_models.extend(
            [
                {
                    "setup": setup,
                    "biomarker": biomarker,
                    "x": month * time_scaling_factor,
                    "y": value,
                    "y_transformed": (
                        value * (biomarker_max - biomarker_min) + biomarker_min
                    ),
                }
                for month, value in zip(months, values)
            ]
        )

df_fitted_models = pd.DataFrame(data_fitted_models)
df_fitted_models

# %%
import numpy as np

estimated_time_shifts_federated = pd.Series(
    get_time_shift_by_subject(
        results["federated"]["estimated_time_shifts"],
        subjects_by_node,
        time_scaling_factor=time_scaling_factor,
    )
).sort_index()

if "centralized" in results:
    estimated_time_shifts_centralized = pd.Series(
        get_time_shift_by_subject(
            results["centralized"]["estimated_time_shifts"],
            subjects_by_node,
            time_scaling_factor=time_scaling_factor,
        )
    ).sort_index()
    # estimated_time_shifts_federated = np.concatenate(
    #     list(results["federated"]["estimated_time_shifts"].values())
    # )
    # estimated_time_shifts_centralized = np.concatenate(
    #     list(results["centralized"]["estimated_time_shifts"].values())
    # )

    print("Correlation between estimated time shifts (federated vs centralized):")
    print(
        np.corrcoef(estimated_time_shifts_federated, estimated_time_shifts_centralized)
    )
else:
    print("Centralized results not available (yet)")

# %%
import seaborn as sns

fig_models = sns.relplot(
    data=df_fitted_models,
    x="x",
    y="y",
    # y="y_transformed",
    hue="biomarker",
    style="setup",
    kind="line",
)

for ax in fig_models.axes.flatten():
    # xticks = ax.get_xticks()
    # ax.set_xticks(xticks)
    # ax.set_xticklabels(xticks * time_scaling_factor)
    ax.set_ylabel("Biomarker value")
    ax.set_xlabel("Months")

# %%
save_fig(fig_models, "group_trajectories")

# %%
import pandas as pd

if fpath_adni_merge is None or not fpath_adni_merge.exists():
    raise ValueError(
        "ADNI merge file path is not set or does not exist. Please set the ADNI_MERGE_FILE environment variable to the correct path."
    )

data_specific_cols = ["rid", "visit"]

index_cols = ["participant_id_int", "months_scaled"] + data_specific_cols

df_adni = pd.read_csv(
    f"{json_content['settings']['dpath_data']}/{TAG}-merged.tsv",
    sep="\t",
    index_col=index_cols,
    dtype={col: str for col in data_specific_cols},
)

df_adni_long = pd.melt(
    df_adni.reset_index(),
    id_vars=index_cols,
    var_name="feature",
    value_name="value",
)
df_adni_long["months_shifted"] = df_adni_long[["months_scaled", "rid"]].apply(
    lambda row: (
        (row["months_scaled"] * time_scaling_factor) + time_shift_by_subject[row["rid"]]
    ),
    axis="columns",
)

df_demographics = pd.read_csv(
    fpath_adni_merge,
    low_memory=False,
    dtype={"RID": str, "Month": str},
)
df_demographics = df_demographics.query('RID in @df_adni.index.get_level_values("rid")')
df_demographics_baseline = df_demographics.query('VISCODE == "bl"')
df_adni_long["group"] = df_adni_long["rid"].map(
    df_demographics_baseline.set_index("RID")["DX_bl"].to_dict()
)

df_adni_long

# %%
import seaborn as sns

sns.set_context("notebook")

fig_model_fits = sns.relplot(
    data=df_fitted_models,
    x="x",
    y="y",
    style="setup",
    col="biomarker",
    col_wrap=2,
    kind="line",
    aspect=2,
    facet_kws={"sharey": False},
)

for i_ax, (biomarker, ax) in enumerate(fig_model_fits.axes_dict.items()):
    df_adni_long_subset = df_adni_long.query(f"feature == '{biomarker}'")
    # palette = {
    #     participant_id: sns.color_palette()[i_ax]
    #     for participant_id in df_adni_long_subset["participant_id_int"].unique()
    # }
    hue_order = ["CN", "SMC", "EMCI", "LMCI", "AD"]
    palette = {group: f"C{hue_order.index(group)}" for group in hue_order}
    sns.lineplot(
        data=df_adni_long_subset,
        x="months_shifted",
        y="value",
        hue="group",
        palette=palette,
        hue_order=hue_order,
        style="participant_id_int",
        alpha=0.2,
        ax=ax,
        legend=False,
        rasterized=True,
    )

    ax.set_ylabel("Biomarker value")
    ax.set_xlabel("Months")

    # xticks = ax.get_xticks()
    # ax.set_xticks(xticks)
    # ax.set_xticklabels(xticks * time_scaling_factor)

# %%
save_fig(fig_model_fits, "biomarker_fits")

# %%
import numpy as np

hue_order = ["CN", "SMC", "EMCI", "LMCI", "AD"]

df_time_shifts = estimated_time_shifts_federated.reset_index(
    name="estimated_time_shift"
)

df_time_shifts["group"] = df_time_shifts["index"].map(
    df_demographics_baseline.set_index("RID")["DX_bl"].to_dict()
)
fig_time_shift_kde = sns.displot(
    data=df_time_shifts,
    x="estimated_time_shift",
    hue="group",
    kind="kde",
    hue_order=hue_order,
)

fig_time_shift_box = sns.catplot(
    data=df_time_shifts.sort_values(
        "group", key=lambda x: x.map({group: i for i, group in enumerate(hue_order)})
    ),
    y="estimated_time_shift",
    x="group",
    kind="box",
    hue="group",
    hue_order=hue_order,
)

# ax = fig_time_shift_kde.ax
# xticks = np.asarray(ax.get_xticks())
# ax.set_xticks(xticks)
# ax.set_xticklabels(xticks * time_scaling_factor)

# ax = fig_time_shift_box.ax
# yticks = np.asarray(ax.get_yticks())
# ax.set_yticks(yticks)
# ax.set_yticklabels(yticks * time_scaling_factor)

# %%
"""
################# For ADNI-Calibrated Simulator ########################
#### UNCOMMENT CELL and RUN ONCE #####

# Save diagnosis-specific centralized time-shift distributions so the ADNI
# simulator (simulate_data_disease_onset.py) can use them as empirical priors.

if "centralized" in results:
    df_time_shifts_centralized = estimated_time_shifts_centralized.reset_index(
        name="estimated_time_shift"
    )

    # Match each fitted subject to their baseline ADNI diagnosis.
    df_time_shifts_centralized["DX_bl"] = df_time_shifts_centralized["index"].map(
        df_demographics_baseline.set_index("RID")["DX_bl"].to_dict()
    )

    df_time_shifts_centralized = df_time_shifts_centralized.dropna(
        subset=["estimated_time_shift", "DX_bl"]
    )

    ''' Sanity check
    hue_order = ["CN", "SMC", "EMCI", "LMCI", "AD"]

    fig_time_shift_kde = sns.displot(
        data=df_time_shifts_centralized,
        x="estimated_time_shift",
        hue="DX_bl",
        kind="kde",
        hue_order=hue_order,
    )

    fig_time_shift_box = sns.catplot(
        data=df_time_shifts_centralized.sort_values(
            "DX_bl", key=lambda x: x.map({group: i for i, group in enumerate(hue_order)})
        ),
        y="estimated_time_shift",
        x="DX_bl",
        kind="box",
        hue="DX_bl",
        hue_order=hue_order,
    )
    '''

    distributions_by_dx = {}

    for diagnosis, group in df_time_shifts_centralized.groupby("DX_bl"):
        samples = group["estimated_time_shift"].astype(float).tolist()

        distributions_by_dx[str(diagnosis)] = {
            # Keeping the empirical samples allows the simulator to preserve
            # skewness or multimodality
            "samples": samples,
            "n": len(samples),
            "mean": float(np.mean(samples)),
            "sd": float(np.std(samples, ddof=1)) if len(samples) > 1 else 0.0,
        }
        print(
            f"{diagnosis}: mean={distributions_by_dx[diagnosis]['mean']:.1f}, \
                sd={distributions_by_dx[diagnosis]['sd']:.1f}"
        )

    time_shift_prior_data = {
        "description": (
            "Diagnosis-specific time-shift distributions estimated "
            "by the DPMoSt fit."
        ),
        "source_tag": TAG,
        "source_setup": "centralized",
        "units": "months",
        "distributions_by_dx": distributions_by_dx,
    }

    fpath_time_shift_priors = (
        DPATH_PROJECT / "data" / "adni" / f"time_shift_priors_centralized_{TAG}.json"
    )
    fpath_time_shift_priors.parent.mkdir(parents=True, exist_ok=True)
    fpath_time_shift_priors.write_text(
        json.dumps(time_shift_prior_data, indent=2)
    )

    print(f"Saved centralized time-shift priors to {fpath_time_shift_priors}")
    """

# %%
data_for_df_params = []
for i_round in results["federated"]["aggregated_params"]:
    round_params = results["federated"]["aggregated_params"][i_round]["params"]
    for i_biomarker in range(len(cols_biomarker)):
        for param_name_prefix in (
            "x0_values",
            "parametrizations.k_values.original",
            "parametrizations.sigma.original",
        ):
            data_for_df_params.append(
                {
                    "round": i_round,
                    "i_biomarker": f"{i_biomarker}",
                    "param_value": round_params[param_name_prefix][i_biomarker],
                    "param_name": param_name_prefix,
                }
            )
    # data_for_df_params.append(
    #     {
    #         "round": i_round,
    #         "i_biomarker": "-1",
    #         "param_value": round_params[],
    #         "param_name": "parametrizations.sigma.original",
    #     }
    # )

df_params = pd.DataFrame(data_for_df_params)
# df_params

# %%
import seaborn as sns

sns.relplot(
    data=df_params,
    x="round",
    y="param_value",
    hue="i_biomarker",
    row="param_name",
    kind="line",
    facet_kws={"sharey": False},
)
