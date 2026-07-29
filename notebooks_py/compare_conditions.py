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

import numpy as np
import pandas as pd
import scipy.stats
import torch
from sklearn.metrics import mean_squared_error

from fl_prog.utils.io import DEFAULT_DPATH_DATA, DEFAULT_DPATH_RESULTS

CONDITIONS = [
    "100percent_3sites_50subjects",
    "050percent_3sites_50subjects",
    "000percent_3sites_50subjects",
    # "100percent_2sites_50subjects_20biomarkers",
    # "050percent_2sites_50subjects_20biomarkers",
    # "000percent_2sites_50subjects_20biomarkers",
    # "100percent_5sites_50subjects_20biomarkers",
    # "050percent_5sites_50subjects_20biomarkers",
    # "000percent_5sites_50subjects_20biomarkers",
    # "100percent_8sites_50subjects_20biomarkers",
    # "050percent_8sites_50subjects_20biomarkers",
    # "000percent_8sites_50subjects_20biomarkers",
]
N_ITERATIONS = 50

# DPATH_DATA = get_dpath_latest(DEFAULT_DPATH_DATA)
# DPATH_RESULTS = get_dpath_latest(DEFAULT_DPATH_RESULTS)
DPATH_DATA = DEFAULT_DPATH_DATA / "2026_02_17"
DPATH_RESULTS = DEFAULT_DPATH_RESULTS / "2026_02_17"


def k_value_relative_error(
    k_values_true, k_values_estimated, x0_values_true, x0_values_estimated
):

    # # get scaling factor
    # min_k_value_true = np.abs(k_values_true).min()
    # min_k_value_estimated = np.abs(k_values_estimated).min()

    # # normalize
    # k_values_estimated_normalized = k_values_estimated / min_k_value_estimated
    # k_values_true_normalized = k_values_true / min_k_value_true

    # return np.mean(
    #     np.abs(
    #         (k_values_estimated_normalized - k_values_true_normalized)
    #         / k_values_true_normalized
    #     )
    # )

    return np.mean(np.abs((k_values_estimated - k_values_true) / k_values_true))


def x0_value_rank_error(
    k_values_true, k_values_estimated, x0_values_true, x0_values_estimated
):
    # return (
    #     len(x0_values_true)
    #     - (np.argsort(x0_values_true) == np.argsort(x0_values_estimated)).sum()
    # )
    k_c = scipy.stats.kendalltau(
        np.argsort(x0_values_true), np.argsort(x0_values_estimated)
    ).statistic
    n = len(x0_values_true)
    k_d = (1 - k_c) * (n * (n - 1)) / 4
    return k_d


data_for_df = []
missing_counts = {condition: 0 for condition in CONDITIONS}
for condition in CONDITIONS:
    for i_iteration in range(1, N_ITERATIONS + 1):
        tag = f"{condition}_{i_iteration}"

        fpath_json_results = DPATH_RESULTS / tag / f"{tag}-estimated_params.json"
        if not fpath_json_results.exists():
            missing_counts[condition] += 1
            continue
        json_results = json.loads(fpath_json_results.read_text())
        results = json_results["results"]

        # load data
        fpath_json_data = DPATH_DATA / tag / f"{tag}.json"
        json_data = json.loads(fpath_json_data.read_text())
        cols = json_data["cols"]
        col_timepoint = cols["col_timepoint"]
        col_subject = cols["col_subject"]
        cols_biomarker = cols["cols_biomarker"]

        fpath_merged_data = DPATH_DATA / tag / f"{tag}-merged.tsv"
        df_merged = pd.read_csv(fpath_merged_data, sep="\t")

        # get original time points (before time shift)
        params = json_data["params"]
        x0_values = np.array(params["x0_values"])
        time_shifts = np.concatenate(params["time_shifts"])
        shifted_time = df_merged[col_timepoint] + time_shifts[df_merged[col_subject]]

        for i_biomarker, col_biomarker in enumerate(cols_biomarker):
            for setup in results:
                estimated_k_values = np.array(results[setup]["estimated_k_values"])
                estimated_x0_values = np.array(results[setup]["estimated_x0_values"])

                mean_x_value_difference = (x0_values - estimated_x0_values).mean()

                y_true = df_merged[col_biomarker].to_numpy()
                y_pred = torch.sigmoid(
                    torch.tensor(
                        estimated_k_values[i_biomarker]
                        * (
                            shifted_time
                            - mean_x_value_difference
                            - estimated_x0_values[i_biomarker]
                        )
                    )
                )

                for metric, score_func in [
                    # ("r2_score", r2_score),
                    ("mean_squared_error", mean_squared_error),
                    # (
                    #     "correlation",
                    #     lambda y_true, y_pred: np.corrcoef(y_true, y_pred)[0, 1],
                    # ),
                ]:
                    data_for_df.append(
                        {
                            "condition": condition,
                            "iteration": i_iteration,
                            "tag": tag,
                            "col_biomarker": col_biomarker,
                            "setup": setup.capitalize(),
                            "metric": metric,
                            "score": score_func(y_true, y_pred.numpy()),
                        }
                    )

                for metric, score_func in [
                    ("k_value_relative_error", k_value_relative_error),
                    ("x0_value_rank_error", x0_value_rank_error),
                ]:
                    data_for_df.append(
                        {
                            "condition": condition,
                            "iteration": i_iteration,
                            "tag": tag,
                            "col_biomarker": col_biomarker,
                            "setup": setup.capitalize(),
                            "metric": metric,
                            "score": score_func(
                                params["k_values"],
                                estimated_k_values,
                                params["x0_values"],
                                estimated_x0_values,
                            ),
                        }
                    )

if sum(missing_counts.values()) == 0:
    print("All expected runs were found!")
else:
    print("Missing runs:")
    for condition, count in missing_counts.items():
        if count > 0:
            print(f"  {condition}: {count}")

df_results = pd.DataFrame(data_for_df)
display(df_results)

# %%
# scipy.stats.kendalltau(np.argsort([1, 2, 3, 4, 5]), np.argsort([1, 3, 2, 4, 5]))

# %%
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams["svg.fonttype"] = "none"

THEME = "paper"  # paper, talk

sns.set_theme(THEME, style="ticks")

LABEL_MAP = {
    "setup": "Setup",
    "federated": "Federated",
    "centralized": "Centralized",
    "r2_score": "R² score",
    "mean_squared_error": "Mean squared error",
    "correlation": "Correlation",
    "k_value_relative_error": "Percent error steepness",
    "x0_value_rank_error": "Kendall tau distance",
    "100percent_3sites_50subjects": "IID",
    "50percent_3sites_50subjects": "50% overlap",
    "0percent_3sites_50subjects": "0% overlap",
    "050percent_3sites_50subjects": "50% overlap",
    "000percent_3sites_50subjects": "0% overlap",
    "100percent_2sites_50subjects_20biomarkers": "IID",
    "050percent_2sites_50subjects_20biomarkers": "50% overlap",
    "000percent_2sites_50subjects_20biomarkers": "0% overlap",
    "100percent_5sites_50subjects_20biomarkers": "IID",
    "050percent_5sites_50subjects_20biomarkers": "50% overlap",
    "000percent_5sites_50subjects_20biomarkers": "0% overlap",
    "100percent_8sites_50subjects_20biomarkers": "IID",
    "050percent_8sites_50subjects_20biomarkers": "50% overlap",
    "000percent_8sites_50subjects_20biomarkers": "0% overlap",
}
LATEX_LABEL_MAP = {
    "r2_score": "$R^2$",
    "mean_squared_error": "MSE",
    "correlation": "Correlation",
    "k_value_relative_error": "Percent error steepness",
    "x0_value_rank_error": "Kendall tau distance",
    "100percent_3sites_50subjects": "IID",
    "050percent_3sites_50subjects": r"50\% overlap",
    "000percent_3sites_50subjects": r"0\% overlap",
}
METRIC_YLIM_MAP = {
    "r2_score": (None, 1.0),
    "mean_squared_error": (0.0, None),
    "correlation": (None, 1.0),
    "k_value_relative_error": (None, None),
    "x0_value_rank_error": (None, None),
}

# average over biomarker curves
df_results_avg = (
    df_results.drop(columns="col_biomarker")
    .groupby(["metric", "condition", "setup", "iteration", "tag"])
    .mean()
    .reset_index()
)

grid_box = sns.catplot(
    data=df_results_avg.sort_values(
        by=["condition"],
        key=lambda s: pd.Series(
            [{"IID": 0, "50% overlap": 1, "0% overlap": 2}[LABEL_MAP[x]] for x in s]
        ),
    ),
    kind="box",
    x="condition",
    y="score",
    hue="setup",
    hue_order=["Centralized", "Federated"],
    # row="metric",
    col="metric",
    col_order=[
        metric
        for metric in [
            "r2_score",
            "mean_squared_error",
            "correlation",
            "k_value_relative_error",
            "x0_value_rank_error",
        ]
        if metric in df_results_avg["metric"].unique()
    ],
    # col_wrap=2,
    height={"paper": 1.8, "talk": 3}[THEME],
    aspect=1.8,
    # aspect=1.75,
    sharey=False,
    showfliers=False,
)

for metric, ax in grid_box.axes_dict.items():
    # ax.set_title("")
    # ax.set_ylabel(LABEL_MAP[metric])
    ax.set_title(LABEL_MAP[metric])
    ax.set_ylabel("")
    ax.set_xticks(ax.get_xticks())
    xticklabels = [
        LABEL_MAP.get(label.get_text(), label.get_text())
        for label in ax.get_xticklabels()
    ]
    if len(xticklabels) > 0:
        ax.set_xticklabels(xticklabels)
    ax.set_xlabel("")
    ax.set_ylim(METRIC_YLIM_MAP[metric])

legend = grid_box.legend
legend.set_title(LABEL_MAP[legend.get_title().get_text()])

# %%
fpath_fig_box = DPATH_RESULTS / f"model_fit_comparison-{THEME}.svg"
# fpath_fig_box = DPATH_RESULTS / "model_fit_comparison-5sites.svg"
grid_box.savefig(fpath_fig_box)
print(fpath_fig_box)

# %%
table_lines = [
    r"\begin{tabular}{p{0.28\textwidth}p{0.15\textwidth}p{0.17\textwidth}p{0.17\textwidth}p{0.17\textwidth}}",
    r"\hline",
    r"\multirow{2}{*}{\textbf{Metric}} & \multirow{2}{*}{\textbf{Setup}} & \multicolumn{3}{c}{\textbf{Condition}} \\",
    r" & ".join(["", ""] + [rf"\textbf{{{LATEX_LABEL_MAP[c]}}}" for c in CONDITIONS])
    + r" \\",
    r"\hline",
]

for metric in [
    m
    for m in [
        "r2_score",
        "mean_squared_error",
        "correlation",
        "k_value_relative_error",
        "x0_value_rank_error",
    ]
    if m in df_results_avg["metric"].unique()
]:
    for setup in df_results_avg["setup"].unique():
        # row
        cells = []
        metric_label = LATEX_LABEL_MAP[metric]
        if metric_label in table_lines[-1]:
            cells.append("")
        else:
            cells.append(rf"\multirow{{2}}{{*}}{{{metric_label}}}")

        cells.append(setup)

        df_results_avg_metric = df_results_avg.query(
            f'metric == "{metric}" and setup == "{setup}"'
        )

        df_results_avg_metric = df_results_avg_metric.pivot_table(
            index=["iteration"],
            columns=["condition"],
            values="score",
        )

        print(f"Setup: {setup}, Metric: {metric}")
        display(df_results_avg_metric.describe().loc[["mean", "std"]])

        for condition in CONDITIONS:
            condition_label = LABEL_MAP[condition]
            mean_score = df_results_avg_metric[condition].mean()
            std_score = df_results_avg_metric[condition].std()
            cells.append(rf"${mean_score:.3f} \pm {std_score:.3f}$")

        table_lines.append(" & ".join(cells) + r" \\")

    table_lines.append(r"\hline")

table_lines.append(r"\end{tabular}")
# print("\n".join(table_lines))
for table_line in table_lines:
    print(table_line)

# %%
import numpy as np

fraction_std_for_margin = 0.75

for metric in df_results_avg["metric"].unique():
    print(f"=== {metric} ===")
    for condition in df_results_avg["condition"].unique():
        print(condition)

        margins = {
            # "r2_score": (-0.005, 0.005),
            # "correlation": (-0.025, 0.025),
            # "x0_value_rank_error": (-1, 1),
            # "mean_squared_error": (-0.0003, 0.0003),
        }

        scores_federated = (
            df_results_avg.query(
                f"condition == '{condition}' and metric == '{metric}' and setup == 'Federated'"
            )
            .set_index("iteration")
            .sort_index()
        )
        scores_centralized = (
            df_results_avg.query(
                f"condition == '{condition}' and metric == '{metric}' and setup == 'Centralized'"
            )
            .set_index("iteration")
            .sort_index()
        )

        score_diffs: pd.Series = scores_federated["score"] - scores_centralized["score"]
        mean_score_diff = score_diffs.mean()
        std_score_diff = score_diffs.std()

        if metric not in margins:
            margins[metric] = (
                -fraction_std_for_margin * std_score_diff,
                fraction_std_for_margin * std_score_diff,
            )
        print(f"\t{margins[metric]=}")

        ci = scipy.stats.bootstrap(
            (score_diffs,),
            np.mean,
            confidence_level=0.95,
            n_resamples=10000,
        ).confidence_interval

        # ci = scipy.stats.wilcoxon.interval(
        #     0.95,
        #     len(score_diffs) - 1,
        #     loc=mean_score_diff,
        #     scale=std_score_diff / np.sqrt(len(score_diffs)),
        # )

        print(f"\t{ci=}")
        if ci[0] > margins[metric][0] and ci[1] < margins[metric][1]:
            print(f"\t***{metric} is within margin of error***")

# %%
import matplotlib.pyplot as plt

scipy.stats.probplot(score_diffs, dist="norm", plot=plt)
# score_diffs.hist()

# %%
df_results_avg.query(
    "condition == '100percent_3sites_50subjects' and metric == 'mean_squared_error' and setup == 'Centralized'"
).describe()
