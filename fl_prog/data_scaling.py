import re
from collections.abc import Iterable
from typing import Any

import pandas as pd

KEY_MIN = "normal"
KEY_MAX = "abnormal"


def scale_min_max(
    df, min_max_by_measure: dict[str, tuple[float, float]]
) -> pd.DataFrame:
    measures = [measure for measure in min_max_by_measure if measure in df.columns]
    min_values = (
        pd.DataFrame(
            data=[x[0] for x in min_max_by_measure.values()],
            index=min_max_by_measure.keys(),
        )
        .squeeze()
        .loc[measures]
    )
    max_values = (
        pd.DataFrame(
            data=[x[1] for x in min_max_by_measure.values()],
            index=min_max_by_measure.keys(),
        )
        .squeeze()
        .loc[measures]
    )
    df.loc[:, measures] = (df[measures] - min_values) / (max_values - min_values)
    return df


def get_min_max_by_measure(
    df: pd.DataFrame,
    measures: list[str],
    scaling_methods: dict[str, dict[str, dict[str, Any]]],
    col_group: str | None = None,
) -> dict[str, Iterable[float, float]]:
    def _get_scaling_config_for_measure(measure: str) -> dict[str, dict[str, Any]]:
        for pattern, config in scaling_methods.items():
            if re.match(pattern, measure):
                return config
        raise ValueError(f"No scaling config found for measure: {measure}")

    min_max_by_measure = {}

    for measure in measures:
        scaling_config = _get_scaling_config_for_measure(measure)
        min_max_by_measure[measure] = [None, None]
        for i_key, key in enumerate((KEY_MIN, KEY_MAX)):
            method_name = scaling_config[key]["method"]
            args = scaling_config[key].get("args", ())
            match method_name:
                case "constant":
                    try:
                        (constant,) = args
                    except TypeError:
                        raise ValueError(
                            f"Expected a single argument 'constant' (float | int) for constant scaling, got {args}"
                        )
                    value = float(constant)
                case "quantile":
                    try:
                        (quantile, group) = args
                    except TypeError:
                        raise ValueError(
                            f"Expected two arguments 'quantile' (float) and 'group' (str) for quantile scaling, got {args}"
                        )
                    if col_group is None and group is not None:
                        raise ValueError(
                            "col_group must be provided for quantile scaling with group filtering."
                        )
                    if group is not None:
                        if isinstance(group, str):
                            group = [group]
                        df_group = df.query(f"{col_group} in @group")
                    else:
                        df_group = df
                    value = df_group[measure].quantile(quantile)
                case "max":
                    value = df[measure].max()
                case "min":
                    value = df[measure].min()
                case _:
                    raise ValueError(f"Unknown scaling method: {method_name}")

            min_max_by_measure[measure][i_key] = value

    return min_max_by_measure
