#!/usr/bin/env python

import warnings
from enum import Enum
from pathlib import Path

import click
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from fl_prog.utils.constants import CLICK_CONTEXT_SETTINGS, COL_ADAPTATION
from fl_prog.utils.io import (
    DEFAULT_DPATH_DATA,
    get_dpath_latest,
    load_json,
    save_json,
)


# train/test split
class SplitMode(str, Enum):
    TIMEPOINT = "timepoint"
    SUBJECT = "subject"


DEFAULT_MODE = SplitMode.TIMEPOINT
DEFAULT_MIN_N_TIMEPOINTS = 4
DEFAULT_TEST_FRACTION = 0.14
SUFFIX_TEST = "-test"


def _is_adaptation_subject(
    df_subject: pd.DataFrame, subject: str, has_adaptation: bool
) -> bool:
    if not has_adaptation:
        return False
    if df_subject[COL_ADAPTATION].all():
        return True
    if df_subject[COL_ADAPTATION].any():
        raise ValueError(
            f"Subject {subject} has a mix of adaptation and non-adaptation rows."
        )
    return False


def _split_site_last_timepoint(
    df_site: pd.DataFrame,
    col_subject: str,
    col_timepoint: str,
    min_n_timepoints: int,
) -> tuple[list[pd.DataFrame], list[pd.DataFrame], list[str]]:
    """Leave out the last timepoint of subjects with enough timepoints."""
    has_adaptation = COL_ADAPTATION in df_site.columns

    dfs_train = []
    dfs_test = []
    subjects = []
    for subject, df_subject in df_site.groupby(col_subject, sort=False):
        if (
            not _is_adaptation_subject(df_subject, subject, has_adaptation)
            and len(df_subject) >= min_n_timepoints
        ):
            df_subject = df_subject.sort_values(col_timepoint, ascending=True)
            dfs_test.append(df_subject.iloc[[-1]])
            df_subject = df_subject.iloc[:-1]

        dfs_train.append(df_subject)
        subjects.append(subject)

    return dfs_train, dfs_test, subjects


def _split_site_subjects(
    df_site: pd.DataFrame,
    col_subject: str,
    n_splits: int,
    random_seed: int | None,
) -> tuple[list[pd.DataFrame], list[pd.DataFrame], list[str]]:
    """Leave out entire subjects, stratified by number of timepoints."""
    has_adaptation = COL_ADAPTATION in df_site.columns

    n_timepoints = pd.Series(
        {
            subject: len(df_subject)
            for subject, df_subject in df_site.groupby(col_subject, sort=False)
            if not _is_adaptation_subject(df_subject, subject, has_adaptation)
        },
        dtype=int,
    )
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    with warnings.catch_warnings():
        # allow fewer subjects than n_splits
        warnings.filterwarnings("ignore", message="The least populated class")
        _, idx_test = next(cv.split(n_timepoints.index, n_timepoints))
    subjects_test = set(n_timepoints.index[idx_test])

    dfs_train = []
    dfs_test = []
    subjects = []
    for subject, df_subject in df_site.groupby(col_subject, sort=False):
        if subject in subjects_test:
            dfs_test.append(df_subject)
        else:
            dfs_train.append(df_subject)
            subjects.append(subject)

    return dfs_train, dfs_test, subjects


def _get_new_tag(
    tag: str,
    mode: SplitMode,
    min_n_timepoints: int,
    n_splits: int,
    random_seed: int | None,
) -> str:
    if mode == SplitMode.TIMEPOINT:
        return f"{tag}_split{min_n_timepoints}"
    new_tag = f"{tag}_heldout{round(100 / n_splits)}"
    if random_seed is not None:
        new_tag += f"_seed{random_seed}"
    return new_tag


def split_train_test(
    tag: str,
    dpath_data: Path,
    mode: SplitMode = DEFAULT_MODE,
    min_n_timepoints: int = DEFAULT_MIN_N_TIMEPOINTS,
    test_fraction: float = DEFAULT_TEST_FRACTION,
    random_seed: int | None = None,
    suffix_test: str = SUFFIX_TEST,
):
    mode = SplitMode(mode)

    # held-out subjects are one fold of a stratified K-fold
    n_splits = round(1 / test_fraction)
    test_fraction_effective = 1 / n_splits
    if mode == SplitMode.SUBJECT and test_fraction_effective != test_fraction:
        click.secho(
            f"WARNING: test fraction {test_fraction} rounded to 1/{n_splits}",
            fg="yellow",
            bold=True,
        )

    old_tag = tag
    tag = _get_new_tag(tag, mode, min_n_timepoints, n_splits, random_seed)
    dpath_out_old = get_dpath_latest(dpath_data) / old_tag
    fpath_json_old = dpath_out_old / f"{old_tag}.json"
    dpath_out_new = get_dpath_latest(dpath_data, use_today=True) / tag

    settings = locals().copy()

    json_data_old = load_json(fpath_json_old)
    node_id_map_old = json_data_old["node_id_map"]
    col_subject = json_data_old["cols"]["col_subject"]
    col_timepoint = json_data_old["cols"]["col_timepoint"]
    col_subject_index = json_data_old["cols"]["col_subject_index"]
    config_old = json_data_old["settings"]["config"]
    settings["config"] = config_old

    dfs_test = []
    node_id_map_new = {}
    subjects_by_node = {}
    scaling_references = {}
    n_samples = 0
    for fname_site, node_id in node_id_map_old.items():
        if fname_site.endswith("-merged.tsv"):
            click.secho(
                "WARNING: merged file found. Make sure to merge the data after this step.",
                fg="yellow",
                bold=True,
            )
            continue

        subjects_by_node[node_id] = []

        fpath_site = dpath_out_old / fname_site
        df_site = pd.read_csv(fpath_site, sep="\t", dtype={col_subject: str})
        n_samples += len(df_site)

        if mode == SplitMode.TIMEPOINT:
            dfs_train, dfs_test_site, subjects = _split_site_last_timepoint(
                df_site, col_subject, col_timepoint, min_n_timepoints
            )
        else:
            dfs_train, dfs_test_site, subjects = _split_site_subjects(
                df_site, col_subject, n_splits, random_seed
            )
            n_subjects_test = sum(df[col_subject].nunique() for df in dfs_test_site)
            print(
                f"Node {node_id}: holding out {n_subjects_test} of "
                f"{df_site[col_subject].nunique()} subjects"
            )
        dfs_test.extend(dfs_test_site)

        subjects_by_node[node_id] = subjects
        df_train = pd.concat(dfs_train)
        df_train[col_subject_index] = df_train[col_subject].map(
            lambda x, subjects=subjects: subjects.index(x)
        )

        fname_site_new = fname_site.replace(old_tag, tag)
        node_id_map_new[fname_site_new] = node_id
        scaling_references[fname_site_new] = fname_site_new

        dpath_out_new.mkdir(parents=True, exist_ok=True)
        fpath_site_new = dpath_out_new / fname_site_new
        df_train.to_csv(fpath_site_new, sep="\t", index=False)
        print(f"{df_site.shape} -> {df_train.shape}: {fpath_site_new}")

    df_test = pd.concat(dfs_test)

    # sanity checks
    if mode == SplitMode.TIMEPOINT:
        if df_test[col_subject].nunique() != len(df_test):
            print(df_test[col_subject].value_counts())
            raise ValueError("Some subjects have multiple timepoints in the test set.")
    else:
        subjects_train = {
            subject for subjects in subjects_by_node.values() for subject in subjects
        }
        subjects_overlap = subjects_train & set(df_test[col_subject])
        if subjects_overlap:
            raise ValueError(
                "Some held-out subjects are also in the train set: "
                f"{sorted(subjects_overlap)[:5]}"
            )
        # held-out subjects have no index in the train data
        df_test = df_test.drop(columns=col_subject_index)

    fpath_test = dpath_out_new / f"{tag}{suffix_test}.tsv"
    df_test.to_csv(fpath_test, sep="\t", index=False)
    print(
        f"Saved test set ({df_test.shape}, {len(df_test) / n_samples:.2%}) to {fpath_test}"
    )
    scaling_references[fpath_test.name] = (
        None  # no scaling reference until site data are merged
    )

    json_data_new = {}
    json_data_new["settings"] = settings
    json_data_new["node_id_map"] = node_id_map_new
    json_data_new["cols"] = json_data_old["cols"]
    json_data_new["subjects_by_node"] = subjects_by_node
    json_data_new["need_scaling"] = json_data_old["need_scaling"]
    json_data_new["scaling_references"] = scaling_references
    json_data_new["do_not_merge"] = json_data_old.get("do_not_merge", [])
    json_data_new["do_not_merge"].append(fpath_test.name)
    fpath_json_new = dpath_out_new / f"{tag}.json"
    save_json(fpath_json_new, json_data_new)
    print(f"Saved new JSON data to {fpath_json_new}")
    print(f"Use new tag --tag {tag} in next steps")


@click.command(context_settings=CLICK_CONTEXT_SETTINGS)
@click.option("--tag", type=str, required=True)
@click.option(
    "--data-dir",
    "dpath_data",
    type=click.Path(path_type=Path, file_okay=False, dir_okay=True),
    default=DEFAULT_DPATH_DATA,
)
@click.option(
    "--mode",
    type=click.Choice(SplitMode, case_sensitive=False),
    default=DEFAULT_MODE.value,
    show_default=True,
    help=(
        f"{SplitMode.TIMEPOINT.value!r}: leave out the last timepoint of each subject. "
        f"{SplitMode.SUBJECT.value!r}: leave out entire subjects."
    ),
)
@click.option(
    "--min-timepoints",
    "min_n_timepoints",
    type=click.IntRange(min=2, min_open=False),
    default=DEFAULT_MIN_N_TIMEPOINTS,
    help=f"Only used in {SplitMode.TIMEPOINT.value!r} mode.",
)
@click.option(
    "--test-fraction",
    type=click.FloatRange(min=0, max=0.5, min_open=True),
    default=DEFAULT_TEST_FRACTION,
    show_default=True,
    help=(
        "Fraction of subjects to hold out per site. Must satisfy 1/K for an integer K."
        f" Only used in {SplitMode.SUBJECT.value!r} mode."
    ),
)
@click.option("--random-seed", type=int, envvar="RNG_SEED", show_envvar=True)
def main(*args, **kwargs):
    """
    Split into train/test sets.

    Either leave out the last timepoint for each subject with enough timepoints, or
    leave out entire subjects (stratified by number of timepoints within each site).

    Run before merging the data across sites.
    """
    split_train_test(*args, **kwargs)


if __name__ == "__main__":
    main()
