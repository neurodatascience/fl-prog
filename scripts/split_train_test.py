#!/usr/bin/env python

import shutil
import warnings
from enum import Enum
from pathlib import Path

import click
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from fl_prog.metrics import PER_SUBJECT_FIELDS
from fl_prog.results import get_subjects, true_params_by_subject
from fl_prog.utils.constants import CLICK_CONTEXT_SETTINGS, COL_ADAPTATION, NODE_ID_TEST
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
# settings of the input data that are copied to the output JSON (if present)
SETTINGS_KEYS_KEPT = ("overlap",)


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


def _choose_test_subjects(
    df: pd.DataFrame,
    col_subject: str,
    n_splits: int,
    random_seed: int | None,
    sort_subjects: bool = False,
) -> set[str]:
    """Choose held-out subjects (one fold), stratified by number of timepoints.

    With ``sort_subjects``, subjects are sorted by ID first so that the result only
    depends on the set of subjects and the random seed (not on row order).
    """
    has_adaptation = COL_ADAPTATION in df.columns

    n_timepoints = pd.Series(
        {
            subject: len(df_subject)
            for subject, df_subject in df.groupby(col_subject, sort=False)
            if not _is_adaptation_subject(df_subject, subject, has_adaptation)
        },
        dtype=int,
    )
    if sort_subjects:
        n_timepoints = n_timepoints.sort_index()
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    with warnings.catch_warnings():
        # allow fewer subjects than n_splits
        warnings.filterwarnings("ignore", message="The least populated class")
        _, idx_test = next(cv.split(n_timepoints.index, n_timepoints))
    return set(n_timepoints.index[idx_test])


def _split_site_subjects(
    df_site: pd.DataFrame,
    col_subject: str,
    subjects_test: set[str],
) -> tuple[list[pd.DataFrame], list[pd.DataFrame], list[str]]:
    """Leave out entire (preselected) subjects."""
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
    pooled: bool = False,
) -> str:
    if mode == SplitMode.TIMEPOINT:
        return f"{tag}_split{min_n_timepoints}"
    new_tag = f"{tag}_heldout{round(100 / n_splits)}"
    if pooled:
        new_tag += "pooled"
    if random_seed is not None:
        new_tag += f"_seed{random_seed}"
    return new_tag


def _get_true_params_by_subject(
    params: dict, subjects_by_node: dict[str, list[str]]
) -> dict[str, dict[str, float]]:
    """Map true per-subject parameters to subject IDs."""
    true_params = true_params_by_subject(subjects_by_node, params)
    subjects = get_subjects(subjects_by_node)
    return {
        param_name: dict(zip(subjects, getattr(true_params, param_name), strict=True))
        for param_name in PER_SUBJECT_FIELDS
    }


def _realign_params(
    params: dict,
    subjects_by_node_old: dict[str, list[str]],
    subjects_by_node_new: dict[str, list[str]],
) -> dict:
    """Realign true per-subject parameters to the new ``subjects_by_node``."""
    params_by_subject = _get_true_params_by_subject(params, subjects_by_node_old)

    params_new = params.copy()
    for param_name, values_by_subject in params_by_subject.items():
        params_new[param_name] = [
            [values_by_subject[subject] for subject in subjects]
            for subjects in subjects_by_node_new.values()
        ]
    return params_new


def split_train_test(
    tag: str,
    dpath_data: Path,
    mode: SplitMode = DEFAULT_MODE,
    min_n_timepoints: int = DEFAULT_MIN_N_TIMEPOINTS,
    test_fraction: float = DEFAULT_TEST_FRACTION,
    random_seed: int | None = None,
    pooled: bool = False,
    overwrite: bool = False,
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
    tag = _get_new_tag(tag, mode, min_n_timepoints, n_splits, random_seed, pooled)
    dpath_out_old = get_dpath_latest(dpath_data) / old_tag
    fpath_json_old = dpath_out_old / f"{old_tag}.json"
    dpath_out_new = get_dpath_latest(dpath_data, use_today=True) / tag

    settings = locals().copy()

    json_data_old = load_json(fpath_json_old)

    # avoid mixing outputs (e.g. merged data) from a previous run
    if dpath_out_new.exists():
        if not overwrite:
            click.secho(
                f"{dpath_out_new} already exists. Use --overwrite to overwrite.",
                fg="red",
                bold=True,
            )
            return
        click.secho(f"Removing existing {dpath_out_new}", fg="yellow")
        shutil.rmtree(dpath_out_new)

    node_id_map_old = json_data_old["node_id_map"]
    col_subject = json_data_old["cols"]["col_subject"]
    col_timepoint = json_data_old["cols"]["col_timepoint"]
    col_subject_index = json_data_old["cols"]["col_subject_index"]
    # settings of the input data needed downstream
    settings["config"] = json_data_old["settings"].get("config", {})
    for key in SETTINGS_KEYS_KEPT:
        if key in json_data_old["settings"]:
            settings[key] = json_data_old["settings"][key]

    fnames_site = []
    for fname_site in node_id_map_old:
        if fname_site.endswith("-merged.tsv"):
            click.secho(
                "WARNING: merged file found. Make sure to merge the data after this step.",
                fg="yellow",
                bold=True,
            )
            continue
        fnames_site.append(fname_site)
    dfs_site = {
        fname_site: pd.read_csv(
            dpath_out_old / fname_site, sep="\t", dtype={col_subject: str}
        )
        for fname_site in fnames_site
    }

    subjects_test_pooled = None
    if mode == SplitMode.SUBJECT and pooled:
        # same held-out subjects regardless of how subjects are split into sites
        subjects_test_pooled = _choose_test_subjects(
            pd.concat(dfs_site.values()),
            col_subject,
            n_splits,
            random_seed,
            sort_subjects=True,
        )

    dfs_test = []
    node_id_map_new = {}
    subjects_by_node = {}
    scaling_references = {}
    n_samples = 0
    for fname_site, df_site in dfs_site.items():
        node_id = node_id_map_old[fname_site]
        subjects_by_node[node_id] = []

        n_samples += len(df_site)

        if mode == SplitMode.TIMEPOINT:
            dfs_train, dfs_test_site, subjects = _split_site_last_timepoint(
                df_site, col_subject, col_timepoint, min_n_timepoints
            )
        else:
            if subjects_test_pooled is not None:
                subjects_test = subjects_test_pooled
            else:
                subjects_test = _choose_test_subjects(
                    df_site, col_subject, n_splits, random_seed
                )
            dfs_train, dfs_test_site, subjects = _split_site_subjects(
                df_site, col_subject, subjects_test
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
        subjects_by_node[NODE_ID_TEST] = df_test[col_subject].unique().tolist()

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
    if "biomarker_names" in json_data_old:
        json_data_new["biomarker_names"] = json_data_old["biomarker_names"]
    json_data_new["subjects_by_node"] = subjects_by_node
    if "params" in json_data_old:
        json_data_new["params"] = _realign_params(
            json_data_old["params"], json_data_old["subjects_by_node"], subjects_by_node
        )
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
@click.option(
    "--pooled/--per-site",
    default=False,
    show_default=True,
    help=(
        "Choose held-out subjects across all sites pooled (same test set regardless "
        "of the site split) or within each site."
        f" Only used in {SplitMode.SUBJECT.value!r} mode."
    ),
)
@click.option(
    "--overwrite/--no-overwrite",
    default=False,
    help="Delete the output directory first if it already exists.",
)
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
