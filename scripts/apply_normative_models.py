#!/usr/bin/env python

import sys
import zipfile
from collections.abc import Iterable
from functools import partial
from pathlib import Path

import click
import httpx
import numpy as np
import pandas as pd
from pcntoolkit import NormativeModel, NormData
from sklearn.model_selection import train_test_split

from fl_prog.freesurfer import _rename_and_drop_cols, get_df_idp
from fl_prog.utils.constants import CLICK_CONTEXT_SETTINGS
from fl_prog.utils.io import DEFAULT_DPATH_DATA, load_json, save_json

FNAME_SETTINGS = "settings.json"
FNAME_HARMONIZED = "harmonized.csv"
DNAME_MODEL = "pcntoolkit"

# from https://pcntoolkit.readthedocs.io/en/dev/tutorials/12_transfer_pretrained.html
SURF_BASE_URL = "https://surfdrive.surf.nl/public.php/webdav/zip/"
SURF_SHARE_TOKEN = "Mb6mZyFmJeCaPcZ"
SURF_PASSWORD = ""

COL_SUBJECT_ADNIMERGE = "RID"
COL_SESSION_ADNIMERGE = "VISCODE"
COL_GROUP_ADNIMERGE = "DX_bl"
COL_AGE_ADNIMERGE = "AGE"
COL_SEX_ADNIMERGE = "PTGENDER"
COL_SITE_ADNIMERGE = "SITE"

COL_ROW_ID = "row_id"
COL_BATCH_ID = "batch_id"
COL_ADAPTATION = "adaptation"

COL_HARMONIZED_PCNTOOLKIT = "Y_harmonized"
COL_ROW_ID_PCNTOOLKIT = "subject_ids"

SEX_MAP_PER_MODEL = {
    "HBR_Sb_ct_DK_lifespan_79K_100sites.zip": {
        "Female": "0.0",
        "Male": "1.0",
    },
    "HBR_Sb_sa_lifespan_DK_46K_59sites.zip": {
        "Female": "F",
        "Male": "M",
    },
    "HBR_Sb_sc_lifespan_79K_100sites.zip": {
        "Female": "0.0",
        "Male": "1.0",
    },
}
COL_NAMES_PER_MODEL = {
    "HBR_Sb_ct_DK_lifespan_79K_100sites.zip": {
        COL_AGE_ADNIMERGE: "age",
        COL_SEX_ADNIMERGE: "sex",
        COL_SITE_ADNIMERGE: "site_id2",
    },
    "HBR_Sb_sa_lifespan_DK_46K_59sites.zip": {
        COL_AGE_ADNIMERGE: "age",
        COL_SEX_ADNIMERGE: "sex",
        COL_SITE_ADNIMERGE: "site",
    },
    "HBR_Sb_sc_lifespan_79K_100sites.zip": {
        COL_AGE_ADNIMERGE: "age",
        COL_SEX_ADNIMERGE: "sex",
        COL_SITE_ADNIMERGE: "site_id2",
    },
}

DF_TRANSFORM_PER_MODEL = {
    "HBR_Sb_ct_DK_lifespan_79K_100sites.zip": partial(
        _rename_and_drop_cols,
        left="L_",
        right="R_",
        suffix_to_strip="_thickness",
        suffix_to_drop="_area",
    ),
    "HBR_Sb_sa_lifespan_DK_46K_59sites.zip": partial(
        _rename_and_drop_cols,
        left="L_",
        right="R_",
        suffix_to_strip="_area",
        suffix_to_drop="_thickness",
    ),
}

DEFAULT_DPATH_HARMONIZED_DATA = DEFAULT_DPATH_DATA / "_harmonized_data"
DEFAULT_DPATH_MODELS = DEFAULT_DPATH_DATA / "_normative_models"
DEFAULT_MODEL_NAMES = (
    "HBR_Sb_ct_DK_lifespan_79K_100sites.zip",  # cortical thickness
    "HBR_Sb_sa_lifespan_DK_46K_59sites.zip",  # surface area
    "HBR_Sb_sc_lifespan_79K_100sites.zip",  # subcortical volume
)
DEFAULT_MIN_BATCH_SIZE = 10
DEFAULT_ADAPTATION_GROUPS = ("CN",)
DEFAULT_ADAPTATION_FRAC = 0.5
DEFAULT_DROP_ADAPTATION = False


class KnownError(Exception):
    pass


def _get_merged_df(
    fpath_idps: Path, fpath_adni_merge: Path, fpath_config: Path
) -> pd.DataFrame:
    config = load_json(fpath_config)
    col_subject_original = config["col_subject_original"]
    col_session_original = config["col_session_original"]
    session_timepoint_map = config["session_timepoint_map"]

    df_idps = get_df_idp(
        fpath_idps,
        merge_hemispheres=False,
        col_subject_original=col_subject_original,
        col_session_original=col_session_original,
        session_timepoint_map=session_timepoint_map,
    )
    df_idps.index = df_idps.index.rename([COL_SUBJECT_ADNIMERGE, COL_SESSION_ADNIMERGE])

    df_adnimerge = pd.read_csv(
        fpath_adni_merge,
        dtype={
            COL_SUBJECT_ADNIMERGE: str,
            COL_SESSION_ADNIMERGE: str,
            COL_SITE_ADNIMERGE: str,
        },
        low_memory=False,
    )
    df_adnimerge[COL_SESSION_ADNIMERGE] = df_adnimerge[
        COL_SESSION_ADNIMERGE
    ].str.upper()
    df_adnimerge = df_adnimerge.set_index(
        [COL_SUBJECT_ADNIMERGE, COL_SESSION_ADNIMERGE]
    )
    df_adnimerge = df_adnimerge[
        [COL_AGE_ADNIMERGE, COL_SEX_ADNIMERGE, COL_SITE_ADNIMERGE, COL_GROUP_ADNIMERGE]
    ]

    df_merged = df_idps.merge(
        df_adnimerge,
        how="inner",
        left_index=True,
        right_index=True,
        suffixes=("_duplicate", ""),
    )

    df_merged[COL_ROW_ID] = df_merged.index.to_frame().apply(
        lambda x: f"{x[COL_SUBJECT_ADNIMERGE]}_{x[COL_SESSION_ADNIMERGE]}",
        axis="columns",
    )
    df_merged = df_merged.reset_index(drop=False)
    df_merged = df_merged.set_index(COL_ROW_ID)

    # TODO (maybe): include age in stratification? How?
    # How? Within each sex/site combination, create age bins
    # Then include age bins in batch ID

    df_merged[COL_BATCH_ID] = df_merged[[COL_SEX_ADNIMERGE, COL_SITE_ADNIMERGE]].apply(
        lambda x: f"{x[COL_SEX_ADNIMERGE]}_{x[COL_SITE_ADNIMERGE]}", axis="columns"
    )

    return df_merged


def _drop_batches(
    df: pd.DataFrame,
    min_batch_size: int = DEFAULT_MIN_BATCH_SIZE,
    groups: Iterable[str] = DEFAULT_ADAPTATION_GROUPS,
) -> pd.DataFrame:
    return df.groupby(COL_BATCH_ID).filter(
        lambda df: (
            (df_group := df[COL_GROUP_ADNIMERGE].isin(groups)).any()
            and df_group.sum() >= min_batch_size
        )
    )


def _get_idx_adaptation(
    df: pd.DataFrame,
    groups: Iterable[str] = DEFAULT_ADAPTATION_GROUPS,
    frac: float = DEFAULT_ADAPTATION_FRAC,
    rng_seed: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_adaptation = df.loc[df[COL_GROUP_ADNIMERGE].isin(groups)]
    idx = np.arange(len(df_adaptation))
    idx_adaptation, _ = train_test_split(
        idx,
        train_size=frac,
        random_state=rng_seed,
        shuffle=True,
        stratify=df_adaptation[COL_BATCH_ID],
    )
    df_adaptation = df_adaptation.iloc[idx_adaptation]
    return df_adaptation.index


def _download_model(model_name: str, dpath_models: Path) -> Path:
    dpath_models.mkdir(parents=True, exist_ok=True)
    fpath_model_zipped = dpath_models / model_name
    dpath_model_unzipped = dpath_models / model_name.removesuffix(".zip")

    if dpath_model_unzipped.exists():
        return dpath_model_unzipped

    if not fpath_model_zipped.exists():
        with httpx.stream(
            "GET",
            f"{SURF_BASE_URL}{model_name}",
            auth=(SURF_SHARE_TOKEN, SURF_PASSWORD),
        ) as response:
            response.raise_for_status()
            with open(fpath_model_zipped, "wb") as f:
                f.writelines(response.iter_bytes(chunk_size=8192))

        click.secho(
            f"Model {model_name} downloaded to {fpath_model_zipped}.", fg="green"
        )

    with zipfile.ZipFile(fpath_model_zipped, "r") as z:
        z.extractall(dpath_models)

    if not dpath_model_unzipped.exists():
        dpath_candidate = dpath_models / "model"
        if dpath_candidate.exists():
            dpath_model_unzipped.mkdir(parents=True, exist_ok=True)
            dpath_candidate.rename(dpath_model_unzipped / dpath_candidate.name)
        else:
            raise RuntimeError("Error: Unzipped model directory not found.")

    click.secho(f"Model {model_name} extracted to {dpath_model_unzipped}.", fg="green")

    return dpath_model_unzipped


# AI-generated fix for incompatibility between pcntoolkit 1.1.1 and 1.2.0.post1
def _repair_basis_function(basis_function) -> None:
    is_legacy = (
        isinstance(basis_function.min, dict)
        or isinstance(basis_function.max, dict)
        or isinstance(basis_function.basis_column, list)
    )
    if not is_legacy:
        return
    if basis_function.basis_name != "bspline":
        raise RuntimeError(
            f"Cannot repair legacy basis function of type {basis_function.basis_name}."
        )
    if basis_function.knot_method != "uniform":
        raise RuntimeError(
            "Cannot repair legacy basis function with non-uniform knot method."
        )
    if isinstance(basis_function.basis_column, list):
        basis_function.basis_column = basis_function.basis_column[0]
    if isinstance(basis_function.min, dict):
        basis_function.min = next(iter(basis_function.min.values()))
    if isinstance(basis_function.max, dict):
        basis_function.max = next(iter(basis_function.max.values()))
    basis_function.knots = None
    basis_function._fit([0.0])


# AI-generated
def _repair_model_basis_functions(model: NormativeModel) -> None:
    for responsevar in model.response_vars:
        likelihood = model[responsevar].likelihood
        for parameter in ("mu", "sigma"):
            prior = getattr(likelihood, parameter)
            if hasattr(prior, "basis_function"):
                _repair_basis_function(prior.basis_function)


def _get_pretrained_model(model_name: str, dpath_models: Path) -> NormativeModel:
    dpath_model = _download_model(model_name, dpath_models)
    model = NormativeModel.load(str(dpath_model))
    _repair_model_basis_functions(model)
    return model


def _apply_model(
    df_original: pd.DataFrame,
    idx_adaptation: np.array,
    model_name: str,
    dpath_models: Path,
    save_dir: Path,
    drop_adaptation: bool = DEFAULT_DROP_ADAPTATION,
) -> pd.DataFrame:
    df = df_original.copy()
    # recode sex
    df[COL_SEX_ADNIMERGE] = df[COL_SEX_ADNIMERGE].map(SEX_MAP_PER_MODEL[model_name])
    # rename/drop FreeSurfer columns
    df, col_map = DF_TRANSFORM_PER_MODEL.get(model_name, lambda df: (df, None))(df)
    # rename covariate/batch effect columns
    df = df.rename(columns=COL_NAMES_PER_MODEL[model_name])

    pretrained_model = _get_pretrained_model(model_name, dpath_models)
    pretrained_model.name = model_name.removesuffix(".zip")
    covariates = pretrained_model.covariates
    batch_effects = list(pretrained_model.unique_batch_effects)
    response_vars = [
        var
        for var in pretrained_model.response_vars
        if var in df.columns and pretrained_model[var].is_fitted
    ]

    if col_map is None:
        # identity map
        col_map = {var: var for var in response_vars}

    click.secho(model_name, fg="magenta")
    click.secho(f"\tCovariate(s): {covariates}")
    click.secho(f"\tBatch effect(s): {batch_effects}")
    click.secho(
        f"\tN response variables: {len(response_vars)}/{len(pretrained_model.response_vars)}"
    )

    if len(response_vars) == 0:
        print(df.columns)
        raise RuntimeError(
            f"None of the response variables in the model are in the dataframe: {pretrained_model.response_vars}"
        )

    df = df.dropna(
        subset=covariates + batch_effects + response_vars, axis="index", how="any"
    )

    df_adaptation = df.loc[idx_adaptation]
    if drop_adaptation:
        df_to_harmonize = df.drop(df_adaptation.index)
    else:
        df_to_harmonize = df

    data_adaptation = NormData.from_dataframe(
        "adaptation",
        df_adaptation.reset_index(drop=False),
        covariates=covariates,
        batch_effects=batch_effects,
        response_vars=response_vars,
        subject_ids=COL_ROW_ID,
    )
    data_to_harmonize = NormData.from_dataframe(
        "to_harmonize",
        df_to_harmonize.reset_index(drop=False),
        covariates=covariates,
        batch_effects=batch_effects,
        response_vars=response_vars,
        subject_ids=COL_ROW_ID,
    )

    new_model = pretrained_model.transfer(data_adaptation, save_dir=str(save_dir))
    data_harmonized = new_model.harmonize(data_to_harmonize)

    df_harmonized = data_harmonized.to_dataframe()[
        [COL_HARMONIZED_PCNTOOLKIT, COL_ROW_ID_PCNTOOLKIT]
    ]

    # map back to original names
    df_harmonized.columns = df_harmonized.columns.droplevel(0)
    df_harmonized = df_harmonized.rename(columns=col_map)
    df_harmonized = df_harmonized.rename(columns={COL_ROW_ID_PCNTOOLKIT: COL_ROW_ID})
    df_harmonized = df_harmonized.set_index(COL_ROW_ID)

    df_original.loc[df_harmonized.index, df_harmonized.columns] = df_harmonized

    return df_original, response_vars


def apply_normative_models(
    fpath_idps: Path,
    fpath_adni_merge: Path,
    fpath_config: Path,
    dpath_models: Path,
    dpath_out: Path,
    model_names: Iterable[str] = DEFAULT_MODEL_NAMES,
    min_batch_size: int = DEFAULT_MIN_BATCH_SIZE,
    adaptation_groups: Iterable[str] = DEFAULT_ADAPTATION_GROUPS,
    adaptation_frac: float = DEFAULT_ADAPTATION_FRAC,
    drop_adaptation: bool = DEFAULT_DROP_ADAPTATION,
    rng_seed: int | None = None,
    overwrite: bool = False,
):
    tag = "-".join(
        [
            fpath_idps.name,
            str(min_batch_size),
            *adaptation_groups,
            str(adaptation_frac),
            str(rng_seed) if rng_seed is not None else "no_seed",
            *[model_name.removesuffix(".zip") for model_name in model_names],
        ]
    )
    dpath_transferred_model = dpath_out / tag / DNAME_MODEL
    fpath_settings = dpath_out / tag / FNAME_SETTINGS
    fpath_harmonized = dpath_out / tag / FNAME_HARMONIZED
    settings = locals().copy()

    if fpath_harmonized.exists() and not overwrite:
        raise KnownError(
            f"Output file already exists: {fpath_harmonized}. Use --overwrite to overwrite."
        )

    df_data = _get_merged_df(fpath_idps, fpath_adni_merge, fpath_config)
    click.secho(f"Merged dataframe: {df_data.shape}")

    df_data = _drop_batches(
        df_data, min_batch_size=min_batch_size, groups=adaptation_groups
    )

    click.secho(f"After dropping batches: {df_data.shape}")

    idx_adaptation = _get_idx_adaptation(
        df_data,
        groups=adaptation_groups,
        frac=adaptation_frac,
        rng_seed=rng_seed,
    )

    click.secho(f"N adaptation samples: {len(idx_adaptation)}")

    df_harmonized = df_data.copy()
    harmonized_vars_map = {}
    for model_name in model_names:
        df_harmonized, harmonized_vars_model = _apply_model(
            df_data,
            idx_adaptation,
            model_name,
            dpath_models,
            dpath_transferred_model,
            drop_adaptation,
        )
        harmonized_vars_map[model_name] = harmonized_vars_model

    df_harmonized[COL_ADAPTATION] = False
    df_harmonized.loc[idx_adaptation, COL_ADAPTATION] = True

    settings["harmonized_vars_map"] = harmonized_vars_map

    dpath_out.mkdir(parents=True, exist_ok=True)
    save_json(fpath_settings, settings)
    click.secho(f"Settings saved to {fpath_settings}.", fg="green")
    df_harmonized.to_csv(fpath_harmonized, index=True)
    click.secho(f"Harmonized data saved to {fpath_harmonized}.", fg="green")

    return df_data


@click.command(context_settings=CLICK_CONTEXT_SETTINGS)
@click.option(
    "--idps",
    "fpath_idps",
    type=click.Path(path_type=Path, exists=True, file_okay=True, dir_okay=False),
    required=True,
    envvar="ADNI_IDP_FILE",
    help="Path to the CSV file containing the ADNI IDPs.",
)
@click.option(
    "--adni-merge",
    "fpath_adni_merge",
    type=click.Path(path_type=Path, exists=True, file_okay=True, dir_okay=False),
    required=True,
    envvar="ADNI_MERGE_FILE",
    help="Path to the ADNIMERGE CSV file.",
)
@click.option(
    "--config",
    "fpath_config",
    type=click.Path(path_type=Path, file_okay=True, dir_okay=False),
    required=True,
    envvar="ADNI_CONFIG_FILE",
)
@click.option(
    "--dpath-models",
    type=click.Path(path_type=Path, file_okay=False, dir_okay=True),
    default=DEFAULT_DPATH_MODELS,
    help="Path to the directory where the normative models are stored.",
)
@click.option(
    "--dpath-out",
    type=click.Path(path_type=Path, file_okay=False, dir_okay=True),
    default=DEFAULT_DPATH_HARMONIZED_DATA,
    help="Path to the output directory.",
)
@click.option(
    "--model",
    "model_names",
    multiple=True,
    type=str,
    default=DEFAULT_MODEL_NAMES,
    help="Normative model name(s).",
)
@click.option(
    "--min-batch-size",
    type=int,
    default=DEFAULT_MIN_BATCH_SIZE,
    help="Batches with smaller than this will be dropped.",
)
@click.option(
    "--adaptation-groups",
    multiple=True,
    type=str,
    default=DEFAULT_ADAPTATION_GROUPS,
    help="Group(s) to use for adaptation.",
)
@click.option(
    "--adaptation-frac",
    type=float,
    default=DEFAULT_ADAPTATION_FRAC,
    help="Fraction of adaptation samples to use.",
)
@click.option(
    "--drop-adaptation/--keep-adaptation",
    is_flag=True,
    default=DEFAULT_DROP_ADAPTATION,
    help="Whether to drop adaptation samples from the harmonized data.",
)
@click.option(
    "--rng-seed",
    type=int,
    default=None,
    envvar="RNG_SEED",
    help="Random state for reproducibility.",
)
@click.option(
    "--overwrite",
    is_flag=True,
    default=False,
    help="Overwrite existing output files.",
)
def main(**params):
    try:
        apply_normative_models(**params)
    except KnownError as exception:
        click.secho(
            str(exception),
            fg="red",
            bold=True,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
