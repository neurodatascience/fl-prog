#!/usr/bin/env python
"""Merge per-lane results of a multi-seed run (see ``run_adni_seeds.sh``) into a
single dated results directory.

Input layout::

    <staging-dir>/<lane>/<date>/<tag>/...

Output layout (same as a regular, non-staged run)::

    <results-dir>/<date>/<tag>/...

Fed-BioMed runs that only differ by setup (e.g. federated and centralized fits on
the same IID tag, run concurrently in separate lanes) are merged into a single
``*-estimated_params.json`` (and ``*-fit_quality*.tsv``/``*-param_recovery*.tsv``),
as if both setups had been run in a single ``run_fedbiomed.py`` call. Files are copied,
never moved.
"""

import filecmp
import re
import shutil
from pathlib import Path

import click
import pandas as pd

from fl_prog.utils.constants import CLICK_CONTEXT_SETTINGS
from fl_prog.utils.io import DEFAULT_DPATH_RESULTS, load_json, save_json

SUFFIX_ESTIMATED_PARAMS = "-estimated_params.json"
KEYS_BY_SETUP = ("results", "individual_parameters_test")
# per-setup TSVs, to be merged across lanes (e.g. federated + centralized on the same tag)
SUBSTRINGS_TSV_BY_SETUP = ("-fit_quality", "-param_recovery")
PATTERN_DATE = re.compile(r"^\d{4}_\d{2}_\d{2}$")


def _replace_paths(obj, pattern: re.Pattern, replacement: str):
    """Recursively replace path prefixes in strings."""
    if isinstance(obj, str):
        return pattern.sub(replacement, obj)
    if isinstance(obj, list):
        return [_replace_paths(x, pattern, replacement) for x in obj]
    if isinstance(obj, dict):
        return {k: _replace_paths(v, pattern, replacement) for k, v in obj.items()}
    return obj


def _is_contained(dst, src) -> bool:
    """Whether all (nested) entries of ``src`` are already in ``dst``."""
    if isinstance(dst, dict) and isinstance(src, dict):
        return all(k in dst and _is_contained(dst[k], v) for k, v in src.items())
    return dst == src


def _deep_merge(dst: dict, src: dict, context: str) -> dict:
    for key, value in src.items():
        if key not in dst:
            dst[key] = value
        elif isinstance(dst[key], dict) and isinstance(value, dict):
            _deep_merge(dst[key], value, f"{context}.{key}")
        elif dst[key] != value:
            raise ValueError(f"Conflicting values for {context}.{key}")
    return dst


def _merge_json(
    fpath_src: Path, fpath_dst: Path, pattern: re.Pattern, replacement: str
):
    json_src = load_json(fpath_src)
    json_src["settings"] = _replace_paths(json_src["settings"], pattern, replacement)

    if not fpath_dst.exists():
        save_json(fpath_dst, json_src)
        return

    json_dst = load_json(fpath_dst)
    if json_dst == json_src or all(
        _is_contained(json_dst.get(key, {}), json_src.get(key, {}))
        for key in KEYS_BY_SETUP
    ):
        return  # already copied/merged
    if not fpath_dst.name.startswith("fedbiomed"):
        raise FileExistsError(f"Cannot merge non-Fed-BioMed results: {fpath_dst}")
    setups_dst = set(json_dst["results"])
    setups_src = set(json_src["results"])
    if setups_dst & setups_src:
        raise ValueError(
            f"Setups {sorted(setups_dst & setups_src)} in both {fpath_src} and "
            f"{fpath_dst}"
        )
    for key in KEYS_BY_SETUP:
        json_dst[key] = _deep_merge(
            json_dst.get(key, {}), json_src.get(key, {}), context=key
        )
    json_dst["settings"]["setups"] = sorted(
        set(json_dst["settings"].get("setups", []))
        | set(json_src["settings"]["setups"])
    )
    save_json(fpath_dst, json_dst)
    print(f"Merged setups {sorted(setups_src)} into {fpath_dst}")


def _merge_tsv(fpath_src: Path, fpath_dst: Path):
    df_src = pd.read_csv(fpath_src, sep="\t")
    if fpath_dst.exists():
        df_dst = pd.read_csv(fpath_dst, sep="\t")
        if "setup" not in df_src.columns:
            raise FileExistsError(f"Cannot merge {fpath_src} into {fpath_dst}")
        df_src = pd.concat(
            [df_dst, df_src.loc[~df_src["setup"].isin(set(df_dst["setup"]))]],
            ignore_index=True,
        )
    df_src.to_csv(fpath_dst, sep="\t", index=False)


def _copy(fpath_src: Path, fpath_dst: Path):
    if fpath_src.is_dir():
        shutil.copytree(fpath_src, fpath_dst, dirs_exist_ok=True)
    elif fpath_dst.exists():
        if not filecmp.cmp(fpath_src, fpath_dst, shallow=False):
            raise FileExistsError(f"{fpath_dst} already exists and differs")
    else:
        shutil.copy2(fpath_src, fpath_dst)


@click.command(context_settings=CLICK_CONTEXT_SETTINGS)
@click.option(
    "--staging-dir",
    "dpath_staging",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    required=True,
)
@click.option(
    "--results-dir",
    "dpath_results",
    type=click.Path(path_type=Path, file_okay=False),
    default=DEFAULT_DPATH_RESULTS,
)
def main(dpath_staging: Path, dpath_results: Path):
    for dpath_lane in sorted(p for p in dpath_staging.iterdir() if p.is_dir()):
        # only <date> dirs (skips "latest" symlinks and e.g. logs/summary dirs)
        dpaths_dated = [
            p
            for p in dpath_lane.iterdir()
            if p.is_dir() and not p.is_symlink() and PATTERN_DATE.match(p.name)
        ]
        for dpath_dated in sorted(dpaths_dated):
            dpath_out_dated = dpath_results / dpath_dated.name
            # <any prefix>/<staging dir name>/<lane> -> <results dir>
            # (settings can contain different aliases of the same absolute path)
            pattern = re.compile(
                r"[^\s\"']*/"
                + re.escape(dpath_staging.resolve().name)
                + "/"
                + re.escape(dpath_lane.name)
                + r"(?=/|$)"
            )
            replacement = str(dpath_results.resolve())
            for dpath_tag in sorted(p for p in dpath_dated.iterdir() if p.is_dir()):
                dpath_out = dpath_out_dated / dpath_tag.name
                dpath_out.mkdir(parents=True, exist_ok=True)
                for fpath in sorted(dpath_tag.iterdir()):
                    fpath_out = dpath_out / fpath.name
                    if fpath.name.endswith(SUFFIX_ESTIMATED_PARAMS):
                        _merge_json(fpath, fpath_out, pattern, replacement)
                    elif fpath.suffix == ".tsv" and any(
                        s in fpath.name for s in SUBSTRINGS_TSV_BY_SETUP
                    ):
                        _merge_tsv(fpath, fpath_out)
                    else:
                        _copy(fpath, fpath_out)
            print(f"Copied {dpath_lane.name}: {dpath_dated} -> {dpath_out_dated}")


if __name__ == "__main__":
    main()
