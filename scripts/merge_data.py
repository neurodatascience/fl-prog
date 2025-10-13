#!/usr/bin/env python

from pathlib import Path

import click
import pandas as pd

from fl_prog.utils.constants import CLICK_CONTEXT_SETTINGS
from fl_prog.utils.io import get_dpath_latest

PREFIX = "simulated_data"
FNAME_MERGED = f"{PREFIX}-merged.tsv"


@click.command(context_settings=CLICK_CONTEXT_SETTINGS)
@click.argument(
    "dpath_data",
    type=click.Path(path_type=Path, file_okay=False, dir_okay=True),
    envvar="DPATH_DATA",
)
def merge_data(dpath_data):
    dpath_out = get_dpath_latest(dpath_data, use_today=True)

    fpaths_tsv = [
        fpath
        for fpath in sorted(dpath_out.glob(f"{PREFIX}*.tsv"))
        if fpath.name != FNAME_MERGED
    ]
    dfs = [pd.read_csv(fpath, sep="\t") for fpath in fpaths_tsv]
    df = pd.concat(dfs)

    fpath_out = dpath_out / FNAME_MERGED
    df.to_csv(fpath_out, sep="\t", index=False)
    print(f"Saved merged data to {fpath_out}")


if __name__ == "__main__":
    merge_data()
