import configparser
from pathlib import Path

import json
import pandas as pd


def get_fpath_config(dpath_node: Path) -> Path:
    return dpath_node / "etc" / "config.ini"


def get_node_config(dpath_node: Path) -> configparser.ConfigParser:
    fpath_config = get_fpath_config(dpath_node)
    if not fpath_config.exists():
        raise FileNotFoundError(
            f"Node config file not found: {fpath_config}"
            ". Make sure node has been initialized."
        )

    config = configparser.ConfigParser()
    config.read(str(fpath_config))
    return config


def get_fpath_db(dpath_node: Path) -> Path:
    config = get_node_config(dpath_node)
    fpath_db = (get_fpath_config(dpath_node).parent / config["default"]["db"]).resolve()
    return fpath_db


def load_node_db(
    *, fpath_db: Path | None = None, dpath_node: Path | None = None
) -> pd.DataFrame:

    if fpath_db is None:
        if dpath_node is None:
            raise ValueError("Either fpath_db or dpath_node must be provided.")
        fpath_db = get_fpath_db(dpath_node)

    if not fpath_db.exists():
        raise FileNotFoundError(f"Node database file not found: {fpath_db}")

    try:
        db_json = json.loads(fpath_db.read_text())
        df_datasets = pd.DataFrame(db_json["Datasets"]).T
    except (json.JSONDecodeError, KeyError) as exception:
        raise RuntimeError(
            f"Error parsing the database file {fpath_db}: {type(exception)} {exception}"
        ) from exception

    return df_datasets
