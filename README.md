# fl-prog

Federated disease progression modelling.

## Setup

Check `pyproject.toml` for supported Python version(s).

In a Python environment managed by `uv`:
```shell
uv sync
```

In other types of Python environments (`venv`, `conda`, etc.):
```shell
pip install -e .
```

## Sample commands

Generate synthetic data:

```shell
# iid case
./scripts/simulate_data.py ./data --tag iid

# non-iid cases
./scripts/simulate_data.py ./data --t0-min 0 --t0-max 0.32 --t0-min 0.33 --t0-max 0.65 --t0-min 0.66 --t0-max 1 --tag non_overlapping_t0
```

Merge data for centralized case:

```shell
./scripts/merge_data.py ./data --tag iid
./scripts/merge_data.py ./data --tag non_overlapping_t0
```

Create Fed-BioMed nodes:
```shell
# TODO
```

Add data to nodes:
```shell
./scripts/add_datasets_to_nodes.py ./data --tag iid
./scripts/add_datasets_to_nodes.py ./data --tag non_overlapping_t0
```

Run model fitting:
```shell
./scripts/run_fedbiomed.py ./data --tag iid
./scripts/run_fedbiomed.py ./data --tag non_overlapping_t0
```

Plot: run cells in `./notebooks/figs.ipynb`

