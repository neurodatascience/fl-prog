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
./scripts/simulate_data.py --tag iid

# non-iid cases
./scripts/simulate_data.py --t0-min 0 --t0-max 0.32 --t0-min 0.33 --t0-max 0.65 --t0-min 0.66 --t0-max 1 --tag non_overlapping_t0
./scripts/simulate_data.py --sigma 0.05 --sigma 0.1 --sigma 0.15 --tag unequal_sigma
```

Merge data for centralized case:

```shell
./scripts/merge_data.py --tag iid
./scripts/merge_data.py --tag non_overlapping_t0
./scripts/merge_data.py --tag unequal_sigma
```

Create Fed-BioMed nodes and update their `config.ini`:

```shell
scripts/create_nodes.py --tag iid
scripts/create_nodes.py --tag non_overlapping_t0
scripts/create_nodes.py --tag unequal_sigma
```

Add data to nodes:

```shell
./scripts/add_datasets_to_nodes.py --tag iid
./scripts/add_datasets_to_nodes.py --tag non_overlapping_t0
./scripts/add_datasets_to_nodes.py --tag unequal_sigma
```

Start the nodes in separate processes:
- For each node, make sure you are in the node directory, e.g. `./fedbiomed/node-1`
- Then run `fedbiomed node -p . start`

Run model fitting:

```shell
./scripts/run_fedbiomed.py --tag iid
./scripts/run_fedbiomed.py --tag non_overlapping_t0
./scripts/run_fedbiomed.py --tag unequal_sigma
```

Plot: run cells in `./notebooks/figs.ipynb`

