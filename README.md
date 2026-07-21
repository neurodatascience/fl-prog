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

## Steps

These scripts should be run sequentially.

### Get data

#### Synthetic

```shell
# iid case
./scripts/simulate_data.py --tag iid

# non-iid cases
./scripts/simulate_data.py --t0-min 0 --t0-max 0.32 --t0-min 0.33 --t0-max 0.65 --t0-min 0.66 --t0-max 1 --tag non_overlapping_t0
./scripts/simulate_data.py --sigma 0.05 --sigma 0.1 --sigma 0.15 --tag unequal_sigma
```

#### ADNI

```shell
./scripts/get_adni_data.py --tag adni_iid --iid
./scripts/get_adni_data.py --tag adni_noniid --non-iid
```

<!-- ### Split into train and test sets

```shell
./scripts/split_train_test.py --tag adni_iid
./scripts/split_train_test.py --tag adni_noniid
``` -->

### Optional: Regress biological age out from biomarkers

```shell
./scripts/regress_age_out.py --tag adni_iid --mode pooled-sites

./scripts/regress_age_out.py --tag adni_noniid --mode site
# with optional --min-max
```
Then, must run all downstream scripts (e.g., merge_data.py, create_nodes.py, etc.)
with --tag adni_\<iid / non_iid\>_age_adjusted_\<minmax / NOTHING\>

### Merge data for centralized case

```shell
./scripts/merge_data.py --tag iid
./scripts/merge_data.py --tag non_overlapping_t0
./scripts/merge_data.py --tag unequal_sigma

./scripts/merge_data.py --tag adni_iid
./scripts/merge_data.py --tag adni_noniid
```

### Create Fed-BioMed nodes and update their `config.ini`

This only needs to be run if some of the nodes haven't been created yet.

```shell
./scripts/create_nodes.py --tag iid
./scripts/create_nodes.py --tag non_overlapping_t0
./scripts/create_nodes.py --tag unequal_sigma

./scripts/create_nodes.py --tag adni_iid
./scripts/create_nodes.py --tag adni_noniid
```

### Add data to nodes

If needed, use `--wipe` to clear existing datasets from each node.

```shell
./scripts/add_datasets_to_nodes.py --tag iid
./scripts/add_datasets_to_nodes.py --tag non_overlapping_t0
./scripts/add_datasets_to_nodes.py --tag unequal_sigma

./scripts/add_datasets_to_nodes.py --tag adni_iid
./scripts/add_datasets_to_nodes.py --tag adni_noniid
```

### Start the nodes in separate processes

Start each node in a separate Terminal.
These need to be running for the next script (`run_fedbiomed.py`) to work.
Use Ctrl+C to stop a node when done.

```bash
# replace <NODE_ID> by '1', '2', '3', ..., or 'centralized'
fedbiomed node -p ./fedbiomed/node-<NODE_ID> start
```

### Run model fitting

```shell
./scripts/run_fedbiomed.py --tag iid --n-rounds 5 --n-updates 100 --learning-rate 0.05 --time-shift-range 0 1
./scripts/run_fedbiomed.py --tag non_overlapping_t0 --n-rounds 5 --n-updates 100 --learning-rate 0.05 --time-shift-range 0 1
./scripts/run_fedbiomed.py --tag unequal_sigma --n-rounds 5 --n-updates 100 --learning-rate 0.05 --time-shift-range 0 1

./scripts/run_fedbiomed.py --tag adni_iid --learning-rate 0.05 --n-rounds 6 --n-updates 25 --time-shift-range 0 3 --lambda 10 --training-replies --aggregated-params
./scripts/run_fedbiomed.py --tag adni_noniid --learning-rate 0.05 --n-rounds 6 --n-updates 25 --time-shift-range 0 3 --lambda 10 --training-replies --aggregated-params
```

### Plot

#### Synthetic data experiments

Run cells in `./notebooks/model_fits_synthetic.ipynb`

#### ADNI data experiments

Run cells in `./notebooks/model_fits_adni.ipynb`
