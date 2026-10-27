# Logistic SDE online-smoothing experiment

This directory contains the numerical experiment for the logistic continuous-discrete state-space model. Run the experiment with `script.py`, then use `logistic_sde_results.ipynb` to load and plot the saved outputs.

## Prerequisites

Set up and activate the project's Python environment from the repository root (see the root [`README.md`](../../README.md) for the environment setup). The environment needs the project itself, `particles`, pandas, matplotlib, seaborn, and Jupyter installed.

Run the commands below from this directory. This is important: both the script and the notebook use paths relative to `paper/logistic_sde_o_smoothing/`.

```shell
cd paper/logistic_sde_o_smoothing
```

## Run an experiment

Configurations are JSON files in `config/`. Pass a configuration filename (not its path) with `-c`:

```shell
python script.py -c logistic_int_config_final.json
```

For the companion guided-filter experiment used in the results notebook, run:

```shell
python script.py -c logistic_int_g_config_final.json
```

The script creates missing output directories before saving. A local run writes:

```text
results/smc/local/r_<config filename>
```

For example, the first command writes `results/smc/local/r_logistic_int_config_final.json`. If a configuration enables `run_lgssm`, the script also writes its ground-truth output to:

```text
results/ground_truth/local/r_<config filename>
```

The supplied final configurations use substantial particle counts and 96 runs, so they are intended as full experiments rather than quick smoke tests. To test a change, copy a configuration and reduce `T`, `M`, and the algorithm `N` values before running it.

## Visualise the results

Start Jupyter from this directory and open the notebook:

```shell
jupyter lab logistic_sde_results.ipynb
```

The notebook's results-loading cell currently assumes results generated on the remote machine:

```python
boot_results = tools.load('logistic_int_config_final.json', remote=True)
guided_results = tools.load('logistic_int_g_config_final.json', remote=True)
```

After running the commands above locally, change both `remote=True` arguments to `remote=False`, then run the notebook cells from top to bottom. The notebook loads the two saved JSON files, converts the estimates to long form, and produces the logistic SDE and estimator-comparison figures.

To save the PDFs produced by the notebook, create its output directory once before executing cells with `savefig=True`:

```shell
mkdir -p figures
```

The exported files are `figures/fig_8_logistic_cdssm.pdf` and `figures/fig_9_logistic_sde_results.pdf`.