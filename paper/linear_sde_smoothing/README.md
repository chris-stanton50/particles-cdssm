# Smoothing Experiment

Re-run the Linear SDEs smoothing experiment for the paper "Particle-based inference for continuous-discrete state space models" here.

The details of the experiment are outlined in Section 6.1.2 of the paper.

## Running the experiment

Experiment parameters are stored in individual JSON files in [`config/`](./config). Each config specifies the simulation settings, CD-SSM, random seed, particle counts, quantiles, and FK models to run.

To run every configured experiment, execute the runner from this directory:

```bash
bash run_smoothing_experiments.sh
```

The runner discovers every `*.json` file in `config/` and invokes `smoothing_experiment.py` once for each config. The full experiment takes a few hours.

To run one config manually, pass its filename without the `.json` extension:

```bash
python smoothing_experiment.py o_MV_OU_0.05
```

Results are stored in `./results` using the config name:

```text
res_<config_name>_part_1.json
res_<config_name>_part_2.json
res_<config_name>_part_3.json
res_<config_name>_meta.pkl
```

The results plot can be reproduced in `smoothing_experiment_results.ipynb` and is saved to `./figures/fig_3_smoothing_results.pdf`.