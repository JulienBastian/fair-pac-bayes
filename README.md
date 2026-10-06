# PAC-Bayesian Generalization Guarantees for Fairness

This repository contains the code used for the experiments presented in
**“PAC-Bayesian Generalization Guarantees for Fairness on Stochastic and
Deterministic Classifiers.”**

It includes the proposed PAC-Bayesian methods, the FERM and Agarwal baselines,
the data preparation utilities, and the notebook used to generate the final
figures and tables. The detailed experimental protocol is described in the
paper.

## Installation

The reference environment uses Python 3.8.

```bash
python3.8 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```


## Reproducing the experiments

Download and cache the datasets:

```bash
python scripts/prepare_raw_data.py --raw-root data/raw
```

The complete experimental grids are defined in the SLURM scripts:

```bash
sbatch run/run_forests.sh
sbatch run/run_ferm.sh
sbatch run/run_agarwal_cart.sh
```

These scripts reproduce respectively:

- the proposed PAC-Bayesian methods and the O1/O2 comparisons;
- the linear FERM baseline;
- the Agarwal reduction with a decision trees voters.

Individual experiments can also be launched directly with `training.py`,
`ferm_training.py`, or `agarwal_training.py` using Hydra command-line
overrides.

For example, the following command runs one PAC-Bayesian configuration:

```bash
python training.py num_trials=1 dataset=GERMAN model.n=100 model.prior=adjusted model.pred=RandomForests training.seed=541944 training.risk=Triple training.fairness_metric=EP training.sensible_variable=GENDER training.distribution=categorical training.alpha=0.5 fairness_bound.type=ours_stochastic_fairness
```

All outputs are written under:

```text
results/
```

Each configuration contains its aggregated results and one directory per
random seed.

## Reproducing the figures and tables


To generate figures and LaTeX tables use notebook `analysis/final_benchmark_analysis.ipynb`,results are written to:

```text
analysis/exports/
```
