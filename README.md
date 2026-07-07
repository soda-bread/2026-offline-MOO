# 2026-HV-misleading-issue

Experiments for studying when the hypervolume (HV) indicator can give a
misleading view of offline data-driven multi-objective optimization performance.
The repository compares surrogate-assisted methods and recent probabilistic
baselines on the same benchmark set, with shared evaluation, plotting, and
BlueBEAR execution scripts.

## Repository Layout

```text
.
|-- baseline/              # Adapted baseline implementations and batch helpers
|-- bluebear/              # Standalone Python scripts for cluster execution
|-- experiments/           # Notebook workflows for Exp1-Exp8 and plotting
|-- results/               # Collected result tables and solution-objective files
|-- src/                   # Shared data, model, optimizer, metric, and plot code
|-- README.md
`-- requirements.txt
```

Generated Python caches, local environments, logs, notebook checkpoints,
AutoGluon artifacts, and common OS/editor files are ignored by Git.

## Experiment Set

The main experiment notebooks are in `experiments/`:

- `Exp1_GPR_RBF.ipynb`
- `Exp2_GPR_Matern.ipynb`
- `Exp3_Autogluon_QR.ipynb`
- `Exp4_BNN.ipynb`
- `Exp5_Prob_RVEA_2022.ipynb`
- `Exp6_Prob_MOEAD_2022.ipynb`
- `Exp7_TGPR_MO_2023.ipynb`
- `Exp8_DDMOEA_GAN_2024.ipynb`

Selected-problem test notebooks are also provided for the first four surrogate
methods:

- `Test_Exp1_GPR_RBF_selected_problems.ipynb`
- `Test_Exp2_GPR_Matern_selected_problems.ipynb`
- `Test_Exp3_Autogluon_QR_selected_problems.ipynb`
- `Test_Exp4_BNN_selected_problems.ipynb`

Surrogate-versus-real objective plotting notebooks live under
`experiments/plot_sur_real/`.

## Shared Configuration

`experiments/config.yaml` defines the benchmark and optimizer settings used by
the notebook workflows. The current configuration includes:

- Problems: DTLZ1-DTLZ7, Omni-test, truss2d, and welded beam.
- Objective count: 2.
- DTLZ decision variables: 10.
- Optimizers: NSGA-II, Dual-Ranking+NSGA-II, MOEA/D, and SMS-EMOA.
- Seeds: 1-30 for optimizer runs, with separate train/test seeds.
- Surrogates: GPR-RBF, GPR-Matern, AutoGluon quantile regression, and BNN.

`bluebear/config.yaml` mirrors the cluster execution settings for the standalone
scripts in `bluebear/`.

## Source Modules

```text
src/
|-- data.py             # Train/validation/test data generation
|-- experiment.py       # Optimization loops, evaluation, and result recording
|-- metrics.py          # HV, IGD+, and supporting metric utilities
|-- models.py           # GPR, AutoGluon QR, and BNN model wrappers
|-- nearest_offline.py  # Nearest-offline-sample utilities
|-- opt_problem.py      # pymoo problem wrapper for real/surrogate objectives
|-- other_functions.py  # Shared helper functions
|-- plotting.py         # Pareto, HV, uncertainty, and comparison plots
|-- survival.py         # Standard and dual-ranking survival operators
`-- uncertainty.py      # Coverage and calibration helpers
```

## Baselines

The `baseline/` directory contains adapted code and helpers for:

- Prob-RVEA (2022)
- Prob-MOEA/D (2022)
- TGPR-MO (2023)
- DDMOEA-GAN (2024)

Use `baseline/batch_experiments.py` for local baseline batch execution where
applicable. See the copied baseline subdirectories for their original project
metadata and dependencies.

## BlueBEAR Runs

The `bluebear/` directory contains notebook-style Python scripts for running
Exp1-Exp8 on BlueBEAR:

```bash
python bluebear/bluebear_exp1_gpr_rbf.py
python bluebear/bluebear_exp2_gpr_matern.py
python bluebear/bluebear_exp3_autogluon_qr.py
python bluebear/bluebear_exp4_bnn.py
python bluebear/bluebear_exp5_prob_rvea_2022.py
python bluebear/bluebear_exp6_prob_moead_2022.py
python bluebear/bluebear_exp7_tgpr_mo_2023.py
python bluebear/bluebear_exp8_ddmoea_gan_2024.py
```

Each script writes progress to the terminal and to a matching `.log` file.
Cluster outputs are written beside `bluebear/config.yaml` as `result.csv` and
`result_exp1-8.txt`.

## Setup

Recommended Python version: 3.10+.

Install the core dependencies:

```bash
python -m pip install -r requirements.txt
```

Main packages include `numpy`, `pandas`, `matplotlib`, `plotly`,
`scikit-learn`, `pymoo`, `GPy`, `autogluon.tabular`, `PyYAML`, `jupyter`, and
`pyro-ppl`.

## Quick Checks

Run a syntax check for the shared source modules:

```bash
python -m py_compile src/*.py
```

Check the Git working tree before committing:

```bash
git status --short
```
