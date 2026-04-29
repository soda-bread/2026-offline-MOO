# agent.md

## Purpose

This file records the expected workflow and result-recording requirements for experiments in this repository.

## Experiment Workflow

Use this workflow when running `Exp1_GPR-RBF[experiment].py` or related experiment notebooks.

1. Work from the repository root.
2. Load experiment inputs from `configs/exp1_gpr_rbf.yaml`.
3. Run experiments from the repository root so relative config and output paths resolve correctly.
4. Keep generated outputs under the configured `output_dir`.
5. Do not commit generated outputs unless explicitly requested.

## Required Result Records

Every experiment result table should include `problem_name` so results from `dtlz1` to `dtlz7` can be compared after batch runs.

### Bias-Variance Experiment

The bias-variance experiment should record both per-training rows and summary rows.

Per-training records should include:

- `problem_name`
- `model_type`
- `run`
- `train_seed`
- `lengthscale_weight`
- `test_mse`
- `f1_lengthscale`
- `f1_kernel_variance`
- `f1_noise`
- `f2_lengthscale`
- `f2_kernel_variance`
- `f2_noise`

Summary records should include:

- `problem_name`
- `model_type`
- `n_runs`
- `lengthscale_weight`
- `bias_squared`
- `variance`
- `test_mse_mean`
- `test_mse_std`

Use `bias_squared` for the quantity computed as:

```python
np.mean((mean_pred - y_test_ref) ** 2)
```

Do not label this quantity as plain `bias`. If true bias is needed, compute:

```python
np.sqrt(bias_squared)
```

### Optimization Experiment

The optimization summary for seed-42 bias/variance models should include:

- `problem_name`
- `model_type`
- `train_seed`
- `lengthscale_weight`
- `bias_squared`
- `variance`
- `mse_mean`
- `mse_std`
- `igd_plus_mean`
- `igd_plus_std`
- `hv_surrogate_mean`
- `hv_surrogate_std`
- `hv_real_mean`
- `hv_real_std`

Formatting expectations:

- `MSE`, `IGD+`, `bias_squared`, and `variance` should be recorded in scientific notation with 3 decimals, for example `1.234e-03`.
- `Sur HV` and `Real HV` should be recorded as fixed-point decimals with 3 decimals, for example `0.875`.
- Internal Python return values should remain raw numeric values. Apply formatting only when printing or writing CSV rows.

## Expected Output Files

When `run_bias_variance_ea` is enabled, write:

- `bias_variance_training_records.csv`
- `bias_variance_summary.csv`
- `optimization_summary_seed42_bias_variance_models.csv`

These files should be placed under the configured `output_dir`, for example:

```text
outputs/exp1_gpr_rbf/dtlz1/
```

## GitHub Update Workflow

When updating GitHub:

1. Commit only intended source and documentation files.
2. Do not commit `.venv/`, `__pycache__/`, `.DS_Store`, logs, or generated outputs unless explicitly requested.
3. Prefer normal `git push` when credentials are available.
4. If local GitHub credentials are unavailable, use the GitHub integration as a fallback.
