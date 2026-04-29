"""Python script version of `Exp1_GPR-RBF[experiment].ipynb`.

This script keeps the notebook's core workflow and adds a command-line entry
for running the 30x bias-variance experiment.
"""

import argparse
import csv
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from pymoo.operators.sampling.lhs import LHS
from sklearn.metrics import mean_squared_error

from src.data import generate_data
from src.experiment import run_experiment
from src.metrics import get_metrics
from src.models import GPR_RBF, gpr_pred_mean_std
from src.opt_problem import build_problem
from src.other_functions import mean_std, print_gpr_params
from src.plotting import plot_obj_2d, plot_y_true_pred, plot_z_score
from src.survival import Survival_dual_ranking, Survival_standard
from src.uncertainty import coverage, find_alpha

warnings.filterwarnings("ignore", message=".*load_learner.*pickle.*")
np.set_printoptions(precision=4, suppress=True)

DEFAULT_CONFIG_PATH = Path("configs/exp1_gpr_rbf.yaml")


def load_experiment_config(config_path):
    with Path(config_path).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    if "bias_variance_configs" not in config:
        raise ValueError("YAML config must define bias_variance_configs.")
    return config


def prepare_problem(problem_name: str = "dtlz1", n_var: int = 10, n_obj: int = 2):
    problem = build_problem(problem_name=problem_name, n_var=n_var, n_obj=n_obj)

    print(f"Problem name: {problem_name}")
    print(f"Cons: {problem.n_constr}")
    print(f"Var: {n_var}")
    print(f"Obj: {n_obj}")

    hv, igd_plus, obj_min, obj_max, ref_point = get_metrics(
        problem_name=problem_name,
        problem=problem,
        n_var=n_var,
        n_obj=n_obj,
    )
    print("\nMin-Max normalization -> Min:", obj_min)
    print("Min-Max normalization -> Max:", obj_max)
    print("HV Reference points:", ref_point)

    return problem, hv, igd_plus, obj_min, obj_max


def sample_data(problem, sample_size: int, train_seed: int = 42, test_seed: int = 1):
    X_train, y_train, X_val, y_val, X_test, y_test = generate_data(
        problem=problem,
        sample_size=sample_size,
        sampling=LHS(),
        train_seed=train_seed,
        val_size=100,
        test_size=100,
        test_seed=test_seed,
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


def run_single_gpr(X_train, y_train, X_test, y_test, plot: bool = True):
    model_f1 = GPR_RBF()
    model_f2 = GPR_RBF()

    model_f1.fit(X_train, y_train[:, 0])
    model_f2.fit(X_train, y_train[:, 1])
    print_gpr_params(model_f1, model_f2)

    pred_mean, pred_std, mean_f1, std_f1, mean_f2, std_f2 = gpr_pred_mean_std(
        model_f1, model_f2, X_test, noiseless=True
    )
    print(f"\nGPR(RBF) MSE: {mean_squared_error(y_test, pred_mean):.3e}\n")

    if plot:
        plot_y_true_pred(y_test, pred_mean)

    return model_f1, model_f2, pred_mean, pred_std, mean_f1, std_f1, mean_f2, std_f2


def run_high_bias_or_variance(X_train, y_train, X_test, y_test, lengthscale_weight: float, plot: bool = True):
    model_f1 = GPR_RBF()
    model_f2 = GPR_RBF()
    model_f1.fit_high_bias_variance(X_train, y_train[:, 0], lengthscale_weight=lengthscale_weight)
    model_f2.fit_high_bias_variance(X_train, y_train[:, 1], lengthscale_weight=lengthscale_weight)
    print_gpr_params(model_f1, model_f2)

    pred_mean, pred_std, _, _, _, _ = gpr_pred_mean_std(model_f1, model_f2, X_test, noiseless=True)
    print(f"\nGPR(RBF) MSE: {mean_squared_error(y_test, pred_mean):.3e}")

    if plot:
        plot_y_true_pred(y_test, pred_mean)

    return model_f1, model_f2, pred_mean, pred_std


def run_bias_variance_experiment(
    problem,
    sample_size: int,
    lengthscale_weight: float,
    model_type: str,
    n_runs: int = 30,
    return_records: bool = False,
):
    preds = []
    y_test_ref = None
    training_records = []

    for seed in range(1, n_runs + 1):
        X_train_i, y_train_i, _, _, X_test_i, y_test_i = sample_data(
            problem=problem,
            sample_size=sample_size,
            train_seed=seed,
            test_seed=1,
        )

        model_f1_i = GPR_RBF()
        model_f2_i = GPR_RBF()
        model_f1_i.fit_high_bias_variance(
            X_train_i,
            y_train_i[:, 0],
            lengthscale_weight=lengthscale_weight,
        )
        model_f2_i.fit_high_bias_variance(
            X_train_i,
            y_train_i[:, 1],
            lengthscale_weight=lengthscale_weight,
        )

        pred_mean_i, _, _, _, _, _ = gpr_pred_mean_std(model_f1_i, model_f2_i, X_test_i, noiseless=True)
        preds.append(pred_mean_i)

        if return_records:
            training_records.append({
                "model_type": model_type,
                "run": seed,
                "train_seed": seed,
                "lengthscale_weight": lengthscale_weight,
                "test_mse": mean_squared_error(y_test_i, pred_mean_i),
                "f1_lengthscale": np.array2string(
                    model_f1_i.model.kern.lengthscale.values,
                    precision=8,
                    separator=" ",
                ),
                "f1_kernel_variance": float(model_f1_i.model.kern.variance.values[0]),
                "f1_noise": float(model_f1_i.model.Gaussian_noise.variance.values[0]),
                "f2_lengthscale": np.array2string(
                    model_f2_i.model.kern.lengthscale.values,
                    precision=8,
                    separator=" ",
                ),
                "f2_kernel_variance": float(model_f2_i.model.kern.variance.values[0]),
                "f2_noise": float(model_f2_i.model.Gaussian_noise.variance.values[0]),
            })

        if y_test_ref is None:
            y_test_ref = y_test_i

    preds = np.array(preds)  # shape: (n_runs, n_test, n_obj)
    mean_pred = preds.mean(axis=0)
    bias_squared = np.mean((mean_pred - y_test_ref) ** 2)
    variance = np.mean(np.var(preds, axis=0))

    if return_records:
        return bias_squared, variance, training_records
    return bias_squared, variance


def run_uncertainty_eval(y_test, pred_mean, pred_std, X_val, y_val, model_f1, model_f2, mean_f1, std_f1, mean_f2, std_f2, plot: bool = True):
    for k in [1.645]:
        per_dim, overall = coverage(y_test, pred_mean, pred_std, k=k)
        print(f"k={k}: per_dim={per_dim * 100}%, overall={overall * 100:.1f}%")

    if plot:
        plot_z_score(y_test, pred_mean, pred_std)

    alpha_c90_f1 = find_alpha(X_val, y_val[:, 0], model_f1, target_coverage=0.9)
    alpha_c90_f2 = find_alpha(X_val, y_val[:, 1], model_f2, target_coverage=0.9)
    print(f"alpha_c90_f1={alpha_c90_f1:.2f}, alpha_c90_f2={alpha_c90_f2:.2f}")

    f_upper_c90 = np.stack(
        [mean_f1 + alpha_c90_f1 * std_f1, mean_f2 + alpha_c90_f2 * std_f2],
        axis=1,
    )
    print("F_upper_c90\n", f_upper_c90[0:5], "\n")

    return alpha_c90_f1, alpha_c90_f2


def summarize_ea_results(results):
    mean_mse, std_mse = mean_std(results["mse_list"])
    mean_igd, std_igd = mean_std(results["igd_list"])
    mean_hv_real, std_hv_real = mean_std(results["hv_real_list"])
    mean_hv_surrogate, std_hv_surrogate = mean_std(results["hv_surrogate_list"])

    return {
        "mse_mean": mean_mse,
        "mse_std": std_mse,
        "igd_plus_mean": mean_igd,
        "igd_plus_std": std_igd,
        "hv_surrogate_mean": mean_hv_surrogate,
        "hv_surrogate_std": std_hv_surrogate,
        "hv_real_mean": mean_hv_real,
        "hv_real_std": std_hv_real,
    }


def print_ea_summary(summary):
    print(f"MSE: Mean = {summary['mse_mean']:.3e}, Std = {summary['mse_std']:.3e}")
    print(f"IGD+: Mean = {summary['igd_plus_mean']:.3e}, Std = {summary['igd_plus_std']:.3e}")
    print(f"Sur HV: Mean = {summary['hv_surrogate_mean']:.3f}, Std = {summary['hv_surrogate_std']:.3f}")
    print(f"Real HV: Mean = {summary['hv_real_mean']:.3f}, Std = {summary['hv_real_std']:.3f}")


def format_ea_summary_for_record(summary):
    return {
        key: f"{value:.3f}" if key.startswith("hv_") else f"{value:.3e}"
        for key, value in summary.items()
    }


def format_scientific_3(value):
    return f"{value:.3e}"


def run_ea_experiment(problem, problem_name, n_gen, pop_size, model_f1, model_f2, obj_min, obj_max, hv, igd_plus, seeds=range(1, 31)):
    results = run_experiment(
        problem=problem,
        problem_name=problem_name,
        n_gen=n_gen,
        pop_size=pop_size,
        model_f1=model_f1,
        model_f2=model_f2,
        obj_min=obj_min,
        obj_max=obj_max,
        hv=hv,
        igd_plus=igd_plus,
        use_surrogate="GPR_uncertainty",
        survival_function=Survival_standard(),
        use_callback=False,
        seeds=seeds,
    )

    summary = summarize_ea_results(results)
    print_ea_summary(summary)
    return summary, results


def write_rows_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_ea_for_seed42_bias_variance_models(
    problem,
    problem_name,
    n_gen,
    pop_size,
    seed42_models_by_type,
    obj_min,
    obj_max,
    hv,
    igd_plus,
    output_dir,
    seeds,
):
    output_dir = Path(output_dir)
    summary_rows = []

    for model_type, model_info in seed42_models_by_type.items():
        print(f"\n=== EA for {problem_name} | seed=42 {model_type} model ===")

        summary, _ = run_ea_experiment(
            problem=problem,
            problem_name=problem_name,
            n_gen=n_gen,
            pop_size=pop_size,
            model_f1=model_info["model_f1"],
            model_f2=model_info["model_f2"],
            obj_min=obj_min,
            obj_max=obj_max,
            hv=hv,
            igd_plus=igd_plus,
            seeds=seeds,
        )

        row = {
            "problem_name": problem_name,
            "model_type": model_type,
            "train_seed": 42,
            "lengthscale_weight": model_info["lengthscale_weight"],
            "bias_squared": format_scientific_3(model_info["bias_squared"]),
            "variance": format_scientific_3(model_info["variance"]),
            **format_ea_summary_for_record(summary),
        }
        summary_rows.append(row)
        write_rows_csv(output_dir / "optimization_summary_seed42_bias_variance_models.csv", summary_rows)

    return summary_rows


def main():
    parser = argparse.ArgumentParser(description="Run Exp1 GPR (RBF) notebook workflow as Python script.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="YAML experiment config file.")
    parser.add_argument("--problem_name", default=None)
    parser.add_argument("--n_var", type=int, default=None)
    parser.add_argument("--n_obj", type=int, default=None)
    parser.add_argument("--n_gen", type=int, default=None)
    parser.add_argument("--pop_size", type=int, default=None)
    parser.add_argument("--n_runs", type=int, default=None, help="Runs for bias-variance experiment.")
    parser.add_argument("--no_plot", action="store_true", default=None, help="Disable plotting calls.")
    parser.add_argument("--run_ea", action="store_true", default=None, help="Run expensive EA experiment section.")
    parser.add_argument(
        "--run_bias_variance_ea",
        action="store_true",
        default=None,
        help="After the 30-run bias-variance experiment, run EA for seed=42 configured bias/variance models.",
    )
    parser.add_argument("--ea_seed_start", type=int, default=None)
    parser.add_argument("--ea_seed_end", type=int, default=None, help="Exclusive end for EA seeds; default runs seeds 1..30.")
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    problem_name = args.problem_name if args.problem_name is not None else config.get("problem_name", "dtlz1")
    n_var = args.n_var if args.n_var is not None else config.get("n_var", 10)
    n_obj = args.n_obj if args.n_obj is not None else config.get("n_obj", 2)
    n_gen = args.n_gen if args.n_gen is not None else config.get("n_gen", 100)
    pop_size = args.pop_size if args.pop_size is not None else config.get("pop_size", 100)
    n_runs = args.n_runs if args.n_runs is not None else config.get("n_runs", 30)
    no_plot = args.no_plot if args.no_plot is not None else config.get("no_plot", False)
    run_ea = args.run_ea if args.run_ea is not None else config.get("run_ea", False)
    run_bias_variance_ea = (
        args.run_bias_variance_ea
        if args.run_bias_variance_ea is not None
        else config.get("run_bias_variance_ea", False)
    )
    ea_seed_start = args.ea_seed_start if args.ea_seed_start is not None else config.get("ea_seed_start", 1)
    ea_seed_end = args.ea_seed_end if args.ea_seed_end is not None else config.get("ea_seed_end", 31)
    output_dir = args.output_dir if args.output_dir is not None else config.get("output_dir", "outputs/exp1_gpr_rbf")
    bias_variance_configs = config["bias_variance_configs"]

    print(f"Loaded config: {args.config}")
    plot = not no_plot
    ea_seeds = range(ea_seed_start, ea_seed_end)
    record_bias_variance_runs = run_bias_variance_ea

    problem, hv, igd_plus, obj_min, obj_max = prepare_problem(
        problem_name=problem_name,
        n_var=n_var,
        n_obj=n_obj,
    )

    sample_size = 11 * n_var - 1
    X_train, y_train, X_val, y_val, X_test, y_test = sample_data(
        problem=problem,
        sample_size=sample_size,
        train_seed=42,
        test_seed=1,
    )

    print(f"Sampling X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_val shape: {y_val.shape}")
    print(f"y_test shape: {y_test.shape}")

    if plot:
        plot_obj_2d(y_train, xlim=(-100, 700), ylim=(-100, 700))

    model_f1, model_f2, pred_mean, pred_std, mean_f1, std_f1, mean_f2, std_f2 = run_single_gpr(
        X_train, y_train, X_test, y_test, plot=plot
    )

    seed42_models_by_type = {}
    bias_variance_summary_rows = []
    bias_variance_training_rows = []

    for bv_config in bias_variance_configs:
        model_type = bv_config["model_type"]
        lengthscale_weight = bv_config["lengthscale_weight"]

        print(f"\n=== {model_type} model (lengthscale_weight={lengthscale_weight:g}) ===")
        seed42_model_f1, seed42_model_f2, _, _ = run_high_bias_or_variance(
            X_train,
            y_train,
            X_test,
            y_test,
            lengthscale_weight=lengthscale_weight,
            plot=plot,
        )

        result = run_bias_variance_experiment(
            problem=problem,
            sample_size=sample_size,
            lengthscale_weight=lengthscale_weight,
            model_type=model_type,
            n_runs=n_runs,
            return_records=record_bias_variance_runs,
        )

        if record_bias_variance_runs:
            bias_squared, variance, training_records = result
            bias_variance_training_rows.extend(training_records)
        else:
            bias_squared, variance = result

        print(f"{model_type} model -> Bias^2: {bias_squared:.3e}, Variance: {variance:.3e}")

        bias_variance_summary_rows.append({
            "problem_name": problem_name,
            "model_type": model_type,
            "n_runs": n_runs,
            "lengthscale_weight": lengthscale_weight,
            "bias_squared": format_scientific_3(bias_squared),
            "variance": format_scientific_3(variance),
        })
        seed42_models_by_type[model_type] = {
            "model_f1": seed42_model_f1,
            "model_f2": seed42_model_f2,
            "lengthscale_weight": lengthscale_weight,
            "bias_squared": bias_squared,
            "variance": variance,
        }

    if record_bias_variance_runs:
        output_dir = Path(output_dir)
        write_rows_csv(
            output_dir / "bias_variance_training_records.csv",
            bias_variance_training_rows,
        )
        write_rows_csv(
            output_dir / "bias_variance_summary.csv",
            bias_variance_summary_rows,
        )

    alpha_c90_f1, alpha_c90_f2 = run_uncertainty_eval(
        y_test=y_test,
        pred_mean=pred_mean,
        pred_std=pred_std,
        X_val=X_val,
        y_val=y_val,
        model_f1=model_f1,
        model_f2=model_f2,
        mean_f1=mean_f1,
        std_f1=std_f1,
        mean_f2=mean_f2,
        std_f2=std_f2,
        plot=plot,
    )

    if run_ea:
        print("\n=== Running EA experiment (this may take a while) ===")
        run_ea_experiment(
            problem=problem,
            problem_name=problem_name,
            n_gen=n_gen,
            pop_size=pop_size,
            model_f1=model_f1,
            model_f2=model_f2,
            obj_min=obj_min,
            obj_max=obj_max,
            hv=hv,
            igd_plus=igd_plus,
            seeds=ea_seeds,
        )

        # Optional: keep variables used in notebook dual-ranking section.
        _ = Survival_dual_ranking(alpha_f1=alpha_c90_f1, alpha_f2=alpha_c90_f2)

    if run_bias_variance_ea:
        print("\n=== Running EA for seed=42 bias-variance models ===")
        run_ea_for_seed42_bias_variance_models(
            problem=problem,
            problem_name=problem_name,
            n_gen=n_gen,
            pop_size=pop_size,
            seed42_models_by_type=seed42_models_by_type,
            obj_min=obj_min,
            obj_max=obj_max,
            hv=hv,
            igd_plus=igd_plus,
            output_dir=output_dir,
            seeds=ea_seeds,
        )

    if plot:
        plt.show()


if __name__ == "__main__":
    main()

