"""Batch runners for Exp5-8 using the uploaded baseline method implementations."""

from __future__ import annotations

import importlib.util
import sys
import time
import types
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.lhs import LHS
from pymoo.optimize import minimize
from sklearn.metrics import mean_squared_error


REPO_ROOT = Path(__file__).resolve().parents[1]
PROB_VENDOR_ROOT = REPO_ROOT / "baseline" / "Prob-RVEA and Prob-MOEA-D 2022"
TGPR_VENDOR_ROOT = REPO_ROOT / "baseline" / "TGPR-MO 2023"
DEFAULT_CONFIG_PATH = REPO_ROOT / "experiments" / "config.yaml"

TOTAL_FUNCTION_EVALUATIONS = 10_000
POPULATION_SIZE = 100
PROB_MOEAD_POPULATION_SIZE = 50
DDMOEA_GAN_N_GEN = 100
MAJORITY_SHIFT_K_VALUES = (1, 2)
POINTWISE_EUCLIDEAN_K_VALUES = (1, 2)


def install_pydoe2_imp_compatibility():
    """Allow the last pyDOE2 release to import on Python 3.12 and newer."""
    if "imp" not in sys.modules and importlib.util.find_spec("imp") is None:
        sys.modules["imp"] = types.ModuleType("imp")


install_pydoe2_imp_compatibility()


def _activate_vendor(vendor_root):
    vendor_root = str(Path(vendor_root).resolve())
    for existing_root in (str(PROB_VENDOR_ROOT.resolve()), str(TGPR_VENDOR_ROOT.resolve())):
        while existing_root in sys.path:
            sys.path.remove(existing_root)
    sys.path.insert(0, vendor_root)

    prefixes = ("desdeo_emo", "desdeo_problem", "desdeo_tools", "framework")
    for module_name in list(sys.modules):
        if any(
            module_name == prefix or module_name.startswith(prefix + ".")
            for prefix in prefixes
        ):
            del sys.modules[module_name]


def load_config(config_path=None):
    resolved_path = Path(config_path or DEFAULT_CONFIG_PATH).resolve()
    with open(resolved_path, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    print(f"Loaded baseline experiment config: {resolved_path}")
    return config


def build_benchmark_problem(problem_name, config):
    from src.opt_problem import build_problem

    pname = str(problem_name).lower()
    if pname.startswith("dtlz"):
        return build_problem(
            problem_name=problem_name,
            n_var=config["dtlz_n_var"],
            n_obj=config["dtlz_n_obj"],
        )
    if pname == "omnitest":
        return build_problem(
            problem_name=problem_name,
            n_var=config["omnitest_n_var"],
        )
    return build_problem(problem_name=problem_name)


def configured_sample_size(problem, config):
    sample_size = config.get("sample_size")
    return (
        int(sample_size)
        if sample_size is not None
        else max(11 * int(problem.n_var) - 1, POPULATION_SIZE)
    )


def generate_offline_dataset(problem, config):
    sample_size = configured_sample_size(problem, config)
    train_seed = int(config["train_seed"])
    np.random.seed(train_seed)
    x_data = LHS()(problem, sample_size, seed=train_seed).get("X")
    y_data = problem.evaluate(x_data, return_values_of=["F"])
    print(
        f"Offline dataset: LHS | sample_size={sample_size} | "
        f"train_seed={train_seed}"
    )
    return x_data, y_data


def generate_offline_test_dataset(problem, config):
    sample_size = configured_sample_size(problem, config)
    test_seed = int(config["test_seed"])
    np.random.seed(test_seed)
    x_test = LHS()(problem, sample_size, seed=test_seed).get("X")
    y_test = problem.evaluate(x_test, return_values_of=["F"])
    return x_test, y_test


def initial_population_from_offline_dataset(x_data, population_size=POPULATION_SIZE):
    x_data = np.asarray(x_data)
    population_size = int(population_size)
    if x_data.shape[0] < population_size:
        raise ValueError(
            f"Offline dataset contains {x_data.shape[0]} points, but "
            f"{population_size} initial solutions are required."
        )
    return x_data[:population_size].copy()


def _repair_decision_vectors(decision_vectors, lower_bounds, upper_bounds):
    decision_vectors = np.asarray(decision_vectors, dtype=float)
    lower_bounds = np.asarray(lower_bounds, dtype=float)
    upper_bounds = np.asarray(upper_bounds, dtype=float)
    midpoint = (lower_bounds + upper_bounds) / 2
    decision_vectors = np.where(np.isfinite(decision_vectors), decision_vectors, midpoint)
    return np.minimum(np.maximum(decision_vectors, lower_bounds), upper_bounds)


def _repair_decision_vectors_for_evaluation(
    decision_vectors,
    lower_bounds,
    upper_bounds,
    relative_epsilon=1e-9,
):
    decision_vectors = np.asarray(decision_vectors, dtype=float)
    lower_bounds = np.asarray(lower_bounds, dtype=float)
    upper_bounds = np.asarray(upper_bounds, dtype=float)
    span = upper_bounds - lower_bounds
    epsilon = np.where(np.isfinite(span) & (span > 0.0), span * relative_epsilon, 0.0)
    safe_lower = lower_bounds + epsilon
    safe_upper = upper_bounds - epsilon
    return _repair_decision_vectors(decision_vectors, safe_lower, safe_upper)


def _repair_and_predict_final_population(surrogate_problem, individuals):
    solution = _repair_decision_vectors_for_evaluation(
        individuals,
        surrogate_problem.get_variable_lower_bounds(),
        surrogate_problem.get_variable_upper_bounds(),
    )
    obj = _predict_desdeo_surrogate(surrogate_problem, solution)
    return solution, obj


def _install_prob_population_numeric_compatibility():
    population_module = importlib.import_module("desdeo_emo.population.Population")
    population_class = population_module.Population
    if getattr(population_class, "_offline_finite_repair_installed", False):
        return

    original_add = population_class.add

    def add(self, offsprings, use_surrogates=False):
        offsprings = _repair_decision_vectors(
            offsprings,
            self.lower_limits,
            self.upper_limits,
        )
        return original_add(self, offsprings, use_surrogates)

    population_class.add = add
    population_class._offline_finite_repair_installed = True


def _install_tgpr_numeric_compatibility():
    population_module = importlib.import_module("desdeo_emo.population.Population")
    population_class = population_module.Population
    if not getattr(population_class, "_offline_finite_repair_installed", False):
        original_add = population_class.add

        def add(self, offsprings, use_surrogates=False):
            offsprings = _repair_decision_vectors(
                offsprings,
                self.lower_limits,
                self.upper_limits,
            )
            return original_add(self, offsprings, use_surrogates)

        population_class.add = add
        population_class._offline_finite_repair_installed = True

    surrogate_module = importlib.import_module(
        "desdeo_problem.surrogatemodels.surrogate_treedGP"
    )
    surrogate_class = surrogate_module.treeGP
    if not getattr(surrogate_class, "_offline_finite_repair_installed", False):
        original_predict = surrogate_class.predict

        def predict(self, x):
            x = np.asarray(x, dtype=float)
            training_x = np.asarray(self.X, dtype=float)
            training_min = np.nanmin(training_x, axis=0)
            training_max = np.nanmax(training_x, axis=0)
            training_midpoint = (training_min + training_max) / 2
            x = np.where(np.isfinite(x), x, training_midpoint)
            x = np.minimum(np.maximum(x, training_min), training_max)
            try:
                y_mean, y_stdev = original_predict(self, x)
            except ValueError:
                y_mean = np.asarray(self.regr.predict(X=x), dtype=float)
                y_stdev = None
            y_mean = np.asarray(y_mean, dtype=float)
            non_finite = ~np.isfinite(y_mean)
            if np.any(non_finite):
                tree_prediction = np.asarray(self.regr.predict(X=x), dtype=float)
                y_mean = np.where(non_finite, tree_prediction, y_mean)
            return y_mean, y_stdev

        surrogate_class.predict = predict
        surrogate_class._offline_finite_repair_installed = True


def _install_prob_moead_numeric_compatibility(population):
    def repair(self, individual):
        return _repair_decision_vectors(
            individual,
            self.problem.get_variable_lower_bounds(),
            self.problem.get_variable_upper_bounds(),
        )

    population.repair = types.MethodType(repair, population)


def _tgpr_default_population_size(surrogate_problem):
    n_objectives = int(surrogate_problem.n_of_objectives)
    lattice_res_options = [49, 13, 7, 5, 4, 3, 3, 3, 3]
    if n_objectives < 11:
        lattice_resolution = lattice_res_options[n_objectives - 2]
    else:
        lattice_resolution = 3
    return comb(lattice_resolution + n_objectives - 1, n_objectives - 1)


def _build_tgpr_rvea(RVEA, surrogate_problem, initial_population):
    default_population_size = _tgpr_default_population_size(surrogate_problem)
    init_pop = np.asarray(initial_population, dtype=float)[:default_population_size].copy()
    return RVEA(
        surrogate_problem,
        use_surrogates=True,
        population_params={
            "design": "InitSamples",
            "init_pop": init_pop,
        },
        total_function_evaluations=TOTAL_FUNCTION_EVALUATIONS,
    )


def _build_kriging_surrogate(benchmark_problem, x_data, y_data):
    _activate_vendor(PROB_VENDOR_ROOT)
    from desdeo_problem.Problem import DataProblem
    from desdeo_problem.surrogatemodels.SurrogateKriging import SurrogateKriging

    n_var = benchmark_problem.n_var
    n_obj = benchmark_problem.n_obj
    x_names = [f"x{i}" for i in range(1, n_var + 1)]
    y_names = [f"f{i}" for i in range(1, n_obj + 1)]
    bounds = pd.DataFrame(
        np.vstack((benchmark_problem.xl, benchmark_problem.xu)),
        columns=x_names,
        index=["lower_bound", "upper_bound"],
    )
    data = pd.DataFrame(np.hstack((x_data, y_data)), columns=x_names + y_names)
    surrogate_problem = DataProblem(
        data=data,
        variable_names=x_names,
        objective_names=y_names,
        bounds=bounds,
    )
    surrogate_problem.train(SurrogateKriging)
    return surrogate_problem


def _metric_context(problem_name, benchmark_problem):
    from src.metrics import get_metrics, get_problem_clip_min

    hv, igd_plus, obj_min, obj_max, _ = get_metrics(
        problem_name=problem_name,
        problem=benchmark_problem,
        n_var=benchmark_problem.n_var,
        n_obj=benchmark_problem.n_obj,
    )
    problem_y_min = get_problem_clip_min(
        problem_name,
        n_var=benchmark_problem.n_var,
    )
    if problem_y_min is None:
        problem_y_min = obj_min
    return hv, igd_plus, obj_min, obj_max, problem_y_min


def _evaluate_solution(
    benchmark_problem,
    solution,
    obj,
    hv,
    igd_plus,
    obj_min,
    obj_max,
):
    solution = np.asarray(solution, dtype=float)
    obj = np.asarray(obj, dtype=float)
    f_real = benchmark_problem.evaluate(solution, return_values_of=["F"])
    f_real = np.asarray(f_real, dtype=float)
    finite_mask = (
        np.all(np.isfinite(solution), axis=1)
        & np.all(np.isfinite(obj), axis=1)
        & np.all(np.isfinite(f_real), axis=1)
    )
    non_finite_candidate_count = int(np.count_nonzero(~finite_mask))
    if non_finite_candidate_count:
        solution = solution[finite_mask]
        obj = obj[finite_mask]
        f_real = f_real[finite_mask]
    if solution.shape[0] == 0:
        raise ValueError(
            "All optimized candidates have non-finite decision variables or "
            "objective values. This can happen when a benchmark objective is "
            "singular at its variable bounds."
        )
    obj_normalized = (obj - obj_min) / (obj_max - obj_min)
    f_real_normalized = (f_real - obj_min) / (obj_max - obj_min)
    result = {
        "solution": solution,
        "obj": obj,
        "obj_sur": obj,
        "f_real": f_real,
        "obj_real": f_real,
        "igd_plus": float(igd_plus(f_real)),
        "igd_plus_real": float(igd_plus(f_real)),
        "igd_plus_surrogate": float(igd_plus(obj)),
        "igd_plus_indicator": igd_plus,
        "hv_surrogate": float(hv.do(obj_normalized)),
        "hv_real": float(hv.do(f_real_normalized)),
        "non_finite_candidate_count": non_finite_candidate_count,
    }
    result["hv_sur_gap"] = abs(result["hv_real"] - result["hv_surrogate"])
    return result


def _predict_desdeo_surrogate(surrogate_problem, x_data):
    evaluation = surrogate_problem.evaluate(x_data, use_surrogate=True)
    return np.asarray(evaluation.objectives, dtype=float)


def _finite_mean_squared_error(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    finite_mask = np.all(np.isfinite(y_true), axis=1) & np.all(
        np.isfinite(y_pred),
        axis=1,
    )
    if not np.any(finite_mask):
        return float("nan")
    return float(mean_squared_error(y_true[finite_mask], y_pred[finite_mask]))


def _compute_offline_test_mse(benchmark_problem, config, predict_fn):
    x_test, y_test = generate_offline_test_dataset(benchmark_problem, config)
    y_pred = np.asarray(predict_fn(x_test), dtype=float)
    return _finite_mean_squared_error(y_test, y_pred)


def _evaluate_pointwise_euclidean_variant(
    result,
    x_train,
    y_train,
    f_train_pred,
    hv,
    obj_min,
    obj_max,
    problem_y_min,
    k,
):
    from src.experiment import (
        _clip_calibration_result_to_problem_bounds,
        compute_local_residual_calibration,
        normalized_hv,
    )

    calibration_result = compute_local_residual_calibration(
        obj=result["obj"],
        y_train=y_train,
        f_train_mean=f_train_pred,
        k=k,
        distance_train=x_train,
        distance_query=result["solution"],
    )
    _clip_calibration_result_to_problem_bounds(
        calibration_result,
        problem_y_min,
        clip_source="problem_y_threshold",
    )
    return normalized_hv(
        hv,
        calibration_result["sur_calibration"],
        obj_min,
        obj_max,
    )


def _add_hv_variants(
    result,
    x_train,
    y_train,
    f_train_pred,
    hv,
    obj_min,
    obj_max,
    problem_y_min,
    config,
):
    from src.experiment import (
        evaluate_majority_shift_x_variant,
        compute_R_indicator,
        normalized_hv,
    )
    from src.plotting import (
        HV_XY_DIRECTION_COUNTS_IMPROVEMENT_LABEL,
        evaluate_majority_shift_y_k1_diagnostic_items,
        evaluate_xy_direction_counts_hv_values,
    )

    hv_variant_values = {}
    majority_shift_x_result = None
    obj_clipped = np.maximum(
        np.asarray(result["obj"], dtype=float),
        np.asarray(problem_y_min, dtype=float),
    )
    hv_clip = normalized_hv(hv, obj_clipped, obj_min, obj_max)
    hv_clip_gap = abs(result["hv_real"] - hv_clip)
    result.update(
        {
            "hv_clip": hv_clip,
            "hv_clip_gap": hv_clip_gap,
            "hv_clip_count": int(obj_clipped.shape[0]),
        }
    )
    covariance_ridge = float(config.get("covariance_ridge", 1.0e-8))
    for k in MAJORITY_SHIFT_K_VALUES:
        for distance_metric, label in (
            ("normalized_euclidean", "HV_majority_shift_x"),
            ("mahalanobis", "HV_majority_shift_x_mahalanobis"),
        ):
            shift_eval = evaluate_majority_shift_x_variant(
                x=result["solution"],
                f_sur=result["obj"],
                f_real=result["f_real"],
                x_train=x_train,
                y_train=y_train,
                f_train_pred=f_train_pred,
                hv=hv,
                hv_real=result["hv_real"],
                hv_sur_gap=result["hv_sur_gap"],
                obj_min=obj_min,
                obj_max=obj_max,
                clip_obj_min=problem_y_min,
                k=k,
                distance_metric=distance_metric,
                covariance_ridge=covariance_ridge,
                clip_source="problem_y_threshold",
            )
            hv_value = shift_eval["hv"]
            hv_variant_values[f"{label} k={k}"] = hv_value
            if k == 1 and distance_metric == "normalized_euclidean":
                majority_shift_x_result = shift_eval["result"]

    no_clip_eval = evaluate_majority_shift_x_variant(
        x=result["solution"],
        f_sur=result["obj"],
        f_real=result["f_real"],
        x_train=x_train,
        y_train=y_train,
        f_train_pred=f_train_pred,
        hv=hv,
        hv_real=result["hv_real"],
        hv_sur_gap=result["hv_sur_gap"],
        obj_min=obj_min,
        obj_max=obj_max,
        clip_obj_min=problem_y_min,
        k=1,
        distance_metric="normalized_euclidean",
        covariance_ridge=covariance_ridge,
        use_clip=False,
        clip_source="problem_y_threshold",
    )
    hv_variant_values["HV_no_clip (ablation)"] = no_clip_eval["hv"]

    for k in POINTWISE_EUCLIDEAN_K_VALUES:
        hv_variant_values[f"HV_calibration_euclidean k={k} (pointwise)"] = (
            _evaluate_pointwise_euclidean_variant(
                result,
                x_train=x_train,
                y_train=y_train,
                f_train_pred=f_train_pred,
                hv=hv,
                obj_min=obj_min,
                obj_max=obj_max,
                problem_y_min=problem_y_min,
                k=k,
            )
        )

    diagnostic_context = {
        "X_train": x_train,
        "y_train": y_train,
        "f_train_mean": f_train_pred,
        "hv": hv,
        "obj_min": obj_min,
        "obj_max": obj_max,
        "problem_y_min": problem_y_min,
        "clip_source": "problem_y_threshold",
    }
    result["majority_shift_x_result"] = majority_shift_x_result
    y_space_diagnostic_items = evaluate_majority_shift_y_k1_diagnostic_items(
        {"run_details": [result]},
        diagnostic_context,
    )
    xy_direction_counts_hv_values = evaluate_xy_direction_counts_hv_values(
        {"run_details": [result]},
        y_space_diagnostic_items,
        diagnostic_context,
    )
    hv_variant_values[HV_XY_DIRECTION_COUNTS_IMPROVEMENT_LABEL] = (
        float(xy_direction_counts_hv_values[0])
        if len(xy_direction_counts_hv_values)
        else float("nan")
    )

    hv_majority_shift_x = hv_variant_values["HV_majority_shift_x k=1"]
    hv_majority_shift_x_gap = abs(result["hv_real"] - hv_majority_shift_x)
    result.update(
        {
            "hv_variant_values": hv_variant_values,
            "majority_shift_x_result": majority_shift_x_result,
            "majority_shift_y_k1_diagnostic": (
                y_space_diagnostic_items[0] if y_space_diagnostic_items else None
            ),
            "_problem_context": diagnostic_context,
            "hv_majority_shift_x": hv_majority_shift_x,
            "hv_majority_shift_x_gap": hv_majority_shift_x_gap,
            "hv_majority_shift_x_gap_improvement_pct": compute_R_indicator(
                result["hv_sur_gap"],
                hv_majority_shift_x_gap,
            ),
        }
    )
    return result


def _evaluate_solution_with_majority_shift_x(
    benchmark_problem,
    solution,
    obj,
    hv,
    igd_plus,
    obj_min,
    obj_max,
    problem_y_min,
    x_train,
    y_train,
    f_train_pred,
    config,
):
    result = _evaluate_solution(
        benchmark_problem,
        solution,
        obj,
        hv,
        igd_plus,
        obj_min,
        obj_max,
    )
    return _add_hv_variants(
        result,
        x_train=x_train,
        y_train=y_train,
        f_train_pred=f_train_pred,
        hv=hv,
        obj_min=obj_min,
        obj_max=obj_max,
        problem_y_min=problem_y_min,
        config=config,
    )


def build_gap_improvement_table(all_results):
    from src.plotting import build_exp5_8_gap_improvement_table

    return build_exp5_8_gap_improvement_table(all_results)


def print_gap_improvement_table(all_results, method_name):
    from src.plotting import print_exp5_8_gap_improvement_table

    return print_exp5_8_gap_improvement_table(all_results, method_name)


def append_compact_summary_record(
    method_name,
    all_results,
    problem_names=None,
    output_path=None,
):
    """Append an Exp1-style compact summary to a txt file beside this module."""
    if output_path is None:
        output_dir = Path(__file__).resolve().parent
    else:
        output_dir = Path(output_path).resolve().parent
    from src.plotting import append_exp5_8_compact_summary_outputs

    return append_exp5_8_compact_summary_outputs(
        method_name=method_name,
        all_results=all_results,
        problem_names=problem_names,
        output_dir=output_dir,
    )


def _run_suite_problem_worker(args):
    (
        method_name,
        run_problem,
        problem_name,
        config,
        seed_start,
        seed_end,
        output_dir,
    ) = args
    from src.plotting import (
        build_bluebear_seed_records,
        summarize_result_records,
    )

    seeds = range(int(seed_start), int(seed_end))
    print(f"\nPreparing {problem_name} for {method_name}")
    benchmark_problem = build_benchmark_problem(problem_name, config)
    run_results = run_problem(
        problem_name,
        benchmark_problem,
        config,
        seeds,
        solution_output_dir=output_dir,
    )
    summary_rows = []
    if run_results:
        context = run_results[0]["_problem_context"]
        y_space_items = [
            item.get("majority_shift_y_k1_diagnostic")
            for item in run_results
        ]
        records = build_bluebear_seed_records(
            {"run_details": run_results},
            context,
            mse_test=run_results[0].get("offline_test_mse", float("nan")),
            y_space_items=y_space_items,
        )
        summary_rows.append(
            summarize_result_records(
                exp_name=method_name,
                method_name=method_name,
                problem_name=problem_name,
                optimizer_name=method_name,
                records=records,
                output_dir=output_dir,
            )
        )
    return problem_name, summary_rows


def _run_suite(method_name, run_problem, config_path=None, parallel_problems=1):
    from src.plotting import (
        append_result_summary_outputs,
        write_result_summary_temp,
    )

    config = load_config(config_path)
    output_dir = Path(config_path or DEFAULT_CONFIG_PATH).resolve().parent / "output"
    problem_names = list(config["problem_names"])
    seed_start = int(config["seed_start"])
    seed_end = int(config["seed_end"])
    all_results = {}
    result_summary_rows = []
    summary_rows_by_problem = {}

    def ordered_summary_rows():
        rows = []
        for configured_problem_name in problem_names:
            rows.extend(summary_rows_by_problem.get(configured_problem_name, []))
        return rows

    def record_completed_problem(problem_name, summary_rows):
        all_results[problem_name] = None
        summary_rows_by_problem[problem_name] = list(summary_rows)
        result_summary_rows[:] = ordered_summary_rows()
        write_result_summary_temp(
            result_summary_rows,
            output_dir=output_dir,
            method_name=method_name,
        )

    max_workers = min(int(parallel_problems or 1), len(problem_names))
    print(f"Problem-level parallel workers for {method_name}: {max_workers}")
    worker_args = [
        (
            method_name,
            run_problem,
            problem_name,
            config,
            seed_start,
            seed_end,
            output_dir,
        )
        for problem_name in problem_names
    ]
    if max_workers <= 1:
        for args in worker_args:
            problem_name, summary_rows = _run_suite_problem_worker(args)
            record_completed_problem(problem_name, summary_rows)
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_problem = {
                executor.submit(_run_suite_problem_worker, args): args[2]
                for args in worker_args
            }
            for future in as_completed(future_to_problem):
                problem_name = future_to_problem[future]
                completed_problem_name, summary_rows = future.result()
                print(f"Completed parallel problem: {completed_problem_name}")
                record_completed_problem(completed_problem_name, summary_rows)
    if result_summary_rows:
        append_result_summary_outputs(
            result_summary_rows,
            output_dir=output_dir,
            title=f"Result summary | {method_name}",
        )
    return all_results


def _run_prob_rvea_problem(
    problem_name,
    benchmark_problem,
    config,
    seeds,
    solution_output_dir=None,
):
    from src.experiment import save_seed_solution_output
    from src.plotting import print_seed_result

    x_data, y_data = generate_offline_dataset(benchmark_problem, config)
    initial_population = initial_population_from_offline_dataset(x_data)
    surrogate_problem = _build_kriging_surrogate(benchmark_problem, x_data, y_data)
    f_train_pred = _predict_desdeo_surrogate(surrogate_problem, x_data)
    offline_test_mse = _compute_offline_test_mse(
        benchmark_problem,
        config,
        lambda x_test: _predict_desdeo_surrogate(surrogate_problem, x_test),
    )
    from desdeo_emo.EAs.ProbRVEA import ProbRVEA_v3

    use_truss2d_repair = str(problem_name).lower() == "truss2d"
    if use_truss2d_repair:
        _install_prob_population_numeric_compatibility()
    hv, igd_plus, obj_min, obj_max, problem_y_min = _metric_context(
        problem_name,
        benchmark_problem,
    )

    run_results = []
    for seed in seeds:
        start_time = time.time()
        evolver = ProbRVEA_v3(
            surrogate_problem,
            use_surrogates=True,
            population_size=POPULATION_SIZE,
            population_params={
                "design": "InitSamples",
                "init_pop": initial_population.copy(),
            },
            total_function_evaluations=TOTAL_FUNCTION_EVALUATIONS,
        )
        while evolver.continue_evolution():
            evolver.iterate()
        elapsed = time.time() - start_time
        if use_truss2d_repair:
            final_solution, final_obj = _repair_and_predict_final_population(
                surrogate_problem,
                evolver.population.individuals,
            )
        else:
            final_solution = evolver.population.individuals
            final_obj = evolver.population.objectives
        result = _evaluate_solution_with_majority_shift_x(
            benchmark_problem,
            final_solution,
            final_obj,
            hv,
            igd_plus,
            obj_min,
            obj_max,
            problem_y_min,
            x_data,
            y_data,
            f_train_pred,
            config,
        )
        result["offline_test_mse"] = offline_test_mse
        run_results.append(result)
        seed_record, _ = print_seed_result(
            result,
            seed=seed,
            elapsed=elapsed,
            problem_name=problem_name,
            method_name="Prob-RVEA",
            output_dir=solution_output_dir,
        )
        result["solution_output_paths"] = save_seed_solution_output(
            solution_output_dir,
            problem_name=problem_name,
            optimizer_name="Prob-RVEA",
            seed=seed,
            solution=result["solution"],
            obj_sur=result["obj"],
            obj_real=result["f_real"],
            method_name="Prob-RVEA",
            seed_record=seed_record,
        )
    return run_results


def _run_prob_moead_problem(
    problem_name,
    benchmark_problem,
    config,
    seeds,
    solution_output_dir=None,
):
    from src.experiment import save_seed_solution_output
    from src.plotting import print_seed_result

    x_data, y_data = generate_offline_dataset(benchmark_problem, config)
    initial_population = initial_population_from_offline_dataset(
        x_data,
        population_size=PROB_MOEAD_POPULATION_SIZE,
    )
    surrogate_problem = _build_kriging_surrogate(benchmark_problem, x_data, y_data)
    f_train_pred = _predict_desdeo_surrogate(surrogate_problem, x_data)
    offline_test_mse = _compute_offline_test_mse(
        benchmark_problem,
        config,
        lambda x_test: _predict_desdeo_surrogate(surrogate_problem, x_test),
    )
    from desdeo_emo.EAs.ProbMOEAD import ProbMOEAD_v3

    hv, igd_plus, obj_min, obj_max, problem_y_min = _metric_context(
        problem_name,
        benchmark_problem,
    )

    run_results = []
    for seed in seeds:
        start_time = time.time()
        evolver = ProbMOEAD_v3(
            surrogate_problem,
            use_surrogates=True,
            population_size=PROB_MOEAD_POPULATION_SIZE,
            population_params={
                "design": "InitSamples",
                "init_pop": initial_population.copy(),
            },
            total_function_evaluations=TOTAL_FUNCTION_EVALUATIONS,
        )
        _install_prob_moead_numeric_compatibility(evolver.population)
        while evolver.continue_evolution():
            evolver.iterate()
        elapsed = time.time() - start_time
        result = _evaluate_solution_with_majority_shift_x(
            benchmark_problem,
            evolver.population.individuals,
            evolver.population.objectives,
            hv,
            igd_plus,
            obj_min,
            obj_max,
            problem_y_min,
            x_data,
            y_data,
            f_train_pred,
            config,
        )
        result["offline_test_mse"] = offline_test_mse
        run_results.append(result)
        seed_record, _ = print_seed_result(
            result,
            seed=seed,
            elapsed=elapsed,
            problem_name=problem_name,
            method_name="Prob-MOEA/D",
            output_dir=solution_output_dir,
        )
        result["solution_output_paths"] = save_seed_solution_output(
            solution_output_dir,
            problem_name=problem_name,
            optimizer_name="Prob-MOEA/D",
            seed=seed,
            solution=result["solution"],
            obj_sur=result["obj"],
            obj_real=result["f_real"],
            method_name="Prob-MOEA/D",
            seed_record=seed_record,
        )
    return run_results


def _run_tgpr_mo_problem(
    problem_name,
    benchmark_problem,
    config,
    seeds,
    solution_output_dir=None,
):
    from src.experiment import save_seed_solution_output
    from src.plotting import print_seed_result

    _activate_vendor(TGPR_VENDOR_ROOT)
    from desdeo_emo.EAs.RVEA import RVEA
    from framework.treedGP_framework import run_treed_GP

    _install_tgpr_numeric_compatibility()
    x_data, y_data = generate_offline_dataset(benchmark_problem, config)
    initial_population = initial_population_from_offline_dataset(x_data)
    surrogate_problem, _, _ = run_treed_GP(
        x_data,
        y_data,
        benchmark_problem.xl,
        benchmark_problem.xu,
    )
    f_train_pred = _predict_desdeo_surrogate(surrogate_problem, x_data)
    offline_test_mse = _compute_offline_test_mse(
        benchmark_problem,
        config,
        lambda x_test: _predict_desdeo_surrogate(surrogate_problem, x_test),
    )
    hv, igd_plus, obj_min, obj_max, problem_y_min = _metric_context(
        problem_name,
        benchmark_problem,
    )

    run_results = []
    for seed in seeds:
        start_time = time.time()
        evolver = _build_tgpr_rvea(RVEA, surrogate_problem, initial_population)
        while evolver.continue_evolution():
            evolver.iterate()
        elapsed = time.time() - start_time
        result = _evaluate_solution_with_majority_shift_x(
            benchmark_problem,
            evolver.population.individuals,
            evolver.population.objectives,
            hv,
            igd_plus,
            obj_min,
            obj_max,
            problem_y_min,
            x_data,
            y_data,
            f_train_pred,
            config,
        )
        result["offline_test_mse"] = offline_test_mse
        run_results.append(result)
        seed_record, _ = print_seed_result(
            result,
            seed=seed,
            elapsed=elapsed,
            problem_name=problem_name,
            method_name="TGPR-MO",
            output_dir=solution_output_dir,
        )
        result["solution_output_paths"] = save_seed_solution_output(
            solution_output_dir,
            problem_name=problem_name,
            optimizer_name="TGPR-MO",
            seed=seed,
            solution=result["solution"],
            obj_sur=result["obj"],
            obj_real=result["f_real"],
            method_name="TGPR-MO",
            seed_record=seed_record,
        )
    return run_results


def _run_ddmoea_gan_problem(
    problem_name,
    benchmark_problem,
    config,
    seeds,
    solution_output_dir=None,
):
    from src.experiment import save_seed_solution_output
    from src.plotting import print_seed_result

    from baseline.ddmoea_gan import (
        DDMOEAGANProblem,
        construct_surrogate_pool_with_gan,
        surrogate_predict_with_ensemble,
        train_wgan_gp,
    )

    x_init, f_init = generate_offline_dataset(benchmark_problem, config)
    initial_population = initial_population_from_offline_dataset(x_init)
    x_min = x_init.min(axis=0)
    x_max = x_init.max(axis=0)
    f_min = f_init.min(axis=0)
    f_max = f_init.max(axis=0)
    x_normalized = 2.0 * (x_init - x_min) / (x_max - x_min + 1e-12) - 1.0
    f_normalized = 2.0 * (f_init - f_min) / (f_max - f_min + 1e-12) - 1.0
    joint_init = np.hstack([x_normalized, f_normalized])

    generator, discriminator, device = train_wgan_gp(
        joint_init=joint_init,
        d_dim=joint_init.shape[1],
        n_epochs=2000,
        batch_size=64,
        z_dim=32,
        lambda_gp=10.0,
        n_critic=5,
        lr=1e-4,
        verbose=False,
    )
    surrogate_pools, norm_bounds = construct_surrogate_pool_with_gan(
        X_init=x_init,
        F_init=f_init,
        generator=generator,
        discriminator=discriminator,
        device=device,
        n_models=benchmark_problem.n_var,
        select_ratio=0.2,
        poly_degree=2,
        gamma_rbfn=0.5,
        lambda_rbfn=1e-6,
        verbose=False,
    )
    x_min, x_max, f_min, f_max = norm_bounds
    dd_problem = DDMOEAGANProblem(
        n_var=benchmark_problem.n_var,
        n_obj=benchmark_problem.n_obj,
        xl=benchmark_problem.xl,
        xu=benchmark_problem.xu,
        surrogate_pools=surrogate_pools,
        discriminator=discriminator,
        device=device,
        x_min=x_min,
        x_max=x_max,
        f_min=f_min,
        f_max=f_max,
        alpha_critic=0.1,
    )
    algorithm = NSGA2(
        pop_size=POPULATION_SIZE,
        sampling=initial_population,
        crossover=SBX(prob=1.0, eta=20),
        mutation=PM(prob=1.0 / benchmark_problem.n_var, eta=20),
        eliminate_duplicates=True,
    )
    f_train_pred = dd_problem.evaluate(x_init, return_values_of=["F"])
    offline_test_mse = _compute_offline_test_mse(
        benchmark_problem,
        config,
        lambda x_test: surrogate_predict_with_ensemble(x_test, surrogate_pools),
    )
    hv, igd_plus, obj_min, obj_max, problem_y_min = _metric_context(
        problem_name,
        benchmark_problem,
    )

    run_results = []
    for seed in seeds:
        start_time = time.time()
        result_minimize = minimize(
            dd_problem,
            algorithm,
            ("n_gen", DDMOEA_GAN_N_GEN),
            seed=seed,
            verbose=False,
            save_history=True,
        )
        elapsed = time.time() - start_time
        final_opt = result_minimize.history[-1].opt
        result = _evaluate_solution_with_majority_shift_x(
            benchmark_problem,
            final_opt.get("X"),
            final_opt.get("F"),
            hv,
            igd_plus,
            obj_min,
            obj_max,
            problem_y_min,
            x_init,
            f_init,
            f_train_pred,
            config,
        )
        result["offline_test_mse"] = offline_test_mse
        run_results.append(result)
        seed_record, _ = print_seed_result(
            result,
            seed=seed,
            elapsed=elapsed,
            problem_name=problem_name,
            method_name="DDMOEA-GAN",
            output_dir=solution_output_dir,
        )
        result["solution_output_paths"] = save_seed_solution_output(
            solution_output_dir,
            problem_name=problem_name,
            optimizer_name="DDMOEA-GAN",
            seed=seed,
            solution=result["solution"],
            obj_sur=result["obj"],
            obj_real=result["f_real"],
            method_name="DDMOEA-GAN",
            seed_record=seed_record,
        )
    return run_results


def run_prob_rvea_suite(config_path=None, parallel_problems=1):
    return _run_suite(
        "Prob-RVEA",
        _run_prob_rvea_problem,
        config_path=config_path,
        parallel_problems=parallel_problems,
    )


def run_prob_moead_suite(config_path=None, parallel_problems=1):
    return _run_suite(
        "Prob-MOEA/D",
        _run_prob_moead_problem,
        config_path=config_path,
        parallel_problems=parallel_problems,
    )


def run_tgpr_mo_suite(config_path=None):
    return _run_suite("TGPR-MO", _run_tgpr_mo_problem, config_path=config_path)


def run_ddmoea_gan_suite(config_path=None):
    return _run_suite(
        "DDMOEA-GAN",
        _run_ddmoea_gan_problem,
        config_path=config_path,
    )
