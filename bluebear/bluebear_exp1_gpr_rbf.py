# Auto-generated BlueBEAR Python version of the corresponding experiment notebook.
# Run on BlueBEAR from the repository root or with the server path below available.

from pathlib import Path
import os
import random
import sys
import warnings


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


def _setup_log():
    log_path = Path(__file__).with_suffix(".log")
    log_file = open(log_path, "a", encoding="utf-8", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, log_file)
    sys.stderr = _Tee(sys.__stderr__, log_file)
    print(f"Logging to: {log_path}", flush=True)


_setup_log()

os.environ.setdefault("DISABLE_HV_PLOTS", "1")
sys.dont_write_bytecode = True
warnings.filterwarnings("ignore", message=".*load_learner.*pickle.*")

code_path = Path("/rds/projects/w/wangsu-building-automation/Huanbo/2026_new_metric")
repo_candidates = [Path.cwd().resolve(), Path.cwd().resolve().parent, code_path]
for repo_root in repo_candidates:
    if (repo_root / "src").exists():
        repo_root_string = str(repo_root)
        if repo_root_string not in sys.path:
            sys.path.insert(0, repo_root_string)
        break
else:
    raise FileNotFoundError("Could not locate repository root containing src/.")

import yaml
import numpy as np
from src.experiment import (
    compute_surrogate_test_mse,
    run_experiment,
)
from src.metrics import get_metrics, get_problem_clip_min
from src.models import train_gpr_rbf_for_calibration
from src.opt_problem import build_problem
from src.plotting import (
    append_result_summary_outputs,
    evaluate_majority_shift_y_k1_diagnostic_items,
    build_bluebear_seed_records,
    summarize_result_records,
    write_result_summary_temp,
)
from src.survival import Survival_dual_ranking, Survival_standard, find_upper_alpha

warnings.filterwarnings("ignore", message=".*load_learner.*pickle.*")
np.set_printoptions(precision=4, suppress=True)

CONFIG_FILE_NAME = "config.yaml"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def _resolve_experiment_config_path(filename=CONFIG_FILE_NAME):
    roots = [
        Path(__file__).resolve().parent,
        Path.cwd().resolve(),
        Path.cwd().resolve().parent,
    ]
    try:
        roots.append(code_path)
    except NameError:
        pass

    seen = set()
    for root in roots:
        for candidate in (root / "experiments" / filename, root / filename):
            candidate = candidate.resolve()
            if candidate in seen:
                continue
            seen.add(candidate)
            if candidate.exists():
                return candidate

    searched = "\n".join(str(path) for path in sorted(seen))
    raise FileNotFoundError(f"Could not find {filename}. Searched:\n{searched}")


config_path = _resolve_experiment_config_path()
with open(config_path, "r", encoding="utf-8") as config_file:
    experiment_config = yaml.safe_load(config_file)

problem_names = experiment_config["problem_names"]
dtlz_n_var = experiment_config["dtlz_n_var"]
dtlz_n_obj = experiment_config["dtlz_n_obj"]
omnitest_n_var = experiment_config["omnitest_n_var"]
n_gen = experiment_config["n_gen"]
pop_size = experiment_config["pop_size"]
seed_start = experiment_config["seed_start"]
seed_end = experiment_config["seed_end"]
train_seed = experiment_config["train_seed"]
test_seed = experiment_config["test_seed"]
sample_size = experiment_config["sample_size"]
val_size = experiment_config["val_size"]
test_size = experiment_config["test_size"]
calibration_k = experiment_config["calibration_k"]
knn_threshold_method = experiment_config["knn_threshold_method"]
show_seed_output = experiment_config["show_seed_output"]
optimizer_run_specs = experiment_config["optimizer_run_specs"]
optimizer_names = [spec["result_name"] for spec in optimizer_run_specs]
dual_ranking_target_coverage = experiment_config.get("dual_ranking_target_coverage", 0.90)
dual_ranking_alpha_max = experiment_config.get("dual_ranking_alpha_max", 500.0)
dual_ranking_alpha_step = experiment_config.get("dual_ranking_alpha_step", 0.01)
surrogate_config = experiment_config["surrogates"]["gpr_rbf"]
method_name = "GPR_RBF"

print(f"Loaded experiment config: {config_path}")
print(f"Problems: {len(problem_names)} | seeds: range({seed_start}, {seed_end}) | n_gen: {n_gen} | pop_size: {pop_size}")
print(f"Optimizers: {optimizer_names}")


use_surrogate = surrogate_config["use_surrogate"]
def reset_experiment_random_state(seed, label=None):
    seed = int(seed)
    np.random.seed(seed)
def train_model_for_calibration(problem, sample_size, train_seed=42, test_seed=1):
    (
        model_f1,
        model_f2,
        X_train,
        y_train,
        f_train_mean,
        X_val,
        y_val,
        X_test,
        y_test,
    ) = train_gpr_rbf_for_calibration(
        problem=problem,
        sample_size=sample_size,
        train_seed=train_seed,
        test_seed=test_seed,
        val_size=val_size,
        test_size=test_size,
    )
    return model_f1, model_f2, X_train, y_train, f_train_mean, X_val, y_val, X_test, y_test


def build_benchmark_problem(problem_name, dtlz_n_var=10, dtlz_n_obj=2, omnitest_n_var=2):
    pname = problem_name.lower()

    if pname.startswith("dtlz"):
        problem = build_problem(problem_name=problem_name, n_var=dtlz_n_var, n_obj=dtlz_n_obj)
    elif pname == "omnitest":
        problem = build_problem(problem_name=problem_name, n_var=omnitest_n_var)
    else:
        problem = build_problem(problem_name=problem_name)

    return problem



def run_problem(problem_name):
    reset_experiment_random_state(train_seed, f"{problem_name} surrogate/data")
    problem = build_benchmark_problem(
        problem_name,
        dtlz_n_var=dtlz_n_var,
        dtlz_n_obj=dtlz_n_obj,
        omnitest_n_var=omnitest_n_var,
    )
    hv, igd_plus, obj_min, obj_max, _ = get_metrics(
        problem_name=problem_name,
        problem=problem,
        n_var=problem.n_var,
        n_obj=problem.n_obj,
    )

    current_sample_size = sample_size if sample_size is not None else max(11 * problem.n_var - 1, 100)
    problem_y_min = get_problem_clip_min(problem_name, n_var=problem.n_var)
    if problem_y_min is None:
        problem_y_min = obj_min

    model_f1, model_f2, X_train, y_train, f_train_mean, X_val, y_val, X_test, y_test = train_model_for_calibration(
        problem=problem,
        sample_size=current_sample_size,
        train_seed=train_seed,
        test_seed=test_seed,
    )


    offline_test_mse = compute_surrogate_test_mse(
        problem=problem,
        problem_name=problem_name,
        model_f1=model_f1,
        model_f2=model_f2,
        use_surrogate=use_surrogate,
        x_test=X_test,
        y_test=y_test,
    )

    diagnostic_context = {
        'X_train': X_train,
        'y_train': y_train,
        'f_train_mean': f_train_mean,
        'hv': hv,
        'obj_min': obj_min,
        'obj_max': obj_max,
        'problem_y_min': problem_y_min,
        'offline_test_mse': offline_test_mse,
    }
    majority_shift_y_k1_diagnostics = {}
    problem_results = {}
    dual_ranking_survival = None
    for optimizer_spec in optimizer_run_specs:
        result_name = optimizer_spec["result_name"]
        optimizer_name = optimizer_spec["optimizer_name"]
        if optimizer_spec["use_dual_ranking"]:
            if dual_ranking_survival is None:
                dual_ranking_alpha_f1, dual_ranking_coverage_f1 = find_upper_alpha(
                    model_f1,
                    X_val,
                    y_val[:, 0],
                    target_coverage=dual_ranking_target_coverage,
                    alpha_max=dual_ranking_alpha_max,
                    alpha_step=dual_ranking_alpha_step,
                )
                dual_ranking_alpha_f2, dual_ranking_coverage_f2 = find_upper_alpha(
                    model_f2,
                    X_val,
                    y_val[:, 1],
                    target_coverage=dual_ranking_target_coverage,
                    alpha_max=dual_ranking_alpha_max,
                    alpha_step=dual_ranking_alpha_step,
                )
                print(
                    f"Dual-ranking upper bounds: "
                    f"alpha_f1={dual_ranking_alpha_f1:.2f} (coverage={dual_ranking_coverage_f1:.1%}), "
                    f"alpha_f2={dual_ranking_alpha_f2:.2f} (coverage={dual_ranking_coverage_f2:.1%})"
                )
                dual_ranking_survival = Survival_dual_ranking(
                    alpha_f1=dual_ranking_alpha_f1,
                    alpha_f2=dual_ranking_alpha_f2,
                )
            survival_function = dual_ranking_survival
        else:
            survival_function = Survival_standard()

        run_kwargs = dict(
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
            use_surrogate=use_surrogate,
            survival_function=survival_function,
            use_callback=False,
            seeds=range(seed_start, seed_end),
            optimizer_name=optimizer_name,
            distance_x=X_train,
            distance_y=y_train,
            problem_y_min=problem_y_min,
            knn_distance_space="x",
            f_train_mean=f_train_mean,
            calibration_k=calibration_k,
            knn_threshold_method=knn_threshold_method,
            compact_seed_output=True,
            compact_seed_context=diagnostic_context,
            compact_seed_mse_test=offline_test_mse,
            solution_output_dir=OUTPUT_DIR,
            solution_output_method_name=f"{method_name}+{result_name}",
        )

        results = run_experiment(**run_kwargs)

        problem_results[result_name] = results
        try:
            y_space_diagnostic_items = evaluate_majority_shift_y_k1_diagnostic_items(
                results,
                diagnostic_context,
            )
            print(f"\n============================== {problem_name} | {result_name} ==============================")
            seed_records = build_bluebear_seed_records(
                results,
                diagnostic_context,
                mse_test=offline_test_mse,
                y_space_items=y_space_diagnostic_items,
            )
            result_summary_rows.append(
                summarize_result_records(
                    exp_name=method_name,
                    method_name=method_name,
                    problem_name=problem_name,
                    optimizer_name=result_name,
                    records=seed_records,
                    output_dir=OUTPUT_DIR,
                )
            )
            write_result_summary_temp(
                result_summary_rows,
                output_dir=OUTPUT_DIR,
                method_name=method_name,
            )
            majority_shift_y_k1_diagnostics[result_name] = y_space_diagnostic_items
        except Exception as err:
            print(f"\n============================== {problem_name} | {result_name} ==============================")
            print(f"Problem finished, but summary printing failed: {type(err).__name__}: {err}")
            print(f"Available result keys: {sorted(results.keys())}")
            raise

    problem_contexts[problem_name] = {
        'offline_test_mse': offline_test_mse,
        'X_train': X_train,
        'y_train': y_train,
        'f_train_mean': f_train_mean,
        'hv': hv,
        'obj_min': obj_min,
        'obj_max': obj_max,
        'problem_y_min': problem_y_min,
        'majority_shift_y_k1_diagnostics': majority_shift_y_k1_diagnostics,
    }

    return problem_results


all_results = {}
problem_contexts = {}
result_summary_rows = []

for problem_index, problem_name in enumerate(problem_names, start=1):
    all_results[problem_name] = run_problem(problem_name)


# Per-seed values and final result summary are printed during run_problem.
# Aggregate gap-improvement table printing is skipped.
gap_improvement_tables = None


result_summary_table = append_result_summary_outputs(
    result_summary_rows,
    output_dir=OUTPUT_DIR,
    title=f"Result summary | {method_name}",
)
