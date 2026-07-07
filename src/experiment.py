import gc
import json
import random
import time
from pathlib import Path
import numpy as np
from scipy import stats
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import NearestNeighbors
from pymoo.algorithms.moo.moead import ParallelMOEAD
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.constraints.as_penalty import ConstraintsAsPenalty
from pymoo.core.individual import calc_cv
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.util.misc import from_dict
from src.survival import Survival_standard 
from src.opt_problem import Benchmark_Problem, EvaluatePreRealCallback, evaluate_pre_real
from src.nearest_offline import (
    compute_majority_shift_calibration_2d,
    estimate_shrinkage_mahalanobis_metric,
)


ALL_HV_VARIANTS = {
    "HV_clip",
    "HV_majority_shift_x",
    "HV_majority_shift_x_mahalanobis",
    "HV_calibration_euclidean",
    "HV_calibration_mahalanobis",
}


R_ZERO_DENOMINATOR_SUFFIX = "_zero_denominator"


def _format_seed_record_value_for_txt(seed_record, key, value):
    if key.endswith(R_ZERO_DENOMINATOR_SUFFIX):
        return None
    if (
        key.endswith("_improvement")
        and bool(seed_record.get(f"{key}{R_ZERO_DENOMINATOR_SUFFIX}", False))
    ):
        return f"{value}*"
    return value


def save_seed_solution_output(
    output_dir,
    problem_name,
    optimizer_name,
    seed,
    solution,
    obj_sur,
    obj_real,
    method_name=None,
    seed_record=None,
):
    """Append one seed's metrics, final decisions, and objective values to txt."""
    if output_dir is None:
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    solution = np.asarray(solution, dtype=float)
    obj_sur = np.asarray(obj_sur, dtype=float)
    obj_real = np.asarray(obj_real, dtype=float)
    if solution.ndim != 2 or obj_sur.ndim != 2 or obj_real.ndim != 2:
        raise ValueError("solution, obj_sur, and obj_real must be 2D arrays.")
    if not (solution.shape[0] == obj_sur.shape[0] == obj_real.shape[0]):
        raise ValueError("solution, obj_sur, and obj_real must have the same number of rows.")

    method_label = str(method_name or optimizer_name)
    filename_method = "".join(
        char if char.isalnum() or char in ("-", "_", ".") else "_"
        for char in method_label
    ).strip("._")
    if not filename_method:
        filename_method = "method"

    txt_path = output_dir / f"{filename_method}_results_solution_obj.txt"
    record = {
        "solution": solution.tolist(),
        "obj_sur": obj_sur.tolist(),
        "obj_real": obj_real.tolist(),
    }
    with open(txt_path, "a", encoding="utf-8") as output_file:
        try:
            import fcntl
        except ImportError:
            fcntl = None
        if fcntl is not None:
            fcntl.flock(output_file.fileno(), fcntl.LOCK_EX)
        try:
            output_file.write(
                f"problem={problem_name} | method={method_label} | seed={int(seed)}\n"
            )
            if seed_record is not None:
                for key, value in seed_record.items():
                    text_value = _format_seed_record_value_for_txt(seed_record, key, value)
                    if text_value is None:
                        continue
                    output_file.write(f"{key}: {text_value}\n")
            output_file.write(
                json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
        finally:
            if fcntl is not None:
                fcntl.flock(output_file.fileno(), fcntl.LOCK_UN)
    return {"txt": txt_path}


class BroadcastConstraintsAsPenalty(ConstraintsAsPenalty):
    def do(self, X, return_values_of, *args, **kwargs):
        out = self.__object__.do(X, return_values_of, *args, **kwargs)
        F, G, H = from_dict(out, "F", "G", "H")
        out["__F__"], out["__G__"], out["__H__"] = F, G, H

        CV = calc_cv(G=G, H=H)
        penalty = np.asarray(CV, dtype=float).reshape(-1, 1)
        out["F"] = F + self.penalty * penalty

        out.pop("G", None)
        out.pop("H", None)
        return out


def _normalize_optimizer_name(optimizer_name):
    name = str(optimizer_name).strip().lower()
    return name.replace("_", "").replace("-", "").replace(" ", "")


def _prepare_initial_population(problem, pop_size, initial_population=None):
    configured_pop_size = int(pop_size)
    if configured_pop_size < 2:
        raise ValueError("pop_size must be at least 2.")
    if initial_population is None:
        return configured_pop_size, None

    initial_population = np.asarray(initial_population, dtype=float)
    if initial_population.ndim != 2:
        raise ValueError("initial_population must be a 2D array.")
    if initial_population.shape[1] != problem.n_var:
        raise ValueError(
            "initial_population must have one column per problem variable: "
            f"expected {problem.n_var}, got {initial_population.shape[1]}."
        )
    if initial_population.shape[0] < 2:
        raise ValueError("initial_population must contain at least 2 points.")
    if not np.all(np.isfinite(initial_population)):
        raise ValueError("initial_population must contain only finite values.")

    initial_population = np.array(
        initial_population[:configured_pop_size],
        dtype=float,
        copy=True,
    )
    return initial_population.shape[0], initial_population


def build_optimization_algorithm(
    optimizer_name,
    problem,
    pop_size,
    survival_function=None,
    initial_population=None,
):
    effective_pop_size, initial_population = _prepare_initial_population(
        problem,
        pop_size,
        initial_population=initial_population,
    )
    optimizer_key = _normalize_optimizer_name(optimizer_name)
    crossover = SBX(prob=1.0, eta=20)
    mutation = PM(prob=1 / problem.n_var, eta=20)
    sampling_kwargs = (
        {}
        if initial_population is None
        else {"sampling": initial_population}
    )

    if optimizer_key in {"nsga2", "nsgaii"}:
        kwargs = {
            "pop_size": effective_pop_size,
            "crossover": crossover,
            "mutation": mutation,
            "eliminate_duplicates": True,
            **sampling_kwargs,
        }
        if survival_function is not None:
            kwargs["survival"] = survival_function
        return NSGA2(**kwargs)

    if optimizer_key == "moead":
        if problem.n_obj != 2:
            raise ValueError("MOEAD ref_dirs are configured for two-objective problems.")
        ref_dirs = get_reference_directions(
            "uniform",
            problem.n_obj,
            n_partitions=max(effective_pop_size - 1, 1),
        )
        # Use pymoo's synchronous variant so surrogate inference is evaluated
        # once per generation as a population-sized batch.
        return ParallelMOEAD(
            ref_dirs=ref_dirs,
            n_neighbors=min(20, len(ref_dirs)),
            n_offsprings=effective_pop_size,
            prob_neighbor_mating=0.9,
            crossover=crossover,
            mutation=mutation,
            **sampling_kwargs,
        )

    if optimizer_key in {"smsemoa", "sms", "sms emoa"}:
        return SMSEMOA(
            pop_size=effective_pop_size,
            crossover=crossover,
            mutation=mutation,
            eliminate_duplicates=True,
            **sampling_kwargs,
        )

    raise ValueError(
        "optimizer_name must be one of 'NSGA-II', 'MOEAD', or 'SMS-EMOA'."
    )


def normalize_by_train_objective_range(F, y_train):
    F = np.asarray(F, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    y_min = np.min(y_train, axis=0)
    y_max = np.max(y_train, axis=0)
    scale = np.where(y_max == y_min, 1.0, y_max - y_min)
    return (F - y_min) / scale


def _min_max_info(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError("values must be a 2D array.")
    values_min = np.min(values, axis=0)
    values_max = np.max(values, axis=0)
    values_range = np.where(values_max == values_min, 1.0, values_max - values_min)
    return values_min, values_max, values_range


def _array_text(values):
    return np.array2string(np.asarray(values, dtype=float), precision=6, suppress_small=False)


def compute_normalized_knn_threshold(y_train, k=5):
    y_train = np.asarray(y_train, dtype=float)
    if y_train.ndim != 2:
        raise ValueError("y_train must be a 2D array.")
    if y_train.shape[0] < 2:
        raise ValueError("Need at least two training samples for KNN threshold.")

    k_eff = min(int(k), y_train.shape[0] - 1)
    if k_eff < 1:
        raise ValueError("k must be at least 1.")

    y_train_norm = normalize_by_train_objective_range(y_train, y_train)

    nbrs_train = NearestNeighbors(n_neighbors=k_eff + 1).fit(y_train_norm)
    dist_train, _ = nbrs_train.kneighbors(y_train_norm)

    dist_train_no_self = dist_train[:, 1:]
    local_mean_dist = np.mean(dist_train_no_self, axis=1)
    dataset_mean_knn_dist = float(np.mean(local_mean_dist))

    return {
        "dataset_mean_knn_dist": dataset_mean_knn_dist,
        "local_mean_dist": local_mean_dist,
        "k": k_eff,
    }


def estimate_train_mahalanobis_metric(
    y_train,
    covariance_ridge=1e-8,
    covariance_source="train",
):
    y_train = np.asarray(y_train, dtype=float)
    if y_train.ndim != 2:
        raise ValueError("y_train must be a 2D array.")
    if y_train.shape[0] < 2:
        raise ValueError("Need at least two training samples to estimate covariance.")

    covariance_info = estimate_shrinkage_mahalanobis_metric(
        y_train,
        covariance_ridge=covariance_ridge,
    )
    covariance_info["covariance_source"] = covariance_source
    return covariance_info


def _check_train_mahalanobis_metric(covariance_info, n_obj):
    covariance_source = covariance_info.get("covariance_source")
    if covariance_source not in {"train", "y_train", "x_train", "distance_y", "distance_x"}:
        raise ValueError("Mahalanobis covariance_info must be estimated from training data.")
    covariance_inv = np.asarray(covariance_info.get("covariance_inv"), dtype=float)
    if covariance_inv.shape != (n_obj, n_obj):
        raise ValueError("covariance_info['covariance_inv'] has incompatible shape.")


def pairwise_mahalanobis_distance(X, Y, covariance_inv):
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    covariance_inv = np.asarray(covariance_inv, dtype=float)

    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y must be 2D arrays.")
    if X.shape[1] != Y.shape[1]:
        raise ValueError("X and Y must have the same number of columns.")
    if covariance_inv.shape != (X.shape[1], X.shape[1]):
        raise ValueError("covariance_inv has incompatible shape.")

    diff = X[:, None, :] - Y[None, :, :]
    dist_sq = np.einsum("...i,ij,...j->...", diff, covariance_inv, diff)
    return np.sqrt(np.maximum(dist_sq, 0.0))


def compute_mahalanobis_knn_threshold(
    y_train,
    k=5,
    covariance_info=None,
    covariance_ridge=1e-8,
    covariance_source="train",
):
    y_train = np.asarray(y_train, dtype=float)
    if y_train.ndim != 2:
        raise ValueError("y_train must be a 2D array.")
    if y_train.shape[0] < 2:
        raise ValueError("Need at least two training samples for KNN threshold.")

    k_eff = min(int(k), y_train.shape[0] - 1)
    if k_eff < 1:
        raise ValueError("k must be at least 1.")

    if covariance_info is None:
        covariance_info = estimate_train_mahalanobis_metric(
            y_train,
            covariance_ridge=covariance_ridge,
            covariance_source=covariance_source,
        )
    _check_train_mahalanobis_metric(covariance_info, y_train.shape[1])

    dist_train = pairwise_mahalanobis_distance(
        y_train,
        y_train,
        covariance_info["covariance_inv"],
    )
    np.fill_diagonal(dist_train, np.inf)
    sorted_dist = np.sort(dist_train, axis=1)[:, :k_eff]
    local_mean_dist = np.mean(sorted_dist, axis=1)
    dataset_mean_knn_dist = float(np.mean(local_mean_dist))

    return {
        "dataset_mean_knn_dist": dataset_mean_knn_dist,
        "local_mean_dist": local_mean_dist,
        "k": k_eff,
        "distance_metric": "mahalanobis",
        "covariance_info": covariance_info,
    }


def _distance_weighted_residual_mean(diff_y_train, neighbor_idx, neighbor_dist, bandwidth):
    bandwidth = max(float(bandwidth), np.finfo(float).eps)
    distance_weight = np.exp(-0.5 * (neighbor_dist / bandwidth) ** 2)
    weight_sum = np.sum(distance_weight)
    if weight_sum <= np.finfo(float).eps:
        return np.mean(diff_y_train[neighbor_idx], axis=0), weight_sum
    return (
        distance_weight[:, None] * diff_y_train[neighbor_idx]
    ).sum(axis=0) / weight_sum, weight_sum


def compute_local_residual_calibration(
    obj,
    y_train,
    f_train_mean,
    k=5,
    knn_info=None,
    dataset_mean_knn_dist=None,
    knn_threshold_method="mean",
    distance_train=None,
    distance_query=None,
):
    obj = np.asarray(obj, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    f_train_mean = np.asarray(f_train_mean, dtype=float)

    if obj.ndim != 2 or y_train.ndim != 2:
        raise ValueError("obj and y_train must be 2D arrays.")
    if f_train_mean.shape != y_train.shape:
        raise ValueError("f_train_mean must have the same shape as y_train.")
    if obj.shape[1] != y_train.shape[1]:
        raise ValueError("obj and y_train must have the same number of objectives.")
    if distance_train is None:
        distance_train = y_train
    if distance_query is None:
        distance_query = obj
    distance_train = np.asarray(distance_train, dtype=float)
    distance_query = np.asarray(distance_query, dtype=float)
    if distance_train.ndim != 2 or distance_query.ndim != 2:
        raise ValueError("distance_train and distance_query must be 2D arrays.")
    if distance_train.shape[0] != y_train.shape[0]:
        raise ValueError("distance_train must have the same number of rows as y_train.")
    if distance_query.shape[0] != obj.shape[0]:
        raise ValueError("distance_query must have the same number of rows as obj.")
    if distance_train.shape[1] != distance_query.shape[1]:
        raise ValueError("distance_train and distance_query must have the same number of columns.")

    if knn_info is None and dataset_mean_knn_dist is None:
        knn_info = compute_normalized_knn_threshold(distance_train, k=k)

    k_eff = min(int(k), y_train.shape[0])
    if k_eff < 1:
        raise ValueError("k must be at least 1.")

    threshold = (
        float(dataset_mean_knn_dist)
        if dataset_mean_knn_dist is not None
        else knn_info["dataset_mean_knn_dist"]
    )

    y_train_norm = normalize_by_train_objective_range(distance_train, distance_train)
    obj_norm = normalize_by_train_objective_range(distance_query, distance_train)

    nbrs = NearestNeighbors(n_neighbors=k_eff).fit(y_train_norm)
    dist, idx = nbrs.kneighbors(obj_norm)
    neighbor_mask = dist < threshold
    neighbor_counts = np.sum(neighbor_mask, axis=1)
    valid_mask = neighbor_counts > 0
    valid_obj_idx = np.flatnonzero(valid_mask)

    diff_y_train = y_train - f_train_mean
    residual_mean_y_train = np.zeros_like(obj, dtype=float)
    residual_distance_weight_sum = np.zeros(obj.shape[0], dtype=float)
    bandwidth = max(float(threshold), np.finfo(float).eps)

    for i in valid_obj_idx:
        valid_neighbor_idx = idx[i, neighbor_mask[i]]
        valid_neighbor_dist = dist[i, neighbor_mask[i]]
        residual_mean_y_train[i], residual_distance_weight_sum[i] = (
            _distance_weighted_residual_mean(
                diff_y_train,
                valid_neighbor_idx,
                valid_neighbor_dist,
                bandwidth,
            )
        )

    sur_calibration_all = obj + residual_mean_y_train
    sur_calibration = sur_calibration_all
    sur_calibration_valid = sur_calibration_all[valid_obj_idx]

    return {
        "sur_calibration": sur_calibration,
        "sur_calibration_all": sur_calibration_all,
        "sur_calibration_valid": sur_calibration_valid,
        "valid_obj_idx": valid_obj_idx,
        "valid_mask": valid_mask,
        "neighbor_counts": neighbor_counts,
        "residual_mean_y_train": residual_mean_y_train,
        "residual_distance_weight_sum": residual_distance_weight_sum,
        "residual_bandwidth": bandwidth,
        "dist": dist,
        "idx": idx,
        "threshold": threshold,
        "knn_threshold_method": "mean",
        "knn_info": knn_info,
    }


def compute_local_residual_calibration_mahalanobis(
    obj,
    y_train,
    f_train_mean,
    k=5,
    knn_info=None,
    dataset_mean_knn_dist=None,
    knn_threshold_method="mean",
    covariance_info=None,
    covariance_ridge=1e-8,
    covariance_source="train",
    distance_train=None,
    distance_query=None,
):
    obj = np.asarray(obj, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    f_train_mean = np.asarray(f_train_mean, dtype=float)

    if obj.ndim != 2 or y_train.ndim != 2:
        raise ValueError("obj and y_train must be 2D arrays.")
    if f_train_mean.shape != y_train.shape:
        raise ValueError("f_train_mean must have the same shape as y_train.")
    if obj.shape[1] != y_train.shape[1]:
        raise ValueError("obj and y_train must have the same number of objectives.")
    if distance_train is None:
        distance_train = y_train
    if distance_query is None:
        distance_query = obj
    distance_train = np.asarray(distance_train, dtype=float)
    distance_query = np.asarray(distance_query, dtype=float)
    if distance_train.ndim != 2 or distance_query.ndim != 2:
        raise ValueError("distance_train and distance_query must be 2D arrays.")
    if distance_train.shape[0] != y_train.shape[0]:
        raise ValueError("distance_train must have the same number of rows as y_train.")
    if distance_query.shape[0] != obj.shape[0]:
        raise ValueError("distance_query must have the same number of rows as obj.")
    if distance_train.shape[1] != distance_query.shape[1]:
        raise ValueError("distance_train and distance_query must have the same number of columns.")

    if covariance_info is None and knn_info is not None:
        covariance_info = knn_info.get("covariance_info")
    if covariance_info is None:
        covariance_info = estimate_train_mahalanobis_metric(
            distance_train,
            covariance_ridge=covariance_ridge,
            covariance_source=covariance_source,
        )
    _check_train_mahalanobis_metric(covariance_info, distance_train.shape[1])

    if knn_info is None and dataset_mean_knn_dist is None:
        knn_info = compute_mahalanobis_knn_threshold(
            distance_train,
            k=k,
            covariance_info=covariance_info,
            covariance_ridge=covariance_ridge,
            covariance_source=covariance_source,
        )

    k_eff = min(int(k), y_train.shape[0])
    if k_eff < 1:
        raise ValueError("k must be at least 1.")

    threshold = (
        float(dataset_mean_knn_dist)
        if dataset_mean_knn_dist is not None
        else knn_info["dataset_mean_knn_dist"]
    )

    dist_all = pairwise_mahalanobis_distance(
        distance_query,
        distance_train,
        covariance_info["covariance_inv"],
    )
    idx = np.argsort(dist_all, axis=1)[:, :k_eff]
    dist = np.take_along_axis(dist_all, idx, axis=1)
    neighbor_mask = dist < threshold
    neighbor_counts = np.sum(neighbor_mask, axis=1)
    valid_mask = neighbor_counts > 0
    valid_obj_idx = np.flatnonzero(valid_mask)

    diff_y_train = y_train - f_train_mean
    residual_mean_y_train = np.zeros_like(obj, dtype=float)
    residual_distance_weight_sum = np.zeros(obj.shape[0], dtype=float)
    bandwidth = max(float(threshold), np.finfo(float).eps)

    for i in valid_obj_idx:
        valid_neighbor_idx = idx[i, neighbor_mask[i]]
        valid_neighbor_dist = dist[i, neighbor_mask[i]]
        residual_mean_y_train[i], residual_distance_weight_sum[i] = (
            _distance_weighted_residual_mean(
                diff_y_train,
                valid_neighbor_idx,
                valid_neighbor_dist,
                bandwidth,
            )
        )

    sur_calibration_all = obj + residual_mean_y_train
    sur_calibration = sur_calibration_all
    sur_calibration_valid = sur_calibration_all[valid_obj_idx]

    return {
        "sur_calibration": sur_calibration,
        "sur_calibration_all": sur_calibration_all,
        "sur_calibration_valid": sur_calibration_valid,
        "valid_obj_idx": valid_obj_idx,
        "valid_mask": valid_mask,
        "neighbor_counts": neighbor_counts,
        "residual_mean_y_train": residual_mean_y_train,
        "residual_distance_weight_sum": residual_distance_weight_sum,
        "residual_bandwidth": bandwidth,
        "dist": dist,
        "idx": idx,
        "threshold": threshold,
        "knn_threshold_method": "mean",
        "knn_info": knn_info,
        "distance_metric": "mahalanobis",
        "covariance_info": covariance_info,
    }


def normalized_hv(hv, F, obj_min, obj_max):
    F = np.asarray(F, dtype=float)
    if F.size == 0:
        return np.nan
    F_normalization = (F - obj_min) / (obj_max - obj_min)
    return float(hv.do(F_normalization))


def normalized_objectives(F, obj_min, obj_max):
    F = np.asarray(F, dtype=float)
    obj_min = np.asarray(obj_min, dtype=float)
    obj_max = np.asarray(obj_max, dtype=float)
    scale = np.where(obj_max == obj_min, 1.0, obj_max - obj_min)
    return (F - obj_min) / scale


def clip_objectives_to_problem_bounds(F, obj_min):
    F = np.asarray(F, dtype=float)
    obj_min = np.asarray(obj_min, dtype=float)
    return np.maximum(F, obj_min)


def _clip_calibration_result_to_problem_bounds(
    calibration_result,
    obj_min,
    clip_source="objective_clip_bounds",
):
    if calibration_result is None:
        return None

    for key in (
        "sur_calibration",
        "sur_calibration_all",
        "sur_calibration_valid",
        "sur_calibration_filtered",
        "sur_calibration_full_residual_all",
    ):
        value = calibration_result.get(key)
        if value is not None:
            calibration_result[f"{key}_unclipped"] = np.asarray(value, dtype=float).copy()
            calibration_result[key] = clip_objectives_to_problem_bounds(value, obj_min)

    calibration_result["objective_clip_bounds"] = {
        "obj_min": np.asarray(obj_min, dtype=float).tolist(),
        "source": clip_source,
        "objective_clipping": "maximum(F, obj_min)",
    }
    return calibration_result


def _clip_majority_shift_result_to_problem_bounds(
    shift_result,
    obj_min,
    clip_source="objective_clip_bounds",
):
    if shift_result is None:
        return None

    value = shift_result.get("f_sur_shifted")
    if value is not None:
        value_arr = np.asarray(value, dtype=float)
        shift_result["f_sur_shifted_unclipped"] = value_arr.copy()
        clipped = clip_objectives_to_problem_bounds(
            value_arr,
            obj_min,
        )
        shift_result["f_sur_shifted"] = clipped

    shift_result["objective_clip_bounds"] = {
        "obj_min": np.asarray(obj_min, dtype=float).tolist(),
        "source": clip_source,
        "objective_clipping": "maximum(F, obj_min)",
        "filtered_dimensions_keep_unclipped": False,
    }
    return shift_result


def _evaluate_majority_shift_hv(
    shift_result,
    hv,
    f_real,
    hv_real,
    hv_sur_gap,
    obj_min,
    obj_max,
):
    hv_value = normalized_hv(
        hv,
        shift_result["f_sur_shifted"],
        obj_min,
        obj_max,
    )
    hv_gap = abs(hv_real - hv_value)
    return {
        "hv": hv_value,
        "gap": hv_gap,
        "gap_reduction_pct": compute_R_indicator(hv_sur_gap, hv_gap),
        "count": int(shift_result["f_sur_shifted"].shape[0]),
    }


def evaluate_majority_shift_x_variant(
    x,
    f_sur,
    f_real,
    x_train,
    y_train,
    f_train_pred,
    hv,
    hv_real,
    hv_sur_gap,
    obj_min,
    obj_max,
    clip_obj_min,
    k,
    distance_metric="normalized_euclidean",
    covariance_ridge=1e-8,
    use_clip=True,
    clip_source="problem_y_threshold",
):
    """Evaluate the shared Exp1-style x-space majority-shift HV variant."""
    shift_result = compute_majority_shift_calibration_2d(
        x=x,
        f_sur=f_sur,
        f_real=f_real,
        x_train=x_train,
        y_train=y_train,
        f_train_pred=f_train_pred,
        distance_space="x",
        distance_metric=distance_metric,
        k=k,
        covariance_ridge=covariance_ridge,
    )
    if use_clip:
        _clip_majority_shift_result_to_problem_bounds(
            shift_result,
            clip_obj_min,
            clip_source=clip_source,
        )
    return {
        **_evaluate_majority_shift_hv(
            shift_result,
            hv,
            f_real,
            hv_real,
            hv_sur_gap,
            obj_min,
            obj_max,
        ),
        "result": shift_result,
    }


def non_dominated_mask(F):
    F = np.asarray(F, dtype=float)
    mask = np.zeros(F.shape[0], dtype=bool)
    if F.size == 0:
        return mask
    nd_idx = NonDominatedSorting().do(F, only_non_dominated_front=True)
    mask[np.asarray(nd_idx, dtype=int)] = True
    return mask


def hv_contribution_vector(hv, F, obj_min, obj_max):
    F = np.asarray(F, dtype=float)
    if F.size == 0:
        return np.asarray([], dtype=float)

    F_normalization = normalized_objectives(F, obj_min, obj_max)
    total_hv = float(hv.do(F_normalization))
    contributions = np.zeros(F.shape[0], dtype=float)

    for i in range(F.shape[0]):
        if F.shape[0] == 1:
            hv_without_i = 0.0
        else:
            hv_without_i = float(hv.do(np.delete(F_normalization, i, axis=0)))
        contributions[i] = total_hv - hv_without_i

    return contributions


def compute_candidate_set_diagnostics(
    hv,
    obj,
    f_real,
    obj_min,
    obj_max,
    normalization_min=None,
    normalization_max=None,
    normalization_source="obj_min_obj_max",
):
    obj = np.asarray(obj, dtype=float)
    f_real = np.asarray(f_real, dtype=float)
    if obj.shape != f_real.shape:
        raise ValueError("obj and f_real must have the same shape.")

    hv_sur = normalized_hv(hv, obj, obj_min, obj_max)
    hv_real = normalized_hv(hv, f_real, obj_min, obj_max)
    sur_nd_mask = non_dominated_mask(obj)
    real_nd_mask = non_dominated_mask(f_real)
    overlap_mask = sur_nd_mask & real_nd_mask
    union_mask = sur_nd_mask | real_nd_mask

    sur_contribution = hv_contribution_vector(hv, obj, obj_min, obj_max)
    real_contribution = hv_contribution_vector(hv, f_real, obj_min, obj_max)
    contribution_error = sur_contribution - real_contribution

    normalization_min = obj_min if normalization_min is None else normalization_min
    normalization_max = obj_max if normalization_max is None else normalization_max
    obj_norm = normalized_objectives(obj, normalization_min, normalization_max)
    real_norm = normalized_objectives(f_real, normalization_min, normalization_max)
    ref_point = np.asarray(getattr(hv, "ref_point", np.ones(obj.shape[1])), dtype=float)
    sur_direction = ref_point - obj_norm
    real_direction = ref_point - real_norm
    direction_dot = np.sum(sur_direction * real_direction, axis=1)
    direction_norm = (
        np.linalg.norm(sur_direction, axis=1)
        * np.linalg.norm(real_direction, axis=1)
    )
    direction_cosine = np.divide(
        direction_dot,
        direction_norm,
        out=np.full_like(direction_dot, np.nan, dtype=float),
        where=direction_norm > 0,
    )
    direction_sign_match = np.mean(
        np.sign(sur_direction) == np.sign(real_direction),
        axis=1,
    )

    sur_nd_count = int(np.sum(sur_nd_mask))
    real_nd_count = int(np.sum(real_nd_mask))
    overlap_count = int(np.sum(overlap_mask))
    union_count = int(np.sum(union_mask))

    return {
        "delta_hv": hv_sur - hv_real,
        "sur_nd_count": sur_nd_count,
        "real_nd_count": real_nd_count,
        "sur_nd_real_nd_count": overlap_count,
        "pf_overlap_count": overlap_count,
        "pf_overlap_sur_ratio": (
            overlap_count / sur_nd_count if sur_nd_count > 0 else np.nan
        ),
        "pf_overlap_jaccard": (
            overlap_count / union_count if union_count > 0 else np.nan
        ),
        "false_positive_pareto_count": int(np.sum(sur_nd_mask & ~real_nd_mask)),
        "false_negative_pareto_count": int(np.sum(real_nd_mask & ~sur_nd_mask)),
        "hv_contribution_error": contribution_error,
        "hv_contribution_error_mean": float(np.mean(contribution_error)),
        "hv_contribution_error_mae": float(np.mean(np.abs(contribution_error))),
        "hv_contribution_error_max_abs": float(np.max(np.abs(contribution_error))),
        "direction_cosine": direction_cosine,
        "direction_cosine_mean": float(np.nanmean(direction_cosine)),
        "direction_cosine_positive_rate": float(np.nanmean(direction_cosine > 0)),
        "direction_sign_match_rate": float(np.mean(direction_sign_match)),
        "normalization_source": normalization_source,
        "normalization_min": np.asarray(normalization_min, dtype=float).tolist(),
        "normalization_max": np.asarray(normalization_max, dtype=float).tolist(),
    }


def mse_or_nan(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size == 0 or y_pred.size == 0:
        return np.nan
    return mean_squared_error(y_true, y_pred)


def compute_surrogate_test_mse(
    problem,
    problem_name,
    model_f1,
    model_f2,
    use_surrogate,
    x_test,
    y_test,
):
    surrogate_problem = Benchmark_Problem(
        model_f1=model_f1,
        model_f2=model_f2,
        n_var=problem.n_var,
        n_obj=problem.n_obj,
        xl=problem.xl,
        xu=problem.xu,
        problem_name=problem_name,
        use_surrogate=use_surrogate,
    )
    f_test_pred = surrogate_problem.evaluate(
        np.asarray(x_test, dtype=float),
        return_values_of=["F"],
    )
    return mse_or_nan(y_test, f_test_pred)


def compute_R_indicator(gap_sur, gap_xy_shift):
    if gap_sur is None:
        return np.nan
    try:
        gap_sur = float(gap_sur)
    except (TypeError, ValueError):
        return np.nan
    if not np.isfinite(gap_sur):
        return np.nan
    if gap_sur == 0:
        return 0.0

    if gap_xy_shift is None:
        return np.nan
    try:
        gap_xy_shift = float(gap_xy_shift)
    except (TypeError, ValueError):
        return np.nan
    if not np.isfinite(gap_xy_shift):
        return np.nan
    R_indicator = 100.0 * (gap_sur - gap_xy_shift) / gap_sur
    return R_indicator


def is_R_indicator_zero_denominator(gap_sur):
    if gap_sur is None:
        return False
    try:
        gap_sur = float(gap_sur)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(gap_sur) and gap_sur == 0)


def format_percent(value):
    if value is None or not np.isfinite(value):
        return "NA%"
    return f"{value:.1f}%"


def wilcoxon_gap_test(hv_sur_gap_list, hv_calibration_gap_list):
    hv_sur_gap = np.asarray(hv_sur_gap_list, dtype=float)
    hv_calibration_gap = np.asarray(hv_calibration_gap_list, dtype=float)

    if hv_sur_gap.shape != hv_calibration_gap.shape:
        raise ValueError("HV gap arrays must have the same shape.")
    if hv_sur_gap.size < 2:
        return np.nan

    try:
        result = stats.wilcoxon(
            hv_calibration_gap,
            hv_sur_gap,
            alternative="two-sided",
            zero_method="wilcox",
        )
    except ValueError:
        return np.nan

    return float(result.pvalue)


def run_experiment(
    problem,
    problem_name,
    n_gen,
    pop_size,
    use_surrogate,
    model_f1,
    model_f2,
    survival_function,
    obj_min,
    obj_max,
    hv,
    igd_plus,
    use_callback,
    seeds,
    optimizer_name="NSGA-II",
    hv_variants=None,
    x_train=None,
    distance_x=None,
    distance_y=None,
    knn_distance_space="auto",
    y_train=None,
    f_train_mean=None,
    problem_y_min=None,
    calibration_k=5,
    knn_threshold_method="mean",
    knn_info=None,
    dataset_mean_knn_dist=None,
    covariance_info=None,
    covariance_ridge=1e-8,
    majority_shift_k_values=(1, 2),
    print_normalization_info=True,
    initial_population=None,
    compact_seed_output=False,
    compact_seed_context=None,
    compact_seed_mse_test=None,
    compact_seed_method_name=None,
    solution_output_dir=None,
    solution_output_method_name=None):

    minimize_kwargs = dict(
      termination=get_termination("n_gen", n_gen),
      save_history=False,
      verbose=False)

    if hv_variants is None:
        hv_variant_set = set(ALL_HV_VARIANTS)
    else:
        hv_variant_set = set(hv_variants)
        unknown_variants = hv_variant_set - ALL_HV_VARIANTS
        if unknown_variants:
            raise ValueError(
                f"Unknown HV variants: {sorted(unknown_variants)}. "
                f"Known variants: {sorted(ALL_HV_VARIANTS)}"
            )

    use_majority_shift_x = (
        "HV_majority_shift_x" in hv_variant_set
        or "HV_majority_shift_x_mahalanobis" in hv_variant_set
    )
    majority_shift_k_values = tuple(
        dict.fromkeys(int(k) for k in majority_shift_k_values)
    )
    if not majority_shift_k_values or any(k < 1 for k in majority_shift_k_values):
        raise ValueError("majority_shift_k_values must contain positive integers.")
    if 1 not in majority_shift_k_values:
        majority_shift_k_values = (1, *majority_shift_k_values)
    use_residual_calibration_variants = any(
        variant in hv_variant_set
        for variant in (
            "HV_calibration_euclidean",
            "HV_calibration_mahalanobis",
        )
    )

    # Callback
    if use_callback:
      callback_standard = EvaluatePreRealCallback(
          true_problem=problem,
          plot_every=10,
          use_opt=True,
          dynamic_show=False,
          prefix=f"{optimizer_name}-standard",
          obj_min=obj_min,
          obj_max=obj_max,
          hv_indicator=hv)
      minimize_kwargs["callback"] = callback_standard

    igd_list = []
    hv_surrogate_list = []
    hv_clipped_list = []
    hv_real_list = []
    hv_calibration_list = []
    hv_calibration_mahalanobis_list = []
    hv_majority_shift_x_list = []
    hv_majority_shift_x_mahalanobis_list = []
    hv_surrogate_count_list = []
    hv_clipped_count_list = []
    hv_real_count_list = []
    hv_calibration_count_list = []
    hv_calibration_mahalanobis_count_list = []
    hv_majority_shift_x_count_list = []
    hv_majority_shift_x_mahalanobis_count_list = []
    hv_sur_gap_list = []
    hv_clipped_gap_list = []
    hv_clipped_gap_reduction_pct_list = []
    hv_calibration_gap_list = []
    hv_gap_reduction_pct_list = []
    hv_calibration_mahalanobis_gap_list = []
    hv_gap_reduction_mahalanobis_pct_list = []
    hv_majority_shift_x_gap_list = []
    hv_majority_shift_x_gap_reduction_pct_list = []
    hv_majority_shift_x_mahalanobis_gap_list = []
    hv_majority_shift_x_mahalanobis_gap_reduction_pct_list = []
    majority_shift_x_result_list = []
    majority_shift_x_by_k_list = []
    majority_shift_x_mahalanobis_result_list = []
    candidate_set_diagnostics_list = []
    nearest_quadrant_summary_list = []
    run_details = []
    if distance_x is None and x_train is not None:
        distance_x = x_train
    if y_train is None and distance_y is not None:
        y_train = distance_y
    if distance_y is None:
        distance_y = y_train
    use_calibration = y_train is not None and f_train_mean is not None

    if initial_population is None:
        initial_population = distance_x
        initial_population_source = (
            "distance_x_first_pop_size"
            if distance_x is not None
            else "optimizer_default_random_sampling"
        )
    else:
        initial_population_source = "explicit_first_pop_size"
    initial_population_array = (
        None if initial_population is None else np.asarray(initial_population)
    )
    initial_population_available_count = (
        None
        if initial_population_array is None or initial_population_array.ndim == 0
        else int(initial_population_array.shape[0])
    )
    effective_pop_size, initial_population = _prepare_initial_population(
        problem,
        pop_size,
        initial_population=initial_population,
    )
    initialization_info = {
        "source": initial_population_source,
        "selection": (
            None
            if initial_population is None
            else "first_configured_pop_size_points"
        ),
        "configured_pop_size": int(pop_size),
        "available_offline_points": initial_population_available_count,
        "effective_pop_size": int(effective_pop_size),
        "initial_population_x": (
            None
            if initial_population is None
            else np.array(initial_population, dtype=float, copy=True)
        ),
    }

    x_norm_min = x_norm_max = None
    if distance_x is not None:
        x_norm_min, x_norm_max, _ = _min_max_info(distance_x)
    y_norm_ref = distance_y if distance_y is not None else y_train
    y_norm_min = y_norm_max = None
    if y_norm_ref is not None:
        y_norm_min, y_norm_max, _ = _min_max_info(y_norm_ref)

    problem_y_min = obj_min if problem_y_min is None else problem_y_min
    clip_obj_min = np.asarray(problem_y_min, dtype=float)
    clip_source = "problem_y_threshold"
    objective_norm_min = (
        y_norm_min if y_norm_min is not None else np.asarray(obj_min, dtype=float)
    )
    objective_norm_max = (
        y_norm_max if y_norm_max is not None else np.asarray(obj_max, dtype=float)
    )
    objective_norm_source = (
        "offline_y_train_minmax"
        if y_norm_min is not None
        else "obj_min_obj_max_fallback"
    )
    normalization_info = {
        "hv_obj_min": np.asarray(obj_min, dtype=float).tolist(),
        "hv_obj_max": np.asarray(obj_max, dtype=float).tolist(),
        "x_train_min": None if x_norm_min is None else x_norm_min.tolist(),
        "x_train_max": None if x_norm_max is None else x_norm_max.tolist(),
        "y_train_min": None if y_norm_min is None else y_norm_min.tolist(),
        "y_train_max": None if y_norm_max is None else y_norm_max.tolist(),
        "objective_normalization_min": np.asarray(
            objective_norm_min,
            dtype=float,
        ).tolist(),
        "objective_normalization_max": np.asarray(
            objective_norm_max,
            dtype=float,
        ).tolist(),
        "objective_normalization_source": objective_norm_source,
        "clip_min": np.asarray(clip_obj_min, dtype=float).tolist(),
        "clip_source": clip_source,
        "objective_clipping": "maximum(F, clip_min)",
        "optimizer_name": optimizer_name,
        "hv_variants": sorted(hv_variant_set),
    }
    if print_normalization_info:
        print(
            f"[{problem_name}] initial population: {initial_population_source}, "
            f"configured pop_size={int(pop_size)}, "
            f"effective pop_size={effective_pop_size}"
        )
        print(f"[{problem_name}] HV obj_min: {_array_text(obj_min)}")
        print(f"[{problem_name}] HV obj_max: {_array_text(obj_max)}")
        if x_norm_min is not None:
            print(f"[{problem_name}] X_train normalization min: {_array_text(x_norm_min)}")
            print(f"[{problem_name}] X_train normalization max: {_array_text(x_norm_max)}")
        if y_norm_min is not None:
            print(f"[{problem_name}] y_train normalization min: {_array_text(y_norm_min)}")
            print(f"[{problem_name}] y_train normalization max: {_array_text(y_norm_max)}")
        print(f"[{problem_name}] clipping min ({clip_source}): {_array_text(clip_obj_min)}")

    knn_distance_space = knn_distance_space.lower()
    if knn_distance_space == "auto":
        knn_distance_space = "x" if distance_x is not None else "y"
    if knn_distance_space in ("x", "decision", "decision_x"):
        if distance_x is None:
            raise ValueError("distance_x is required when knn_distance_space='x'.")
        distance_train = distance_x
        distance_source = "distance_x"
    elif knn_distance_space in ("y", "objective", "objective_y"):
        if distance_y is None:
            raise ValueError("distance_y is required when knn_distance_space='y'.")
        distance_train = distance_y
        distance_source = "distance_y"
    else:
        raise ValueError("knn_distance_space must be 'auto', 'x', or 'y'.")

    use_residual_calibration = use_calibration and use_residual_calibration_variants
    mahalanobis_knn_mean_info = None
    knn_mean_info = None
    if use_residual_calibration and knn_info is None and dataset_mean_knn_dist is None:
      knn_info = compute_normalized_knn_threshold(distance_train, k=calibration_k)
    if use_residual_calibration:
      knn_mean_info = knn_info
      if covariance_info is None:
        covariance_info = estimate_train_mahalanobis_metric(
            distance_train,
            covariance_ridge=covariance_ridge,
            covariance_source=distance_source,
        )
      else:
        _check_train_mahalanobis_metric(covariance_info, distance_train.shape[1])
      mahalanobis_knn_mean_info = compute_mahalanobis_knn_threshold(
          distance_train,
          k=calibration_k,
          covariance_info=covariance_info,
          covariance_ridge=covariance_ridge,
          covariance_source=distance_source,
      )

    for seed in seeds:
      random.seed(seed)
      np.random.seed(seed)
        
      # Benchmark problem
      benchmark_problem_GPR = Benchmark_Problem(
          model_f1=model_f1,
          model_f2=model_f2,
          n_var=problem.n_var,
          n_obj=problem.n_obj,
          xl=problem.xl,
          xu=problem.xu,
          problem_name=problem_name,
          use_surrogate=use_surrogate)


      # Optimization
      start_time = time.time()
      algorithm = build_optimization_algorithm(
          optimizer_name=optimizer_name,
          problem=problem,
          pop_size=pop_size,
          survival_function=survival_function,
          initial_population=initial_population,
      )
      optimization_problem = benchmark_problem_GPR
      constraint_handling = None
      if (
          _normalize_optimizer_name(optimizer_name) == "moead"
          and benchmark_problem_GPR.has_constraints()
      ):
          optimization_problem = BroadcastConstraintsAsPenalty(
              benchmark_problem_GPR,
              penalty=1e6,
          )
          constraint_handling = "BroadcastConstraintsAsPenalty(penalty=1e6)"

      res = minimize(
          optimization_problem,
          algorithm,
          seed=seed,
          **minimize_kwargs)

      end_time = time.time()

      # Final solutions. Avoid save_history=True because history retains heavy
      # surrogate/problem objects across seeds, especially for GPy models.
      final_opt = getattr(res, "opt", None)
      if final_opt is None:
          raise RuntimeError("pymoo result does not contain final opt solutions.")
      solution = final_opt.get("X")
      obj = benchmark_problem_GPR.evaluate(solution, return_values_of=["F"])
      distance_query = solution if knn_distance_space in ("x", "decision", "decision_x") else obj
      f_real = problem.evaluate(solution, return_values_of=["F"])
      solution_output_paths = None

      # IGD+
      igd_plus_real = float(igd_plus(f_real))
      igd_plus_surrogate = float(igd_plus(obj))
      igd_list.append(igd_plus_real)

      # HV
      hv_real = normalized_hv(hv, f_real, obj_min, obj_max)
      hv_surrogate = normalized_hv(hv, obj, obj_min, obj_max)
      obj_clipped = clip_objectives_to_problem_bounds(obj, clip_obj_min)
      hv_clipped = normalized_hv(hv, obj_clipped, obj_min, obj_max)
      hv_sur_gap = abs(hv_real - hv_surrogate)
      hv_clipped_gap = abs(hv_real - hv_clipped)
      hv_clipped_gap_reduction_pct = compute_R_indicator(
          hv_sur_gap,
          hv_clipped_gap,
      )
      candidate_set_diagnostics = compute_candidate_set_diagnostics(
          hv,
          obj,
          f_real,
          obj_min,
          obj_max,
          normalization_min=objective_norm_min,
          normalization_max=objective_norm_max,
          normalization_source=objective_norm_source,
      )
      hv_real_count = int(f_real.shape[0])
      hv_surrogate_count = int(obj.shape[0])
      hv_clipped_count = int(obj_clipped.shape[0])
      hv_calibration = None
      hv_calibration_mahalanobis = None
      hv_calibration_count = 0
      hv_calibration_mahalanobis_count = 0
      hv_calibration_gap = None
      hv_gap_reduction_pct = None
      hv_calibration_mahalanobis_gap = None
      hv_gap_reduction_mahalanobis_pct = None
      hv_majority_shift_x = None
      hv_majority_shift_x_mahalanobis = None
      hv_majority_shift_x_gap = None
      hv_majority_shift_x_mahalanobis_gap = None
      hv_majority_shift_x_gap_reduction_pct = None
      hv_majority_shift_x_mahalanobis_gap_reduction_pct = None
      hv_majority_shift_x_count = 0
      hv_majority_shift_x_mahalanobis_count = 0
      majority_shift_x_result = None
      majority_shift_x_by_k = {}
      majority_shift_x_mahalanobis_result = None
      nearest_quadrant_summary = None
      calibration_result = None
      calibration_mean_result = None
      calibration_mahalanobis_result = None
      calibration_mahalanobis_mean_result = None

      if use_calibration and use_majority_shift_x and distance_x is not None:
          for majority_shift_k in majority_shift_k_values:
              majority_shift_x_by_k[majority_shift_k] = evaluate_majority_shift_x_variant(
                  x=solution,
                  f_sur=obj,
                  f_real=f_real,
                  x_train=distance_x,
                  y_train=y_train,
                  f_train_pred=f_train_mean,
                  hv=hv,
                  hv_real=hv_real,
                  hv_sur_gap=hv_sur_gap,
                  obj_min=obj_min,
                  obj_max=obj_max,
                  clip_obj_min=clip_obj_min,
                  distance_metric="normalized_euclidean",
                  k=majority_shift_k,
                  covariance_ridge=covariance_ridge,
                  clip_source=clip_source,
              )

          majority_shift_x_result = majority_shift_x_by_k[1]["result"]
          hv_majority_shift_x = majority_shift_x_by_k[1]["hv"]
          hv_majority_shift_x_gap = majority_shift_x_by_k[1]["gap"]
          hv_majority_shift_x_gap_reduction_pct = majority_shift_x_by_k[1][
              "gap_reduction_pct"
          ]
          hv_majority_shift_x_count = majority_shift_x_by_k[1]["count"]

          majority_shift_x_mahalanobis_eval = evaluate_majority_shift_x_variant(
              x=solution,
              f_sur=obj,
              f_real=f_real,
              x_train=distance_x,
              y_train=y_train,
              f_train_pred=f_train_mean,
              hv=hv,
              hv_real=hv_real,
              hv_sur_gap=hv_sur_gap,
              obj_min=obj_min,
              obj_max=obj_max,
              clip_obj_min=clip_obj_min,
              distance_metric="mahalanobis",
              k=1,
              covariance_ridge=covariance_ridge,
              clip_source=clip_source,
          )
          majority_shift_x_mahalanobis_result = majority_shift_x_mahalanobis_eval["result"]
          hv_majority_shift_x_mahalanobis = majority_shift_x_mahalanobis_eval["hv"]
          hv_majority_shift_x_mahalanobis_gap = majority_shift_x_mahalanobis_eval["gap"]
          hv_majority_shift_x_mahalanobis_gap_reduction_pct = majority_shift_x_mahalanobis_eval["gap_reduction_pct"]
          hv_majority_shift_x_mahalanobis_count = majority_shift_x_mahalanobis_eval["count"]

          nearest_quadrant_summary = {
              "x": majority_shift_x_result["nearest_offline_summary"],
              "x_mahalanobis": majority_shift_x_mahalanobis_result["nearest_offline_summary"],
          }

      if use_residual_calibration:
          calibration_mean_result = compute_local_residual_calibration(
              obj=obj,
              y_train=y_train,
              f_train_mean=f_train_mean,
              k=calibration_k,
              knn_info=knn_mean_info,
              dataset_mean_knn_dist=dataset_mean_knn_dist,
              knn_threshold_method="mean",
              distance_train=distance_train,
              distance_query=distance_query,
          )
          _clip_calibration_result_to_problem_bounds(
              calibration_mean_result,
              clip_obj_min,
              clip_source=clip_source,
          )
          calibration_result = calibration_mean_result
          hv_calibration = normalized_hv(
              hv,
              calibration_result["sur_calibration"],
              obj_min,
              obj_max,
          )
          hv_calibration_count = int(calibration_result["sur_calibration"].shape[0])
          hv_calibration_gap = abs(hv_real - hv_calibration)
          hv_gap_reduction_pct = compute_R_indicator(
              hv_sur_gap,
              hv_calibration_gap,
          )
          calibration_mahalanobis_mean_result = compute_local_residual_calibration_mahalanobis(
              obj=obj,
              y_train=y_train,
              f_train_mean=f_train_mean,
              k=calibration_k,
              knn_info=mahalanobis_knn_mean_info,
              knn_threshold_method="mean",
              covariance_info=covariance_info,
              covariance_ridge=covariance_ridge,
              covariance_source=distance_source,
              distance_train=distance_train,
              distance_query=distance_query,
          )
          _clip_calibration_result_to_problem_bounds(
              calibration_mahalanobis_mean_result,
              clip_obj_min,
              clip_source=clip_source,
          )
          calibration_mahalanobis_result = calibration_mahalanobis_mean_result
          hv_calibration_mahalanobis = normalized_hv(
              hv,
              calibration_mahalanobis_result["sur_calibration"],
              obj_min,
              obj_max,
          )
          hv_calibration_mahalanobis_count = int(calibration_mahalanobis_result["sur_calibration"].shape[0])
          hv_calibration_mahalanobis_gap = abs(hv_real - hv_calibration_mahalanobis)
          hv_gap_reduction_mahalanobis_pct = compute_R_indicator(
              hv_sur_gap,
              hv_calibration_mahalanobis_gap,
          )

      hv_real_list.append(hv_real)
      hv_surrogate_list.append(hv_surrogate)
      hv_clipped_list.append(hv_clipped)
      hv_real_count_list.append(hv_real_count)
      hv_surrogate_count_list.append(hv_surrogate_count)
      hv_clipped_count_list.append(hv_clipped_count)
      hv_sur_gap_list.append(hv_sur_gap)
      hv_clipped_gap_list.append(hv_clipped_gap)
      hv_clipped_gap_reduction_pct_list.append(hv_clipped_gap_reduction_pct)
      candidate_set_diagnostics_list.append(candidate_set_diagnostics)
      if nearest_quadrant_summary is not None:
          nearest_quadrant_summary_list.append(nearest_quadrant_summary)
      if majority_shift_x_result is not None:
          hv_majority_shift_x_list.append(hv_majority_shift_x)
          hv_majority_shift_x_gap_list.append(hv_majority_shift_x_gap)
          hv_majority_shift_x_gap_reduction_pct_list.append(
              hv_majority_shift_x_gap_reduction_pct
          )
          hv_majority_shift_x_count_list.append(hv_majority_shift_x_count)
          majority_shift_x_result_list.append(majority_shift_x_result)
          majority_shift_x_by_k_list.append(majority_shift_x_by_k)
      if majority_shift_x_mahalanobis_result is not None:
          hv_majority_shift_x_mahalanobis_list.append(hv_majority_shift_x_mahalanobis)
          hv_majority_shift_x_mahalanobis_gap_list.append(hv_majority_shift_x_mahalanobis_gap)
          hv_majority_shift_x_mahalanobis_gap_reduction_pct_list.append(
              hv_majority_shift_x_mahalanobis_gap_reduction_pct
          )
          hv_majority_shift_x_mahalanobis_count_list.append(hv_majority_shift_x_mahalanobis_count)
          majority_shift_x_mahalanobis_result_list.append(majority_shift_x_mahalanobis_result)
      if use_residual_calibration:
          hv_calibration_list.append(hv_calibration)
          hv_calibration_mahalanobis_list.append(hv_calibration_mahalanobis)
          hv_calibration_count_list.append(hv_calibration_count)
          hv_calibration_mahalanobis_count_list.append(hv_calibration_mahalanobis_count)
          hv_calibration_gap_list.append(hv_calibration_gap)
          hv_gap_reduction_pct_list.append(hv_gap_reduction_pct)
          hv_calibration_mahalanobis_gap_list.append(hv_calibration_mahalanobis_gap)
          hv_gap_reduction_mahalanobis_pct_list.append(hv_gap_reduction_mahalanobis_pct)
      max_obj = np.max(obj, axis=0)
      max_obj_real = np.max(f_real, axis=0)

      majority_shift_extra_detail = {
          "majority_shift_x_by_k": majority_shift_x_by_k,
          "hv_majority_shift_x_mahalanobis": hv_majority_shift_x_mahalanobis,
          "hv_majority_shift_x_mahalanobis_count": hv_majority_shift_x_mahalanobis_count,
          "hv_majority_shift_x_mahalanobis_gap": hv_majority_shift_x_mahalanobis_gap,
          "hv_majority_shift_x_mahalanobis_gap_reduction_pct": hv_majority_shift_x_mahalanobis_gap_reduction_pct,
          "majority_shift_x_mahalanobis_result": majority_shift_x_mahalanobis_result,
      }
 
      if use_callback:
          run_details.append({
              "seed": seed,
              "time": end_time - start_time,
              "solution": solution,
              "obj": obj,
              "obj_sur": obj,
              "f_real": f_real,
              "obj_real": f_real,
              "solution_output_paths": solution_output_paths,
              "igd_plus": igd_plus_real,
              "igd_plus_real": igd_plus_real,
              "igd_plus_surrogate": igd_plus_surrogate,
              "igd_plus_indicator": igd_plus,
              "hv_surrogate": hv_surrogate,
              "hv_clipped": hv_clipped,
              "hv_calibration": hv_calibration,
              "hv_calibration_mahalanobis": hv_calibration_mahalanobis,
              "hv_majority_shift_x": hv_majority_shift_x,
              **majority_shift_extra_detail,
              "hv_real": hv_real,
              "hv_surrogate_count": hv_surrogate_count,
              "hv_clipped_count": hv_clipped_count,
              "hv_real_count": hv_real_count,
              "hv_calibration_count": hv_calibration_count,
              "hv_calibration_mahalanobis_count": hv_calibration_mahalanobis_count,
              "hv_majority_shift_x_count": hv_majority_shift_x_count,
              "hv_sur_gap": hv_sur_gap,
              "hv_clipped_gap": hv_clipped_gap,
              "hv_clipped_gap_reduction_pct": hv_clipped_gap_reduction_pct,
              "candidate_set_diagnostics": candidate_set_diagnostics,
              "hv_calibration_gap": hv_calibration_gap,
              "hv_gap_reduction_pct": hv_gap_reduction_pct,
              "hv_calibration_mahalanobis_gap": hv_calibration_mahalanobis_gap,
              "hv_gap_reduction_mahalanobis_pct": hv_gap_reduction_mahalanobis_pct,
              "hv_majority_shift_x_gap": hv_majority_shift_x_gap,
              "hv_majority_shift_x_gap_reduction_pct": hv_majority_shift_x_gap_reduction_pct,
              "max_obj": max_obj,
              "max_f_real": max_obj_real,
              "obj_clipped": obj_clipped,
              "calibration_result": calibration_result,
              "calibration_mean_result": calibration_mean_result,
              "calibration_mahalanobis_result": calibration_mahalanobis_result,
              "calibration_mahalanobis_mean_result": calibration_mahalanobis_mean_result,
              "majority_shift_x_result": majority_shift_x_result,
              "nearest_quadrant_summary": nearest_quadrant_summary,
              "normalization_info": normalization_info,
              "constraint_handling": constraint_handling,
              "obj_calibration": None if calibration_result is None else calibration_result["sur_calibration"],
              "obj_calibration_mahalanobis": None if calibration_mahalanobis_result is None else calibration_mahalanobis_result["sur_calibration"],
              "calibrated_count": 0 if calibration_result is None else len(calibration_result["valid_obj_idx"]),
              "calibrated_mahalanobis_count": 0 if calibration_mahalanobis_result is None else len(calibration_mahalanobis_result["valid_obj_idx"]),
              "gen_history": callback_standard.gen_list,
              "hv_sur_history": callback_standard.hv_sur_list,
              "hv_real_history": callback_standard.hv_real_list})
      else:
          run_details.append({
              "seed": seed,
              "time": end_time - start_time,
              "solution": solution,
              "obj": obj,
              "obj_sur": obj,
              "f_real": f_real,
              "obj_real": f_real,
              "solution_output_paths": solution_output_paths,
              "igd_plus": igd_plus_real,
              "igd_plus_real": igd_plus_real,
              "igd_plus_surrogate": igd_plus_surrogate,
              "igd_plus_indicator": igd_plus,
              "hv_surrogate": hv_surrogate,
              "hv_clipped": hv_clipped,
              "hv_calibration": hv_calibration,
              "hv_calibration_mahalanobis": hv_calibration_mahalanobis,
              "hv_majority_shift_x": hv_majority_shift_x,
              **majority_shift_extra_detail,
              "hv_real": hv_real,
              "hv_surrogate_count": hv_surrogate_count,
              "hv_clipped_count": hv_clipped_count,
              "hv_real_count": hv_real_count,
              "hv_calibration_count": hv_calibration_count,
              "hv_calibration_mahalanobis_count": hv_calibration_mahalanobis_count,
              "hv_majority_shift_x_count": hv_majority_shift_x_count,
              "hv_sur_gap": hv_sur_gap,
              "hv_clipped_gap": hv_clipped_gap,
              "hv_clipped_gap_reduction_pct": hv_clipped_gap_reduction_pct,
              "candidate_set_diagnostics": candidate_set_diagnostics,
              "hv_calibration_gap": hv_calibration_gap,
              "hv_gap_reduction_pct": hv_gap_reduction_pct,
              "hv_calibration_mahalanobis_gap": hv_calibration_mahalanobis_gap,
              "hv_gap_reduction_mahalanobis_pct": hv_gap_reduction_mahalanobis_pct,
              "hv_majority_shift_x_gap": hv_majority_shift_x_gap,
              "hv_majority_shift_x_gap_reduction_pct": hv_majority_shift_x_gap_reduction_pct,
              "max_obj": max_obj,
              "max_f_real": max_obj_real,
              "obj_clipped": obj_clipped,
              "calibration_result": calibration_result,
              "calibration_mean_result": calibration_mean_result,
              "calibration_mahalanobis_result": calibration_mahalanobis_result,
              "calibration_mahalanobis_mean_result": calibration_mahalanobis_mean_result,
              "majority_shift_x_result": majority_shift_x_result,
              "nearest_quadrant_summary": nearest_quadrant_summary,
              "normalization_info": normalization_info,
              "constraint_handling": constraint_handling,
              "obj_calibration": None if calibration_result is None else calibration_result["sur_calibration"],
              "obj_calibration_mahalanobis": None if calibration_mahalanobis_result is None else calibration_mahalanobis_result["sur_calibration"],
              "calibrated_count": 0 if calibration_result is None else len(calibration_result["valid_obj_idx"]),
              "calibrated_mahalanobis_count": 0 if calibration_mahalanobis_result is None else len(calibration_mahalanobis_result["valid_obj_idx"])})

      seed_record = None
      if compact_seed_output:
          from src.plotting import print_seed_result

          seed_context = compact_seed_context
          if seed_context is None:
              seed_context = {
                  "X_train": distance_x,
                  "y_train": y_train,
                  "f_train_mean": f_train_mean,
                  "hv": hv,
                  "igd_plus": igd_plus,
                  "obj_min": obj_min,
                  "obj_max": obj_max,
                  "problem_y_min": problem_y_min,
                  "clip_source": clip_source,
              }
          seed_record, _ = print_seed_result(
              run_details[-1],
              seed_context,
              mse_test=compact_seed_mse_test,
              problem_name=problem_name,
              method_name=compact_seed_method_name or optimizer_name,
              plot_on_seed=1,
              output_dir=solution_output_dir,
          )
      solution_output_paths = save_seed_solution_output(
          solution_output_dir,
          problem_name=problem_name,
          optimizer_name=optimizer_name,
          seed=seed,
          solution=solution,
          obj_sur=obj,
          obj_real=f_real,
          method_name=solution_output_method_name,
          seed_record=seed_record,
      )
      run_details[-1]["solution_output_paths"] = solution_output_paths

      del res, algorithm, benchmark_problem_GPR, optimization_problem
      gc.collect()



    return {
        "igd_list": igd_list,
        "hv_surrogate_list": hv_surrogate_list,
        "hv_clipped_list": hv_clipped_list,
        "hv_calibration_list": hv_calibration_list,
        "hv_calibration_mahalanobis_list": hv_calibration_mahalanobis_list,
        "hv_majority_shift_x_list": hv_majority_shift_x_list,
        "hv_majority_shift_x_mahalanobis_list": hv_majority_shift_x_mahalanobis_list,
        "hv_surrogate_count_list": hv_surrogate_count_list,
        "hv_clipped_count_list": hv_clipped_count_list,
        "hv_real_count_list": hv_real_count_list,
        "hv_calibration_count_list": hv_calibration_count_list,
        "hv_calibration_mahalanobis_count_list": hv_calibration_mahalanobis_count_list,
        "hv_majority_shift_x_count_list": hv_majority_shift_x_count_list,
        "hv_majority_shift_x_mahalanobis_count_list": hv_majority_shift_x_mahalanobis_count_list,
        "hv_real_list": hv_real_list,
        "hv_sur_gap_list": hv_sur_gap_list,
        "hv_clipped_gap_list": hv_clipped_gap_list,
        "hv_clipped_gap_reduction_pct_list": hv_clipped_gap_reduction_pct_list,
        "hv_calibration_gap_list": hv_calibration_gap_list,
        "hv_gap_reduction_pct_list": hv_gap_reduction_pct_list,
        "hv_calibration_mahalanobis_gap_list": hv_calibration_mahalanobis_gap_list,
        "hv_gap_reduction_mahalanobis_pct_list": hv_gap_reduction_mahalanobis_pct_list,
        "hv_majority_shift_x_gap_list": hv_majority_shift_x_gap_list,
        "hv_majority_shift_x_gap_reduction_pct_list": hv_majority_shift_x_gap_reduction_pct_list,
        "hv_majority_shift_x_mahalanobis_gap_list": hv_majority_shift_x_mahalanobis_gap_list,
        "hv_majority_shift_x_mahalanobis_gap_reduction_pct_list": hv_majority_shift_x_mahalanobis_gap_reduction_pct_list,
        "majority_shift_x_result_list": majority_shift_x_result_list,
        "majority_shift_x_by_k_list": majority_shift_x_by_k_list,
        "majority_shift_x_mahalanobis_result_list": majority_shift_x_mahalanobis_result_list,
        "candidate_set_diagnostics_list": candidate_set_diagnostics_list,
        "nearest_quadrant_summary_list": nearest_quadrant_summary_list,
        "normalization_info": normalization_info,
        "initialization_info": initialization_info,
        "run_details": run_details}
