import numpy as np
from sklearn.covariance import LedoitWolf


def validate_nearest_offline_arrays(x, f_sur, f_real, x_train, y_train, f_train_pred):
    x = np.asarray(x, dtype=float)
    f_sur = np.asarray(f_sur, dtype=float)
    f_real = np.asarray(f_real, dtype=float)
    x_train = np.asarray(x_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    f_train_pred = np.asarray(f_train_pred, dtype=float)

    if x.ndim != 2 or x_train.ndim != 2:
        raise ValueError("x and x_train must be 2D arrays.")
    if x.shape[1] != x_train.shape[1]:
        raise ValueError("x and x_train must have the same number of columns.")

    objective_arrays = {
        "f_sur": f_sur,
        "f_real": f_real,
        "y_train": y_train,
        "f_train_pred": f_train_pred,
    }
    for name, values in objective_arrays.items():
        if values.ndim != 2 or values.shape[1] != 2:
            raise ValueError(f"{name} must have shape (n, 2).")
    if f_sur.shape[0] != x.shape[0] or f_real.shape[0] != x.shape[0]:
        raise ValueError("f_sur and f_real must have one row per x.")
    if y_train.shape[0] != x_train.shape[0] or f_train_pred.shape[0] != x_train.shape[0]:
        raise ValueError("y_train and f_train_pred must have one row per x_train.")

    return x, f_sur, f_real, x_train, y_train, f_train_pred


def nearest_offline_query_train(x, f_sur, x_train, f_train_pred, distance_space):
    distance_space = distance_space.lower()
    if distance_space in ("x", "decision", "decision_x"):
        return x, x_train, "x"
    if distance_space in (
        "y",
        "y_sur",
        "pred",
        "sur",
        "surrogate",
        "objective",
        "objective_sur",
    ):
        return f_sur, f_train_pred, "y_pred"
    raise ValueError("distance_space must be 'x' or 'y'.")


def estimate_shrinkage_mahalanobis_metric(train_points, covariance_ridge=1e-8):
    train_points = np.asarray(train_points, dtype=float)
    if train_points.ndim != 2:
        raise ValueError("train_points must be a 2D array.")
    if train_points.shape[0] < 2:
        raise ValueError("Need at least two training samples to estimate covariance.")

    n_dim = train_points.shape[1]
    covariance_estimator = LedoitWolf().fit(train_points)
    covariance = np.asarray(covariance_estimator.covariance_, dtype=float)
    covariance = 0.5 * (covariance + covariance.T)
    scale = float(np.trace(covariance) / max(n_dim, 1))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0

    covariance_regularized = (
        covariance
        + float(covariance_ridge) * scale * np.eye(n_dim)
    )
    covariance_inv = np.linalg.pinv(covariance_regularized)
    return {
        "covariance": covariance,
        "covariance_regularized": covariance_regularized,
        "covariance_inv": covariance_inv,
        "covariance_ridge": float(covariance_ridge),
        "covariance_estimator": "ledoit_wolf",
        "covariance_shrinkage": float(covariance_estimator.shrinkage_),
    }


def pairwise_nearest_offline_distance(
    query_points,
    train_points,
    k=1,
    distance_metric="normalized_euclidean",
    covariance_ridge=1e-8,
    normalization_points=None,
):
    query_points = np.asarray(query_points, dtype=float)
    train_points = np.asarray(train_points, dtype=float)
    if normalization_points is not None:
        normalization_points = np.asarray(normalization_points, dtype=float)
    distance_metric = distance_metric.lower()

    if query_points.ndim != 2 or train_points.ndim != 2:
        raise ValueError("query_points and train_points must be 2D arrays.")
    if query_points.shape[1] != train_points.shape[1]:
        raise ValueError("query_points and train_points must have the same number of columns.")
    if normalization_points is not None:
        if normalization_points.ndim != 2:
            raise ValueError("normalization_points must be a 2D array.")
        if normalization_points.shape[1] != train_points.shape[1]:
            raise ValueError("normalization_points and train_points must have the same number of columns.")
    k_eff = min(max(int(k), 1), train_points.shape[0])

    if distance_metric in ("normalized_euclidean", "normalized", "norm_euclidean"):
        normalization_ref = train_points if normalization_points is None else normalization_points
        normalization_min = np.min(normalization_ref, axis=0)
        normalization_max = np.max(normalization_ref, axis=0)
        scale = np.where(
            normalization_max == normalization_min,
            1.0,
            normalization_max - normalization_min,
        )
        query_scaled = (query_points - normalization_min) / scale
        train_scaled = (train_points - normalization_min) / scale
        distances = np.linalg.norm(
            query_scaled[:, None, :] - train_scaled[None, :, :],
            axis=2,
        )
        distance_metric_label = "normalized_euclidean"
    elif distance_metric in ("euclidean", "l2"):
        distances = np.linalg.norm(
            query_points[:, None, :] - train_points[None, :, :],
            axis=2,
        )
        distance_metric_label = "euclidean"
    elif distance_metric in ("mahalanobis", "maha"):
        metric_info = estimate_shrinkage_mahalanobis_metric(
            train_points,
            covariance_ridge=covariance_ridge,
        )
        covariance_inv = metric_info["covariance_inv"]
        diff = query_points[:, None, :] - train_points[None, :, :]
        dist_sq = np.einsum("...i,ij,...j->...", diff, covariance_inv, diff)
        distances = np.sqrt(np.maximum(dist_sq, 0.0))
        distance_metric_label = "mahalanobis"
    else:
        raise ValueError(
            "distance_metric must be 'normalized_euclidean', 'euclidean', or 'mahalanobis'."
        )

    nearest_idx = np.argsort(distances, axis=1)[:, :k_eff].astype(int)
    nearest_distance = np.take_along_axis(distances, nearest_idx, axis=1)
    if k_eff == 1:
        nearest_idx = nearest_idx[:, 0]
        nearest_distance = nearest_distance[:, 0]
    return nearest_idx, nearest_distance, distance_metric_label


def real_sur_quadrants(real, sur):
    real = np.asarray(real, dtype=float)
    sur = np.asarray(sur, dtype=float)
    if real.shape != sur.shape:
        raise ValueError("real and sur must have the same shape.")
    if real.ndim != 2 or real.shape[1] != 2:
        raise ValueError("real and sur must have shape (n, 2).")

    delta = real - sur

    right = delta[:, 0] >= 0
    upper = delta[:, 1] >= 0
    quadrant = np.empty(delta.shape[0], dtype=object)
    quadrant[right & upper] = "right_upper"
    quadrant[right & ~upper] = "right_lower"
    quadrant[~right & upper] = "left_upper"
    quadrant[~right & ~upper] = "left_lower"
    return quadrant, delta


def offline_real_sur_quadrants(y_train, f_train_pred, nearest_idx):
    nearest_idx = np.asarray(nearest_idx, dtype=int)
    nearest_idx_flat = nearest_idx.reshape(-1)
    nearest_sur = np.asarray(f_train_pred, dtype=float)[nearest_idx_flat]
    nearest_real = np.asarray(y_train, dtype=float)[nearest_idx_flat]
    quadrant, delta = real_sur_quadrants(nearest_real, nearest_sur)
    return quadrant, delta, nearest_sur, nearest_real


def quadrant_counts(quadrant):
    quadrant = np.asarray(quadrant, dtype=object)
    return {
        "right_upper": int(np.sum(quadrant == "right_upper")),
        "right_lower": int(np.sum(quadrant == "right_lower")),
        "left_upper": int(np.sum(quadrant == "left_upper")),
        "left_lower": int(np.sum(quadrant == "left_lower")),
    }


def residual_direction_counts(delta, threshold=0.0, include_zero_positive=True):
    delta = np.asarray(delta, dtype=float)
    if delta.ndim != 2 or delta.shape[1] != 2:
        raise ValueError("delta must have shape (n, 2).")

    threshold = float(threshold)
    if threshold < 0:
        raise ValueError("threshold must be non-negative.")

    f1_delta = delta[:, 0]
    f2_delta = delta[:, 1]
    if include_zero_positive and threshold == 0.0:
        right = f1_delta >= 0
        left = f1_delta < 0
        upper = f2_delta >= 0
        lower = f2_delta < 0
    else:
        right = f1_delta > threshold
        left = f1_delta < -threshold
        upper = f2_delta > threshold
        lower = f2_delta < -threshold

    return {
        "right": int(np.sum(right)),
        "left": int(np.sum(left)),
        "upper": int(np.sum(upper)),
        "lower": int(np.sum(lower)),
    }


def residual_direction_count_line(counts):
    return (
        f"right={counts['right']}, "
        f"left={counts['left']}, "
        f"upper={counts['upper']}, "
        f"lower={counts['lower']}"
    )


def nearest_train_residual_direction_counts_from_summaries(
    summaries,
    threshold=0.0,
    include_zero_positive=True,
):
    deltas = []
    for summary in summaries:
        if "nearest_train_real_minus_nearest_train_sur" not in summary:
            continue
        delta = np.asarray(
            summary["nearest_train_real_minus_nearest_train_sur"],
            dtype=float,
        ).reshape(-1, 2)
        deltas.append(delta)
    if deltas:
        delta = np.vstack(deltas)
    else:
        delta = np.empty((0, 2), dtype=float)
    return residual_direction_counts(
        delta,
        threshold=threshold,
        include_zero_positive=include_zero_positive,
    )


def print_nearest_train_residual_direction_counts(
    summaries,
    threshold=1.0e-8,
    prefix="nearest train_real relative to train_sur",
):
    raw_counts = nearest_train_residual_direction_counts_from_summaries(
        summaries,
        threshold=0.0,
        include_zero_positive=True,
    )
    filtered_counts = nearest_train_residual_direction_counts_from_summaries(
        summaries,
        threshold=threshold,
        include_zero_positive=False,
    )
    print(
        f"{prefix} direction counts: "
        f"{residual_direction_count_line(raw_counts)}"
    )
    print(
        f"{prefix} direction counts abs residual > {float(threshold):.1e}: "
        f"{residual_direction_count_line(filtered_counts)}"
    )
    return {
        "raw": raw_counts,
        "filtered": filtered_counts,
    }


def summarize_nearest_offline_quadrants_2d(
    x,
    f_sur,
    f_real,
    x_train,
    y_train,
    f_train_pred,
    distance_space="x",
    distance_metric="normalized_euclidean",
    k=1,
    covariance_ridge=1e-8,
    print_summary=True,
):
    """Count candidate and nearest-offline real-vs-surrogate quadrants.

    The train counts are not computed row-by-row over the training set. For each
    optimized solution, this first finds the nearest offline training point in
    the requested distance space, then compares that nearest point's y_train
    against the paired f_train_pred.
    """
    x, f_sur, f_real, x_train, y_train, f_train_pred = validate_nearest_offline_arrays(
        x,
        f_sur,
        f_real,
        x_train,
        y_train,
        f_train_pred,
    )
    query_points, train_points, distance_space_label = nearest_offline_query_train(
        x,
        f_sur,
        x_train,
        f_train_pred,
        distance_space,
    )
    normalization_points = x_train if distance_space_label == "x" else y_train
    nearest_idx, nearest_distance, distance_metric_label = pairwise_nearest_offline_distance(
        query_points,
        train_points,
        k=k,
        distance_metric=distance_metric,
        covariance_ridge=covariance_ridge,
        normalization_points=normalization_points,
    )
    train_quadrant, train_delta, nearest_sur, nearest_real = offline_real_sur_quadrants(
        y_train,
        f_train_pred,
        nearest_idx,
    )
    candidate_quadrant, candidate_delta = real_sur_quadrants(f_real, f_sur)
    candidate_counts = quadrant_counts(candidate_quadrant)
    train_counts = quadrant_counts(train_quadrant)
    result = {
        "distance_space": distance_space_label,
        "distance_metric": distance_metric_label,
        "total": int(candidate_quadrant.shape[0]),
        "nearest_total": int(train_quadrant.shape[0]),
        "k": min(max(int(k), 1), train_points.shape[0]),
        "counts": train_counts,
        "f_real_vs_f_sur_counts": candidate_counts,
        "train_real_vs_train_sur_counts": train_counts,
        "nearest_train_real_vs_nearest_train_sur_counts": train_counts,
        "nearest_idx": nearest_idx,
        "nearest_distance": nearest_distance,
        "quadrant": train_quadrant,
        "delta_real_minus_sur": train_delta,
        "f_real_vs_f_sur_quadrant": candidate_quadrant,
        "f_real_minus_f_sur": candidate_delta,
        "train_real_vs_train_sur_quadrant": train_quadrant,
        "train_real_minus_train_sur": train_delta,
        "nearest_train_real_vs_nearest_train_sur_quadrant": train_quadrant,
        "nearest_train_real_minus_nearest_train_sur": train_delta,
        "nearest_sur": nearest_sur,
        "nearest_real": nearest_real,
        "normalization_min": np.min(normalization_points, axis=0),
        "normalization_max": np.max(normalization_points, axis=0),
    }

    if print_summary:
        print(
            "Quadrant counts "
            f"({distance_space_label}, {distance_metric_label}, k={result['k']})"
        )
        print(f"total: {result['total']}")
        print("optimized f_real relative to optimized f_sur:")
        print(f"right_upper: {candidate_counts['right_upper']}")
        print(f"right_lower: {candidate_counts['right_lower']}")
        print(f"left_upper: {candidate_counts['left_upper']}")
        print(f"left_lower: {candidate_counts['left_lower']}")
        print("nearest train_real relative to nearest train_sur:")
        print(f"nearest total: {result['nearest_total']}")
        print(f"right_upper: {train_counts['right_upper']}")
        print(f"right_lower: {train_counts['right_lower']}")
        print(f"left_upper: {train_counts['left_upper']}")
        print(f"left_lower: {train_counts['left_lower']}")

    return result


def compute_majority_shift_calibration_2d(
    x,
    f_sur,
    f_real,
    x_train,
    y_train,
    f_train_pred,
    distance_space="x",
    distance_metric="normalized_euclidean",
    k=1,
    covariance_ridge=1e-8,
    mean_abs_delta_source="nearest",
):
    """Shift every f_sur point by the majority train_real - train_sur direction.

    The nearest offline indices are selected in x space or y-pred space. The
    offset magnitude is either the nearest-paired residual mean or the global
    offline residual mean, depending on ``mean_abs_delta_source``.
    The offset sign is the majority quadrant direction of those paired residuals.
    """
    y_train = np.asarray(y_train, dtype=float)
    f_train_pred = np.asarray(f_train_pred, dtype=float)
    summary = summarize_nearest_offline_quadrants_2d(
        x=x,
        f_sur=f_sur,
        f_real=f_real,
        x_train=x_train,
        y_train=y_train,
        f_train_pred=f_train_pred,
        distance_space=distance_space,
        distance_metric=distance_metric,
        k=k,
        covariance_ridge=covariance_ridge,
        print_summary=False,
    )
    nearest_delta = np.asarray(summary["nearest_train_real_minus_nearest_train_sur"], dtype=float)
    offline_delta = y_train - f_train_pred
    counts = summary["nearest_train_real_vs_nearest_train_sur_counts"]
    dominant_quadrant = max(
        ("right_upper", "right_lower", "left_upper", "left_lower"),
        key=lambda key: counts[key],
    )
    dominant_direction = np.array(
        [
            1.0 if dominant_quadrant.startswith("right") else -1.0,
            1.0 if dominant_quadrant.endswith("upper") else -1.0,
        ],
        dtype=float,
    )
    mean_abs_delta_source = mean_abs_delta_source.lower()
    if mean_abs_delta_source in ("nearest", "nearest_offline"):
        mean_abs_delta = np.mean(np.abs(nearest_delta), axis=0)
        mean_abs_delta_source_label = "nearest_offline_abs_y_train_minus_f_train_pred"
    elif mean_abs_delta_source in ("all", "offline", "offline_all", "global", "global_mean"):
        mean_abs_delta = np.mean(np.abs(offline_delta), axis=0)
        mean_abs_delta_source_label = "all_offline_abs_y_train_minus_f_train_pred"
    else:
        raise ValueError(
            "mean_abs_delta_source must be 'nearest' or 'offline_all'."
        )
    y_min = np.min(y_train, axis=0)
    y_max = np.max(y_train, axis=0)
    y_scale = np.where(y_max == y_min, 1.0, y_max - y_min)
    normalized_mean_abs_delta = mean_abs_delta / y_scale
    normalized_mean_abs_delta_norm = float(np.linalg.norm(normalized_mean_abs_delta))
    signed_shift = dominant_direction * mean_abs_delta
    f_sur_shifted = np.asarray(f_sur, dtype=float) + signed_shift

    return {
        "f_sur_shifted": f_sur_shifted,
        "signed_shift": signed_shift,
        "mean_abs_delta": mean_abs_delta,
        "mean_abs_delta_source": mean_abs_delta_source_label,
        "nearest_mean_abs_delta": np.mean(np.abs(nearest_delta), axis=0),
        "offline_mean_abs_delta": np.mean(np.abs(offline_delta), axis=0),
        "normalized_mean_abs_delta": normalized_mean_abs_delta,
        "normalized_mean_abs_delta_norm": normalized_mean_abs_delta_norm,
        "calibration_applied": True,
        "dominant_direction": dominant_direction,
        "dominant_quadrant": dominant_quadrant,
        "counts": counts,
        "nearest_idx": summary["nearest_idx"],
        "nearest_distance": summary["nearest_distance"],
        "nearest_offline_summary": summary,
        "distance_space": summary["distance_space"],
        "distance_metric": summary["distance_metric"],
        "k": summary["k"],
    }


def summarize_nearest_offline_quadrants_x_y_2d(
    x,
    f_sur,
    f_real,
    x_train,
    y_train,
    f_train_pred,
    distance_metric="normalized_euclidean",
    k=1,
    covariance_ridge=1e-8,
    print_summary=True,
):
    """Run quadrant counts once with x distance and once with y distance."""
    x_summary = summarize_nearest_offline_quadrants_2d(
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
        print_summary=print_summary,
    )
    y_summary = summarize_nearest_offline_quadrants_2d(
        x=x,
        f_sur=f_sur,
        f_real=f_real,
        x_train=x_train,
        y_train=y_train,
        f_train_pred=f_train_pred,
        distance_space="y",
        distance_metric=distance_metric,
        k=k,
        covariance_ridge=covariance_ridge,
        print_summary=print_summary,
    )
    return {
        "x": x_summary,
        "y": y_summary,
    }
