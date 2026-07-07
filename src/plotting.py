import numpy as np
import os
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from plotly.subplots import make_subplots
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from src.nearest_offline import (
    nearest_offline_query_train,
    offline_real_sur_quadrants,
    pairwise_nearest_offline_distance,
    real_sur_quadrants,
    validate_nearest_offline_arrays,
)


RESULT_SUMMARY_XLSX_FILENAME = "result.xlsx"
RESULT_TEMP_DIRNAME = "temp"
R_ZERO_DENOMINATOR_SUFFIX = "_zero_denominator"
RESULT_METRIC_KEYS = (
    "MSE_test",
    "MSE_sur",
    "MSE_clip",
    "MSE_xy_shift",
    "MSE_x_shift",
    "MSE_y_shift",
    "MSE_no_clip",
    "IGD+_real",
    "IGD+_sur",
    "IGD+_clip",
    "IGD+_xy_shift",
    "IGD+_x_shift",
    "IGD+_y_shift",
    "IGD+_no_clip",
    "HV_sur",
    "HV_real",
    "HV_clip",
    "HV_xy_shift",
    "HV_x_shift",
    "HV_y_shift",
    "HV_no_clip",
)
RESULT_TABLE_COLUMNS = (
    "timestamp",
    "method",
    "Problem",
    "MSE_test",
    "MSE_sur",
    "MSE_clip",
    "MSE_clip (improvement)",
    "MSE_xy_shift",
    "MSE_xy_shift (improvement)",
    "MSE_x_shift",
    "MSE_x_shift (improvement)",
    "MSE_y_shift",
    "MSE_y_shift (improvement)",
    "MSE_no_clip",
    "MSE_no_clip (improvement)",
    "IGD+_real",
    "IGD+_sur",
    "IGD+_clip",
    "IGD+_clip (improvement)",
    "IGD+_xy_shift",
    "IGD+_xy_shift (improvement)",
    "IGD+_x_shift",
    "IGD+_x_shift (improvement)",
    "IGD+_y_shift",
    "IGD+_y_shift (improvement)",
    "IGD+_no_clip",
    "IGD+_no_clip (improvement)",
    "HV_sur",
    "HV_real",
    "HV_clip",
    "HV_clip (improvement)",
    "HV_xy_shift",
    "HV_xy_shift (improvement)",
    "HV_x_shift",
    "HV_x_shift (improvement)",
    "HV_y_shift",
    "HV_y_shift (improvement)",
    "HV_no_clip",
    "HV_no_clip (improvement)",
    "pvalue_HV_xy_shift_vs_HV_sur",
)
RESULT_VARIANT_LABELS = {
    "HV_clip": "HV_clip",
    "HV_xy_shift": "HV_xy_shift",
    "HV_x_shift": "HV_x_shift",
    "HV_y_shift": "HV_y_shift",
    "HV_no_clip": "HV_no_clip",
}
RESULT_ID_COLUMNS = ("timestamp", "method", "Problem")
MAIN_RESULT_COLUMNS = (
    *RESULT_ID_COLUMNS,
    "MSE_test",
    "MSE_sur",
    "MSE_xy_shift",
    "MSE_xy_shift (improvement)",
    "IGD+_real",
    "IGD+_sur",
    "IGD+_xy_shift",
    "IGD+_xy_shift (improvement)",
    "HV_sur",
    "HV_real",
    "HV_xy_shift",
    "HV_xy_shift (improvement)",
    "pvalue_HV_xy_shift_vs_HV_sur",
)
ABLATION_RESULT_COLUMNS = (
    *RESULT_ID_COLUMNS,
    "MSE_sur",
    "MSE_clip",
    "MSE_clip (improvement)",
    "MSE_x_shift",
    "MSE_x_shift (improvement)",
    "MSE_y_shift",
    "MSE_y_shift (improvement)",
    "MSE_no_clip",
    "MSE_no_clip (improvement)",
    "IGD+_sur",
    "IGD+_clip",
    "IGD+_clip (improvement)",
    "IGD+_x_shift",
    "IGD+_x_shift (improvement)",
    "IGD+_y_shift",
    "IGD+_y_shift (improvement)",
    "IGD+_no_clip",
    "IGD+_no_clip (improvement)",
    "HV_sur",
    "HV_real",
    "HV_clip",
    "HV_clip (improvement)",
    "HV_x_shift",
    "HV_x_shift (improvement)",
    "HV_y_shift",
    "HV_y_shift (improvement)",
    "HV_no_clip",
    "HV_no_clip (improvement)",
)
MAJORITY_SHIFT_K_VALUES = (1, 2)
POINTWISE_EUCLIDEAN_K_VALUES = (1, 2)
HV_SHIFT_SUR_COLUMN = "HV_shift_sur"
HV_SHIFT_SUR_IMPROVEMENT_COLUMN = "HV_shift_sur（improvement）"
HV_SHIFT_SUR_VS_HV_SUR_DETAIL_LABEL = "HV_shift_sur_vs_HV_sur（improvement）"
HV_SHIFT_SUR_VS_CLIP_COLUMN = "HV_shift_sur_vs_HV_clip（improvement）"
HV_XY_DIRECTION_COUNTS_IMPROVEMENT_LABEL = (
    "HV_xy_direction_counts_vs_HV_sur (improvement)"
)

# Main plot entry points used by experiments/notebooks:
# - plot_hv_majority_shift_x_latest_run: Exp1-8 seed=1 majority-shift figure.
# - plot_nearest_offline_diagnostics: selected-problem x/y nearest diagnostics.
# - plot_hv_xy_shift_seed_2d: compact f_sur/f_real/HV_xy_shift figure.


def _format_summary_float(value, digits=3):
    if value is None or not np.isfinite(value):
        return "nan"
    return f"{float(value):.{int(digits)}f}"


def _format_summary_percent(value):
    if value is None or not np.isfinite(value):
        return "NA%"
    return f"{float(value):.1f}%"


def _mean_gap_improvement(results, hv_values):
    from src.experiment import compute_R_indicator

    hv_values = np.asarray(hv_values, dtype=float)
    hv_real = np.asarray(results["hv_real_list"], dtype=float)
    hv_sur_gap = np.asarray(results["hv_sur_gap_list"], dtype=float)
    valid = np.isfinite(hv_values) & np.isfinite(hv_real) & np.isfinite(hv_sur_gap)
    if not np.any(valid):
        return "NA%"
    variant_gap = np.abs(hv_real[valid] - hv_values[valid])
    improvements = np.asarray(
        [
            compute_R_indicator(sur_gap, var_gap)
            for sur_gap, var_gap in zip(hv_sur_gap[valid], variant_gap)
        ],
        dtype=float,
    )
    improvements = improvements[np.isfinite(improvements)]
    if improvements.size == 0:
        return "NA%"
    return _format_summary_percent(float(np.nanmean(improvements)))


def _mean_gap_improvement_against_values(results, hv_values, reference_values):
    from src.experiment import compute_R_indicator

    hv_values = np.asarray(hv_values, dtype=float)
    reference_values = np.asarray(reference_values, dtype=float)
    hv_real = np.asarray(results["hv_real_list"], dtype=float)
    valid = (
        np.isfinite(hv_values)
        & np.isfinite(reference_values)
        & np.isfinite(hv_real)
    )
    if not np.any(valid):
        return "NA%"
    reference_gap = np.abs(hv_real[valid] - reference_values[valid])
    variant_gap = np.abs(hv_real[valid] - hv_values[valid])
    improvements = np.asarray(
        [
            compute_R_indicator(ref_gap, var_gap)
            for ref_gap, var_gap in zip(reference_gap, variant_gap)
        ],
        dtype=float,
    )
    improvements = improvements[np.isfinite(improvements)]
    if improvements.size == 0:
        return "NA%"
    return _format_summary_percent(float(np.nanmean(improvements)))


def _xy_direction_counts_hv_values_for_optimizer(
    results,
    context,
    optimizer_name=None,
):
    diagnostics_by_optimizer = context.get("majority_shift_y_k1_diagnostics", {})
    y_space_items = None
    if isinstance(diagnostics_by_optimizer, dict) and optimizer_name is not None:
        y_space_items = diagnostics_by_optimizer.get(optimizer_name)

    if y_space_items is None:
        y_space_items = evaluate_majority_shift_y_k1_diagnostic_items(
            results,
            context,
        )
    if not y_space_items:
        return np.asarray([], dtype=float)
    return evaluate_xy_direction_counts_hv_values(results, y_space_items, context)


def _summary_values(values):
    values = np.asarray(values, dtype=float)
    return values.reshape(-1)


def _summary_mean_std_text(values, digits=3):
    values = _summary_values(values)
    if values.size == 0:
        return f"Mean = {float('nan'):.{digits}f}, Std = {float('nan'):.{digits}f}"
    return (
        f"Mean = {float(np.nanmean(values)):.{digits}f}, "
        f"Std = {float(np.nanstd(values)):.{digits}f}"
    )


def _summary_count_line(counts):
    return (
        f"right_upper={counts['right_upper']}, "
        f"right_lower={counts['right_lower']}, "
        f"left_upper={counts['left_upper']}, "
        f"left_lower={counts['left_lower']}"
    )


def _summary_sum_quadrant_counts(summaries, key):
    total = {
        "right_upper": 0,
        "right_lower": 0,
        "left_upper": 0,
        "left_lower": 0,
    }
    for summary in summaries:
        counts = summary[key]
        for count_key in total:
            total[count_key] += int(counts[count_key])
    return total


def _direction_counts_from_deltas(delta):
    delta = np.asarray(delta, dtype=float)
    if delta.size == 0:
        return {"right": 0, "left": 0, "upper": 0, "lower": 0}
    delta = delta.reshape(-1, delta.shape[-1])
    return {
        "right": int(np.sum(delta[:, 0] >= 0.0)),
        "left": int(np.sum(delta[:, 0] < 0.0)),
        "upper": int(np.sum(delta[:, 1] >= 0.0)),
        "lower": int(np.sum(delta[:, 1] < 0.0)),
    }


def _direction_counts_from_summaries(summaries):
    if not summaries:
        return {"right": 0, "left": 0, "upper": 0, "lower": 0}
    nearest_deltas = np.vstack([
        np.asarray(
            summary["nearest_train_real_minus_nearest_train_sur"],
            dtype=float,
        )
        for summary in summaries
    ])
    return _direction_counts_from_deltas(nearest_deltas)


def _direction_count_line(counts):
    return (
        f"right={int(counts['right'])}, "
        f"left={int(counts['left'])}, "
        f"upper={int(counts['upper'])}, "
        f"lower={int(counts['lower'])}"
    )


def _majority_direction_from_counts(counts):
    directions = []
    right = int(counts["right"])
    left = int(counts["left"])
    upper = int(counts["upper"])
    lower = int(counts["lower"])
    if right > left:
        directions.append("right")
    elif left > right:
        directions.append("left")
    if upper > lower:
        directions.append("upper")
    elif lower > upper:
        directions.append("lower")
    return "_".join(directions) if directions else "none"


def _signed_direction_vector_from_counts(counts):
    return np.asarray(
        [
            np.sign(int(counts["right"]) - int(counts["left"])),
            np.sign(int(counts["upper"]) - int(counts["lower"])),
        ],
        dtype=float,
    )


def _summary_mean_vector_text(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return "[nan, nan]"
    mean_values = np.nanmean(values, axis=0)
    return f"[{mean_values[0]:.3e}, {mean_values[1]:.3e}]"


def _summary_series_from_results(results, exp1_key, exp5_key):
    if isinstance(results, dict):
        return _summary_values(results.get(exp1_key, []))
    return _summary_values([
        item[exp5_key]
        for item in results
        if exp5_key in item
    ])


def _summary_shift_results(results):
    if isinstance(results, dict):
        return [
            item
            for item in results.get("majority_shift_x_result_list", [])
            if item is not None
        ]
    return [
        item.get("majority_shift_x_result")
        for item in results
        if item.get("majority_shift_x_result") is not None
    ]


def _display_gap_improvement(label, improvement_pct):
    from src.experiment import format_percent

    percent_text = format_percent(improvement_pct)
    try:
        is_negative = np.isfinite(float(improvement_pct)) and float(improvement_pct) < 0
    except (TypeError, ValueError):
        is_negative = False

    if is_negative:
        html_value = (
            f"<span style='color:#d00000;font-weight:700'>{percent_text}</span>"
        )
        ansi_value = f"\033[1;31m{percent_text}\033[0m"
    else:
        html_value = f"<span style='font-weight:700'>{percent_text}</span>"
        ansi_value = f"\033[1m{percent_text}\033[0m"

    try:
        from html import escape
        from IPython.display import HTML, display

        display(HTML(f"{escape(label)}: {html_value}"))
    except Exception:
        print(f"{label}: {ansi_value}")


def _styled_percent_text(value):
    from src.experiment import format_percent

    percent_text = format_percent(value)
    try:
        is_negative = np.isfinite(float(value)) and float(value) < 0
    except (TypeError, ValueError):
        is_negative = False

    if is_negative:
        return (
            f"<span style='color:#d00000;font-weight:700'>{percent_text}</span>",
            f"\033[1;31m{percent_text}\033[0m",
        )
    return (
        f"<span style='font-weight:700'>{percent_text}</span>",
        f"\033[1m{percent_text}\033[0m",
    )


def _print_hv_value_with_styled_improvement(label, hv_value, improvement_pct):
    html_percent, ansi_percent = _styled_percent_text(improvement_pct)
    try:
        hv_value_float = float(hv_value)
    except (TypeError, ValueError):
        hv_value_float = float("nan")
    value_text = f"{hv_value_float:.3f}" if np.isfinite(hv_value_float) else "nan"
    try:
        from html import escape
        from IPython.display import HTML, display

        display(
            HTML(
                f"{escape(label)}: {escape(value_text)} "
                f"({html_percent})"
            )
        )
    except Exception:
        print(f"{label}: {value_text} ({ansi_percent})")


def _print_hv_variant_vs_hv_sur_improvement(
    label,
    hv_values,
    hv_sur_values,
    hv_real_values,
):
    from src.experiment import compute_R_indicator

    hv_values = _summary_values(hv_values)
    if hv_values.size == 0:
        print(f"{label}: no results available.")
        return

    n = min(
        hv_values.size,
        hv_sur_values.size,
        hv_real_values.size,
    )
    hv_values = hv_values[:n]
    hv_sur_values = hv_sur_values[:n]
    hv_real_values = hv_real_values[:n]
    variant_real_gap_values = np.abs(hv_values - hv_real_values)
    hv_sur_real_gap_values = np.abs(hv_sur_values - hv_real_values)

    _display_gap_improvement(
        f"{label}_vs_HV_sur (improvement)",
        compute_R_indicator(
            float(np.nanmean(hv_sur_real_gap_values)),
            float(np.nanmean(variant_real_gap_values)),
        ),
    )


def print_hv_majority_shift_x_k1_summary(
    problem_name,
    method_name,
    results,
    y_space_direction_counts=None,
    xy_direction_counts_hv_values=None,
):
    from src.experiment import compute_R_indicator, wilcoxon_gap_test

    hv_n_values = _summary_series_from_results(
        results,
        "hv_surrogate_count_list",
        "hv_surrogate_count",
    )
    hv_sur_values = _summary_series_from_results(
        results,
        "hv_surrogate_list",
        "hv_surrogate",
    )
    hv_real_values = _summary_series_from_results(
        results,
        "hv_real_list",
        "hv_real",
    )
    hv_clip_values = _summary_series_from_results(
        results,
        "hv_clipped_list",
        "hv_clip",
    )
    hv_shift_values = _summary_series_from_results(
        results,
        "hv_majority_shift_x_list",
        "hv_majority_shift_x",
    )
    shift_results = _summary_shift_results(results)

    if method_name is not None:
        print(f"\n=== {problem_name} | {method_name} ===")

    x_space_direction_counts = None
    if shift_results:
        summaries = [item["nearest_offline_summary"] for item in shift_results]
        first = shift_results[0]
        candidate_counts = _summary_sum_quadrant_counts(
            summaries,
            "f_real_vs_f_sur_counts",
        )
        total = sum(int(summary["total"]) for summary in summaries)
        print("Counts:")
        print(
            f"{first['distance_space']} counts "
            f"({first['distance_metric']}, k={first['k']}):"
        )
        print(
            f"optimized f_real relative to f_sur total={total}: "
            f"{_summary_count_line(candidate_counts)}"
        )
        x_space_direction_counts = _direction_counts_from_summaries(summaries)
        print(
            "x-space nearest train_real relative to train_sur direction counts: "
            f"{_direction_count_line(x_space_direction_counts)}"
        )
        if y_space_direction_counts is not None:
            print(
                "y-space nearest train_real relative to train_sur direction counts: "
                f"{_direction_count_line(y_space_direction_counts)}"
            )

    hv_sur_real_gap_values = np.abs(hv_sur_values - hv_real_values)
    hv_clip_real_gap_values = np.abs(hv_clip_values - hv_real_values)
    hv_shift_real_gap_values = np.abs(hv_shift_values - hv_real_values)

    print(f"HV_n: {_summary_mean_std_text(hv_n_values, digits=1)}")
    print(f"HV_sur: {_summary_mean_std_text(hv_sur_values)}")
    print(f"HV_real: {_summary_mean_std_text(hv_real_values)}")
    print(f"HV_clip: {_summary_mean_std_text(hv_clip_values)}")

    if not shift_results:
        print("HV_majority_shift_x diagnostics: no results available.")
        return

    dominant_quadrants = ", ".join(
        dict.fromkeys(item["dominant_quadrant"] for item in shift_results)
    )
    mean_hv_sur_gap = float(np.nanmean(hv_sur_real_gap_values))
    mean_hv_clip_gap = float(np.nanmean(hv_clip_real_gap_values))
    mean_hv_shift_real_gap = float(np.nanmean(hv_shift_real_gap_values))
    shift_vs_sur_improvement_pct = compute_R_indicator(
        mean_hv_sur_gap,
        mean_hv_shift_real_gap,
    )
    shift_vs_clip_improvement_pct = compute_R_indicator(
        mean_hv_clip_gap,
        mean_hv_shift_real_gap,
    )

    print(
        "HV_majority_shift_x signed shift mean & dominant quadrant: "
        f"{_summary_mean_vector_text([item['signed_shift'] for item in shift_results])} "
        f"& {dominant_quadrants}"
    )
    print(f"HV_majority_shift_x: {_summary_mean_std_text(hv_shift_values)}")
    _display_gap_improvement(
        HV_SHIFT_SUR_VS_HV_SUR_DETAIL_LABEL,
        shift_vs_sur_improvement_pct,
    )
    _display_gap_improvement(
        HV_SHIFT_SUR_VS_CLIP_COLUMN,
        shift_vs_clip_improvement_pct,
    )
    if x_space_direction_counts is not None:
        print(
            "x-space nearest train_real relative to train_sur majority direction: "
            f"{_majority_direction_from_counts(x_space_direction_counts)}"
        )
    if y_space_direction_counts is not None:
        print(
            "y-space nearest train_real relative to train_sur majority direction: "
            f"{_majority_direction_from_counts(y_space_direction_counts)}"
        )
    if xy_direction_counts_hv_values is not None:
        _print_hv_variant_vs_hv_sur_improvement(
            "HV_xy_direction_counts",
            xy_direction_counts_hv_values,
            hv_sur_values,
            hv_real_values,
        )
    print()
    wilcoxon_p_value = wilcoxon_gap_test(
        hv_sur_real_gap_values,
        hv_shift_real_gap_values,
    )
    print(
        "Wilcoxon signed-rank "
        f"(HV_majority_shift_x, two-sided) p-value: {wilcoxon_p_value:.6g}"
    )


def _returned_majority_shift_hv_values(results, k):
    return [
        by_k[int(k)]["hv"]
        for by_k in results["majority_shift_x_by_k_list"]
    ]


def _baseline_mean_result_value(run_results, key):
    return float(np.nanmean([result[key] for result in run_results]))


def _baseline_mean_hv_variant_value(run_results, variant_label):
    return float(np.nanmean([
        result["hv_variant_values"][variant_label]
        for result in run_results
    ]))


def _baseline_mean_gap_improvement_for_variant(run_results, variant_label):
    from src.experiment import compute_R_indicator

    improvements = []
    for result in run_results:
        hv_sur_gap = result.get("hv_sur_gap", np.nan)
        variant_value = result.get("hv_variant_values", {}).get(variant_label, np.nan)
        if not (
            np.isfinite(hv_sur_gap)
            and np.isfinite(result.get("hv_real", np.nan))
            and np.isfinite(variant_value)
        ):
            continue
        variant_gap = abs(result["hv_real"] - variant_value)
        improvement = compute_R_indicator(hv_sur_gap, variant_gap)
        if np.isfinite(improvement):
            improvements.append(improvement)
    if not improvements:
        return "NA%"
    return _format_summary_percent(float(np.nanmean(improvements)))


def _baseline_mean_gap_improvement_against_key(run_results, variant_label, reference_key):
    from src.experiment import compute_R_indicator

    improvements = []
    for result in run_results:
        hv_real = result.get("hv_real", np.nan)
        reference_value = result.get(reference_key, np.nan)
        variant_value = result.get("hv_variant_values", {}).get(variant_label, np.nan)
        if not (
            np.isfinite(hv_real)
            and np.isfinite(reference_value)
            and np.isfinite(variant_value)
        ):
            continue
        reference_gap = abs(hv_real - reference_value)
        variant_gap = abs(hv_real - variant_value)
        improvement = compute_R_indicator(reference_gap, variant_gap)
        if np.isfinite(improvement):
            improvements.append(improvement)
    if not improvements:
        return "NA%"
    return _format_summary_percent(float(np.nanmean(improvements)))


def _evaluate_majority_shift_shared_variant(
    run_detail,
    context,
    k,
    distance_metric,
    use_clip=True,
):
    from src.experiment import evaluate_majority_shift_x_variant

    shift_eval = evaluate_majority_shift_x_variant(
        x=run_detail["solution"],
        f_sur=run_detail["obj"],
        f_real=run_detail["f_real"],
        x_train=context["X_train"],
        y_train=context["y_train"],
        f_train_pred=context["f_train_mean"],
        hv=context["hv"],
        hv_real=run_detail["hv_real"],
        hv_sur_gap=run_detail["hv_sur_gap"],
        obj_min=context["obj_min"],
        obj_max=context["obj_max"],
        clip_obj_min=context["problem_y_min"],
        k=k,
        distance_metric=distance_metric,
        covariance_ridge=float(context.get("covariance_ridge", 1.0e-8)),
        use_clip=use_clip,
        clip_source=context.get("clip_source", "problem_y_threshold"),
    )
    return shift_eval["hv"]


def _evaluate_pointwise_euclidean_variant(run_detail, context, k):
    from src.experiment import (
        _clip_calibration_result_to_problem_bounds,
        compute_local_residual_calibration,
        normalized_hv,
    )

    result = compute_local_residual_calibration(
        obj=run_detail["obj"],
        y_train=context["y_train"],
        f_train_mean=context["f_train_mean"],
        k=k,
        distance_train=context["X_train"],
        distance_query=run_detail["solution"],
    )
    _clip_calibration_result_to_problem_bounds(
        result,
        context["problem_y_min"],
        clip_source=context.get("clip_source", "problem_y_threshold"),
    )
    return normalized_hv(
        context["hv"],
        result["sur_calibration"],
        context["obj_min"],
        context["obj_max"],
    )


def gap_improvement_variant_rows(
    results,
    context,
    optimizer_name=None,
):
    rows = []

    for k in MAJORITY_SHIFT_K_VALUES:
        for distance_metric, label in (
            ("normalized_euclidean", "HV_majority_shift_x"),
            ("mahalanobis", "HV_majority_shift_x_mahalanobis"),
        ):
            hv_values = [
                _evaluate_majority_shift_shared_variant(
                    run_detail,
                    context,
                    k=k,
                    distance_metric=distance_metric,
                )
                for run_detail in results["run_details"]
            ]
            rows.append((f"{label} k={k}", _mean_gap_improvement(results, hv_values)))

    no_clip_hv_values = [
        _evaluate_majority_shift_shared_variant(
            run_detail,
            context,
            k=1,
            distance_metric="normalized_euclidean",
            use_clip=False,
        )
        for run_detail in results["run_details"]
    ]
    rows.append(("HV_no_clip (ablation)", _mean_gap_improvement(results, no_clip_hv_values)))

    xy_direction_counts_hv_values = (
        _xy_direction_counts_hv_values_for_optimizer(
            results,
            context,
            optimizer_name=optimizer_name,
        )
    )
    rows.append((
        HV_XY_DIRECTION_COUNTS_IMPROVEMENT_LABEL,
        _mean_gap_improvement(results, xy_direction_counts_hv_values),
    ))

    for k in POINTWISE_EUCLIDEAN_K_VALUES:
        hv_values = [
            _evaluate_pointwise_euclidean_variant(run_detail, context, k=k)
            for run_detail in results["run_details"]
        ]
        rows.append((
            f"HV_calibration_euclidean k={k} (pointwise)",
            _mean_gap_improvement(results, hv_values),
        ))

    return rows


def build_exp1_4_gap_improvement_tables(
    optimizer_names,
    problem_names,
    all_results,
    problem_contexts,
):
    import pandas as pd

    gap_improvement_tables = {}
    for optimizer_name in optimizer_names:
        rows = {}
        for problem_name in problem_names:
            results = all_results.get(problem_name, {}).get(optimizer_name)
            context = problem_contexts.get(problem_name)
            if results is None or context is None:
                continue
            for variant_label, improvement in gap_improvement_variant_rows(
                results,
                context,
                optimizer_name=optimizer_name,
            ):
                rows.setdefault(variant_label, {})[problem_name] = improvement

        table = pd.DataFrame.from_dict(rows, orient="index")
        table.index.name = "hv_variant"
        gap_improvement_tables[optimizer_name] = table
    return gap_improvement_tables


def print_exp1_4_gap_improvement_tables(
    method_name,
    optimizer_names,
    problem_names,
    all_results,
    problem_contexts,
):
    tables = build_exp1_4_gap_improvement_tables(
        optimizer_names=optimizer_names,
        problem_names=problem_names,
        all_results=all_results,
        problem_contexts=problem_contexts,
    )
    try:
        from IPython.display import display
    except ImportError:
        display = None

    for optimizer_name, table in tables.items():
        print(f"Gap improvement | {method_name} | {optimizer_name}")
        if display is None:
            print(table.to_string())
        else:
            display(table)
    return tables


def build_exp5_8_gap_improvement_table(all_results):
    import pandas as pd

    rows = {}
    for problem_name, run_results in all_results.items():
        if not run_results:
            continue
        for variant_label in run_results[0]["hv_variant_values"]:
            rows.setdefault(variant_label, {})[problem_name] = (
                _baseline_mean_gap_improvement_for_variant(run_results, variant_label)
            )
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "hv_variant"
    return table


def print_exp5_8_gap_improvement_table(all_results, method_name):
    table = build_exp5_8_gap_improvement_table(all_results)
    print(f"Gap improvement | {method_name}")
    try:
        from IPython.display import display
    except ImportError:
        display = None
    if display is None:
        print(table.to_string())
    else:
        display(table)
    return table


def build_exp1_4_compact_summary_table(
    method_name,
    optimizer_names,
    problem_names,
    all_results,
    problem_contexts,
):
    import pandas as pd

    summary_metrics = (
        "MSE",
        "HV_sur",
        "HV_real",
        "HV_clip",
        HV_SHIFT_SUR_COLUMN,
        HV_SHIFT_SUR_IMPROVEMENT_COLUMN,
        HV_XY_DIRECTION_COUNTS_IMPROVEMENT_LABEL,
        HV_SHIFT_SUR_VS_CLIP_COLUMN,
    )
    summary_columns = [("Method", "")]
    for problem_name in problem_names:
        summary_columns.extend((problem_name, metric) for metric in summary_metrics)

    summary_rows = []
    for optimizer_name in optimizer_names:
        row = {("Method", ""): f"{method_name}+{optimizer_name}"}
        for problem_name in problem_names:
            results = all_results[problem_name][optimizer_name]
            context = problem_contexts[problem_name]
            k1_hv_values = _returned_majority_shift_hv_values(results, k=1)
            k1_gap_improvement = _mean_gap_improvement(results, k1_hv_values)
            xy_direction_counts_hv_values = (
                _xy_direction_counts_hv_values_for_optimizer(
                    results,
                    context,
                    optimizer_name=optimizer_name,
                )
            )
            row[(problem_name, "MSE")] = _format_summary_float(
                context["offline_test_mse"]
            )
            row[(problem_name, "HV_sur")] = _format_summary_float(
                np.nanmean(results["hv_surrogate_list"])
            )
            row[(problem_name, "HV_real")] = _format_summary_float(
                np.nanmean(results["hv_real_list"])
            )
            row[(problem_name, "HV_clip")] = _format_summary_float(
                np.nanmean(results["hv_clipped_list"])
            )
            row[(problem_name, HV_SHIFT_SUR_COLUMN)] = _format_summary_float(
                np.nanmean(k1_hv_values)
            )
            row[(problem_name, HV_SHIFT_SUR_IMPROVEMENT_COLUMN)] = k1_gap_improvement
            row[(problem_name, HV_XY_DIRECTION_COUNTS_IMPROVEMENT_LABEL)] = (
                _mean_gap_improvement(
                    results,
                    xy_direction_counts_hv_values,
                )
            )
            row[(problem_name, HV_SHIFT_SUR_VS_CLIP_COLUMN)] = (
                _mean_gap_improvement_against_values(
                    results,
                    k1_hv_values,
                    results["hv_clipped_list"],
                )
            )
        summary_rows.append(row)

    return pd.DataFrame(
        summary_rows,
        columns=pd.MultiIndex.from_tuples(summary_columns),
    )


def build_exp5_8_compact_summary_table(method_name, all_results, problem_names=None):
    import pandas as pd

    if problem_names is None:
        problem_names = list(all_results.keys())

    summary_metrics = (
        "MSE",
        "HV_sur",
        "HV_real",
        "HV_clip",
        HV_SHIFT_SUR_COLUMN,
        HV_SHIFT_SUR_IMPROVEMENT_COLUMN,
        HV_XY_DIRECTION_COUNTS_IMPROVEMENT_LABEL,
        HV_SHIFT_SUR_VS_CLIP_COLUMN,
    )
    summary_columns = [("Method", "")]
    for problem_name in problem_names:
        summary_columns.extend((problem_name, metric) for metric in summary_metrics)

    row = {("Method", ""): method_name}
    for problem_name in problem_names:
        run_results = all_results.get(problem_name, [])
        if not run_results:
            continue

        k1_label = "HV_majority_shift_x k=1"
        k1_gap_improvement = _baseline_mean_gap_improvement_for_variant(
            run_results,
            k1_label,
        )
        row[(problem_name, "MSE")] = _format_summary_float(
            _baseline_mean_result_value(run_results, "offline_test_mse")
        )
        row[(problem_name, "HV_sur")] = _format_summary_float(
            _baseline_mean_result_value(run_results, "hv_surrogate")
        )
        row[(problem_name, "HV_real")] = _format_summary_float(
            _baseline_mean_result_value(run_results, "hv_real")
        )
        row[(problem_name, "HV_clip")] = _format_summary_float(
            _baseline_mean_result_value(run_results, "hv_clip")
        )
        row[(problem_name, HV_SHIFT_SUR_COLUMN)] = _format_summary_float(
            _baseline_mean_hv_variant_value(run_results, k1_label)
        )
        row[(problem_name, HV_SHIFT_SUR_IMPROVEMENT_COLUMN)] = k1_gap_improvement
        row[(problem_name, HV_XY_DIRECTION_COUNTS_IMPROVEMENT_LABEL)] = (
            _baseline_mean_gap_improvement_for_variant(
                run_results,
                HV_XY_DIRECTION_COUNTS_IMPROVEMENT_LABEL,
            )
            if HV_XY_DIRECTION_COUNTS_IMPROVEMENT_LABEL
            in run_results[0].get("hv_variant_values", {})
            else "NA%"
        )
        row[(problem_name, HV_SHIFT_SUR_VS_CLIP_COLUMN)] = (
            _baseline_mean_gap_improvement_against_key(
                run_results,
                k1_label,
                "hv_clip",
            )
        )
    return pd.DataFrame([row], columns=pd.MultiIndex.from_tuples(summary_columns))


def append_compact_summary_outputs(table, title, txt_path=None, csv_path=None):
    summary_text = table.to_string(index=False)
    print(title)
    print(summary_text)
    print("Compact summary file write skipped; final mean/std summary is written to result.xlsx only.")
    return table


def append_exp1_4_compact_summary_outputs(
    method_name,
    optimizer_names,
    problem_names,
    all_results,
    problem_contexts,
    output_dir,
):
    table = build_exp1_4_compact_summary_table(
        method_name=method_name,
        optimizer_names=optimizer_names,
        problem_names=problem_names,
        all_results=all_results,
        problem_contexts=problem_contexts,
    )
    return append_compact_summary_outputs(
        table=table,
        title=f"Compact summary | {method_name}",
    )


def append_exp5_8_compact_summary_outputs(
    method_name,
    all_results,
    problem_names,
    output_dir,
):
    table = build_exp5_8_compact_summary_table(
        method_name=method_name,
        all_results=all_results,
        problem_names=problem_names,
    )
    return append_compact_summary_outputs(
        table=table,
        title=f"Compact summary | {method_name}",
    )


def plot_hv_majority_shift_x_latest_run(problem_name, method_name, results):
    if isinstance(results, dict):
        run_details = results.get("run_details", [])
    else:
        run_details = list(results)

    latest_detail = None
    for detail in reversed(run_details):
        if detail.get("majority_shift_x_result") is not None:
            latest_detail = detail
            break

    if latest_detail is None:
        print("HV_majority_shift_x plot skipped: no run detail available.")
        return None

    title_parts = [str(problem_name)]
    if method_name is not None:
        title_parts.append(str(method_name))
    title_parts.append("latest seed")
    return _plot_hv_majority_shift_before_after_2d(
        f_sur=latest_detail["obj"],
        f_real=latest_detail["f_real"],
        majority_shift_result=latest_detail["majority_shift_x_result"],
        title=" | ".join(title_parts),
    )

# Plot: 2 Objs and pareto front
def plot_obj_2d(F, xlim=(0, 1), ylim=(0, 1)):
    n_obj = F.shape[1]
    if n_obj == 2:
        nds = NonDominatedSorting()
        front_idx = nds.do(F, only_non_dominated_front=True)

        pareto_F = F[front_idx]
        non_pareto_F = np.delete(F, front_idx, axis=0)

        fig = go.Figure(
            data=go.Scatter(
                x=F[:, 0],
                y=F[:, 1],
                mode='markers',
                name='Objective Values',
                marker=dict(size=6, color='#FF7F0E', opacity=0.7)
            )
        )
        fig.add_trace(go.Scatter(
            x=pareto_F[:, 0],
            y=pareto_F[:, 1],
            mode='markers',
            name='Pareto Front',
            marker=dict(size=7, color='#B07AA1', opacity=0.9, symbol='diamond')
        ))
        fig.update_layout(
            xaxis_title='f1',
            yaxis_title='f2',
            width=600,
            height=600,
            xaxis=dict(range=list(xlim)),
            yaxis=dict(range=list(ylim))
        )
        fig.show()


def plot_z_score(y_test, pred_mean, pred_std, bins=30, eps=1e-12):
    y_test = np.asarray(y_test)
    pred_mean = np.asarray(pred_mean)
    pred_std = np.asarray(pred_std)

    z = (y_test - pred_mean) / np.maximum(pred_std, eps)

    n_obj = z.shape[1]
    fig, axes = plt.subplots(1, n_obj, figsize=(5 * n_obj, 4), sharey=True)

    if n_obj == 1:
        axes = [axes]

    for i in range(n_obj):
        axes[i].hist(z[:, i], bins=bins)
        axes[i].set_title(f"Z distribution - f{i+1}")
        axes[i].set_xlabel("z")

    axes[0].set_ylabel("count")
    plt.tight_layout()
    plt.show()


def plot_y_true_pred(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    plt.figure(figsize=(6, 6))
    plt.scatter(y_pred[:, 0], y_pred[:, 1], label="y_pred", color='#87CEEB')
    plt.scatter(y_true[:, 0], y_true[:, 1], label="y_true", color='#FF7F0E')
    plt.xlabel("y1")
    plt.ylabel("y2")
    plt.legend()
    plt.tight_layout()
    plt.show()


def _plot_sur_real_train_2d(
    f_sur,
    f_real,
    y_train,
    f_train_pred,
    xlim=None,
    ylim=None,
    title=None,
    width=700,
    height=650,
    show=True,
):
    f_sur = np.asarray(f_sur, dtype=float)
    f_real = np.asarray(f_real, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    f_train_pred = np.asarray(f_train_pred, dtype=float)

    arrays = {
        "f_sur": f_sur,
        "f_real": f_real,
        "y_train": y_train,
        "f_train_pred": f_train_pred,
    }
    for name, values in arrays.items():
        if values.ndim != 2 or values.shape[1] != 2:
            raise ValueError(f"{name} must have shape (n, 2).")

    all_points = np.vstack([f_sur, f_real, y_train, f_train_pred])
    if xlim is None:
        x_min, x_max = np.nanmin(all_points[:, 0]), np.nanmax(all_points[:, 0])
        x_pad = 0.05 * (x_max - x_min) if x_max > x_min else 1.0
        xlim = (x_min - x_pad, x_max + x_pad)
    if ylim is None:
        y_min, y_max = np.nanmin(all_points[:, 1]), np.nanmax(all_points[:, 1])
        y_pad = 0.05 * (y_max - y_min) if y_max > y_min else 1.0
        ylim = (y_min - y_pad, y_max + y_pad)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=f_sur[:, 0],
        y=f_sur[:, 1],
        mode="markers",
        name="f_sur",
        marker=dict(size=7, color="#87CEEB", opacity=0.75, symbol="circle"),
    ))
    fig.add_trace(go.Scatter(
        x=f_real[:, 0],
        y=f_real[:, 1],
        mode="markers",
        name="f_real",
        marker=dict(size=7, color="#FF7F0E", opacity=0.75, symbol="circle"),
    ))
    fig.add_trace(go.Scatter(
        x=y_train[:, 0],
        y=y_train[:, 1],
        mode="markers",
        name="train_real",
        marker=dict(size=7, color="#B07AA1", opacity=0.8, symbol="diamond"),
    ))
    fig.add_trace(go.Scatter(
        x=f_train_pred[:, 0],
        y=f_train_pred[:, 1],
        mode="markers",
        name="train_pred",
        marker=dict(size=7, color="#1565C0", opacity=0.75, symbol="x"),
    ))
    fig.update_layout(
        title=title,
        xaxis_title="f1",
        yaxis_title="f2",
        width=width,
        height=height,
        xaxis=dict(range=list(xlim)),
        yaxis=dict(range=list(ylim)),
    )

    if show:
        fig.show()
    return fig


def _plot_selected_solution_nearest_offline_2d(
    x,
    f_sur,
    f_real,
    x_train,
    y_train,
    f_train_pred,
    selected_idx=1,
    k=1,
    distance_space="x",
    distance_metric="normalized_euclidean",
    covariance_ridge=1e-8,
    xlim=None,
    ylim=None,
    title=None,
    width=700,
    height=650,
    show=True,
    print_quadrant=True,
    show_x_space_nearest=False,
):
    """Highlight one optimized solution and its nearest offline prediction pair.

    The nearest offline index is selected in the requested distance space.
    The highlighted offline real point is always y_train[nearest_idx], paired
    with the highlighted offline pred point f_train_pred[nearest_idx].
    """
    x, f_sur, f_real, x_train, y_train, f_train_pred = validate_nearest_offline_arrays(
        x,
        f_sur,
        f_real,
        x_train,
        y_train,
        f_train_pred,
    )

    selected_idx = int(selected_idx)
    if selected_idx < 0 or selected_idx >= x.shape[0]:
        raise IndexError("selected_idx is outside x.")
    query_points, train_points, distance_space_label = nearest_offline_query_train(
        x,
        f_sur,
        x_train,
        f_train_pred,
        distance_space,
    )
    normalization_points = x_train if distance_space_label == "x" else y_train
    nearest_idx, nearest_distance, distance_metric_label = pairwise_nearest_offline_distance(
        query_points[[selected_idx]],
        train_points,
        k=k,
        distance_metric=distance_metric,
        covariance_ridge=covariance_ridge,
        normalization_points=normalization_points,
    )
    nearest_train_idx_arr = np.asarray(nearest_idx, dtype=int).reshape(-1)
    nearest_train_idx = (
        int(nearest_train_idx_arr[0])
        if nearest_train_idx_arr.size == 1
        else nearest_train_idx_arr
    )
    plot_distance_space_label = "y-space" if distance_space_label == "y_pred" else "x-space"
    x_space_nearest_idx_arr = np.asarray([], dtype=int)
    if show_x_space_nearest and distance_space_label != "x":
        x_space_nearest_idx, _, _ = pairwise_nearest_offline_distance(
            x[[selected_idx]],
            x_train,
            k=1,
            distance_metric="normalized_euclidean",
            covariance_ridge=covariance_ridge,
            normalization_points=x_train,
        )
        x_space_nearest_idx_arr = np.asarray(x_space_nearest_idx, dtype=int).reshape(-1)
    quadrant, delta, nearest_sur, nearest_real = offline_real_sur_quadrants(
        y_train,
        f_train_pred,
        nearest_train_idx_arr,
    )
    selected_quadrant, selected_delta = real_sur_quadrants(
        f_real[[selected_idx]],
        f_sur[[selected_idx]],
    )
    if print_quadrant:
        print(
            f"selected x[{selected_idx}] nearest offline train{nearest_train_idx_arr.tolist()} "
            f"({distance_space_label}, {distance_metric_label}, k={nearest_train_idx_arr.size})"
        )
        print(
            "f_real relative to f_sur: "
            f"{selected_quadrant[0]}"
        )
        if nearest_train_idx_arr.size == 1:
            print(
                "train_real relative to train_sur: "
                f"{quadrant[0]}"
            )
        else:
            print(
                "train_real relative to train_sur counts: "
                f"right_upper={int(np.sum(quadrant == 'right_upper'))}, "
                f"right_lower={int(np.sum(quadrant == 'right_lower'))}, "
                f"left_upper={int(np.sum(quadrant == 'left_upper'))}, "
                f"left_lower={int(np.sum(quadrant == 'left_lower'))}"
            )

    fig = _plot_sur_real_train_2d(
        f_sur=f_sur,
        f_real=f_real,
        y_train=y_train,
        f_train_pred=f_train_pred,
        xlim=xlim,
        ylim=ylim,
        title=title,
        width=width,
        height=height,
        show=False,
    )

    fig.add_trace(go.Scatter(
        x=[f_sur[selected_idx, 0]],
        y=[f_sur[selected_idx, 1]],
        mode="markers",
        name=f"selected f_sur x[{selected_idx}]",
        marker=dict(size=16, color="#8B0000", opacity=1.0, symbol="x"),
    ))
    fig.add_trace(go.Scatter(
        x=[f_real[selected_idx, 0]],
        y=[f_real[selected_idx, 1]],
        mode="markers",
        name=f"selected f_real x[{selected_idx}]",
        marker=dict(
            size=14,
            color="#B00020",
            opacity=1.0,
            symbol="circle",
            line=dict(color="#7F0000", width=2),
        ),
    ))
    fig.add_trace(go.Scatter(
        x=f_train_pred[nearest_train_idx_arr, 0],
        y=f_train_pred[nearest_train_idx_arr, 1],
        mode="markers",
        name=f"{plot_distance_space_label} nearest offline_y_pred",
        marker=dict(
            size=17,
            color="#6D28D9",
            opacity=1.0,
            symbol="cross",
            line=dict(color="#3B0764", width=3),
        ),
    ))
    fig.add_trace(go.Scatter(
        x=y_train[nearest_train_idx_arr, 0],
        y=y_train[nearest_train_idx_arr, 1],
        mode="markers",
        name=f"{plot_distance_space_label} paired offline_y_true",
        marker=dict(
            size=15,
            color="#6D28D9",
            opacity=1.0,
            symbol="square",
            line=dict(color="#3B0764", width=2),
        ),
    ))
    if x_space_nearest_idx_arr.size:
        fig.add_trace(go.Scatter(
            x=f_train_pred[x_space_nearest_idx_arr, 0],
            y=f_train_pred[x_space_nearest_idx_arr, 1],
            mode="markers",
            name="x-space nearest offline_y_pred",
            marker=dict(
                size=18,
                color="#0EA5E9",
                opacity=1.0,
                symbol="cross",
                line=dict(color="#075985", width=3),
            ),
        ))
        fig.add_trace(go.Scatter(
            x=y_train[x_space_nearest_idx_arr, 0],
            y=y_train[x_space_nearest_idx_arr, 1],
            mode="markers",
            name="x-space paired offline_y_true",
            marker=dict(
                size=16,
                color="#0EA5E9",
                opacity=1.0,
                symbol="square",
                line=dict(color="#075985", width=2),
            ),
        ))
    fig.update_layout(
        title=title or (
            f"selected x[{selected_idx}] and nearest {nearest_train_idx_arr.size} offline train points "
            f"({distance_space_label}, {distance_metric_label})"
        )
    )

    if show:
        fig.show()
    return fig, nearest_train_idx


def _finite_2d_points(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("values must have shape (n, 2).")
    return values[np.all(np.isfinite(values), axis=1)]


def _add_hv_non_dominated_panel(fig, f_sur, f_real, row, col, sur_label="f_sur"):
    f_sur = _finite_2d_points(f_sur)
    f_real = _finite_2d_points(f_real)
    fig.add_trace(go.Scatter(
        x=f_sur[:, 0],
        y=f_sur[:, 1],
        mode="markers",
        name=sur_label,
        marker=dict(size=7, color="#87CEEB", opacity=0.30, symbol="circle"),
    ), row=row, col=col)
    fig.add_trace(go.Scatter(
        x=f_real[:, 0],
        y=f_real[:, 1],
        mode="markers",
        name="f_real",
        marker=dict(size=7, color="#FF7F0E", opacity=0.30, symbol="circle"),
    ), row=row, col=col)

    nds = NonDominatedSorting()
    for values, name, color in (
        (f_sur, f"{sur_label} HV ND", "#005B8F"),
        (f_real, "f_real HV ND", "#9A3412"),
    ):
        if values.shape[0] == 0:
            continue
        front_idx = nds.do(values, only_non_dominated_front=True)
        front = values[front_idx]
        fig.add_trace(go.Scatter(
            x=front[:, 0],
            y=front[:, 1],
            mode="markers",
            name=name,
            marker=dict(size=10, color=color, opacity=1.0, symbol="diamond"),
        ), row=row, col=col)


def _plot_hv_majority_shift_before_after_2d(
    f_sur,
    f_real,
    majority_shift_result,
    title=None,
    width=1200,
    height=650,
    show=True,
):
    """Plot original surrogate objectives beside k=1 majority-shift objectives."""
    f_sur = np.asarray(f_sur, dtype=float)
    f_real = np.asarray(f_real, dtype=float)
    f_sur_shifted = np.asarray(
        majority_shift_result["f_sur_shifted"],
        dtype=float,
    )
    if f_sur.ndim != 2 or f_sur.shape[1] != 2:
        raise ValueError("f_sur must have shape (n, 2).")
    if f_real.ndim != 2 or f_real.shape[1] != 2:
        raise ValueError("f_real must have shape (n, 2).")
    if f_sur_shifted.shape != f_sur.shape:
        raise ValueError("majority-shifted objectives must have the same shape as f_sur.")

    all_points = np.vstack((f_sur, f_real, f_sur_shifted))
    finite_points = all_points[np.all(np.isfinite(all_points), axis=1)]
    if finite_points.shape[0] == 0:
        return None
    x_min, x_max = np.min(finite_points[:, 0]), np.max(finite_points[:, 0])
    y_min, y_max = np.min(finite_points[:, 1]), np.max(finite_points[:, 1])
    x_pad = 0.05 * (x_max - x_min) if x_max > x_min else 1.0
    y_pad = 0.05 * (y_max - y_min) if y_max > y_min else 1.0
    x_range = [x_min - x_pad, x_max + x_pad]
    y_range = [y_min - y_pad, y_max + y_pad]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Original f_sur and f_real",
            "HV_majority_shift_x k=1 and f_real",
        ),
        horizontal_spacing=0.10,
    )
    _add_hv_non_dominated_panel(
        fig,
        f_sur,
        f_real,
        row=1,
        col=1,
        sur_label="f_sur",
    )
    _add_hv_non_dominated_panel(
        fig,
        f_sur_shifted,
        f_real,
        row=1,
        col=2,
        sur_label="HV_majority_shift_x k=1",
    )
    for col in (1, 2):
        fig.update_xaxes(title_text="f1", range=x_range, row=1, col=col)
        fig.update_yaxes(title_text="f2", range=y_range, row=1, col=col)
    fig.update_layout(
        title=title,
        width=width,
        height=height,
    )
    if show:
        fig.show()
    return fig


def _plot_selected_solutions_nearest_offline_hv_majority_shift_2d(
    x,
    f_sur,
    f_real,
    x_train,
    y_train,
    f_train_pred,
    majority_shift_result,
    majority_shift_reference_result=None,
    selected_indices=(10, 30, 80),
    k=1,
    distance_space="x",
    distance_metric="normalized_euclidean",
    covariance_ridge=1e-8,
    xlim=None,
    ylim=None,
    title=None,
    width=1800,
    height=650,
    show=True,
    print_quadrant=False,
    show_x_space_nearest=False,
):
    """Plot selected neighbors above majority-shifted objective values."""
    f_sur = np.asarray(f_sur, dtype=float)
    f_real = np.asarray(f_real, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    f_train_pred = np.asarray(f_train_pred, dtype=float)
    f_sur_shifted = np.asarray(
        majority_shift_result["f_sur_shifted"],
        dtype=float,
    )
    if majority_shift_reference_result is None:
        majority_shift_reference_result = majority_shift_result
    f_sur_shifted_reference = np.asarray(
        majority_shift_reference_result["f_sur_shifted"],
        dtype=float,
    )
    if f_sur_shifted.shape != f_sur.shape:
        raise ValueError("majority-shifted objectives must have the same shape as f_sur.")
    if f_sur_shifted_reference.shape != f_sur.shape:
        raise ValueError("reference majority-shifted objectives must have the same shape as f_sur.")
    selected_indices = tuple(int(selected_idx) for selected_idx in selected_indices)
    if len(selected_indices) != 3:
        raise ValueError("selected_indices must contain exactly three solution indices.")
    for selected_idx in selected_indices:
        if selected_idx < 0 or selected_idx >= f_sur.shape[0]:
            raise IndexError(f"selected_idx {selected_idx} is outside f_sur.")

    all_points = np.vstack([
        f_sur,
        f_real,
        y_train,
        f_train_pred,
        f_sur_shifted,
        f_sur_shifted_reference,
    ])
    if xlim is None:
        x_min, x_max = np.nanmin(all_points[:, 0]), np.nanmax(all_points[:, 0])
        x_pad = 0.05 * (x_max - x_min) if x_max > x_min else 1.0
        xlim = (x_min - x_pad, x_max + x_pad)
    if ylim is None:
        y_min, y_max = np.nanmin(all_points[:, 1]), np.nanmax(all_points[:, 1])
        y_pad = 0.05 * (y_max - y_min) if y_max > y_min else 1.0
        ylim = (y_min - y_pad, y_max + y_pad)

    fig = make_subplots(
        rows=2,
        cols=6,
        specs=[
            [
                {"colspan": 2}, None,
                {"colspan": 2}, None,
                {"colspan": 2}, None,
            ],
            [
                {"colspan": 3}, None, None,
                {"colspan": 3}, None, None,
            ],
        ],
        subplot_titles=(
            f"selected x[{selected_indices[0]}] and nearest {k} offline points",
            f"selected x[{selected_indices[1]}] and nearest {k} offline points",
            f"selected x[{selected_indices[2]}] and nearest {k} offline points",
            "HV_majority_shift_x and HV_real",
            "HV_majority_shift_x reference and HV_real",
        ),
        horizontal_spacing=0.05,
        vertical_spacing=0.14,
    )
    nearest_train_indices = {}
    for selected_idx, col in zip(selected_indices, (1, 3, 5)):
        nearest_fig, nearest_train_idx = _plot_selected_solution_nearest_offline_2d(
            x=x,
            f_sur=f_sur,
            f_real=f_real,
            x_train=x_train,
            y_train=y_train,
            f_train_pred=f_train_pred,
            selected_idx=selected_idx,
            k=k,
            distance_space=distance_space,
            distance_metric=distance_metric,
            covariance_ridge=covariance_ridge,
            xlim=xlim,
            ylim=ylim,
            title=title,
            show=False,
            print_quadrant=print_quadrant,
            show_x_space_nearest=show_x_space_nearest,
        )
        nearest_train_indices[selected_idx] = nearest_train_idx
        for trace in nearest_fig.data:
            fig.add_trace(trace, row=1, col=col)

    _add_hv_non_dominated_panel(
        fig,
        f_sur_shifted,
        f_real,
        row=2,
        col=1,
        sur_label="f_sur majority shift",
    )
    _add_hv_non_dominated_panel(
        fig,
        f_sur_shifted_reference,
        f_real,
        row=2,
        col=4,
        sur_label="f_sur majority shift reference",
    )

    for selected_idx in selected_indices:
        fig.add_trace(go.Scatter(
            x=[f_sur[selected_idx, 0], f_sur_shifted[selected_idx, 0]],
            y=[f_sur[selected_idx, 1], f_sur_shifted[selected_idx, 1]],
            mode="lines+markers",
            name=f"selected x[{selected_idx}] majority movement",
            marker=dict(size=10, color="#6D28D9"),
            line=dict(color="#6D28D9", width=2),
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=[f_sur[selected_idx, 0], f_sur_shifted_reference[selected_idx, 0]],
            y=[f_sur[selected_idx, 1], f_sur_shifted_reference[selected_idx, 1]],
            mode="lines+markers",
            name=f"selected x[{selected_idx}] reference majority movement",
            marker=dict(size=10, color="#B00020"),
            line=dict(color="#B00020", width=2),
        ), row=2, col=4)

    for col in (1, 3, 5):
        fig.update_xaxes(title_text="f1", range=list(xlim), row=1, col=col)
        fig.update_yaxes(title_text="f2", range=list(ylim), row=1, col=col)
    for col in (1, 4):
        fig.update_xaxes(title_text="f1", range=list(xlim), row=2, col=col)
        fig.update_yaxes(title_text="f2", range=list(ylim), row=2, col=col)
    fig.update_layout(
        title=title or f"k={k} nearest-offline and majority shift comparison",
        width=width,
        height=max(int(height), 1100),
    )
    if show:
        fig.show()
    return fig, nearest_train_indices


def _count_line(counts):
    return (
        f"right_upper={counts['right_upper']}, "
        f"right_lower={counts['right_lower']}, "
        f"left_upper={counts['left_upper']}, "
        f"left_lower={counts['left_lower']}"
    )


def _sum_nearest_offline_counts(items, key):
    total = {"right_upper": 0, "right_lower": 0, "left_upper": 0, "left_lower": 0}
    for item in items:
        counts = item["result"]["nearest_offline_summary"][key]
        for count_key in total:
            total[count_key] += int(counts[count_key])
    return total


def _normalized_l2_distance(values_a, values_b, normalization_ref):
    values_a = np.asarray(values_a, dtype=float)
    values_b = np.asarray(values_b, dtype=float)
    normalization_ref = np.asarray(normalization_ref, dtype=float)
    ref_min = np.nanmin(normalization_ref, axis=0)
    ref_max = np.nanmax(normalization_ref, axis=0)
    scale = np.where(ref_max == ref_min, 1.0, ref_max - ref_min)
    values_a_scaled = (values_a - ref_min) / scale
    values_b_scaled = (values_b - ref_min) / scale
    return np.linalg.norm(values_a_scaled - values_b_scaled, axis=1)


def _objective_space_nearest_distance_info(run_detail, context, nearest_idx):
    nearest_idx = np.asarray(nearest_idx, dtype=int).reshape(-1)
    f_sur = np.asarray(run_detail["obj"], dtype=float)
    y_train = np.asarray(context["y_train"], dtype=float)
    y_pred_nearest = np.asarray(context["f_train_mean"], dtype=float)[nearest_idx]
    y_true_nearest = y_train[nearest_idx]
    return {
        "nearest_idx": nearest_idx,
        "y_pred_nearest": y_pred_nearest,
        "y_true_nearest": y_true_nearest,
        "f_sur_to_y_pred_distance": _normalized_l2_distance(
            f_sur,
            y_pred_nearest,
            y_train,
        ),
        "y_pred_to_y_true_distance": _normalized_l2_distance(
            y_pred_nearest,
            y_true_nearest,
            y_train,
        ),
    }


def _x_space_nearest_distance_info(run_detail, context):
    x = np.asarray(run_detail["solution"], dtype=float)
    x_train = np.asarray(context["X_train"], dtype=float)
    y_train = np.asarray(context["y_train"], dtype=float)
    f_train_mean = np.asarray(context["f_train_mean"], dtype=float)
    nearest_idx, nearest_distance, _ = pairwise_nearest_offline_distance(
        x,
        x_train,
        k=1,
        distance_metric="normalized_euclidean",
        normalization_points=x_train,
    )
    nearest_idx = np.asarray(nearest_idx, dtype=int).reshape(-1)
    nearest_y_pred = f_train_mean[nearest_idx]
    nearest_y_true = y_train[nearest_idx]
    return {
        "nearest_idx": nearest_idx,
        "nearest_x": x_train[nearest_idx],
        "nearest_y_pred": nearest_y_pred,
        "nearest_y_true": nearest_y_true,
        "nearest_normalized_x_distance": np.asarray(nearest_distance, dtype=float).reshape(-1),
        "nearest_y_pred_to_y_true_distance": _normalized_l2_distance(
            nearest_y_pred,
            nearest_y_true,
            y_train,
        ),
    }


def _evaluate_majority_shift_y_k1_diagnostic(
    run_detail,
    context,
):
    from src.experiment import (
        _clip_majority_shift_result_to_problem_bounds,
        compute_R_indicator,
        mse_or_nan,
        normalized_hv,
    )
    from src.nearest_offline import compute_majority_shift_calibration_2d

    result = compute_majority_shift_calibration_2d(
        x=run_detail["solution"],
        f_sur=run_detail["obj"],
        f_real=run_detail["f_real"],
        x_train=context["X_train"],
        y_train=context["y_train"],
        f_train_pred=context["f_train_mean"],
        distance_space="y",
        distance_metric="normalized_euclidean",
        k=1,
    )
    _clip_majority_shift_result_to_problem_bounds(
        result,
        context["problem_y_min"],
        clip_source="problem_y_min",
    )
    hv_value = normalized_hv(
        context["hv"],
        result["f_sur_shifted"],
        context["obj_min"],
        context["obj_max"],
    )
    hv_gap = abs(run_detail["hv_real"] - hv_value)

    return {
        "result": result,
        "hv": hv_value,
        "hv_gap": hv_gap,
        "gap_reduction_pct": compute_R_indicator(run_detail["hv_sur_gap"], hv_gap),
        "mse": mse_or_nan(run_detail["f_real"], result["f_sur_shifted"]),
        **_objective_space_nearest_distance_info(
            run_detail,
            context,
            result["nearest_idx"],
        ),
    }


def evaluate_majority_shift_y_k1_diagnostic_items(
    results,
    context,
):
    return [
        _evaluate_majority_shift_y_k1_diagnostic(
            run_detail,
            context,
        )
        for run_detail in results.get("run_details", [])
    ]


def y_space_direction_counts_from_diagnostic_items(items):
    summaries = [item["result"]["nearest_offline_summary"] for item in items]
    return _direction_counts_from_summaries(summaries)


def evaluate_xy_direction_counts_hv_values(
    results,
    y_space_items,
    context,
):
    from src.experiment import normalized_hv

    hv_values = []
    run_details = results.get("run_details", [])
    for run_detail, y_space_item in zip(run_details, y_space_items):
        x_shift_result = run_detail.get("majority_shift_x_result")
        if x_shift_result is None:
            hv_values.append(np.nan)
            continue

        x_summary = x_shift_result["nearest_offline_summary"]
        y_summary = y_space_item["result"]["nearest_offline_summary"]
        x_direction = _signed_direction_vector_from_counts(
            _direction_counts_from_summaries([x_summary])
        )
        y_direction = _signed_direction_vector_from_counts(
            _direction_counts_from_summaries([y_summary])
        )
        agreed_direction = np.where(x_direction == y_direction, x_direction, 0.0)

        shift = agreed_direction * np.asarray(x_shift_result["mean_abs_delta"], dtype=float)
        shifted = np.asarray(run_detail["obj"], dtype=float) + shift
        shifted = np.maximum(shifted, np.asarray(context["problem_y_min"], dtype=float))
        hv_values.append(
            normalized_hv(
                context["hv"],
                shifted,
                context["obj_min"],
                context["obj_max"],
            )
        )
    return np.asarray(hv_values, dtype=float)


def _compact_seed_direction_from_summary(summary):
    return _signed_direction_vector_from_counts(
        _direction_counts_from_summaries([summary])
    )


def _compact_seed_shifted_points_with_direction(run_detail, context, direction, use_clip):
    x_shift_result = run_detail.get("majority_shift_x_result")
    if x_shift_result is None:
        return None
    f_sur = np.asarray(run_detail["obj"], dtype=float)
    mean_abs_delta = np.asarray(x_shift_result["mean_abs_delta"], dtype=float)
    shifted = f_sur + np.asarray(direction, dtype=float) * mean_abs_delta
    if use_clip:
        shifted = np.maximum(shifted, np.asarray(context["problem_y_min"], dtype=float))
    return shifted


def _compact_seed_hv_with_direction(run_detail, context, direction, use_clip):
    from src.experiment import normalized_hv

    shifted = _compact_seed_shifted_points_with_direction(
        run_detail,
        context,
        direction,
        use_clip=use_clip,
    )
    if shifted is None:
        return float("nan")
    return float(
        normalized_hv(
            context["hv"],
            shifted,
            context["obj_min"],
            context["obj_max"],
        )
    )


def _compact_seed_xy_shifted_points(run_detail, context, y_space_item=None, use_clip=True):
    x_shift_result = run_detail.get("majority_shift_x_result")
    if x_shift_result is None:
        return None

    x_direction = _compact_seed_direction_from_summary(
        x_shift_result["nearest_offline_summary"]
    )
    if y_space_item is None:
        y_direction = np.asarray([0.0, 0.0], dtype=float)
    else:
        y_direction = _compact_seed_direction_from_summary(
            y_space_item["result"]["nearest_offline_summary"]
    )
    xy_direction = np.where(x_direction == y_direction, x_direction, 0.0)

    return _compact_seed_shifted_points_with_direction(
        run_detail,
        context,
        xy_direction,
        use_clip=use_clip,
    )


def _plot_hv_xy_shift_seed_2d_matplotlib(
    f_sur,
    f_real,
    f_xy_shift,
    title,
    x_range,
    y_range,
    width=760,
    height=650,
    show=True,
    save_svg_path=None,
    separate_legend=False,
    save_legend_svg_path=None,
):
    plt.rcParams["text.usetex"] = False
    series = (
        ("f_sur", f_sur, "#87CEEB", "o", 0.78),
        ("f_real", f_real, "#FF7F0E", "o", 0.78),
        ("HV_xy_shift", f_xy_shift, "#0B3D91", "o", 0.90),
    )
    fig, ax = plt.subplots(
        figsize=(float(width) / 100.0, float(height) / 100.0),
        dpi=100,
    )
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    for label, values, color, marker, alpha in series:
        ax.scatter(
            values[:, 0],
            values[:, 1],
            s=50,
            c=color,
            marker=marker,
            alpha=alpha,
            label=label,
            edgecolors="none",
        )
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.grid(False)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(direction="out", length=5, width=1, colors="black")
    ax.tick_params(labelsize=25)
    if not separate_legend:
        ax.legend(frameon=False)
    fig.tight_layout()

    legend_fig = None
    if separate_legend:
        from matplotlib.lines import Line2D

        legend_handles = [
            Line2D(
                [0],
                [0],
                marker=marker,
                linestyle="None",
                label="",
                markerfacecolor=color,
                markeredgecolor=color,
                markersize=7,
                alpha=alpha,
            )
            for label, _, color, marker, alpha in series
        ]
        legend_fig, legend_ax = plt.subplots(figsize=(3.6, 0.7), dpi=100)
        legend_fig.patch.set_facecolor("white")
        legend_ax.axis("off")
        legend_ax.legend(
            handles=legend_handles,
            loc="center",
            ncol=len(legend_handles),
            frameon=False,
            handlelength=1.0,
            handletextpad=0.0,
            columnspacing=0.8,
        )
        legend_fig.tight_layout(pad=0.05)

    if save_svg_path is not None:
        save_svg_path = Path(save_svg_path)
        save_svg_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_svg_path, format="svg", bbox_inches="tight")
        print(f"Saved HV_xy_shift SVG: {save_svg_path}")
    if save_legend_svg_path is not None and legend_fig is not None:
        save_legend_svg_path = Path(save_legend_svg_path)
        save_legend_svg_path.parent.mkdir(parents=True, exist_ok=True)
        legend_fig.savefig(save_legend_svg_path, format="svg", bbox_inches="tight")
        print(f"Saved HV_xy_shift legend SVG: {save_legend_svg_path}")
        if title:
            print(f"HV_xy_shift title: {title}")

    if show:
        plt.show()
    else:
        plt.close(fig)
        if legend_fig is not None:
            plt.close(legend_fig)
    return (fig, legend_fig) if separate_legend else fig


def plot_hv_xy_shift_seed_2d(
    run_detail,
    context,
    y_space_item=None,
    problem_name=None,
    method_name=None,
    hv_xy_shift=None,
    width=760,
    height=650,
    show=True,
    plain_style=False,
    save_svg_path=None,
    separate_legend=False,
    save_legend_svg_path=None,
):
    """Plot f_sur, f_real, and the current clipped xy-shift points in one panel."""
    f_sur = np.asarray(run_detail["obj"], dtype=float)
    f_real = np.asarray(run_detail["f_real"], dtype=float)
    f_xy_shift = _compact_seed_xy_shifted_points(
        run_detail,
        context,
        y_space_item=y_space_item,
        use_clip=True,
    )
    if f_xy_shift is None:
        print("HV_xy_shift plot skipped: no majority_shift_x_result available.")
        return None

    all_points = np.vstack((f_sur, f_real, f_xy_shift))
    finite_points = all_points[np.all(np.isfinite(all_points), axis=1)]
    if finite_points.shape[0] == 0:
        print("HV_xy_shift plot skipped: no finite 2D points available.")
        return None

    x_min, x_max = np.min(finite_points[:, 0]), np.max(finite_points[:, 0])
    y_min, y_max = np.min(finite_points[:, 1]), np.max(finite_points[:, 1])
    x_pad = 0.05 * (x_max - x_min) if x_max > x_min else 1.0
    y_pad = 0.05 * (y_max - y_min) if y_max > y_min else 1.0

    title_parts = [str(problem_name)] if problem_name is not None else []
    if method_name is not None:
        title_parts.append(str(method_name))
    seed = run_detail.get("seed", None)
    if seed is not None:
        title_parts.append(f"Seed {seed}")
    title = " | ".join(title_parts) if title_parts else "HV_xy_shift"

    if plain_style:
        return _plot_hv_xy_shift_seed_2d_matplotlib(
            f_sur=f_sur,
            f_real=f_real,
            f_xy_shift=f_xy_shift,
            title=title,
            x_range=[x_min - x_pad, x_max + x_pad],
            y_range=[y_min - y_pad, y_max + y_pad],
            width=width,
            height=height,
            show=show,
            save_svg_path=save_svg_path,
            separate_legend=separate_legend,
            save_legend_svg_path=save_legend_svg_path,
        )

    trace_specs = (
        (
            "f_sur",
            dict(size=8, color="#87CEEB", opacity=0.78, symbol="circle"),
        ),
        (
            "f_real",
            dict(size=8, color="#FF7F0E", opacity=0.78, symbol="circle"),
        ),
        (
            "HV_xy_shift",
            dict(size=8, color="#0B3D91", opacity=0.90, symbol="circle"),
        ),
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=f_sur[:, 0],
        y=f_sur[:, 1],
        mode="markers",
        name=trace_specs[0][0],
        marker=trace_specs[0][1],
    ))
    fig.add_trace(go.Scatter(
        x=f_real[:, 0],
        y=f_real[:, 1],
        mode="markers",
        name=trace_specs[1][0],
        marker=trace_specs[1][1],
    ))
    fig.add_trace(go.Scatter(
        x=f_xy_shift[:, 0],
        y=f_xy_shift[:, 1],
        mode="markers",
        name=trace_specs[2][0],
        marker=trace_specs[2][1],
    ))
    layout_kwargs = dict(
        title=title,
        xaxis_title="f1",
        yaxis_title="f2",
        width=width,
        height=height,
        xaxis=dict(range=[x_min - x_pad, x_max + x_pad]),
        yaxis=dict(range=[y_min - y_pad, y_max + y_pad]),
        legend=dict(itemsizing="constant"),
        showlegend=not separate_legend,
    )
    if plain_style:
        layout_kwargs.update(
            template="plotly_white",
            xaxis_title=None,
            yaxis_title=None,
            paper_bgcolor="white",
            plot_bgcolor="white",
            xaxis=dict(
                range=[x_min - x_pad, x_max + x_pad],
                showgrid=True,
                zeroline=False,
            ),
            yaxis=dict(
                range=[y_min - y_pad, y_max + y_pad],
                showgrid=True,
                zeroline=False,
            ),
        )
    fig.update_layout(**layout_kwargs)
    legend_fig = None
    if separate_legend:
        legend_fig = go.Figure()
        for trace_name, marker in trace_specs:
            legend_fig.add_trace(go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                name=trace_name,
                marker=marker,
                showlegend=True,
            ))
        legend_fig.update_layout(
            template="plotly_white" if plain_style else None,
            width=360,
            height=110,
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="white" if plain_style else None,
            plot_bgcolor="white" if plain_style else None,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            legend=dict(
                orientation="h",
                x=0.5,
                y=0.5,
                xanchor="center",
                yanchor="middle",
                itemsizing="constant",
            ),
        )
    if save_svg_path is not None:
        save_svg_path = Path(save_svg_path)
        save_svg_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fig.write_image(str(save_svg_path), format="svg")
            print(f"Saved HV_xy_shift SVG: {save_svg_path}")
        except Exception as err:
            print(
                "HV_xy_shift SVG save skipped: "
                f"{type(err).__name__}: {err}. "
                "Install kaleido if static image export is unavailable."
            )
    if save_legend_svg_path is not None and legend_fig is not None:
        save_legend_svg_path = Path(save_legend_svg_path)
        save_legend_svg_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            legend_fig.write_image(str(save_legend_svg_path), format="svg")
            print(f"Saved HV_xy_shift legend SVG: {save_legend_svg_path}")
        except Exception as err:
            print(
                "HV_xy_shift legend SVG save skipped: "
                f"{type(err).__name__}: {err}. "
                "Install kaleido if static image export is unavailable."
            )
    if show:
        fig.show()
        if legend_fig is not None:
            legend_fig.show()
    return (fig, legend_fig) if separate_legend else fig


def _compact_seed_gap_improvement(hv_sur, hv_real, hv_value):
    from src.experiment import compute_R_indicator

    if not np.isfinite(hv_value):
        return float("nan")
    return compute_R_indicator(abs(hv_real - hv_sur), abs(hv_real - hv_value))


def _R_zero_denominator_marker_key(improvement_key):
    return f"{improvement_key}{R_ZERO_DENOMINATOR_SUFFIX}"


def _compact_seed_R_zero_denominator(gap_sur):
    from src.experiment import is_R_indicator_zero_denominator

    return is_R_indicator_zero_denominator(gap_sur)


def _compact_seed_record(run_detail, context, y_space_item=None, mse_test=None):
    from src.experiment import compute_R_indicator, mse_or_nan

    hv_sur = float(run_detail["hv_surrogate"])
    hv_real = float(run_detail["hv_real"])
    hv_clip = float(run_detail.get("hv_clipped", run_detail.get("hv_clip", np.nan)))
    f_sur = np.asarray(run_detail["obj"], dtype=float)
    f_real = np.asarray(run_detail["f_real"], dtype=float)
    f_clip = np.maximum(f_sur, np.asarray(context["problem_y_min"], dtype=float))
    mse_sur = mse_or_nan(f_real, f_sur)
    igd_plus_indicator = run_detail.get("igd_plus_indicator")
    if igd_plus_indicator is None:
        igd_plus_indicator = context.get("igd_plus")
    igd_plus_real = run_detail.get("igd_plus_real", run_detail.get("igd_plus", np.nan))
    try:
        igd_plus_real = float(igd_plus_real)
    except (TypeError, ValueError):
        igd_plus_real = float("nan")
    if igd_plus_indicator is None:
        igd_plus_sur = float("nan")
    else:
        igd_plus_sur = run_detail.get("igd_plus_surrogate")
        if igd_plus_sur is None:
            igd_plus_sur = igd_plus_indicator(f_sur)
        igd_plus_sur = float(igd_plus_sur)
    try:
        hv_majority_shift_x = float(run_detail.get("hv_majority_shift_x", np.nan))
    except (TypeError, ValueError):
        hv_majority_shift_x = float("nan")
    if not np.isfinite(hv_majority_shift_x):
        hv_majority_shift_x = run_detail.get("hv_variant_values", {}).get(
            "HV_majority_shift_x k=1",
            np.nan,
        )
        try:
            hv_majority_shift_x = float(hv_majority_shift_x)
        except (TypeError, ValueError):
            hv_majority_shift_x = float("nan")
    if mse_test is None:
        mse_test = context.get(
            "offline_test_mse",
            run_detail.get("offline_test_mse", np.nan),
        )

    x_shift_result = run_detail.get("majority_shift_x_result")
    if x_shift_result is None:
        x_direction = np.asarray([0.0, 0.0], dtype=float)
    else:
        x_direction = _compact_seed_direction_from_summary(
            x_shift_result["nearest_offline_summary"]
        )

    if y_space_item is None:
        y_direction = np.asarray([0.0, 0.0], dtype=float)
    else:
        y_direction = _compact_seed_direction_from_summary(
            y_space_item["result"]["nearest_offline_summary"]
        )
    xy_direction = np.where(x_direction == y_direction, x_direction, 0.0)

    hv_xy_shift = _compact_seed_hv_with_direction(
        run_detail,
        context,
        xy_direction,
        use_clip=True,
    )
    hv_x_shift = _compact_seed_hv_with_direction(
        run_detail,
        context,
        x_direction,
        use_clip=True,
    )
    hv_y_shift = _compact_seed_hv_with_direction(
        run_detail,
        context,
        y_direction,
        use_clip=True,
    )
    hv_no_clip = _compact_seed_hv_with_direction(
        run_detail,
        context,
        xy_direction,
        use_clip=False,
    )
    f_clip_variant = f_clip
    f_xy_shift = _compact_seed_xy_shifted_points(
        run_detail,
        context,
        y_space_item=y_space_item,
        use_clip=True,
    )
    f_x_shift = _compact_seed_shifted_points_with_direction(
        run_detail,
        context,
        x_direction,
        use_clip=True,
    )
    f_y_shift = _compact_seed_shifted_points_with_direction(
        run_detail,
        context,
        y_direction,
        use_clip=True,
    )
    f_no_clip = _compact_seed_shifted_points_with_direction(
        run_detail,
        context,
        xy_direction,
        use_clip=False,
    )

    def _mse_variant(points):
        return float("nan") if points is None else mse_or_nan(f_real, points)

    def _igd_plus_variant(points):
        if points is None or igd_plus_indicator is None:
            return float("nan")
        return float(igd_plus_indicator(points))

    mse_clip = _mse_variant(f_clip_variant)
    mse_xy_shift = _mse_variant(f_xy_shift)
    mse_x_shift = _mse_variant(f_x_shift)
    mse_y_shift = _mse_variant(f_y_shift)
    mse_no_clip = _mse_variant(f_no_clip)
    mse_clip_improvement = compute_R_indicator(mse_sur, mse_clip)
    mse_xy_shift_improvement = compute_R_indicator(mse_sur, mse_xy_shift)
    mse_x_shift_improvement = compute_R_indicator(mse_sur, mse_x_shift)
    mse_y_shift_improvement = compute_R_indicator(mse_sur, mse_y_shift)
    mse_no_clip_improvement = compute_R_indicator(mse_sur, mse_no_clip)

    igd_plus_clip = _igd_plus_variant(f_clip_variant)
    igd_plus_xy_shift = _igd_plus_variant(f_xy_shift)
    igd_plus_x_shift = _igd_plus_variant(f_x_shift)
    igd_plus_y_shift = _igd_plus_variant(f_y_shift)
    igd_plus_no_clip = _igd_plus_variant(f_no_clip)
    igd_plus_clip_improvement = compute_R_indicator(igd_plus_sur, igd_plus_clip)
    igd_plus_xy_shift_improvement = compute_R_indicator(
        igd_plus_sur,
        igd_plus_xy_shift,
    )
    igd_plus_x_shift_improvement = compute_R_indicator(
        igd_plus_sur,
        igd_plus_x_shift,
    )
    igd_plus_y_shift_improvement = compute_R_indicator(
        igd_plus_sur,
        igd_plus_y_shift,
    )
    igd_plus_no_clip_improvement = compute_R_indicator(
        igd_plus_sur,
        igd_plus_no_clip,
    )
    hv_sur_gap = abs(hv_real - hv_sur)
    hv_majority_shift_x_k1_improvement = _compact_seed_gap_improvement(
        hv_sur,
        hv_real,
        hv_majority_shift_x,
    )
    hv_clip_improvement = _compact_seed_gap_improvement(
        hv_sur,
        hv_real,
        hv_clip,
    )
    hv_xy_shift_improvement = _compact_seed_gap_improvement(
        hv_sur,
        hv_real,
        hv_xy_shift,
    )
    hv_x_shift_improvement = _compact_seed_gap_improvement(
        hv_sur,
        hv_real,
        hv_x_shift,
    )
    hv_y_shift_improvement = _compact_seed_gap_improvement(
        hv_sur,
        hv_real,
        hv_y_shift,
    )
    hv_no_clip_improvement = _compact_seed_gap_improvement(
        hv_sur,
        hv_real,
        hv_no_clip,
    )

    return {
        "seed": int(run_detail.get("seed", 0)),
        "time": float(run_detail.get("time", run_detail.get("elapsed_seconds", np.nan))),
        "MSE_test": float(mse_test),
        "MSE_sur": float(mse_sur),
        "MSE_clip": float(mse_clip),
        "MSE_xy_shift": float(mse_xy_shift),
        "MSE_x_shift": float(mse_x_shift),
        "MSE_y_shift": float(mse_y_shift),
        "MSE_no_clip": float(mse_no_clip),
        "MSE_clip_improvement": float(mse_clip_improvement),
        "MSE_xy_shift_improvement": float(mse_xy_shift_improvement),
        "MSE_x_shift_improvement": float(mse_x_shift_improvement),
        "MSE_y_shift_improvement": float(mse_y_shift_improvement),
        "MSE_no_clip_improvement": float(mse_no_clip_improvement),
        "IGD+_real": float(igd_plus_real),
        "IGD+_sur": float(igd_plus_sur),
        "IGD+_clip": float(igd_plus_clip),
        "IGD+_xy_shift": float(igd_plus_xy_shift),
        "IGD+_x_shift": float(igd_plus_x_shift),
        "IGD+_y_shift": float(igd_plus_y_shift),
        "IGD+_no_clip": float(igd_plus_no_clip),
        "IGD+_clip_improvement": float(igd_plus_clip_improvement),
        "IGD+_xy_shift_improvement": float(igd_plus_xy_shift_improvement),
        "IGD+_x_shift_improvement": float(igd_plus_x_shift_improvement),
        "IGD+_y_shift_improvement": float(igd_plus_y_shift_improvement),
        "IGD+_no_clip_improvement": float(igd_plus_no_clip_improvement),
        "HV_sur": hv_sur,
        "HV_real": hv_real,
        "HV_clip": hv_clip,
        "HV_majority_shift_x_k1": hv_majority_shift_x,
        "HV_xy_shift": hv_xy_shift,
        "HV_x_shift": hv_x_shift,
        "HV_y_shift": hv_y_shift,
        "HV_no_clip": hv_no_clip,
        "HV_majority_shift_x_k1_improvement": hv_majority_shift_x_k1_improvement,
        "HV_clip_improvement": hv_clip_improvement,
        "HV_xy_shift_improvement": hv_xy_shift_improvement,
        "HV_x_shift_improvement": hv_x_shift_improvement,
        "HV_y_shift_improvement": hv_y_shift_improvement,
        "HV_no_clip_improvement": hv_no_clip_improvement,
        _R_zero_denominator_marker_key("MSE_clip_improvement"): _compact_seed_R_zero_denominator(mse_sur),
        _R_zero_denominator_marker_key("MSE_xy_shift_improvement"): _compact_seed_R_zero_denominator(mse_sur),
        _R_zero_denominator_marker_key("MSE_x_shift_improvement"): _compact_seed_R_zero_denominator(mse_sur),
        _R_zero_denominator_marker_key("MSE_y_shift_improvement"): _compact_seed_R_zero_denominator(mse_sur),
        _R_zero_denominator_marker_key("MSE_no_clip_improvement"): _compact_seed_R_zero_denominator(mse_sur),
        _R_zero_denominator_marker_key("IGD+_clip_improvement"): _compact_seed_R_zero_denominator(igd_plus_sur),
        _R_zero_denominator_marker_key("IGD+_xy_shift_improvement"): _compact_seed_R_zero_denominator(igd_plus_sur),
        _R_zero_denominator_marker_key("IGD+_x_shift_improvement"): _compact_seed_R_zero_denominator(igd_plus_sur),
        _R_zero_denominator_marker_key("IGD+_y_shift_improvement"): _compact_seed_R_zero_denominator(igd_plus_sur),
        _R_zero_denominator_marker_key("IGD+_no_clip_improvement"): _compact_seed_R_zero_denominator(igd_plus_sur),
        _R_zero_denominator_marker_key("HV_majority_shift_x_k1_improvement"): _compact_seed_R_zero_denominator(hv_sur_gap),
        _R_zero_denominator_marker_key("HV_clip_improvement"): _compact_seed_R_zero_denominator(hv_sur_gap),
        _R_zero_denominator_marker_key("HV_xy_shift_improvement"): _compact_seed_R_zero_denominator(hv_sur_gap),
        _R_zero_denominator_marker_key("HV_x_shift_improvement"): _compact_seed_R_zero_denominator(hv_sur_gap),
        _R_zero_denominator_marker_key("HV_y_shift_improvement"): _compact_seed_R_zero_denominator(hv_sur_gap),
        _R_zero_denominator_marker_key("HV_no_clip_improvement"): _compact_seed_R_zero_denominator(hv_sur_gap),
    }


def _format_record_R_percent(record, key):
    marker = "*" if bool(record.get(_R_zero_denominator_marker_key(key), False)) else ""
    return f"{record[key]:.1f}%{marker}"


def _print_compact_seed_record(record):
    print(
        f"Seed {record['seed']} | "
        f"Time: {record['time']:.2f}s | "
        f"MSE_test: {record['MSE_test']:.2e} | "
        f"R_MSE_xy_shift: {_format_record_R_percent(record, 'MSE_xy_shift_improvement')} | "
        f"R_IGD+_xy_shift: {_format_record_R_percent(record, 'IGD+_xy_shift_improvement')} | "
        f"R_HV_xy_shift: {_format_record_R_percent(record, 'HV_xy_shift_improvement')}"
    )


def _attach_R_indicator_to_run_detail(run_detail, record):
    r_indicator = {}
    r_indicator_zero_denominator = {}
    for key, value in record.items():
        if not key.endswith("_improvement"):
            continue
        metric_key = key[: -len("_improvement")]
        try:
            r_value = float(value)
        except (TypeError, ValueError):
            r_value = float("nan")
        zero_denominator = bool(
            record.get(_R_zero_denominator_marker_key(key), False)
        )
        r_indicator[metric_key] = r_value
        r_indicator_zero_denominator[metric_key] = zero_denominator
        run_detail[f"R_{metric_key}"] = r_value
        run_detail[f"R_{metric_key}{R_ZERO_DENOMINATOR_SUFFIX}"] = zero_denominator
    run_detail["R_indicator"] = r_indicator
    run_detail["R_indicator_zero_denominator"] = r_indicator_zero_denominator
    run_detail["seed_result_record"] = dict(record)


def print_seed_result(
    run_detail,
    context=None,
    seed=None,
    elapsed=None,
    mse_test=None,
    y_space_item=None,
    problem_name=None,
    method_name=None,
    plot_on_seed=1,
    output_dir=None,
):
    """Print one completed seed and show seed-1 diagnostics/plot once."""
    if seed is not None:
        run_detail["seed"] = int(seed)
    if elapsed is not None:
        run_detail["time"] = float(elapsed)
    if context is None:
        context = run_detail.get("_problem_context")
    if context is None:
        return None, y_space_item
    if mse_test is None:
        mse_test = run_detail.get("offline_test_mse", None)
    if y_space_item is None:
        y_space_item = run_detail.get("majority_shift_y_k1_diagnostic")
    if y_space_item is None:
        y_space_items = evaluate_majority_shift_y_k1_diagnostic_items(
            {"run_details": [run_detail]},
            context,
        )
        y_space_item = y_space_items[0] if y_space_items else None

    seed = int(run_detail.get("seed", -1))
    disable_plots = os.environ.get("DISABLE_HV_PLOTS", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    plain_plot = os.environ.get("HV_TEST_PLAIN_PLOT", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    save_svg = os.environ.get("HV_TEST_SAVE_SVG", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    separate_legend = os.environ.get("HV_TEST_SEPARATE_LEGEND", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    record = _compact_seed_record(
        run_detail,
        context,
        y_space_item=y_space_item,
        mse_test=mse_test,
    )
    _attach_R_indicator_to_run_detail(run_detail, record)
    if seed == int(plot_on_seed):
        print_shift_direction_statistics(
            {"run_details": [run_detail]},
            y_space_items=[y_space_item] if y_space_item is not None else None,
        )
        print(
            "Plot HV values: "
            f"HV_sur={record['HV_sur']:.3f}, "
            f"HV_real={record['HV_real']:.3f}, "
            f"HV_xy_shift={record['HV_xy_shift']:.3f} "
            f"({_format_record_R_percent(record, 'HV_xy_shift_improvement')})"
        )
        if not disable_plots:
            if plain_plot:
                save_svg_path = None
                save_legend_svg_path = None
                if save_svg:
                    svg_dir = Path(os.environ.get("HV_TEST_SVG_DIR", "plot_sur_real_svg"))
                    svg_stem = _plot_sur_real_filename_stem(problem_name, method_name)
                    save_svg_path = svg_dir / f"{svg_stem}.svg"
                    if separate_legend:
                        save_legend_svg_path = svg_dir / "exp_legend.svg"
                plot_hv_xy_shift_seed_2d(
                    run_detail,
                    context,
                    y_space_item=y_space_item,
                    problem_name=problem_name,
                    method_name=method_name,
                    hv_xy_shift=record["HV_xy_shift"],
                    plain_style=True,
                    save_svg_path=save_svg_path,
                    separate_legend=separate_legend,
                    save_legend_svg_path=save_legend_svg_path,
                )
            else:
                plot_hv_majority_shift_x_latest_run(
                    problem_name,
                    method_name,
                    {"run_details": [run_detail]},
                )

    _print_compact_seed_record(record)
    return record, y_space_item


def print_single_seed_result_with_diagnostics(*args, **kwargs):
    return print_seed_result(*args, **kwargs)


def print_shift_direction_statistics(results, y_space_items=None):
    """Print x/y nearest-neighbor shift direction counts without HV aggregates."""
    if isinstance(results, dict):
        run_details = results.get("run_details", [])
    else:
        run_details = list(results)
    x_summaries = [
        detail["majority_shift_x_result"]["nearest_offline_summary"]
        for detail in run_details
        if detail.get("majority_shift_x_result") is not None
    ]
    if not x_summaries:
        print("Shift direction statistics: no majority-shift results available.")
        return

    first = x_summaries[0]
    candidate_counts = _summary_sum_quadrant_counts(
        x_summaries,
        "f_real_vs_f_sur_counts",
    )
    total = sum(int(summary["total"]) for summary in x_summaries)
    x_space_direction_counts = _direction_counts_from_summaries(x_summaries)

    print("Shift direction statistics:")
    print(
        f"x-space counts ({first['distance_metric']}, k={first['k']}):"
    )
    print(
        f"optimized f_real relative to f_sur total={total}: "
        f"{_summary_count_line(candidate_counts)}"
    )
    print(
        "x-space nearest train_real relative to train_sur direction counts: "
        f"{_direction_count_line(x_space_direction_counts)}"
    )
    print(
        "x-space nearest train_real relative to train_sur majority direction: "
        f"{_majority_direction_from_counts(x_space_direction_counts)}"
    )

    if y_space_items is None:
        return
    y_summaries = [
        item["result"]["nearest_offline_summary"]
        for item in y_space_items
        if item is not None and item.get("result") is not None
    ]
    if not y_summaries:
        return
    y_space_direction_counts = _direction_counts_from_summaries(y_summaries)
    print(
        "y-space nearest train_real relative to train_sur direction counts: "
        f"{_direction_count_line(y_space_direction_counts)}"
    )
    print(
        "y-space nearest train_real relative to train_sur majority direction: "
        f"{_majority_direction_from_counts(y_space_direction_counts)}"
    )


def print_bluebear_seed_results(
    results,
    context,
    mse_test=None,
    y_space_items=None,
):
    """Print bluebear-style per-seed HV values without aggregate statistics."""
    if isinstance(results, dict):
        run_details = results.get("run_details", [])
    else:
        run_details = list(results)
    if y_space_items is None:
        y_space_items = evaluate_majority_shift_y_k1_diagnostic_items(
            {"run_details": run_details},
            context,
        )

    records = build_bluebear_seed_records(
        {"run_details": run_details},
        context,
        mse_test=mse_test,
        y_space_items=y_space_items,
    )
    if run_details and int(run_details[0].get("seed", -1)) == 1:
        print_shift_direction_statistics(
            {"run_details": [run_details[0]]},
            y_space_items=[y_space_items[0]] if y_space_items else None,
        )
    for record in records:
        _print_compact_seed_record(record)
    return records


def build_bluebear_seed_records(
    results,
    context,
    mse_test=None,
    y_space_items=None,
):
    if isinstance(results, dict):
        run_details = results.get("run_details", [])
    else:
        run_details = list(results)
    if y_space_items is None:
        y_space_items = evaluate_majority_shift_y_k1_diagnostic_items(
            {"run_details": run_details},
            context,
        )
    records = []
    for run_detail, y_space_item in zip(run_details, y_space_items):
        record = _compact_seed_record(
            run_detail,
            context,
            y_space_item=y_space_item,
            mse_test=mse_test,
        )
        _attach_R_indicator_to_run_detail(run_detail, record)
        records.append(record)
    return records


def _records_mean_std(records, key):
    values = np.asarray([record.get(key, np.nan) for record in records], dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(finite)), float(np.std(finite))


def _records_any_R_zero_denominator(records, improvement_key):
    marker_key = _R_zero_denominator_marker_key(improvement_key)
    return any(bool(record.get(marker_key, False)) for record in records)


def _result_method_label(method_name, optimizer_name):
    if optimizer_name is None or str(optimizer_name) == str(method_name):
        return str(method_name)
    return f"{method_name}+{optimizer_name}"


def _safe_filename_label(value):
    keep = []
    for char in str(value):
        if char.isalnum() or char in ("-", "_", "."):
            keep.append(char)
        elif char in ("+", " "):
            keep.append("_")
    filename = "".join(keep).strip("._")
    return filename or "method"


def _result_summary_table(rows):
    import pandas as pd

    table = pd.DataFrame(rows)
    for column in RESULT_TABLE_COLUMNS:
        if column not in table.columns:
            table[column] = ""
    return table.loc[:, list(RESULT_TABLE_COLUMNS)]


def write_result_summary_temp(rows, output_dir, method_name):
    output_dir = Path(output_dir)
    temp_dir = output_dir / RESULT_TEMP_DIRNAME
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / f"{_safe_filename_label(method_name)}_temp_result_records.csv"
    table = _result_summary_table(rows)
    table.to_csv(temp_path, index=False)
    print(f"Wrote temporary result records to: {temp_path}")
    return temp_path


def _read_result_summary_temp_tables(output_dir):
    import pandas as pd

    temp_dir = Path(output_dir) / RESULT_TEMP_DIRNAME
    if not temp_dir.exists():
        return []

    tables = []
    for temp_path in sorted(temp_dir.glob("*_temp_result_records.csv")):
        try:
            temp_table = pd.read_csv(temp_path, dtype=str).fillna("")
        except Exception as err:
            print(
                "Temporary result records read skipped: "
                f"{temp_path}: {type(err).__name__}: {err}"
            )
            continue
        tables.append(_result_summary_table(temp_table))
    return tables


def _plot_sur_real_filename_stem(problem_name, method_name):
    prefix = os.environ.get("HV_TEST_SVG_PREFIX", "negative1").strip()
    method_label = str(method_name or "method")
    replacements = {
        "Dual-Ranking+NSGA-II": "NSGA-II_DR",
        "Dual-Ranking+MOEAD": "MOEAD_DR",
        "Dual-Ranking+SMS-EMOA": "SMS-EMOA_DR",
        "Prob-MOEA/D": "Prob-MOEA-D",
    }
    for old, new in replacements.items():
        method_label = method_label.replace(old, new)
    parts = [
        _safe_filename_label(prefix) if prefix else "",
        _safe_filename_label(problem_name or "problem"),
        _safe_filename_label(method_label),
    ]
    return "_".join(part for part in parts if part)


def _format_mean_std_cell(mean_value, std_value, digits=3, notation="fixed"):
    if not np.isfinite(mean_value):
        return "nan"
    number_format = f".{int(digits)}e" if notation == "scientific" else f".{int(digits)}f"
    if not np.isfinite(std_value):
        return f"{float(mean_value):{number_format}}±nan"
    return f"{float(mean_value):{number_format}}±{float(std_value):{number_format}}"


def _format_percent_mean_std_cell(
    mean_value,
    std_value,
    digits=1,
    zero_denominator=False,
):
    if not np.isfinite(mean_value):
        return "NA%"
    marker = "*" if zero_denominator else ""
    if not np.isfinite(std_value):
        return f"{float(mean_value):.{digits}f}±nan%{marker}"
    return f"{float(mean_value):.{digits}f}±{float(std_value):.{digits}f}%{marker}"


def _wilcoxon_pvalue(values_a, values_b):
    values_a = np.asarray(values_a, dtype=float)
    values_b = np.asarray(values_b, dtype=float)
    valid = np.isfinite(values_a) & np.isfinite(values_b)
    if int(np.sum(valid)) < 2:
        return float("nan")
    try:
        from scipy.stats import wilcoxon

        return float(wilcoxon(values_a[valid], values_b[valid], alternative="two-sided").pvalue)
    except Exception:
        return float("nan")


def summarize_result_records(
    exp_name,
    method_name,
    problem_name,
    optimizer_name,
    records,
    output_dir=None,
):
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    method_label = _result_method_label(method_name, optimizer_name)
    row = {column: "" for column in RESULT_TABLE_COLUMNS}
    row["timestamp"] = timestamp
    row["method"] = method_label
    row["Problem"] = problem_name
    print(
        f"\n=== {exp_name} | {method_name} | {problem_name} | {optimizer_name} ==="
    )

    means = {}
    stds = {}
    for key in RESULT_METRIC_KEYS:
        means[key], stds[key] = _records_mean_std(records, key)

    mse_summary_keys = (
        "MSE_test",
        "MSE_sur",
        "MSE_clip",
        "MSE_xy_shift",
        "MSE_x_shift",
        "MSE_y_shift",
        "MSE_no_clip",
    )
    for key in mse_summary_keys:
        row[key] = _format_mean_std_cell(
            means[key],
            stds[key],
            digits=3,
            notation="scientific",
        )
        print(f"{key}: {row[key]}")

    igd_summary_keys = (
        "IGD+_real",
        "IGD+_sur",
        "IGD+_clip",
        "IGD+_xy_shift",
        "IGD+_x_shift",
        "IGD+_y_shift",
        "IGD+_no_clip",
    )
    for key in igd_summary_keys:
        row[key] = _format_mean_std_cell(
            means[key],
            stds[key],
            digits=3,
            notation="scientific",
        )
        print(f"{key}: {row[key]}")

    for key in (
        "HV_sur",
        "HV_real",
    ):
        row[key] = _format_mean_std_cell(means[key], stds[key], digits=3)
        print(f"{key}: {row[key]}")

    for metric_key, column in (
        ("MSE_clip_improvement", "MSE_clip (improvement)"),
        ("MSE_xy_shift_improvement", "MSE_xy_shift (improvement)"),
        ("MSE_x_shift_improvement", "MSE_x_shift (improvement)"),
        ("MSE_y_shift_improvement", "MSE_y_shift (improvement)"),
        ("MSE_no_clip_improvement", "MSE_no_clip (improvement)"),
        ("IGD+_clip_improvement", "IGD+_clip (improvement)"),
        ("IGD+_xy_shift_improvement", "IGD+_xy_shift (improvement)"),
        ("IGD+_x_shift_improvement", "IGD+_x_shift (improvement)"),
        ("IGD+_y_shift_improvement", "IGD+_y_shift (improvement)"),
        ("IGD+_no_clip_improvement", "IGD+_no_clip (improvement)"),
    ):
        improvement_mean, improvement_std = _records_mean_std(records, metric_key)
        row[column] = _format_percent_mean_std_cell(
            improvement_mean,
            improvement_std,
            zero_denominator=_records_any_R_zero_denominator(records, metric_key),
        )
        print(f"{column}: {row[column]}")

    for key, label in RESULT_VARIANT_LABELS.items():
        improvement_key = f"{key}_improvement"
        improvement_mean, improvement_std = _records_mean_std(records, improvement_key)
        improvement_text = _format_percent_mean_std_cell(
            improvement_mean,
            improvement_std,
            zero_denominator=_records_any_R_zero_denominator(records, improvement_key),
        )
        row[label] = _format_mean_std_cell(means[key], stds[key])
        row[f"{label} (improvement)"] = improvement_text
        print(f"{label}: {row[label]} ({improvement_text})")

    pvalue = _wilcoxon_pvalue(
        [record["HV_xy_shift"] for record in records],
        [record["HV_sur"] for record in records],
    )
    row["pvalue_HV_xy_shift_vs_HV_sur"] = (
        "nan" if not np.isfinite(pvalue) else f"{pvalue:.3f}"
    )
    print(
        "Wilcoxon signed-rank (HV_xy_shift vs HV_sur, two-sided) "
        f"p-value: {pvalue:.6g}"
    )
    return row


def _result_subset(table, columns):
    for column in columns:
        if column not in table.columns:
            table[column] = ""
    return table.loc[:, list(columns)]


def _upsert_result_table(existing_table, new_table, columns):
    import pandas as pd

    if existing_table is None:
        table = pd.DataFrame(columns=list(columns))
    else:
        table = existing_table.copy().fillna("")
    for column in columns:
        if column not in table.columns:
            table[column] = ""
        if column not in new_table.columns:
            new_table[column] = ""
    table = table.loc[:, list(columns)]
    new_table = new_table.loc[:, list(columns)]

    for _, new_row in new_table.iterrows():
        method = str(new_row["method"])
        problem = str(new_row["Problem"])
        match = (table["method"].astype(str) == method) & (
            table["Problem"].astype(str) == problem
        )
        if match.any():
            first_idx = table.index[match][0]
            for column in columns:
                table.at[first_idx, column] = new_row[column]
        else:
            table = pd.concat(
                [table, pd.DataFrame([new_row.to_dict()])],
                ignore_index=True,
            )
    return table


def write_result_workbook(table, output_dir):
    import pandas as pd

    output_dir = Path(output_dir)
    xlsx_path = output_dir / RESULT_SUMMARY_XLSX_FILENAME
    main_new = _result_subset(table.copy(), MAIN_RESULT_COLUMNS)
    ablation_new = _result_subset(table.copy(), ABLATION_RESULT_COLUMNS)

    existing_sheets = {}
    if xlsx_path.exists():
        try:
            existing_sheets = pd.read_excel(
                xlsx_path,
                sheet_name=None,
                dtype=str,
            )
        except Exception as err:
            print(
                "Existing result workbook read skipped: "
                f"{type(err).__name__}: {err}. Rewriting workbook."
            )

    main_table = _upsert_result_table(
        existing_sheets.get("main"),
        main_new,
        MAIN_RESULT_COLUMNS,
    )
    ablation_table = _upsert_result_table(
        existing_sheets.get("ablation"),
        ablation_new,
        ABLATION_RESULT_COLUMNS,
    )

    with pd.ExcelWriter(xlsx_path) as writer:
        main_table.to_excel(writer, sheet_name="main", index=False)
        ablation_table.to_excel(writer, sheet_name="ablation", index=False)
    print(f"Wrote result workbook to: {xlsx_path}")
    return xlsx_path


def append_result_summary_outputs(rows, output_dir, title="Result summary"):
    import pandas as pd

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table = _result_summary_table(rows)
    temp_tables = _read_result_summary_temp_tables(output_dir)
    if temp_tables:
        table = _result_summary_table(pd.concat([*temp_tables, table], ignore_index=True))
    write_result_workbook(table, output_dir)
    return table


def _print_selected_nearest_neighbor_distances(
    run_detail,
    item,
    x_info,
    selected_solution_indices,
):
    f_sur = np.asarray(run_detail["obj"], dtype=float)
    objective_nearest_idx = item["nearest_idx"]
    f_sur_to_y_pred_distance = item["f_sur_to_y_pred_distance"]
    y_pred_to_y_true_distance = item["y_pred_to_y_true_distance"]
    x_nearest_idx = x_info["nearest_idx"]
    nearest_normalized_x_distance = x_info["nearest_normalized_x_distance"]
    x_nearest_y_pred_to_y_true_distance = x_info["nearest_y_pred_to_y_true_distance"]
    print("Selected nearest-neighbor distances:")
    for selected_idx in selected_solution_indices:
        if selected_idx < 0 or selected_idx >= f_sur.shape[0]:
            continue
        objective_nn = int(objective_nearest_idx[selected_idx])
        x_nn = int(x_nearest_idx[selected_idx])
        print(f"selected solution {selected_idx}")
        print(f"    y-space nearest offline idx={objective_nn}")
        print(
            "    normalized ||f_sur - offline_y_pred||="
            f"{f_sur_to_y_pred_distance[selected_idx]:.6e}"
        )
        print(
            "    normalized ||offline_y_pred - offline_y_true||="
            f"{y_pred_to_y_true_distance[selected_idx]:.6e}"
        )
        print(f"    x-space nearest offline idx={x_nn}")
        print(
            "    normalized ||x - offline_x||="
            f"{nearest_normalized_x_distance[selected_idx]:.6e}"
        )
        print(
            "    normalized ||offline_y_pred - offline_y_true||="
            f"{x_nearest_y_pred_to_y_true_distance[selected_idx]:.6e}"
        )


def _print_majority_shift_y_k1_diagnostic_summary(
    problem_name,
    optimizer_name,
    results,
    items,
    context,
    selected_solution_indices,
):
    latest_x_space_info = _x_space_nearest_distance_info(results["run_details"][-1], context)
    print(
        f"=== {problem_name} | {optimizer_name} | normalized nearest-neighbor distances ==="
    )
    _print_selected_nearest_neighbor_distances(
        results["run_details"][-1],
        items[-1],
        latest_x_space_info,
        selected_solution_indices,
    )
    print()


def plot_nearest_offline_diagnostics(
    problem_name,
    problem_results,
    context,
    selected_solution_indices=(10, 30, 80),
    diagnostic_items_by_optimizer=None,
    show=True,
):
    """Print and plot y-space k=1 majority-shift diagnostics for one problem.

    ``problem_results`` can be either one optimizer's result dict or a mapping
    from optimizer name to result dict.
    """
    if "run_details" in problem_results:
        optimizer_items = [(None, problem_results)]
    else:
        optimizer_items = list(problem_results.items())

    selected_solution_indices = tuple(int(idx) for idx in selected_solution_indices)
    diagnostic_items_by_optimizer = diagnostic_items_by_optimizer or {}
    diagnostic_results = {}
    for optimizer_name, results in optimizer_items:
        run_details = results.get("run_details", [])
        if not run_details:
            print(f"{problem_name} | {optimizer_name}: no run details for diagnostics.")
            diagnostic_results[optimizer_name] = []
            continue

        items = diagnostic_items_by_optimizer.get(optimizer_name)
        if items is None:
            items = evaluate_majority_shift_y_k1_diagnostic_items(
                results,
                context,
            )
        diagnostic_results[optimizer_name] = items
        _print_majority_shift_y_k1_diagnostic_summary(
            problem_name,
            optimizer_name,
            results,
            items,
            context,
            selected_solution_indices,
        )

        valid_selected_indices = tuple(
            selected_idx
            for selected_idx in selected_solution_indices
            if 0 <= selected_idx < run_details[-1]["solution"].shape[0]
        )
        if len(valid_selected_indices) != 3:
            print(
                "Skip combined y-space diagnostic plot: expected solution indices "
                f"{selected_solution_indices}, got {valid_selected_indices}."
            )
            continue

        _plot_selected_solutions_nearest_offline_hv_majority_shift_2d(
            x=run_details[-1]["solution"],
            f_sur=run_details[-1]["obj"],
            f_real=run_details[-1]["f_real"],
            x_train=context["X_train"],
            y_train=context["y_train"],
            f_train_pred=context["f_train_mean"],
            majority_shift_result=items[-1]["result"],
            majority_shift_reference_result=items[-1]["result"],
            selected_indices=valid_selected_indices,
            k=1,
            distance_space="y",
            distance_metric="normalized_euclidean",
            title=(
                f"{problem_name} {optimizer_name}: f_sur nearest offline_y_pred, "
                "k=1, and paired offline_y_true"
            ),
            show=show,
            show_x_space_nearest=True,
        )

    return diagnostic_results


def plot_hv_history(
    results,
    title="HV over Generations",
    figsize=(7, 6),
    line_width=2.0,
    marker_size=4,
    tick_fontsize=12,
    label_fontsize=12,
    title_fontsize=14,
    legend_fontsize=11,
    show_plot=True,
    save_svg=True,
    svg_path="hv_curve.svg",
    show_legend=True,
    show_axis_labels=True,
    x_label="Generation",
    y_label="HV",
    xlim=(1, 100),
    ylim=(0, 1.3)
):
    gen_list = np.asarray(results["gen_history"])
    hv_sur_list = np.asarray(results["hv_sur_history"], dtype=float)
    hv_real_list = np.asarray(results["hv_real_history"], dtype=float)

    fig, ax = plt.subplots(figsize=figsize)

    sur_label = "HV surrogate" if show_legend else None
    real_label = "HV real" if show_legend else None

    ax.plot(
        gen_list,
        hv_sur_list,
        marker='s',
        color="#1565C0",
        linewidth=line_width,
        markersize=marker_size,
        label=sur_label
    )

    ax.plot(
        gen_list,
        hv_real_list,
        marker='s',
        color="#D55E00",
        linewidth=line_width,
        markersize=marker_size,
        label=real_label
    )

    if show_axis_labels:
        ax.set_xlabel(x_label, fontsize=label_fontsize)
        ax.set_ylabel(y_label, fontsize=label_fontsize)

    if title is not None:
        ax.set_title(title, fontsize=title_fontsize)

    if xlim is not None:
        ax.set_xlim(xlim)

    if ylim is not None:
        ax.set_ylim(ylim)

    ax.tick_params(axis='both', labelsize=tick_fontsize)

    if show_legend:
        ax.legend(fontsize=legend_fontsize)

    plt.tight_layout()

    if save_svg:
        plt.savefig(svg_path, format="svg", bbox_inches="tight")
        print(f"Figure saved as SVG: {svg_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)
