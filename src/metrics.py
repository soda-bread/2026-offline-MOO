import numpy as np
from pymoo.indicators.hv import HV
from pymoo.indicators.igd_plus import IGDPlus
from sklearn.metrics import mean_squared_error


def get_problem_y_bounds(problem_name, n_var=None):
    problem_name = str(problem_name).strip().lower()

    if problem_name == 'dtlz1':
        obj_min = np.array([0,0])
        obj_max = np.array([700,700])
    elif problem_name == 'dtlz2':
        obj_min = np.array([0,0])
        obj_max = np.array([2.78,2.93])
    elif problem_name == 'dtlz3':
        obj_min = np.array([0,0])
        obj_max = np.array([1605.54,1670.48])
    elif problem_name == 'dtlz4':
        obj_min = np.array([0,0])
        obj_max = np.array([2.83,2.78])
    elif problem_name == 'dtlz5':
        obj_min = np.array([0,0])
        obj_max = np.array([2.61,2.70])
    elif problem_name == 'dtlz6':
        obj_min = np.array([0,0])
        obj_max = np.array([9.78,9.78])
    elif problem_name == 'dtlz7':
        obj_min = np.array([0,0])
        obj_max = np.array([1.10,33.43])
    elif problem_name == 'omnitest':
        obj_min = np.array([-2,-2])
        obj_max = np.array([2.4,2.4])
    elif problem_name == 'truss2d':
        obj_min = np.array([0,0])
        obj_max = np.array([0.06,1.5e10])
    elif problem_name == 'welded_beam':
        obj_min = np.array([0,-200])
        obj_max = np.array([300,160])
    elif problem_name == 'cs1':
        obj_min = np.array([0,-300])
        obj_max = np.array([2,-10])
    elif problem_name == 'ct1':
        obj_min = np.array([0,-5])
        obj_max = np.array([2,1])
    else:
        obj_min, obj_max = None, None

    return obj_min, obj_max


def get_problem_clip_min(problem_name, n_var=None):
    problem_name = str(problem_name).strip().lower()

    if problem_name == 'welded_beam':
        return np.array([0,0])

    obj_min, _ = get_problem_y_bounds(problem_name, n_var=n_var)
    return obj_min


def _as_non_empty_pareto_front(pf):
    if pf is None:
        return None
    pf = np.asarray(pf, dtype=float)
    if pf.size == 0:
        return None
    if pf.ndim == 1:
        pf = pf.reshape(-1, 1)
    return pf


def _omnitest_pareto_front(problem, n_var, n_points=1000):
    n_var = int(n_var if n_var is not None else problem.n_var)
    t = np.linspace(0.0, 0.5, int(n_points))
    X_opt = np.tile((1.0 + t).reshape(-1, 1), (1, n_var))
    return problem.evaluate(X_opt)


def get_metrics(problem_name, problem, n_var=None, n_obj=None):
    problem_name = str(problem_name).strip().lower()
    if n_var is None and hasattr(problem, "n_var"):
        n_var = problem.n_var
    if n_obj is None and hasattr(problem, "n_obj"):
        n_obj = problem.n_obj

    # Metrics: HV
    obj_min, obj_max = get_problem_y_bounds(problem_name, n_var=n_var)

    ref_point = np.array([1.1,1.1])
    hv = HV(ref_point=ref_point)
    
    # Metrics: IGD+
    n_points = 200
    if problem_name == 'dtlz5':
        X_opt = np.full((n_points, n_var), 0.5)
        X_opt[:, 0] = np.linspace(0, 1, n_points)
        pf = problem.evaluate(X_opt)
    elif problem_name == 'dtlz6':
        X_opt = np.zeros((n_points, n_var))
        X_opt[:, 0] = np.linspace(0, 1, n_points)
        pf = problem.evaluate(X_opt)
    elif problem_name == 'dtlz7':
        X_opt = np.zeros((n_points, n_var))
        X_opt[:, :n_obj-1] = np.linspace(0, 1, n_points).reshape(-1, 1)
        pf = problem.evaluate(X_opt)
    elif problem_name == 'omnitest':
        pf = _omnitest_pareto_front(problem, n_var=n_var, n_points=max(n_points, 1000))
    else:
        pf = problem.pareto_front()

    pf = _as_non_empty_pareto_front(pf)
    if pf is None:
        try:
            pf = _as_non_empty_pareto_front(problem.pareto_front(max(n_points, 1000)))
        except TypeError:
            pf = None
    if pf is None:
        raise ValueError(
            f"Could not build a non-empty Pareto front for problem '{problem_name}'."
        )

    if obj_min is None or obj_max is None:
        raise ValueError(
            f"Objective bounds are not configured for problem '{problem_name}'."
        )
    
    igd_plus = IGDPlus(pf)
    
    return hv, igd_plus, obj_min, obj_max, ref_point
