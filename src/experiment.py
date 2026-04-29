import time
import numpy as np
from sklearn.metrics import mean_squared_error
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from src.survival import Survival_standard 
from src.opt_problem import Benchmark_Problem, EvaluatePreRealCallback, evaluate_pre_real


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
    seeds):

    minimize_kwargs = dict(
      termination=get_termination("n_gen", n_gen),
      save_history=True,
      verbose=False)
    
    # Algorithm
    algorithm = NSGA2(
        pop_size=pop_size,
        crossover=SBX(prob=1.0, eta=20),
        mutation=PM(prob=1 / problem.n_var, eta=20),
        survival=survival_function,
        eliminate_duplicates=True)

    # Callback
    if use_callback:
      callback_standard = EvaluatePreRealCallback(
          true_problem=problem,
          plot_every=1,
          use_opt=True,
          dynamic_show=False,
          prefix="NSGA2-standard",
          obj_min=obj_min,
          obj_max=obj_max,
          hv_indicator=hv)
      minimize_kwargs["callback"] = callback_standard

    mse_list = []
    igd_list = []
    hv_surrogate_list = []
    hv_real_list = []
    run_details = []

    for seed in seeds:
        
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

      res = minimize(
          benchmark_problem_GPR,
          algorithm,
          seed=seed,
          **minimize_kwargs)

      end_time = time.time()

      # Final solutions
      solution = res.history[-1].opt.get("X")
      obj = res.history[-1].opt.get("F")
      f_real = problem.evaluate(solution, return_values_of=["F"])

      # MSE
      mse = mean_squared_error(f_real, obj)
      mse_list.append(mse)

      # IGD+
      igd_plus_real = float(igd_plus(f_real))
      igd_list.append(igd_plus_real)

      # HV
      f_real_normalization = (f_real - obj_min) / (obj_max - obj_min)
      obj_normalization = (obj - obj_min) / (obj_max - obj_min)

      hv_real = float(hv.do(f_real_normalization))
      hv_surrogate = float(hv.do(obj_normalization))

      hv_real_list.append(hv_real)
      hv_surrogate_list.append(hv_surrogate)

      max_obj = np.max(obj, axis=0)
      max_obj_real = np.max(f_real, axis=0)

      print(
          f"Seed {seed} | Time: {end_time - start_time:.2f}s | "
          f"MSE: {mse:.3e} | "
          f"igd+: {igd_plus_real:.3e} | "
          f"Sur HV: {hv_surrogate:.3f} | "
          f"Real HV: {hv_real:.3f} | "
          f"Max obj: {max_obj} | "
          f"Max f_real: {max_obj_real}")
 
      if use_callback:
          run_details.append({
              "seed": seed,
              "time": end_time - start_time,
              "solution": solution,
              "obj": obj,
              "f_real": f_real,
              "mse": mse,
              "igd_plus": igd_plus_real,
              "hv_surrogate": hv_surrogate,
              "hv_real": hv_real,
              "max_obj": max_obj,
              "max_f_real": max_obj_real,
              "result": res,
              "gen_history": callback_standard.gen_list,
              "hv_sur_history": callback_standard.hv_sur_list,
              "hv_real_history": callback_standard.hv_real_list})
      else:
          run_details.append({
              "seed": seed,
              "time": end_time - start_time,
              "solution": solution,
              "obj": obj,
              "f_real": f_real,
              "mse": mse,
              "igd_plus": igd_plus_real,
              "hv_surrogate": hv_surrogate,
              "hv_real": hv_real,
              "max_obj": max_obj,
              "max_f_real": max_obj_real,
              "result": res})



    return {
        "mse_list": mse_list,
        "igd_list": igd_list,
        "hv_surrogate_list": hv_surrogate_list,
        "hv_real_list": hv_real_list,
        "run_details": run_details}
