# Auto-generated BlueBEAR Python version of the corresponding baseline notebook.
# Run on BlueBEAR from the repository root or with the server path below available.

import importlib
import importlib.util
import os
import sys
import types
from pathlib import Path


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

if "imp" not in sys.modules and importlib.util.find_spec("imp") is None:
    # pyDOE2 1.3.0 imports the removed module but does not use it for LHS.
    sys.modules["imp"] = types.ModuleType("imp")

code_path = Path("/rds/projects/w/wangsu-building-automation/Huanbo/2026_new_metric")
for repo_root in (Path.cwd().resolve(), Path.cwd().resolve().parent, code_path):
    if (repo_root / "baseline" / "batch_experiments.py").exists():
        repo_root_string = str(repo_root)
        while repo_root_string in sys.path:
            sys.path.remove(repo_root_string)
        sys.path.insert(0, repo_root_string)
        print(f"Using repository root: {repo_root}")
        break
else:
    raise FileNotFoundError("Could not locate baseline/batch_experiments.py")

importlib.invalidate_caches()
for module_name in list(sys.modules):
    if module_name == "src" or module_name.startswith("src."):
        sys.modules.pop(module_name, None)
sys.modules.pop("baseline.batch_experiments", None)
sys.modules.pop("baseline", None)
import baseline.batch_experiments as batch_experiments

print(f"Loaded baseline runner: {batch_experiments.__file__}")
run_tgpr_mo_suite = batch_experiments.run_tgpr_mo_suite

bluebear_config_path = Path(__file__).resolve().parent / "config.yaml"
all_results = run_tgpr_mo_suite(config_path=bluebear_config_path)


# Per-seed bluebear-style results are printed by the suite.
# Aggregate gap-improvement table printing is skipped for single-seed runs.
gap_improvement_table = None
