"""Bound each Gotham Ray worker before TensorFlow creates data thread pools."""
import os


def setup_driver():
    # Normal scheduling priority within the two-CPU worker budget. Ray's
    # default nice=15 otherwise starves these workers on this shared host.
    os.environ["RAY_worker_niceness"] = "0"
    if hasattr(os, "sched_getaffinity"):
        available = sorted(os.sched_getaffinity(0))
        os.environ["GOTHAM_ALLOWED_CPUS"] = ",".join(map(str, available))
        if len(available) > 4:
            start = (os.getpid() % ((len(available) + 3) // 4)) * 4
            os.sched_setaffinity(0, available[start:start + 4])


def setup_worker():
    if hasattr(os, "sched_getaffinity"):
        available = (list(map(int, os.environ["GOTHAM_ALLOWED_CPUS"].split(",")))
                     if os.environ.get("GOTHAM_ALLOWED_CPUS") else sorted(os.sched_getaffinity(0)))
        if len(available) > 2:
            slot = os.getpid() % ((len(available) + 1) // 2)
            selected = available[2 * slot:2 * slot + 2]
            os.sched_setaffinity(0, selected)
    for key, value in {"TF_NUM_INTRAOP_THREADS": "2", "TF_NUM_INTEROP_THREADS": "1",
                       "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "1"}.items():
        os.environ[key] = value
