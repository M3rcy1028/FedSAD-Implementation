"""Coordinate immutable source snapshots, training and final artifacts."""
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from zoneinfo import ZoneInfo

from experiment.artifacts import ROOT, dump, now, paths, sha256, writable_output


def environment(config):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1" if config["gpus"] == "cpu" else config["gpus"]
    for key, value in {"MPLBACKEND": "Agg", "TF_FORCE_GPU_ALLOW_GROWTH": "true",
                       "TF_NUM_INTRAOP_THREADS": "2", "TF_NUM_INTEROP_THREADS": "1",
                       "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "1",
                       "PYTHONDONTWRITEBYTECODE": "1", "RAY_worker_niceness": "0"}.items():
        os.environ[key] = value


def worker(config):
    environment(config)
    output = Path(config["output"])
    try:
        dump(output / "status.json", {"stage": "preparing", "started_at": now(), "pid": os.getpid()})
        if config["model"] in {"mvids", "mmfewshotsids"}:
            from experiment.trainers.graph import train
            metrics = train(config)
        elif config["model"] == "fedsad":
            from experiment.trainers.fedsad import train
            metrics = train(config)
        else:
            from experiment.datasets.registry import prepare
            from experiment.trainers.federated import train
            from experiment.evaluation.export import evaluate
            bundle = prepare(config)
            print(f"DATA train={bundle['X_train'].shape}, val={bundle['X_val'].shape}, test={bundle['X_test'].shape}", flush=True)
            model = train(config, bundle)
            metrics = evaluate(model, bundle, config)
        dump(output / "status.json", {"stage": "complete", "finished_at": now(),
             "clients": config["clients"], "rounds": config["rounds"], "epochs": config["epochs"],
             "accuracy": metrics["classification_report"]["accuracy"]})
    except Exception as error:
        dump(output / "status.json", {"stage": "failed", "error": str(error), "updated_at": now()})
        raise


def launch(config):
    output, runtime = paths(config)
    config = {**config, "output": str(output), "runtime": str(runtime),
              "cache_root": str(ROOT / "runtime/cache"), "project_root": str(ROOT), "created_at": now()}
    config = json.loads(json.dumps(config, default=str))
    dump(output / "config.json", config)
    dump(output / "status.json", {"stage": "queued", "created_at": now()})
    source = runtime / "source"
    source.mkdir()
    shutil.copy2(ROOT / "run.py", source / "run.py")
    shutil.copytree(ROOT / "experiment", source / "experiment", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(ROOT / "configs", source / "configs")
    hashes = {str(path.relative_to(source)): sha256(path) for path in source.rglob("*.py")}
    dump(output / "metadata/source.json", hashes)
    logs = runtime / "logs"
    logs.mkdir()
    command = [sys.executable, "-u", "-B", str(source / "run.py"), "--worker", str(output / "config.json")]
    dump(output / "metadata/launch.json", {"command": command, "log": str(logs / "train.log")})
    print(f"OUTPUT: {output}\nLOG: {logs / 'train.log'}", flush=True)
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    with (logs / "train.log").open("w") as stream:
        process = subprocess.Popen(command, cwd=source, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            stream.write(line); stream.flush()
            print(line, end="", flush=True)
        code = process.wait()
    if code:
        raise RuntimeError(f"Experiment failed ({code}); see {logs / 'train.log'}")
    for name, expected in hashes.items():
        if sha256(source / name) != expected:
            raise RuntimeError(f"Source snapshot changed during training: {name}")
    return output


def dispatch(config):
    mode = config["mode"]
    if mode == "train":
        runs = []
        for repeat in range(config["repeats"]):
            for count in config["clients"]:
                name = config.get("name")
                if name and len(config["clients"]) > 1:
                    name = f"{name}_clients{count}"
                if name and config["repeats"] > 1:
                    name = f"{name}_seed{config['seed'] + repeat}"
                run_config = {**config, "clients": count, "name": name, "seed": config["seed"] + repeat}
                runs.append(launch(run_config))
        if len(runs) > 1:
            compare(config, runs)
    elif mode == "plot":
        import pandas as pd
        from experiment.plotting.figures import render
        run = writable_output(config["run"])
        saved = json.loads((run / "config.json").read_text())
        render(run, pd.read_csv(run / "tables/predictions.csv"), saved)
        print(f"Plots updated: {run / 'figures'}")
    elif mode == "evaluate":
        evaluate_saved(config)
    else:
        compare(config, config["runs"])


def compare(config, runs):
    from experiment.plotting.figures import compare as make_comparison
    name = config.get("name") or datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d_%H%M%S_%f")
    if Path(name).name != name:
        raise ValueError("Invalid comparison name")
    target = writable_output(Path(config["output_root"]) / "comparisons" / name)
    make_comparison(runs, target)
    print(f"Comparison: {target}")


def evaluate_saved(request):
    import numpy as np
    from experiment.models.registry import build
    from experiment.evaluation.export import evaluate
    source = Path(request["run"]).expanduser().resolve()
    config = json.loads((source / "config.json").read_text())
    provided = set(request.get("provided_options", []))
    if "percentile" in provided:
        config["percentile"] = request["percentile"]
        config["threshold_policy"] = "fixed"
    if "threshold_policy" in provided:
        config["threshold_policy"] = request["threshold_policy"]
    environment({**config, "gpus": request["gpus"]})
    if config["model"] in {"mvids", "mmfewshotsids"}:
        from experiment.trainers.graph import evaluate_saved as graph_evaluate
        return graph_evaluate(source, request)
    cache = Path(config["prepared_cache"])
    if not cache.exists() or sha256(cache) != config["prepared_cache_sha256"]:
        raise ValueError("The saved run's exact runtime data cache is missing or changed")
    with np.load(cache) as data:
        bundle = {key: data[key] for key in data.files}
    config["name"] = request.get("name")
    config["output_root"] = request["output_root"]
    output, runtime = paths(config)
    config.update(output=str(output), runtime=str(runtime), evaluated_from=str(source))
    dump(output / "config.json", config)
    shutil.copy2(source / "preprocessing.joblib", output / "preprocessing.joblib")
    shutil.copy2(source / "metadata/data.json", output / "metadata/data.json")
    model = build(config, tuple(config["input_shape"]))
    model.load_weights(str(source / "weights/model.h5"))
    shutil.copy2(source / "weights/model.h5", output / "weights/model.h5")
    evaluate(model, bundle, config)
    dump(output / "status.json", {"stage": "complete", "evaluated_from": str(source), "finished_at": now()})
    print(f"Evaluation: {output}")
