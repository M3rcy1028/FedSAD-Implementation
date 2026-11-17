"""Run the selected source TAAE version(s) on a frozen GeNIS split."""
from datetime import datetime
import importlib
import inspect
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
from zoneinfo import ZoneInfo

from utils.genis import file_sha256, prepare_genis

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = [
    "train_taae.py", "taae_current.py", "taae_previous.py", "utils/__init__.py", "utils/arguments.py", "utils/metrics.py",
    "utils/genis.py", "utils/genis_experiment.py", "utils/genis_outputs.py", "utils/taae_checkpoint.py",
    "utils/gotham.py", "utils/gotham_experiment.py", "utils/gotham_runtime.py",
    "utils/threshold_search.py",
]


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str) + "\n")


def runtime_environment():
    for key, value in {
        "TF_FORCE_GPU_ALLOW_GROWTH": "true", "MPLBACKEND": "Agg",
        "TF_NUM_INTRAOP_THREADS": "2", "TF_NUM_INTEROP_THREADS": "1",
        "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "1",
    }.items():
        os.environ.setdefault(key, value)


def frozen_sources(destination):
    destination.mkdir(parents=True)
    hashes = {}
    for name in SOURCE_FILES:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PROJECT_ROOT / name, target)
        hashes[name] = file_sha256(target)
    return hashes


def score(model, x):
    import numpy as np
    predicted = model.predict(x, batch_size=512, verbose=0)
    errors = np.mean(np.square(x - predicted), axis=1)
    if not np.isfinite(errors).all():
        raise ValueError("Nonfinite reconstruction errors")
    return errors


def metric_report(labels, errors, threshold):
    import numpy as np
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
    predicted = (errors > threshold).astype(int)
    return {
        "threshold": float(threshold), "roc_auc": float(roc_auc_score(labels, errors)),
        "classification_report": classification_report(
            labels, predicted, labels=[0, 1], target_names=["Normal", "Anomaly"],
            output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(labels, predicted, labels=[0, 1]).tolist(),
    }


def analyze_thresholds(output, data_path, percentile=82, dataset_name="GeNIS"):
    """Select on validation; independently report test and exploratory test optima."""
    import numpy as np
    import pandas as pd
    from utils.threshold_search import threshold_metrics, best_row
    output = Path(output)
    with np.load(output / "scores.npz") as z:
        train, val, test, y_val, y_test = [z[key] for key in
            ["train_errors", "val_errors", "test_errors", "y_val", "y_test"]]
    grid = np.unique(np.r_[np.round(np.linspace(0, 100, 10001), 2), percentile])
    val_rows = threshold_metrics(train, val, y_val, grid)
    test_rows = threshold_metrics(train, test, y_test, grid)
    pd.DataFrame(val_rows).to_csv(output / "validation_percentile_metrics.csv", index=False)
    pd.DataFrame(test_rows).to_csv(output / "test_percentile_metrics_exploratory.csv", index=False)
    selections = {
        "baseline": next(row for row in val_rows if row["percentile"] == percentile),
        "validation_best_macro_f1": best_row(val_rows, "macro_f1"),
        "validation_best_accuracy": best_row(val_rows, "accuracy"),
        "test_best_macro_f1_exploratory": best_row(test_rows, "macro_f1"),
        "test_best_accuracy_exploratory": best_row(test_rows, "accuracy"),
    }
    eligible = [row for row in val_rows if row["normal_recall"] >= .8]
    if eligible:
        selections["validation_best_normal_precision_recall80"] = best_row(eligible, "normal_precision")
    summary = {"selection_metric": "validation macro F1", "step": .01,
               "test_used_for_primary_threshold_selection": False, "settings": {}}
    per_attack = []
    with np.load(Path(data_path) / "source_rows.npz") as z:
        attack_labels = z["test_labels"]
        test_ids = z["test_ids"]
    for name, row in selections.items():
        metrics = metric_report(y_test, test, row["threshold"])
        summary["settings"][name] = {"percentile": row["percentile"],
                                     "selection_metrics": row, "test_metrics": metrics}
        predictions = (test > row["threshold"]).astype(int)
        for label in sorted(set(attack_labels[y_test == 1])):
            mask = attack_labels == label
            per_attack.append({"setting": name, "attack": label, "support": int(mask.sum()),
                               "recall": float(predictions[mask].mean())})
        if name == "validation_best_macro_f1":
            pd.DataFrame({"source_id": test_ids, "attack": attack_labels, "y_true": y_test,
                          "score": test, "y_pred": predictions}).to_csv(output / "predictions.csv", index=False)
            dump(output / "selected_threshold.json", {"percentile": row["percentile"],
                "threshold": row["threshold"], "selection_set": "validation", "test_metrics": metrics})
    pd.DataFrame(per_attack).to_csv(output / "per_attack_comparison.csv", index=False)
    dump(output / "percentile_summary.json", summary)
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, rows, title in zip(axes, [val_rows, test_rows], ["Validation", "Held-out test"]):
        for key, label in [("macro_f1", "Macro F1"), ("normal_recall", "Normal recall"),
                           ("attack_recall", "Attack recall")]:
            ax.plot(grid, [row[key] for row in rows], label=label)
        ax.axvline(selections["validation_best_macro_f1"]["percentile"], linestyle="--", color="black")
        ax.set(title=title, xlabel="Normal training-error percentile", ylabel="Score", ylim=(0, 1.02))
        ax.grid(alpha=.25); ax.legend()
    fig.tight_layout(); fig.savefig(output / "percentile_tradeoffs.png", dpi=180); plt.close(fig)
    primary = summary["settings"]["validation_best_macro_f1"]
    from sklearn.metrics import classification_report, roc_curve
    predictions = (test > primary["test_metrics"]["threshold"]).astype(int)
    text = (f"Percentile selected on validation: {primary['percentile']:.2f}\n"
            f"Threshold: {primary['test_metrics']['threshold']:.15g}\n"
            f"Test ROC AUC: {primary['test_metrics']['roc_auc']:.6f}\n\n")
    text += classification_report(y_test, predictions, labels=[0, 1],
                                  target_names=["Normal", "Anomaly"], digits=6, zero_division=0)
    (output / "final_report.txt").write_text(text)
    positive = test[test > 0]
    floor = float(positive.min() / 2) if len(positive) else 1e-12
    logged = np.log10(np.maximum(test, floor))
    bins = np.linspace(logged.min(), logged.max() + 1e-12, 80)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(logged[y_test == 0], bins=bins, density=True, histtype="step", label="Normal")
    ax.hist(logged[y_test == 1], bins=bins, density=True, histtype="step", label="Attack")
    ax.axvline(np.log10(max(primary["test_metrics"]["threshold"], floor)), linestyle="--", color="black",
               label=f"Validation-selected p={primary['percentile']:.2f}")
    ax.set(xlabel="log10(Reconstruction MSE)", ylabel="Density", title=f"{dataset_name} held-out test")
    ax.legend(); fig.tight_layout(); fig.savefig(output / "distribution.png", dpi=180); plt.close(fig)
    fpr, tpr, _ = roc_curve(y_test, test)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUC={primary['test_metrics']['roc_auc']:.4f}")
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title=f"{dataset_name} held-out test")
    ax.legend(); fig.tight_layout(); fig.savefig(output / "roc.png", dpi=180); plt.close(fig)
    return summary


def run_single(args):
    runtime_environment()
    dataset_name = "Gotham" if args.eval_dataset == "gotham" else "GeNIS"
    if dataset_name == "Gotham":
        from utils.gotham_runtime import setup_driver
        setup_driver()
    import numpy as np
    import tensorflow as tf
    import flwr as fl
    from sklearn.metrics import accuracy_score
    from utils.metrics import save_and_plot_history
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "model_config.json").exists():
        raise FileExistsError(f"Refusing to overwrite an existing model run: {output}")
    (output / "checkpoints").mkdir(exist_ok=True)
    prepared = args.genis_prepared.resolve()
    data_hash = file_sha256(prepared / "prepared_data.npz")
    with np.load(prepared / "prepared_data.npz") as z:
        train, val, test, y_val, y_test = [z[key] for key in
            ["X_train", "X_val", "X_test", "y_val", "y_test"]]
        client_ids = (z["train_client_ids"] if args.eval_dataset == "gotham"
                      and args.gotham_client_partition == "devices" else None)
    module_name = "taae_previous" if args.taae_version == "previous" else "taae_current"
    module = importlib.import_module(module_name)
    tf.keras.utils.set_random_seed(args.random_seed)
    devices = tf.config.list_physical_devices("GPU")
    for device in devices:
        tf.config.experimental.set_memory_growth(device, True)
    if args.gpu_per_client and not devices:
        raise RuntimeError("No GPU available")
    latent = args.genis_latent_dim
    if latent is None:
        latent = inspect.signature(module.TransformerAAE).parameters["latent_dim"].default
    model_kwargs = {"input_dim": train.shape[1], "latent_dim": latent, "dropout_rate": args.dropout_rate}

    def build_model():
        model = module.TransformerAAE(**model_kwargs)
        model(tf.zeros((1, train.shape[1])), prior_labels=tf.zeros((1, 1)))
        return model

    model = build_model()
    chunks = ([train[client_ids == cid] for cid in range(args.client_nums)] if client_ids is not None
              else np.array_split(train, args.client_nums))
    if any(len(chunk) < 2 for chunk in chunks):
        raise ValueError("Each federated client needs at least two normal training rows")
    source_hashes = {name: file_sha256(PROJECT_ROOT / name) for name in SOURCE_FILES}
    dump(output / "model_config.json", {
        "dataset": dataset_name, "taae_version": args.taae_version, "module": module_name,
        "input_dim": train.shape[1], "latent_dim": latent, "arguments": vars(args),
        "prepared_data": str(prepared), "prepared_data_sha256": data_hash,
        "source_sha256": source_hashes,
        "fit_implementation": f"unmodified {module_name}.FLClient.fit",
        "model_implementation": f"unmodified {module_name}.TransformerAAE",
        "pairing": "client IDs sorted, original RNEP mixing, Python random seed + round",
        "initial_parameters": "central model initialized with the common experiment seed",
        "previous_loss_note": "Previous model receives no prior_labels in fit, as in the original source; adversarial add_loss is inactive during fit",
        "test_used_for_training_or_round_evaluation": False,
        "client_partition": "device groups" if client_ids is not None else "shuffled row chunks",
        "client_training_rows": [len(chunk) for chunk in chunks],
        "client_batch_size": args.batch_size if dataset_name == "Gotham" else "source module default in Ray worker",
    })

    class SourceClient(module.FLClient):
        def fit(self, parameters, config):
            weights, count, metrics = super().fit(parameters, config)
            return weights, count, {**metrics, "local_epochs_completed": self.epochs,
                                     "partition_id": int(self.cid)}

        def evaluate(self, parameters, config):
            self.model.set_weights(parameters)
            threshold = np.percentile(score(self.model, self.X_train), args.percentile)
            errors = score(self.model, self.X_test)
            accuracy = float(accuracy_score(self.y_test, errors > threshold))
            with (output / "client_evaluations.jsonl").open("a") as stream:
                stream.write(json.dumps({"client": int(self.cid), "accuracy": accuracy,
                                         "threshold": float(threshold)}) + "\n")
            return 1 - accuracy, len(self.y_test), {}

    def client_fn(cid: str):
        cid_int = int(cid)
        if dataset_name == "Gotham":
            # The original fit() reads a module global, not a client argument.
            # Ray workers import that module with worker argv, so apply the
            # requested experiment setting explicitly in each worker.
            module.BATCH_SIZE = args.batch_size
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(args.random_seed + cid_int)
        for device in tf.config.list_physical_devices("GPU"):
            try:
                tf.config.experimental.set_memory_growth(device, True)
            except RuntimeError:
                pass
        return SourceClient(cid_int, build_model(), chunks[cid_int], val, y_val,
                            epochs=args.client_epochs, learning_rate=args.learning_rate).to_client()

    class MatchedRNEP(module.SaveEvaluationRNEP):
        def aggregate_fit(self, rnd, results, failures):
            if failures or len(results) != args.client_nums:
                raise RuntimeError(f"Round {rnd}: incomplete client fits: {len(results)}, {len(failures)} failures")
            if any(result.metrics.get("local_epochs_completed") != args.client_epochs for _, result in results):
                raise RuntimeError("A client did not finish the requested local epochs")
            results = sorted(results, key=lambda pair: pair[1].metrics["partition_id"])
            state = random.getstate()
            try:
                random.seed(args.random_seed + rnd)
                result = super().aggregate_fit(rnd, results, failures)
            finally:
                random.setstate(state)
            with (output / "round_counts.jsonl").open("a") as stream:
                order = list(range(args.client_nums))
                random.Random(args.random_seed + rnd).shuffle(order)
                stream.write(json.dumps({"round": rnd, "fits": len(results), "failures": len(failures),
                                         "epochs_per_client": args.client_epochs,
                                         "pairs": [order[i:i + 2] for i in range(0, len(order), 2)]}) + "\n")
            return result

        def evaluate(self, server_round, parameters):
            model.set_weights(fl.common.parameters_to_ndarrays(parameters))
            errors_train = score(model, train)
            errors_val = score(model, val)
            threshold = np.percentile(errors_train, args.percentile)
            metrics = metric_report(y_val, errors_val, threshold)
            with (output / "round_validation_metrics.jsonl").open("a") as stream:
                stream.write(json.dumps({"round": server_round, **metrics}) + "\n")
            if server_round:
                model.save_weights(str(output / "checkpoints" / f"round_{server_round:02d}.h5"))
            return float(errors_val.mean()), {"acc": metrics["classification_report"]["accuracy"],
                                              "roc_auc": metrics["roc_auc"]}

        def aggregate_evaluate(self, rnd, results, failures):
            if failures or len(results) != args.client_nums:
                raise RuntimeError(f"Round {rnd}: incomplete client evaluation")
            with (output / "evaluation_counts.jsonl").open("a") as stream:
                stream.write(json.dumps({"round": rnd, "evaluations": len(results), "failures": 0}) + "\n")
            return super().aggregate_evaluate(rnd, results, failures)

    strategy = MatchedRNEP(
        fraction_fit=1, fraction_evaluate=1, min_fit_clients=args.client_nums,
        min_evaluate_clients=args.client_nums, min_available_clients=args.client_nums,
        accept_failures=False,
        initial_parameters=fl.common.ndarrays_to_parameters(model.get_weights()),
    )
    print(f"[{dataset_name} {args.taae_version}] train={len(train)} val={len(val)} test={len(test)} "
          f"features={train.shape[1]} latent={latent} clients={args.client_nums} "
          f"rounds={args.server_rounds} local_epochs={args.client_epochs}", flush=True)
    ray_options = {"include_dashboard": False, "num_cpus": 40, "num_gpus": len(devices),
                   "object_store_memory": 2 * 1024 ** 3,
                   "runtime_env": {"working_dir": str(PROJECT_ROOT)}}
    if dataset_name == "Gotham":
        from utils.gotham_runtime import setup_worker
        ray_options["num_cpus"] = min(40, args.client_nums * 2)
        # Keep this dataset's temporary files on the spacious data volume.
        ray_options["_temp_dir"] = "/data/SDP_Dataset/.ray-gotham"
        ray_options["runtime_env"]["worker_process_setup_hook"] = setup_worker
        ray_options["runtime_env"]["env_vars"] = {
            "GOTHAM_ALLOWED_CPUS": os.environ["GOTHAM_ALLOWED_CPUS"], "RAY_worker_niceness": "0"}
    try:
        history = fl.simulation.start_simulation(
            client_fn=client_fn, num_clients=args.client_nums, strategy=strategy,
            config=fl.server.ServerConfig(num_rounds=args.server_rounds),
            client_resources={"num_cpus": 2, "num_gpus": args.gpu_per_client},
            ray_init_args=ray_options,
        )
    finally:
        import ray
        ray.shutdown()
    model.set_weights(fl.common.parameters_to_ndarrays(strategy.final_parameters))
    model.save_weights(str(output / "fedsad_weights.h5"))
    save_and_plot_history(history, str(output / "history.csv"), str(output / "history.png"))
    scores = {"train_errors": score(model, train), "val_errors": score(model, val),
              "test_errors": score(model, test), "y_val": y_val, "y_test": y_test}
    np.savez_compressed(output / "scores.npz", **scores)
    threshold = np.percentile(scores["train_errors"], args.percentile)
    dump(output / "baseline_metrics.json", {"percentile": args.percentile,
        **metric_report(y_test, scores["test_errors"], threshold)})
    summary = analyze_thresholds(output, prepared, args.percentile, dataset_name)
    fit_rounds = [json.loads(line) for line in (output / "round_counts.jsonl").read_text().splitlines()]
    eval_rounds = [json.loads(line) for line in (output / "evaluation_counts.jsonl").read_text().splitlines()]
    assert len(fit_rounds) == len(eval_rounds) == args.server_rounds
    assert file_sha256(prepared / "prepared_data.npz") == data_hash
    unchanged = {name: file_sha256(PROJECT_ROOT / name) == digest for name, digest in source_hashes.items()}
    assert all(unchanged.values())
    dump(output / "verification.json", {"rounds_completed": len(fit_rounds),
        "successful_client_fits": sum(row["fits"] for row in fit_rounds),
        "successful_client_evaluations": sum(row["evaluations"] for row in eval_rounds),
        "fit_failures": 0, "evaluation_failures": 0, "requested_epochs_completed": True,
        "prepared_data_unchanged": True, "source_files_unchanged": unchanged,
        "primary_percentile_selected_on_validation": True})
    from utils.genis_outputs import publish_rnep_outputs
    published = args.result_path or PROJECT_ROOT / "results" / dataset_name / (
        "taae" if args.taae_version == "previous" and dataset_name == "GeNIS" else "taae_current")
    publish_rnep_outputs(output, published)
    print(f"Published {dataset_name} outputs: {published}", flush=True)
    print(json.dumps(summary["settings"]["validation_best_macro_f1"], indent=2), flush=True)


def run_genis(args):
    if args.genis_worker:
        if args.taae_version == "both" or args.genis_prepared is None or args.output_dir is None:
            raise ValueError("A model worker needs one TAAE version, --genis_prepared and --output_dir")
        return run_single(args)
    runtime_environment()
    versions = ["previous", "current"] if args.taae_version == "both" else [args.taae_version]
    stamp = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d_%H%M%S_%f")
    model_dir = "taae" if args.taae_version == "previous" else "taae_current"
    published = (args.result_path or PROJECT_ROOT / "results" / "GeNIS" / model_dir).resolve()
    run_name = "comparison" if len(versions) == 2 else "training"
    output = args.output_dir or published / "runs" / f"{run_name}_{stamp}"
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "sources"
    hashes = frozen_sources(snapshot)
    prepared = args.genis_prepared or args.genis_dir / "processed" / f"{run_name}_{stamp}"
    prepared = prepared.resolve()
    dump(output / "status.json", {"stage": "preparing", "prepared_data": prepared})
    if args.genis_prepared is None:
        report = prepare_genis(args.genis_dir, prepared, flow_interval=args.flow_interval,
                               seed=args.random_seed, max_train_samples=args.n_samples)
    else:
        report = json.loads((prepared / "preprocessing_report.json").read_text())
        if report["seed"] != args.random_seed or report["flow_interval_seconds"] != args.flow_interval:
            raise ValueError("Prepared data do not match the requested seed/flow interval")
        if report.get("preprocessor_sha256") != hashes["utils/genis.py"]:
            raise ValueError("Prepared data do not match the frozen preprocessing source")
    dump(output / "preprocessing_report.json", report)
    dump(output / "comparison_config.json", {"arguments": vars(args), "prepared_data": prepared,
        "prepared_sha256": file_sha256(prepared / "prepared_data.npz"), "frozen_sources": hashes,
        "source_model_files_preserved": True})
    print(f"[GeNIS] Normal training={report['normal_training_rows']}, "
          f"validation={report['validation_rows']}, test={report['test_rows']}, "
          f"features={report['input_dim']}; attack classes are evaluation labels.", flush=True)
    print(f"[GeNIS RNEP] Selected TAAE version(s): {', '.join(versions)}", flush=True)
    if args.prepare_only:
        dump(output / "status.json", {"stage": "prepared", "prepared_data": prepared})
        return
    for version in versions:
        folder = output / version
        folder.mkdir()
        command = [sys.executable, "-u", str(snapshot / "train_taae.py"), "--eval_dataset", "genis",
                   "--genis_worker", "--taae_version", version,
                   "--genis_prepared", str(prepared), "--output_dir", str(folder)]
        destination = (published.with_name("taae")
                       if len(versions) == 2 and version == "previous" else published)
        command += ["--result_path", str(destination)]
        for option in ["client_nums", "server_rounds", "client_epochs", "batch_size", "learning_rate",
                       "dropout_rate", "random_seed", "percentile", "gpu_per_client"]:
            command += [f"--{option}", str(getattr(args, option))]
        if args.genis_latent_dim is not None:
            command += ["--genis_latent_dim", str(args.genis_latent_dim)]
        dump(output / "status.json", {"stage": "training", "version": version, "command": command})
        print(f"Starting {version}: log={folder / 'training.log'}", flush=True)
        with (folder / "training.log").open("w") as log:
            result = subprocess.run(command, cwd=snapshot, stdout=log, stderr=subprocess.STDOUT)
        dump(folder / "exit_status.json", {"exit_code": result.returncode})
        if result.returncode:
            dump(output / "status.json", {"stage": "failed", "version": version, "exit_code": result.returncode})
            raise RuntimeError(f"{version} experiment failed; see {folder / 'training.log'}")
    import pandas as pd
    if len(versions) == 2:
        previous_config = json.loads((output / "previous" / "model_config.json").read_text())
        current_config = json.loads((output / "current" / "model_config.json").read_text())
        assert previous_config["prepared_data_sha256"] == current_config["prepared_data_sha256"]
        previous_pairs = [json.loads(line)["pairs"] for line in (output / "previous" / "round_counts.jsonl").read_text().splitlines()]
        current_pairs = [json.loads(line)["pairs"] for line in (output / "current" / "round_counts.jsonl").read_text().splitlines()]
        assert previous_pairs == current_pairs
    comparisons = []
    for version in versions:
        model_config = json.loads((output / version / "model_config.json").read_text())
        summary = json.loads((output / version / "percentile_summary.json").read_text())
        for name, setting in summary["settings"].items():
            metrics = setting["test_metrics"]; report = metrics["classification_report"]
            comparisons.append({"model": version, "latent_dim": model_config["latent_dim"],
                "setting": name, "percentile": setting["percentile"], "threshold": metrics["threshold"],
                "accuracy": report["accuracy"], "macro_f1": report["macro avg"]["f1-score"],
                "normal_precision": report["Normal"]["precision"], "normal_recall": report["Normal"]["recall"],
                "attack_precision": report["Anomaly"]["precision"], "attack_recall": report["Anomaly"]["recall"],
                "roc_auc": metrics["roc_auc"]})
    summary_name = "model_comparison.csv" if len(versions) == 2 else "model_summary.csv"
    pd.DataFrame(comparisons).to_csv(output / summary_name, index=False)
    shutil.copy2(output / summary_name,
                 (published.parent if len(versions) == 2 else published) / summary_name)
    dump(output / "status.json", {"stage": "complete", "results": output, "prepared_data": prepared})
    print(pd.DataFrame(comparisons).to_string(index=False), flush=True)
    print(f"GeNIS RNEP completed ({', '.join(versions)}): {output}", flush=True)
