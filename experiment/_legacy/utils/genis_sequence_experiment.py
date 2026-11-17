"""GeNIS entry points using the existing AE-LSTM and CNN-LSTM fit methods."""
from datetime import datetime
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from zoneinfo import ZoneInfo

from utils.genis import file_sha256
from utils.genis_sequences import DEFAULT_PREPARED

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCES = ["utils/__init__.py", "utils/arguments.py", "utils/genis.py", "utils/genis_sequences.py",
           "utils/genis_outputs.py", "utils/genis_sequence_experiment.py", "utils/metrics.py",
           "AE_LSTM.py", "CNN_LSTM.py", "models/model_aelstm.py", "models/model_cnnlstm.py"]


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str) + "\n")


def make_model(kind, timesteps, features):
    if kind == "aelstm":
        from models.model_aelstm import AE_LSTM
        return AE_LSTM(input_dim=features, timesteps=timesteps, features=features)
    from models.model_cnnlstm import CNN_LSTM
    return CNN_LSTM(timesteps=timesteps, features=features)


def probabilities(model, data, kind, batch_size=512):
    import numpy as np
    values = model.predict(data, batch_size=batch_size, verbose=0)
    if kind == "aelstm":
        return values["pred"].reshape(-1), np.mean(np.square(data - values["decoded"]), axis=(1, 2))
    return np.asarray(values).reshape(-1), None


def evaluate_genis_sequence(model, kind, prepared, destination, percentile=82, batch_size=512):
    import numpy as np
    from utils.genis_outputs import save_detection_outputs
    prepared, destination = Path(prepared), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with np.load(prepared / "prepared_data.npz") as z:
        train, y_train, test, labels = [z[k] for k in ["X_train", "y_train", "X_test", "y_test"]]
    with np.load(prepared / "source_rows.npz") as z:
        attacks, ids = z["test_labels"], z["test_ids"]
    scores, errors = probabilities(model, test, kind, batch_size)
    classification = save_detection_outputs(destination, labels, scores, .5, attacks, ids,
                                            tag="cls" if kind == "aelstm" else "",
                                            score_kind="classifier_probability",
                                            model_name="AE-LSTM" if kind == "aelstm" else "CNN-LSTM")
    result = {"classification": classification}
    saved_scores = {"y_test": labels, "classifier_probability": scores}
    if kind == "aelstm":
        _, normal_errors = probabilities(model, train[y_train == 0], kind, batch_size)
        threshold = float(np.percentile(normal_errors, percentile))
        result["reconstruction"] = save_detection_outputs(
            destination, labels, errors, threshold, attacks, ids, tag="recon", percentile=percentile,
            model_name="AE-LSTM")
        saved_scores.update(normal_train_errors=normal_errors, test_errors=errors)
    np.savez_compressed(destination / "GeNIS_scores.npz", **saved_scores)
    dump(destination / "evaluation.json", result)
    return result


def evaluate_saved_genis(kind, weights, destination=None, percentile=None, batch_size=512):
    import tensorflow as tf
    weights = Path(weights)
    config = json.loads((weights.parent / "model_config.json").read_text())
    if config["model"] != kind:
        raise ValueError(f"Saved weights are for {config['model']}, not {kind}")
    if file_sha256(Path(config["sequence_data_path"]) / "prepared_data.npz") != config["prepared_data_sha256"]:
        raise ValueError("The saved preprocessing cache changed after training")
    model = make_model(kind, config["timesteps"], config["features"])
    model(tf.zeros((1, config["timesteps"], config["features"])))
    model.load_weights(str(weights))
    return evaluate_genis_sequence(model, kind, config["sequence_data_path"], destination or weights.parent,
                                   config["reconstruction_percentile"] if percentile is None else percentile,
                                   batch_size)


def run_worker(args, kind):
    import numpy as np
    import tensorflow as tf
    import flwr as fl
    from utils.metrics import save_and_plot_history
    from sklearn.metrics import classification_report, roc_auc_score
    output, published = args.output_dir.resolve(), args.result_path.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "model_config.json").exists():
        raise FileExistsError(f"Model run already exists: {output}")
    (output / "checkpoints").mkdir()
    prepared = args.genis_prepared.resolve()
    data_hash = file_sha256(prepared / "prepared_data.npz")
    with np.load(prepared / "prepared_data.npz") as z:
        train, y_train, val, y_val = [z[k] for k in ["X_train", "y_train", "X_val", "y_val"]]
    if len(train) // args.client_nums < 2:
        raise ValueError("Each client needs at least two training rows for its internal validation split")
    timesteps, features = train.shape[1:]
    tf.keras.utils.set_random_seed(args.random_seed)
    devices = tf.config.list_physical_devices("GPU")
    for device in devices:
        tf.config.experimental.set_memory_growth(device, True)
    if args.gpu_per_client and not devices:
        raise RuntimeError("GPU requested but no GPU detected; use --gpu_per_client 0 for CPU")
    module = importlib.import_module("models.model_aelstm" if kind == "aelstm" else "models.model_cnnlstm")
    chunks, label_chunks = np.array_split(train, args.client_nums), np.array_split(y_train, args.client_nums)
    model = make_model(kind, timesteps, features)
    model(tf.zeros((1, timesteps, features)))
    source_hashes = {name: file_sha256(PROJECT_ROOT / name) for name in SOURCES}
    config = {"dataset": "GeNIS", "model": kind, "timesteps": timesteps, "features": features,
              "sequence_data_path": str(prepared), "prepared_data_sha256": data_hash,
              "reconstruction_percentile": args.percentile, "arguments": vars(args),
              "client_learning_rate": .001 if kind == "aelstm" else .0001,
              "source_sha256": source_hashes, "fit_implementation": f"unmodified {module.__name__}.FLClient.fit",
              "test_used_for_training_or_round_evaluation": False}
    dump(output / "model_config.json", config)

    class Client(module.FLClient):
        def get_parameters(self, config):
            return self.model.get_weights()

        def fit(self, parameters, config):
            values, count, metrics = super().fit(parameters, config)
            return values, count, {**metrics, "epochs_completed": self.epochs}

        def evaluate(self, parameters, config):
            self.model.set_weights(parameters)
            scores, _ = probabilities(self.model, self.X_test, kind)
            accuracy = float(np.mean((scores > .5) == self.y_test))
            with (output / "client.txt").open("a") as stream:
                stream.write(f"Client {self.cid}: validation accuracy={accuracy:.6f}\n")
            return 1 - accuracy, len(self.y_test), {"accuracy": accuracy}

    def client_fn(cid: str):
        index = int(cid)
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(args.random_seed + index)
        local = make_model(kind, timesteps, features)
        kwargs = {"y_train_cls" if kind == "aelstm" else "y_train": label_chunks[index]}
        return Client(index, local, chunks[index], X_test=val, y_test=y_val,
                      epochs=args.client_epochs, batch_size=args.batch_size, **kwargs).to_client()

    prefix = "ae_lstm" if kind == "aelstm" else "cnn_lstm"
    server_report = output / f"{prefix}_server.txt"
    server_report.write_text("[GeNIS validation reports; final test evaluated after training]\n")

    class Strategy(fl.server.strategy.FedAvg):
        final_parameters = None

        def aggregate_fit(self, server_round, results, failures):
            if failures or len(results) != args.client_nums:
                raise RuntimeError("Some federated clients failed during training")
            if any(r.metrics.get("epochs_completed") != args.client_epochs for _, r in results):
                raise RuntimeError("Incomplete local training")
            self.final_parameters, metrics = super().aggregate_fit(server_round, results, failures)
            with (output / "round_counts.jsonl").open("a") as stream:
                stream.write(json.dumps({"round": server_round, "fits": len(results), "failures": 0}) + "\n")
            return self.final_parameters, metrics

        def aggregate_evaluate(self, server_round, results, failures):
            if failures or len(results) != args.client_nums:
                raise RuntimeError("Some federated clients failed during evaluation")
            return super().aggregate_evaluate(server_round, results, failures)

        def evaluate(self, server_round, parameters):
            model.set_weights(fl.common.parameters_to_ndarrays(parameters))
            scores, _ = probabilities(model, val, kind)
            predicted = (scores > .5).astype(int)
            accuracy = float(np.mean(predicted == y_val))
            with server_report.open("a") as stream:
                stream.write(f"\n[Round {server_round}] Accuracy: {accuracy:.6f}\n")
                stream.write(classification_report(y_val, predicted, labels=[0, 1],
                             target_names=["Normal", "Anomaly"], zero_division=0))
            if server_round:
                model.save_weights(str(output / "checkpoints" / f"round_{server_round:02d}.h5"))
            return 1 - accuracy, {"accuracy": accuracy, "roc_auc": float(roc_auc_score(y_val, scores))}

    strategy = Strategy(fraction_fit=1, fraction_evaluate=1, min_fit_clients=args.client_nums,
                        min_evaluate_clients=args.client_nums, min_available_clients=args.client_nums,
                        accept_failures=False, initial_parameters=fl.common.ndarrays_to_parameters(model.get_weights()))
    try:
        history = fl.simulation.start_simulation(
            client_fn=client_fn, num_clients=args.client_nums, strategy=strategy,
            config=fl.server.ServerConfig(num_rounds=args.server_rounds),
            client_resources={"num_cpus": 2, "num_gpus": args.gpu_per_client},
            ray_init_args={"include_dashboard": False,
                           "num_cpus": min(os.cpu_count() or 1, 40, args.client_nums * 2),
                           "num_gpus": len(devices), "object_store_memory": 2 * 1024 ** 3,
                           "runtime_env": {"working_dir": str(PROJECT_ROOT)}},
        )
    finally:
        import ray
        ray.shutdown()
    model.set_weights(fl.common.parameters_to_ndarrays(strategy.final_parameters))
    model.save_weights(str(output / f"{prefix}_weights.h5"))
    save_and_plot_history(history, str(output / f"{prefix}_history.csv"), str(output / f"{prefix}_history.png"))
    evaluate_genis_sequence(model, kind, prepared, output, args.percentile)
    records = [json.loads(line) for line in (output / "round_counts.jsonl").read_text().splitlines()]
    assert len(records) == args.server_rounds
    assert file_sha256(prepared / "prepared_data.npz") == data_hash
    assert all(file_sha256(PROJECT_ROOT / name) == value for name, value in source_hashes.items())
    dump(output / "verification.json", {"rounds_completed": len(records),
        "successful_client_fits": sum(r["fits"] for r in records), "fit_failures": 0,
        "prepared_data_unchanged": True, "source_files_unchanged": True})
    published.mkdir(parents=True, exist_ok=True)
    for path in output.iterdir():
        if path.is_file() and (path.name.startswith(("GeNIS", prefix)) or
                              path.name in {"model_config.json", "client.txt", "evaluation.json", "verification.json"}):
            shutil.copy2(path, published / path.name)
    shutil.copy2(prepared / "preprocessing.joblib", published / "preprocessing.joblib")
    shutil.copy2(prepared / "preprocessing_report.json", published / "preprocessing_report.json")
    dump(published / "output_manifest.json", {"source_run": str(output), "model": kind, "prepared_data": str(prepared)})
    print(f"GeNIS {kind} results: {published}", flush=True)


def run_genis_sequence(args, kind, loader):
    for key, value in {"TF_FORCE_GPU_ALLOW_GROWTH": "true", "TF_NUM_INTRAOP_THREADS": "2",
                       "TF_NUM_INTEROP_THREADS": "1", "OMP_NUM_THREADS": "2",
                       "OPENBLAS_NUM_THREADS": "1", "MPLBACKEND": "Agg"}.items():
        os.environ.setdefault(key, value)
    if args.genis_worker:
        return run_worker(args, kind)
    prepared = args.genis_prepared or DEFAULT_PREPARED
    *_, metadata = loader(prepared=prepared, random_seed=args.random_seed,
                          attack_fraction=args.genis_attack_fraction, max_normal_samples=args.n_samples,
                          split_mode=args.genis_sequence_split, return_metadata=True)
    shared = Path(metadata["prepared_data"])
    print(f"[GeNIS {kind}] train normal={metadata['train_normal_rows']} attack={metadata['train_attack_rows']}; "
          f"input=({metadata['timesteps']},{metadata['features']}); "
          f"same evaluation rows as RNEP={metadata['same_evaluation_rows_as_rnep']}", flush=True)
    if metadata["unseen_attack_types_at_training"]:
        print(f"Attack types reserved for evaluation: {metadata['unseen_attack_types_at_training']}", flush=True)
    if args.prepare_only:
        print(f"Prepared GeNIS inputs: {shared}", flush=True)
        return
    published = (args.result_path or PROJECT_ROOT / "results" / "GeNIS" / kind).resolve()
    stamp = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d_%H%M%S_%f")
    output = (args.output_dir or published / "runs" / f"training_{stamp}").resolve()
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "sources"
    for name in SOURCES:
        path = snapshot / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PROJECT_ROOT / name, path)
    (snapshot / "models" / "__init__.py").touch()
    entry = "AE_LSTM" if kind == "aelstm" else "CNN_LSTM"
    command = [sys.executable, "-u", "-m", entry, "--eval_dataset", "genis", "--genis_worker",
               "--genis_prepared", str(shared), "--output_dir", str(output), "--result_path", str(published)]
    for name in ["client_nums", "client_epochs", "server_rounds", "batch_size", "percentile", "random_seed", "gpu_per_client"]:
        command += [f"--{name}", str(getattr(args, name))]
    dump(output / "launch_config.json", {"command": command, "prepared_data": str(shared)})
    print(f"Starting {kind} federated training; log={output / 'training.log'}", flush=True)
    with (output / "training.log").open("w") as stream:
        result = subprocess.run(command, cwd=snapshot, stdout=stream, stderr=subprocess.STDOUT)
    dump(output / "exit_status.json", {"exit_code": result.returncode})
    if result.returncode:
        raise RuntimeError(f"{kind} training failed; see {output / 'training.log'}")
    print(f"Saved results to {published}", flush=True)
