"""Existing TAAE/RNEP and sequence/FedAvg training recipes behind one runner."""
import json
import os
from pathlib import Path
import random

from experiment.artifacts import dump, now


def train(config, bundle):
    import numpy as np
    import tensorflow as tf
    import flwr as fl
    from experiment.backends import load, worker_setup
    from experiment.models.registry import build
    from experiment.evaluation.metrics import report, score

    runtime, output = Path(config["runtime"]), Path(config["output"])
    checkpoints = runtime / "checkpoints"
    checkpoints.mkdir()
    tf.keras.utils.set_random_seed(config["seed"])
    devices = tf.config.list_physical_devices("GPU")
    for device in devices:
        tf.config.experimental.set_memory_growth(device, True)
    if config["gpus"] != "cpu" and not devices:
        raise RuntimeError("No visible GPU; use --gpus cpu --execution serial for CPU")
    train_x, train_y = bundle["X_train"], bundle["y_train"]
    val_x, val_y = bundle["X_val"], bundle["y_val"]
    shape = tuple(train_x.shape[1:])
    model = build(config, shape)
    chunks = np.array_split(train_x, config["clients"])
    labels = np.array_split(train_y, config["clients"])
    if config["model"] == "taae":
        backend = load("taae_previous" if config["variant"] == "previous" else "taae_current")
        base_strategy = backend.SaveEvaluationRNEP
    else:
        base_strategy = fl.server.strategy.FedAvg
    rounds = []

    class Client(fl.client.NumPyClient):
        def __init__(self, cid):
            self.cid = cid
            tf.keras.utils.set_random_seed(config["seed"] + cid)
            self.model = build(config, shape)

        def get_parameters(self, parameters):
            return self.model.get_weights()

        def fit(self, parameters, fit_config):
            self.model.set_weights(parameters)
            x, y = chunks[self.cid], labels[self.cid]
            optimizer = tf.keras.optimizers.Adam(config["lr"])
            if config["model"] == "taae":
                self.model.compile(optimizer=optimizer)
                self.model.fit(x, x, epochs=config["epochs"], batch_size=config["batch_size"],
                               verbose=2, validation_split=.2)
                entropy = backend.compute_client_entropy(x)
            elif config["model"] == "aelstm":
                self.model.compile(optimizer=optimizer, loss={"decoded": "mse", "pred": "binary_crossentropy"},
                                   loss_weights={"decoded": 1.0, "pred": .1}, metrics={"pred": ["accuracy"]})
                self.model.fit(x, {"decoded": x, "pred": y}, epochs=config["epochs"],
                               batch_size=config["batch_size"], verbose=2, validation_split=.2)
                entropy = 0.0
            else:
                self.model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
                self.model.fit(x, y, epochs=config["epochs"], batch_size=config["batch_size"],
                               verbose=2, validation_split=.1, class_weight={0: 1, 1: 2})
                entropy = 0.0
            return self.model.get_weights(), len(x), {"entropy": float(entropy), "client_id": self.cid,
                                                    "epochs_completed": config["epochs"]}

        def evaluate(self, parameters, evaluate_config):
            self.model.set_weights(parameters)
            values, _ = score(self.model, val_x, config["model"])
            # Server performs the full validation once; distributed losses are diagnostics.
            return float(np.mean(values)), len(val_y), {}

    def client_fn(context: fl.common.Context):
        cid = int(context.node_config["partition-id"])
        return Client(cid).to_client()

    class Strategy(base_strategy):
        def aggregate_fit(self, server_round, results, failures):
            if failures or len(results) != config["clients"]:
                raise RuntimeError(f"Round {server_round}: incomplete client fits")
            if any(item.metrics["epochs_completed"] != config["epochs"] for _, item in results):
                raise RuntimeError("A client did not finish all requested epochs")
            results = sorted(results, key=lambda item: item[1].metrics["client_id"])
            state = random.getstate()
            try:
                random.seed(config["seed"] + server_round)
                aggregated, metrics = super().aggregate_fit(server_round, results, failures)
            finally:
                random.setstate(state)
            self.final_parameters = aggregated
            with (runtime / "round_counts.jsonl").open("a") as stream:
                stream.write(json.dumps({"round": server_round, "fits": len(results), "failures": 0,
                                         "epochs_per_client": config["epochs"]}) + "\n")
            return aggregated, metrics

        def evaluate(self, server_round, parameters):
            model.set_weights(fl.common.parameters_to_ndarrays(parameters))
            values, _ = score(model, val_x, config["model"])
            row = {"round": server_round, "evaluation_set": "validation",
                   "mean_validation_score": float(np.mean(values))}
            if len(np.unique(val_y)) == 2:
                train_scores, _ = score(model, train_x, config["model"])
                threshold = (float(np.percentile(train_scores, config["percentile"]))
                             if config["model"] == "taae" else .5)
                row.update(report(val_y, values, threshold))
            rounds.append(row)
            dump(output / "metadata/round_metrics.json", rounds)
            if server_round:
                model.save_weights(str(checkpoints / f"round_{server_round:02d}.h5"))
            dump(output / "status.json", {"stage": "training", "completed_rounds": server_round,
                 "total_rounds": config["rounds"], "clients": config["clients"], "updated_at": now()})
            print(f"[Round {server_round}/{config['rounds']}] validation score={row['mean_validation_score']:.7g}", flush=True)
            return row["mean_validation_score"], {}

    strategy = Strategy(fraction_fit=1, fraction_evaluate=1,
                        min_fit_clients=config["clients"], min_evaluate_clients=config["clients"],
                        min_available_clients=config["clients"], accept_failures=False,
                        initial_parameters=fl.common.ndarrays_to_parameters(model.get_weights()))
    if config["execution"] == "ray":
        import ray
        try:
            fl.simulation.start_simulation(
                client_fn=client_fn, num_clients=config["clients"], strategy=strategy,
                config=fl.server.ServerConfig(num_rounds=config["rounds"]),
                client_resources={"num_cpus": 2, "num_gpus": 0 if config["gpus"] == "cpu" else config["gpu_per_client"]},
                ray_init_args={"include_dashboard": False, "num_cpus": min(40, 2 * config["clients"]),
                    "num_gpus": len(devices), "object_store_memory": 2 * 1024 ** 3,
                    "_temp_dir": "/data/SDP_Dataset/.ray-unified",
                    "runtime_env": {"working_dir": str(Path(__file__).resolve().parents[2]),
                         "excludes": ["results/", "outputs/", "runtime/", "docs/", "plot_data/"],
                         "worker_process_setup_hook": worker_setup,
                         "env_vars": {"RAY_worker_niceness": "0", "PYTHONDONTWRITEBYTECODE": "1"}}})
        finally:
            ray.shutdown()
    else:
        strategy.evaluate(0, strategy.initial_parameters)
        parameters = fl.common.parameters_to_ndarrays(strategy.initial_parameters)
        for server_round in range(1, config["rounds"] + 1):
            results = []
            for cid in range(config["clients"]):
                weights, count, metrics = Client(cid).fit(parameters, {})
                fit_result = fl.common.FitRes(status=fl.common.Status(fl.common.Code.OK, ""),
                   parameters=fl.common.ndarrays_to_parameters(weights), num_examples=count, metrics=metrics)
                results.append((None, fit_result))
            aggregated, _ = strategy.aggregate_fit(server_round, results, [])
            strategy.evaluate(server_round, aggregated)
            parameters = fl.common.parameters_to_ndarrays(aggregated)
    if strategy.final_parameters is None:
        raise RuntimeError("Training did not produce final weights")
    model.set_weights(fl.common.parameters_to_ndarrays(strategy.final_parameters))
    model.save_weights(str(output / "weights/model.h5"))
    if len(rounds) != config["rounds"] + 1:
        raise RuntimeError("Not all requested rounds were evaluated")
    dump(output / "metadata/training.json", {"rounds_completed": config["rounds"],
         "clients": config["clients"], "epochs_per_client": config["epochs"],
         "aggregation": "RNEP" if config["model"] == "taae" else "FedAvg",
         "client_training_rows": [len(part) for part in chunks], "test_used_for_training": False})
    return model
