"""Run RNEP on CIC, ITU/ULAK, GeNIS, or Gotham."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

from utils.arguments import get_args

PROJECT_ROOT = Path(__file__).resolve().parent

WEIGHT_PATH = "./FedSAD_Results/fedsad_weights.h5"
MATRIX_PATH = "./FedSAD_Results/fedsad_cm.png"
RESULT_PATH = "./FedSAD_Results/fedsad_server.txt"
ROC_PATH = "./FedSAD_Results/fedsad_roc.png"
CSV_PATH = "./FedSAD_Results/fedsad_history"
PNG_PATH = "./FedSAD_Results/fedsad_history.png"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_training(args):
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2")
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras.optimizers import Adam
    import flwr as fl
    from flwr.common import parameters_to_ndarrays
    from utils.get_datasets import get_datasets_cic_sam, eval_server
    from utils.metrics import save_and_plot_history
    from taae_current import SaveEvaluationRNEP, TransformerAAE, FLClient

    os.makedirs("./FedSAD_Results", exist_ok=True)
    model_kwargs = {}
    client_learning_rate = 1e-5
    X_train_scaled, X_test_scaled, y_test = get_datasets_cic_sam()
    client_data = np.array_split(X_train_scaled, args.client_nums)
    
    # Prepare model/data for server evaluation
    input_dim = X_train_scaled.shape[1]
    print(input_dim)
    central_model = TransformerAAE(input_dim, **model_kwargs)
    _ = central_model(tf.zeros((1, input_dim)), prior_labels=tf.zeros((1,1)))
    central_model.compile(optimizer=Adam(0.0001), loss="mse")
    model_config = {
        "dataset": args.eval_dataset,
        "input_dim": input_dim,
        "latent_dim": central_model.latent_dim,
        "client_learning_rate": client_learning_rate,
        "arguments": vars(args),
        "normal_train_ratio": None,
    }
    with open("./FedSAD_Results/model_config.json", "w") as stream:
        json.dump(model_config, stream, indent=2, default=str)
    print(f"[RNEP] latent_dim={central_model.latent_dim}, "
          f"client_learning_rate={client_learning_rate}, "
          f"clients={args.client_nums}, rounds={args.server_rounds}, "
          f"local_epochs={args.client_epochs}, batch_size={args.batch_size}, "
          f"percentile={args.percentile}", flush=True)

    eval_server_args = {
        "model": central_model,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "y_test": y_test,
        "result_path": RESULT_PATH,
        "matrix_path": MATRIX_PATH,
    }

    strategy = SaveEvaluationRNEP(
        eval_server_args=eval_server_args,
        fraction_fit=0.8,
        fraction_evaluate=0.8,
        min_fit_clients=args.client_nums,
        min_evaluate_clients=args.client_nums,
        min_available_clients=args.client_nums,
        evaluate_fn=None
    )

    # client_fn as closure: captures client_data, X_test_scaled, y_test, input_dim
    def client_fn(cid: str):
        cid_int = int(cid)
        client_model = TransformerAAE(input_dim, **model_kwargs)
        _ = client_model(tf.zeros((1, input_dim)), prior_labels=tf.zeros((1,1)))
        
        return FLClient(
            cid_int,
            client_model,
            client_data[cid_int],
            X_test_scaled,
            y_test,
            epochs=args.client_epochs,
            learning_rate=client_learning_rate,
        )

    # Clear previous results file
    with open(RESULT_PATH, "w") as f:
        f.write("[Server Evaluation Report]\n")

    # start_simulation with safe settings
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=args.client_nums,
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=args.server_rounds),
        client_resources={"num_cpus": 1},
        # num_parallel_clients=args.num_parallel_clients,   # recommended 1~2
        ray_init_args={"include_dashboard": False, "ignore_reinit_error": True},   
        # client_fn_eval=client_fn 
        )

    # Apply the final trained weights to central_model
    if strategy.final_parameters is not None:
        final_weights = parameters_to_ndarrays(strategy.final_parameters)
        central_model.set_weights(final_weights)
    
    save_and_plot_history(
        history, 
        csv_path=CSV_PATH, 
        png_path=PNG_PATH
    )

    eval_server(
        central_model,
        X_train_scaled,
        X_test_scaled,
        y_test,
        result_path=RESULT_PATH,
        matrix_path=MATRIX_PATH,
        roc_path=ROC_PATH
    )
    
    # Save the trained model weights
    central_model.save_weights(WEIGHT_PATH)
    print(f"\nModel weights saved to {WEIGHT_PATH}")


def run_zenodo(args):
    """Use the published ULAK split and the existing RNEP/TAAE implementation."""
    for name, value in {
        "TF_FORCE_GPU_ALLOW_GROWTH": "true", "MPLBACKEND": "Agg",
        "TF_NUM_INTRAOP_THREADS": "2", "TF_NUM_INTEROP_THREADS": "1",
        "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "1",
    }.items():
        os.environ.setdefault(name, value)
    if args.output_dir is None:
        stamp = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d_%H%M%S_%f")
        args.output_dir = PROJECT_ROOT / "results" / "ITU_SDN_ULAK" / f"rnep_{stamp}"
    output = args.output_dir.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Output directory must be new or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "checkpoints").mkdir()
    (output / "sources").mkdir()

    import numpy as np
    import pandas as pd
    import joblib
    import tensorflow as tf
    import flwr as fl
    from taae_current import TransformerAAE, FLClient, SaveEvaluationRNEP, get_datasets_zenodo
    from utils.metrics import save_and_plot_history

    tf.keras.utils.set_random_seed(args.random_seed)
    physical_gpus = tf.config.list_physical_devices("GPU")
    for device in physical_gpus:
        tf.config.experimental.set_memory_growth(device, True)
    if args.gpu_per_client and not physical_gpus:
        raise RuntimeError("No GPU detected; use --gpu_per_client 0 for CPU training")
    sources = {}
    for name in ["taae_current.py", "train_taae.py", "utils/arguments.py", "utils/metrics.py"]:
        src = PROJECT_ROOT / name
        target = output / "sources" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
        sources[name] = sha256(src)
    data_hashes = {name: sha256(args.data_dir / name)
                   for name in ["Train_ULAK.csv", "Test_ULAK.csv"]}
    train, test, labels, metadata = get_datasets_zenodo(
        args.data_dir, random_seed=args.random_seed, max_train_samples=args.n_samples,
        balance_test=True, return_metadata=True,
    )
    if len(train) < args.client_nums:
        raise ValueError("Fewer training rows than clients")
    np.savez_compressed(output / "prepared_data.npz", X_train=train, X_test=test, y_test=labels)
    np.savez_compressed(output / "source_rows.npz", train=metadata["train_source_rows"],
                        test=metadata["test_source_rows"], attack_labels=metadata["test_attack_labels"])
    joblib.dump({"imputer": metadata["imputer"], "scaler": metadata["scaler"],
                 "feature_names": metadata["report"]["feature_names"]}, output / "preprocessing.joblib")
    (output / "preprocessing_report.json").write_text(json.dumps(metadata["report"], indent=2) + "\n")
    config = {
        "dataset": "ITU_SDN_ULAK", "created_at_kst": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
        "input_dim": train.shape[1], "latent_dim": args.latent_dim,
        "client_learning_rate": args.learning_rate, "arguments": vars(args).copy(),
        "data_sha256": data_hashes, "source_sha256": sources,
        "preprocessing": "normal-train-only median imputation and MinMax; no log transform",
        "split": "published Train_ULAK/Test_ULAK; normal-only sampled training; balanced held-out test",
        "loss": "current taae.py custom reconstruction + beta*adversarial loss; beta=0.3",
        "gpu_count": len(physical_gpus), "gpu_per_client": args.gpu_per_client,
        "prediction_batch_size": 512,
        "test_used_for_threshold_selection": False,
    }
    config["arguments"]["data_dir"] = str(args.data_dir.resolve())
    config["arguments"]["output_dir"] = str(output)
    config["arguments"] = {key: str(value) if isinstance(value, Path) else value
                           for key, value in config["arguments"].items()}
    (output / "model_config.json").write_text(json.dumps(config, indent=2) + "\n")
    if args.prepare_only:
        print(f"Preprocessing complete: {output}", flush=True)
        return

    model = TransformerAAE(train.shape[1], latent_dim=args.latent_dim)
    model(tf.zeros((1, train.shape[1])), prior_labels=tf.zeros((1, 1)))
    model.compile(optimizer=tf.keras.optimizers.Adam(args.learning_rate))
    client_data = np.array_split(train, args.client_nums)
    model_kwargs = {"input_dim": train.shape[1], "latent_dim": args.latent_dim}
    client_log = str(output / "client.txt")

    def client_fn(cid: str):
        cid_int = int(cid)
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(args.random_seed + cid_int)
        for gpu in tf.config.list_physical_devices("GPU"):
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass
        print(f"[ZENODO client {cid_int}] CUDA_VISIBLE_DEVICES="
              f"{os.environ.get('CUDA_VISIBLE_DEVICES')} GPUs="
              f"{tf.config.list_physical_devices('GPU')}", flush=True)
        client_model = TransformerAAE(**model_kwargs)
        client_model(tf.zeros((1, model_kwargs["input_dim"])), prior_labels=tf.zeros((1, 1)))
        return FLClient(
            cid_int, client_model, client_data[cid_int], test, labels,
            epochs=args.client_epochs, learning_rate=args.learning_rate,
            batch_size=args.batch_size, result_path=client_log, prediction_batch_size=512,
            percentile=args.percentile,
        ).to_client()

    class CheckpointStrategy(SaveEvaluationRNEP):
        def aggregate_fit(self, rnd, results, failures):
            if failures or len(results) != args.client_nums:
                raise RuntimeError(f"Round {rnd}: {len(results)} fits, {len(failures)} failures")
            value = super().aggregate_fit(rnd, results, failures)
            with (output / "round_counts.jsonl").open("a") as stream:
                stream.write(json.dumps({"round": rnd, "fit_results": len(results),
                                         "fit_failures": len(failures)}) + "\n")
            return value

        def aggregate_evaluate(self, rnd, results, failures):
            if failures or len(results) != args.client_nums:
                raise RuntimeError(f"Round {rnd}: {len(results)} evaluations, {len(failures)} failures")
            with (output / "evaluation_counts.jsonl").open("a") as stream:
                stream.write(json.dumps({"round": rnd, "evaluation_results": len(results),
                                         "evaluation_failures": len(failures)}) + "\n")
            return super().aggregate_evaluate(rnd, results, failures)

        def evaluate(self, server_round, parameters):
            result = super().evaluate(server_round, parameters)
            if server_round > 0:
                self.eval_server_args["model"].save_weights(
                    str(output / "checkpoints" / f"round_{server_round:02d}.h5"))
            return result

    report_path = output / "server_report.txt"
    report_path.write_text("[ITU/ULAK RNEP server evaluation]\n")
    strategy = CheckpointStrategy(
        eval_server_args={"model": model, "X_train_scaled": train, "X_test_scaled": test,
                          "y_test": labels, "result_path": str(report_path),
                          "matrix_path": str(output / "confusion_matrix.png"),
                          "prediction_batch_size": 512, "percentile": args.percentile},
        fraction_fit=.8, fraction_evaluate=.8, min_fit_clients=args.client_nums,
        min_evaluate_clients=args.client_nums, min_available_clients=args.client_nums,
        accept_failures=False,
    )
    ray_args = {"include_dashboard": False, "ignore_reinit_error": True,
                "num_cpus": min(os.cpu_count() or 1, 40), "num_gpus": len(physical_gpus),
                "object_store_memory": 4 * 1024 ** 3}
    print(f"Starting RNEP: {args.client_nums} clients, {args.server_rounds} rounds, "
          f"{args.client_epochs} local epochs, batch={args.batch_size}, "
          f"lr={args.learning_rate}, latent={args.latent_dim}", flush=True)
    try:
        history = fl.simulation.start_simulation(
            client_fn=client_fn, num_clients=args.client_nums, strategy=strategy,
            config=fl.server.ServerConfig(num_rounds=args.server_rounds),
            client_resources={"num_cpus": 2, "num_gpus": args.gpu_per_client}, ray_init_args=ray_args,
        )
    finally:
        import ray
        ray.shutdown()
    if strategy.final_parameters is None:
        raise RuntimeError("Simulation produced no final parameters")
    model.set_weights(fl.common.parameters_to_ndarrays(strategy.final_parameters))
    model.save_weights(str(output / "fedsad_weights.h5"))
    save_and_plot_history(history, csv_path=str(output / "history"), png_path=str(output / "history.png"))
    train_pred = model.predict(train, batch_size=512, verbose=2)
    train_errors = np.mean(np.square(train - train_pred), axis=1)
    test_pred = model.predict(test, batch_size=512, verbose=2)
    errors = np.mean(np.square(test - test_pred), axis=1)
    threshold = float(np.percentile(train_errors, args.percentile))
    if not np.isfinite(train_errors).all() or not np.isfinite(errors).all():
        raise RuntimeError("Nonfinite reconstruction scores")
    from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve
    import matplotlib.pyplot as plt
    from scripts.evaluate_taae import plot_error_distributions
    predicted = (errors > threshold).astype(int)
    report = classification_report(labels, predicted, labels=[0, 1],
                                   target_names=["Normal", "Anomaly"], output_dict=True, zero_division=0)
    np.savez_compressed(output / "scores.npz", train_errors=train_errors, test_errors=errors, y_test=labels)
    pd.DataFrame({"source_row": metadata["test_source_rows"],
                  "attack": metadata["test_attack_labels"], "y_true": labels,
                  "anomaly_score": errors, "y_pred": predicted}).to_csv(output / "predictions.csv", index=False)
    summary = {"percentile": args.percentile, "threshold": threshold,
               "roc_auc": float(roc_auc_score(labels, errors)), "classification_report": report,
               "confusion_matrix": confusion_matrix(labels, predicted, labels=[0, 1]).tolist(),
               "normal_test_rows": int((labels == 0).sum()), "attack_test_rows": int((labels == 1).sum())}
    (output / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    text = classification_report(labels, predicted, labels=[0, 1], target_names=["Normal", "Anomaly"],
                                 digits=6, zero_division=0)
    (output / "final_report.txt").write_text(f"Percentile: {args.percentile}\nThreshold: {threshold}\n\n{text}\n")
    fpr, tpr, _ = roc_curve(labels, errors)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, label=f"AUC={summary['roc_auc']:.4f}"); ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ITU/ULAK RNEP")
    ax.legend(); fig.tight_layout(); fig.savefig(output / "roc.png", dpi=180); plt.close(fig)
    error_by_attack = {}
    per_class = []
    for index, label in enumerate(sorted(set(metadata["test_attack_labels"][labels == 1])), 1):
        mask = metadata["test_attack_labels"] == label
        error_by_attack[index] = (label, errors[mask])
        per_class.append({"attack": label, "support": int(mask.sum()),
                          "recall": float(predicted[mask].mean())})
    pd.DataFrame(per_class).to_csv(output / "per_attack.csv", index=False)
    plot_error_distributions(errors[labels == 0], error_by_attack, threshold,
                             "ITU_SDN_ULAK", output / "distribution.png")
    rounds = [json.loads(line) for line in (output / "round_counts.jsonl").read_text().splitlines()]
    evaluations = [json.loads(line) for line in (output / "evaluation_counts.jsonl").read_text().splitlines()]
    if len(rounds) != args.server_rounds or any(r["fit_failures"] for r in rounds):
        raise RuntimeError("Not all requested rounds completed successfully")
    if len(evaluations) != args.server_rounds or any(r["evaluation_failures"] for r in evaluations):
        raise RuntimeError("Not all requested client evaluations completed successfully")
    verification = {"rounds_completed": len(rounds), "successful_client_fits": sum(r["fit_results"] for r in rounds),
                    "successful_client_evaluations": sum(r["evaluation_results"] for r in evaluations),
                    "evaluation_failures": 0,
                    "fit_failures": 0, "test_balanced": int((labels == 0).sum()) == int((labels == 1).sum()),
                    "source_files_unchanged": all(sha256(PROJECT_ROOT / name) == value for name, value in sources.items()),
                    "raw_data_unchanged": all(sha256(args.data_dir / name) == value for name, value in data_hashes.items())}
    (output / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    print(text, flush=True)
    print(f"ROC AUC={summary['roc_auc']:.6f}; results={output}", flush=True)


def main(argv=None):
    args = get_args(argv)
    if args.eval_dataset == "gotham":
        from utils.gotham_experiment import run_gotham
        return run_gotham(args)
    if args.eval_dataset == "genis":
        from utils.genis_experiment import run_genis
        return run_genis(args)
    if args.eval_dataset == "zenodo":
        return run_zenodo(args)
    return run_training(args)


if __name__ == "__main__":
    main()
