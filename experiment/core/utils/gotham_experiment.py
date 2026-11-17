"""Launch the unmodified previous TAAE with Gotham-prepared inputs."""
from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess
import sys
from zoneinfo import ZoneInfo

from utils.genis import file_sha256
from utils.genis_experiment import PROJECT_ROOT, dump, frozen_sources, run_single, runtime_environment
from utils.gotham import prepare_gotham


def run_gotham(args):
    runtime_environment()
    if args.genis_worker:
        if args.gotham_prepared is None or args.output_dir is None:
            raise ValueError("Gotham worker requires a prepared split and output directory")
        args.genis_prepared = args.gotham_prepared
        return run_single(args)
    stamp = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d_%H%M%S_%f")
    published = (args.result_path or PROJECT_ROOT / "results" / "Gotham" / "taae").resolve()
    output = (args.output_dir or published / "runs" / f"training_{stamp}").resolve()
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "sources"
    hashes = frozen_sources(snapshot)
    prepared = (args.gotham_prepared or args.gotham_dir / "processed" / f"training_{stamp}").resolve()
    dump(output / "status.json", {"stage": "preparing", "dataset": "Gotham", "prepared_data": prepared})
    if args.gotham_prepared is None:
        report = prepare_gotham(args.gotham_dir, prepared, seed=args.random_seed,
            max_train_samples=args.n_samples, eval_normal_samples=args.gotham_eval_samples,
            client_nums=args.client_nums)
    else:
        report = json.loads((prepared / "preprocessing_report.json").read_text())
        if report["dataset"] != "Gotham" or report["seed"] != args.random_seed:
            raise ValueError("Prepared Gotham data do not match the requested dataset/seed")
        if report["preprocessor_sha256"] != hashes["utils/gotham.py"]:
            raise ValueError("Prepared Gotham cache differs from the frozen preprocessing source")
        if len(report["client_training_rows"]) != args.client_nums:
            raise ValueError("Prepared data were assigned to a different client count")
    dump(output / "preprocessing_report.json", report)
    dump(output / "run_config.json", {"arguments": vars(args), "prepared_data": prepared,
         "prepared_data_sha256": file_sha256(prepared / "prepared_data.npz"), "source_sha256": hashes,
         "model": "taae_previous.TransformerAAE", "normal_only_training": True})
    print(f"[Gotham] normal training={report['normal_training_rows']}, validation={report['validation_rows']}, "
          f"test={report['test_rows']}, features={report['input_dim']}; "
          f"clients={args.client_nums} ({args.gotham_client_partition}); model=taae_previous", flush=True)
    if args.prepare_only:
        dump(output / "status.json", {"stage": "prepared", "prepared_data": prepared})
        print(f"Prepared Gotham data: {prepared}", flush=True)
        return
    folder = output / "previous"
    folder.mkdir()
    command = [sys.executable, "-u", str(snapshot / "train_taae.py"), "--eval_dataset", "gotham",
               "--genis_worker", "--taae_version", "previous", "--gotham_prepared", str(prepared),
               "--output_dir", str(folder), "--result_path", str(published),
               "--gotham_client_partition", args.gotham_client_partition]
    for option in ["client_nums", "server_rounds", "client_epochs", "batch_size", "learning_rate",
                   "dropout_rate", "random_seed", "percentile", "gpu_per_client"]:
        command += [f"--{option}", str(getattr(args, option))]
    if args.genis_latent_dim is not None:
        command += ["--genis_latent_dim", str(args.genis_latent_dim)]
    dump(output / "status.json", {"stage": "training", "command": command, "prepared_data": prepared})
    print(f"Starting Gotham RNEP; log={folder / 'training.log'}", flush=True)
    with (folder / "training.log").open("w") as log:
        result = subprocess.run(command, cwd=snapshot, stdout=log, stderr=subprocess.STDOUT)
    dump(folder / "exit_status.json", {"exit_code": result.returncode})
    if result.returncode:
        dump(output / "status.json", {"stage": "failed", "exit_code": result.returncode})
        raise RuntimeError(f"Gotham RNEP failed; see {folder / 'training.log'}")
    shutil.copy2(folder / "percentile_summary.json", published / "percentile_summary.json")
    dump(output / "status.json", {"stage": "complete", "prepared_data": prepared, "results": published})
    print(f"Gotham RNEP finished: {published}", flush=True)
