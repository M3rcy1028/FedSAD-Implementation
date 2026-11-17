"""Public argument names; parsing never imports TensorFlow or PyTorch."""
import argparse
import json
from pathlib import Path
import sys

from experiment.artifacts import ROOT
from experiment.models.registry import MODELS, SUPPORT

ALIASES = {"kdd": "kdd99", "nsl": "nsl-kdd", "cic": "cic2018",
           "cse-cic-ids2018": "cic2018", "unsw": "unsw-nb15", "unsw_nb15": "unsw-nb15",
           "ciciot": "ciciot2023", "zenodo": "itu-ulak"}


def parser():
    defaults = json.loads((ROOT / "configs/defaults.json").read_text())
    p = argparse.ArgumentParser(description="FedSAD: train, evaluate, plot and compare")
    p.add_argument("--config", type=Path, help="JSON experiment settings; explicit CLI options take precedence")
    p.add_argument("--mode", choices=["train", "evaluate", "plot", "compare"], default="train")
    p.add_argument("--model", choices=[*MODELS, "mnids"], default="taae")
    p.add_argument("--dataset", default="kdd99")
    p.add_argument("--epochs", "--epoch", "--local-epochs", type=int, default=defaults["epochs"],
                   help="Epochs per client per round for FL; epochs for centralized models (default: 30)")
    p.add_argument("--clients", type=int, nargs="+", default=[defaults["clients"]])
    p.add_argument("--rounds", "--server-rounds", type=int, default=defaults["rounds"])
    p.add_argument("--batch-size", type=int)
    p.add_argument("--seed", type=int, default=defaults["seed"])
    p.add_argument("--percentile", type=float, default=defaults["percentile"])
    p.add_argument("--threshold-policy", choices=["auto", "fixed", "validation"], default="auto")
    p.add_argument("--variant", choices=["previous", "current"], default=defaults["taae_variant"])
    p.add_argument("--latent-dim", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--dropout", type=float)
    p.add_argument("--data-root", type=Path, default=Path.home() / "IDS_Datasets")
    p.add_argument("--data-dir", type=Path, help="Dataset directory; graph models accept a CSV file")
    p.add_argument("--prepared", type=Path, help="Reuse a frozen GeNIS/Gotham prepared directory")
    p.add_argument("--flow-interval", type=int, choices=[5, 10, 30, 60], default=10)
    p.add_argument("--normal-train-ratio", type=float,
                   help="Defaults: KDD99/NSL-KDD and InSDN AE-LSTM 0.5; others 0.8 (CICIoT sequence models 0.5)")
    p.add_argument("--attack-fraction", type=float,
                   help="Sequence training attack ratio; KDD99 legacy defaults: AE 0.05, CNN 0.2")
    p.add_argument("--max-train-samples", type=int, default=150000,
                   help="Normal-row cap for GeNIS/Gotham/ITU preparation")
    p.add_argument("--gpus", default="0,1,2", help="Visible GPU IDs, or cpu")
    p.add_argument("--gpu-per-client", type=float, default=.2)
    p.add_argument("--execution", choices=["ray", "serial"], default="ray",
                   help="serial executes the same FL clients sequentially, useful on CPU")
    p.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    p.add_argument("--name", help="Unique run directory name")
    p.add_argument("--run", type=Path, help="Saved run for evaluate or plot")
    p.add_argument("--runs", type=Path, nargs="+", help="Saved runs for compare")
    p.add_argument("--dry-run", action="store_true", help="Print the resolved configuration without training")
    p.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--list", action="store_true", help="List supported model/dataset combinations")
    p.add_argument("--repeats", type=int, default=1, help="Independent runs with seed, seed+1, ...")
    p.add_argument("--k-train", type=int, default=15)
    p.add_argument("--k-test", type=int, default=30)
    return p


def main(argv=None):
    p = parser()
    tokens = list(sys.argv[1:] if argv is None else argv)
    provided = set()
    preliminary, _ = p.parse_known_args(argv)
    if preliminary.config:
        settings = json.loads(preliminary.config.read_text())
        permitted = {action.dest for action in p._actions} - {"worker", "help"}
        unknown = set(settings) - permitted
        if unknown:
            p.error(f"Unknown configuration keys: {', '.join(sorted(unknown))}")
        if isinstance(settings.get("clients"), int):
            settings["clients"] = [settings["clients"]]
        provided.update(settings)
        p.set_defaults(**settings)
    args = p.parse_args(argv)
    defaults = json.loads((ROOT / "configs/defaults.json").read_text())
    if args.worker:
        from experiment.runner import worker
        return worker(json.loads(args.worker.read_text()))
    if args.list:
        for model, datasets in SUPPORT.items():
            print(f"{model:16} {', '.join(datasets)}")
        return
    if args.mode in {"evaluate", "plot"} and args.run is None:
        p.error("--run is required for evaluate/plot")
    if args.mode == "compare" and not args.runs:
        p.error("--runs is required for compare")
    config = vars(args).copy()
    # Workers execute from a frozen source directory, so resolve user paths
    # against the invoking shell before starting a subprocess.
    for key in ["config", "data_root", "data_dir", "prepared", "output_root", "run"]:
        if config.get(key) is not None:
            config[key] = Path(config[key]).expanduser().resolve()
    if config.get("runs"):
        config["runs"] = [Path(path).expanduser().resolve() for path in config["runs"]]
    for action in p._actions:
        if any(token.split("=", 1)[0] in action.option_strings for token in tokens):
            provided.add(action.dest)
    config["provided_options"] = sorted(provided)
    config["model"] = "mvids" if args.model == "mnids" else args.model
    config["dataset"] = ALIASES.get(args.dataset.lower(), args.dataset.lower())
    if config["model"] not in MODELS:
        p.error(f"Unknown model: {config['model']}")
    config["batch_size"] = args.batch_size if args.batch_size is not None else {"mvids": 64, "mmfewshotsids": 32}.get(config["model"], defaults["batch_size"])
    config["dropout"] = args.dropout if args.dropout is not None else {"aelstm": .001, "mvids": .3}.get(config["model"], defaults["dropout"])
    if config["model"] == "cnnlstm" and args.dropout is not None:
        p.error("CNN-LSTM preserves its source architecture's dropout rates (0.25/0.5)")
    if config["model"] in {"mvids", "mmfewshotsids"}:
        if {"clients", "rounds"} & provided:
            p.error("--clients/--rounds are FL options; graph models use --epochs and --repeats")
        config.update(clients=[1], rounds=1)
    if config["dataset"] not in SUPPORT[config["model"]] and args.mode == "train":
        p.error(f"{config['model']} supports: {', '.join(SUPPORT[config['model']])}")
    if args.latent_dim is not None and config["model"] not in {"taae", "aelstm"}:
        p.error("--latent-dim is available for TAAE and AE-LSTM")
    if args.mode == "train" and config["model"] == "taae" and args.threshold_policy == "validation" and config["dataset"] not in {"genis", "gotham"}:
        p.error("Validation-selected thresholds require independent labeled validation (GeNIS/Gotham)")
    if min([args.epochs, args.rounds, config["batch_size"], args.max_train_samples, args.repeats, *args.clients]) < 1:
        p.error("Epochs, rounds, clients, batch size and sample counts must be positive")
    if not 0 <= args.percentile <= 100 or not 0 <= config["dropout"] < 1:
        p.error("Invalid percentile or dropout")
    if args.normal_train_ratio is not None and not 0 < args.normal_train_ratio < 1:
        p.error("--normal-train-ratio must be in (0,1)")
    if args.attack_fraction is not None and not 0 <= args.attack_fraction < 1:
        p.error("--attack-fraction must be in [0,1)")
    defaults = json.loads((ROOT / "configs/defaults.json").read_text())
    config["lr"] = args.lr if args.lr is not None else defaults["learning_rates"][config["model"]]
    config["latent_dim"] = (args.latent_dim if args.latent_dim is not None else
                            (32 if config["model"] == "taae" and args.variant == "current" else 64))
    if config["lr"] <= 0 or config["latent_dim"] < 1 or args.gpu_per_client < 0:
        p.error("Invalid learning rate, latent dimension or GPU allocation")
    if args.dataset == "gotham" and (args.model != "taae" or args.variant != "previous"):
        p.error("Gotham uses --model taae --variant previous")
    if args.dry_run:
        print(json.dumps(config, indent=2, default=str))
        return
    from experiment.runner import dispatch
    dispatch(config)
