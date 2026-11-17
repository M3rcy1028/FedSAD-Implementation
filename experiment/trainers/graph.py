"""Adapters for the preserved MV-IDS and MM-FewShot-IDS PyTorch recipes."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

from experiment.artifacts import dump, sha256


def dependencies():
    if importlib.util.find_spec("torch") is None:
        raise RuntimeError("This model needs PyTorch: install requirements/optional-graph.txt in a compatible environment")
    from experiment.backends import activate
    activate()
    import torch
    return torch


def settings(config):
    from experiment.backends import CORE
    import ast
    # Read the original CLI defaults without executing imports or parsing argv.
    filename = "train_mvids.py" if config["model"] == "mvids" else "train_mmfewshotsids.py"
    tree = ast.parse((CORE / filename).read_text())
    defaults = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
            if node.args and isinstance(node.args[0], ast.Constant):
                key = node.args[0].value.lstrip("-")
                for keyword in node.keywords:
                    if keyword.arg == "default":
                        defaults[key] = ast.literal_eval(keyword.value)
    defaults.update(dataset="cic" if config["dataset"] == "cic2018" else "unsw",
                    epochs=config["epochs"], batch_size=config["batch_size"], lr=config["lr"],
                    lr_finetune=config["lr"], seed=config["seed"], repeats=1,
                    device="cpu" if config["gpus"] == "cpu" else "cuda:0",
                    result_path=str(Path(config["runtime"]) / "graph"),
                    k_train=config["k_train"], k_test=config["k_test"])
    if config["model"] == "mmfewshotsids" and defaults["dataset"] == "cic":
        defaults["dataset"] = "cicids2018"
    if config.get("data_dir"):
        defaults["data_path"] = str(config["data_dir"])
    defaults["tensor_path"] = str(Path(defaults["result_path"]) / "traffic_tensor.npy")
    if config["model"] == "mmfewshotsids":
        defaults.update(g_dropout=config["dropout"], s_dropout=config["dropout"], fusion_dropout=config["dropout"])
    Path(defaults["result_path"]).mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(**defaults)


def graph_scores(model, loader, kind, device):
    import numpy as np
    import torch
    probabilities, labels = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            if kind == "mvids":
                views, truth = batch
                logits = model([view.to(device) for view in views])
                values = torch.sigmoid(logits)
            else:
                g, cont, disc, truth = batch
                logits = model(g.to(device), cont.to(device), disc.to(device))
                values = torch.softmax(logits, dim=1)[:, 1]
            probabilities.append(values.cpu().numpy().reshape(-1))
            labels.append(truth.cpu().numpy().astype(int))
    return np.concatenate(labels), np.concatenate(probabilities)


def export(config, labels, scores, ids, attacks):
    import numpy as np
    import pandas as pd
    from experiment.evaluation.metrics import report
    from experiment.evaluation.export import save_predictions
    output = Path(config["output"])
    inclusive = config["model"] == "mvids"
    metrics = report(labels, scores, .5, inclusive=inclusive)
    metrics.update(dataset=config["dataset"], model=config["model"], score_kind="classifier_probability")
    dump(output / "metadata/metrics.json", metrics)
    dump(output / "metadata/threshold.json", {"policy": "classifier_probability", "threshold": .5})
    pd.DataFrame([{"class": key, **metrics["classification_report"][key]}
                  for key in ["Normal", "Anomaly", "macro avg", "weighted avg"]]).to_csv(output / "tables/metrics.csv", index=False)
    predicted = ((scores >= .5) if inclusive else (scores > .5)).astype(int)
    save_predictions(config, labels, scores, predicted, ids, attacks)
    summary = metrics["classification_report"]["Anomaly"]
    print(f"FINAL {config['model']}/{config['dataset']}: Precision={summary['precision']:.4f} "
          f"Recall={summary['recall']:.4f} F1={summary['f1-score']:.4f}", flush=True)
    return metrics


def train(config):
    torch = dependencies()
    import joblib
    import numpy as np
    import pandas as pd
    from experiment.backends import load
    output = Path(config["output"])
    args = settings(config)
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    csv = Path(args.data_path)
    if not csv.is_file():
        raise FileNotFoundError(f"Required feature CSV missing: {csv}; provide --data-dir /path/to/features.csv")
    frame = pd.read_csv(csv)
    if config["model"] == "mvids":
        dataset_module = load("models.mvids.load_data")
        features, labels, preprocessor = dataset_module.load_dataset(args.dataset, str(csv))
        attack_key = "Attack" if args.dataset == "cic" else "attack_cat"
        attack_key = attack_key if attack_key in frame else None
        attack_types = frame[attack_key].to_numpy() if attack_key else None
        loader, _, test, lengths, train_views, names, y_train = dataset_module.make_loaders(
            features, labels, preprocessor, args, attack_types=attack_types)
        kg = load("models.mvids.kg_builder")
        args.y_train = y_train
        kg.make_kg_provider(train_views, names, args)
        model = dataset_module.MultiViewKnowledgeNet(view_lengths=lengths, cnn_channels=32,
                            fusion_hidden=500, kg_dim=0, dropout=config["dropout"])
        load("models.mvids.train").train_loop(model, loader, None, args,
                                             save_path=str(output / "weights/model.pt"))
        from sklearn.model_selection import train_test_split
        train_ids, test_ids = train_test_split(np.arange(len(labels)), test_size=.2,
                    random_state=args.seed, stratify=attack_types if attack_types is not None else labels)
        checkpoint = {"kind": "mvids", "state_dict": model.state_dict(), "view_lengths": lengths}
        joblib.dump(preprocessor, output / "preprocessing.joblib")
        cache = {"views": test.dataset.views, "labels": test.dataset.labels.astype(int)}
    else:
        model_module = load("train_mmfewshotsids")
        data_module = load("models.mmfewshotsids.load_data")
        tensor = load("models.mmfewshotsids.traffic_feature_graph").build_packet_sequence_tensor(
            csv_path=str(csv), output_path=args.tensor_path)
        dataset = data_module.MultimodalDataset(csv_path=str(csv), tensor_data=tensor,
                    label_col="Label" if args.dataset == "cicids2018" else "label",
                    attack_col="Attack" if args.dataset == "cicids2018" else "attack_cat",
                    dataset_type=args.dataset)
        if dataset.classes() != [0, 1]:
            raise ValueError("Unified metrics require binary labels 0=Normal, 1=Anomaly")
        train_subset, test_subset, loader, test = model_module.prepare_loaders(dataset, args)
        adjacency = data_module.build_graph_A(dataset.df, dataset.g_cols, topk=args.kg_topk, thresh=args.kg_thresh)
        params = {"num_cont": dataset.num_cont, "disc_cardinalities": dataset.cardinalities,
                  "num_classes": 2, "fusion_dim": args.fusion_dim, "fusion_type": args.fusion_type,
                  "transformer_layers": args.s_depth, "transformer_heads": args.s_heads,
                  "g_dropout": args.g_dropout, "s_dropout": args.s_dropout,
                  "fusion_dropout": args.fusion_dropout, "g_in_dim": adjacency.shape[0]}
        model = load("models.mmfewshotsids.multimodal_ids").MultiModalIDS(
            **params, A_tensor=adjacency.to(args.device)).to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
        load("models.mmfewshotsids.train").train_self(args, model, loader, optimizer)
        # The source evaluates its last in-memory model; retain that exact state.
        checkpoint = {"kind": "mmfewshotsids", "state_dict": model.state_dict(), "model_params": params,
                      "adjacency": adjacency.cpu()}
        train_ids, test_ids = np.array(train_subset.indices), np.array(test_subset.indices)
        attack_key = dataset.attack_col
        joblib.dump({"cont_stats": dataset._cont_stats, "disc_mappings": dataset._disc_mappings,
                     "cont_cols": dataset.cont_cols, "disc_cols": dataset.disc_cols,
                     "g_cols": dataset.g_cols, "label_names": dataset.label_names},
                    output / "preprocessing.joblib")
        cache = {"subset": test_subset}
    torch.save(checkpoint, output / "weights/model.pt")
    model.to(args.device)
    labels, scores = graph_scores(model, test, config["model"], args.device)
    ids = np.array([f"{csv.name}:{i}" for i in test_ids])
    attacks = frame.iloc[test_ids][attack_key].astype(str).to_numpy() if attack_key else np.where(labels == 0, "Normal", "Anomaly")
    cache.update(ids=ids, attacks=attacks)
    joblib.dump(cache, Path(config["runtime"]) / "graph_eval.joblib")
    config["graph_eval_cache"] = str(Path(config["runtime"]) / "graph_eval.joblib")
    config["graph_eval_sha256"] = sha256(config["graph_eval_cache"])
    dump(output / "config.json", config)
    dump(output / "metadata/data.json", {"raw_csv": str(csv), "raw_sha256": sha256(csv),
         "training_rows": len(train_ids), "test_rows": len(test_ids), "source_recipe_preserved": True})
    dump(output / "metadata/graph_arguments.json", {key: value for key, value in vars(args).items() if key != "y_train"})
    return export(config, labels, scores, ids, attacks)


def evaluate_saved(source, request):
    torch = dependencies()
    import joblib
    from torch.utils.data import DataLoader
    from experiment.artifacts import paths, now
    from experiment.backends import load
    config = json.loads((source / "config.json").read_text())
    if sha256(config["graph_eval_cache"]) != config["graph_eval_sha256"]:
        raise ValueError("Graph evaluation cache changed")
    cache = joblib.load(config["graph_eval_cache"])
    checkpoint = torch.load(source / "weights/model.pt", map_location="cpu", weights_only=False)
    device = "cpu" if request["gpus"] == "cpu" else "cuda:0"
    if config["model"] == "mvids":
        module = load("models.mvids.load_data")
        model = module.MultiViewKnowledgeNet(view_lengths=checkpoint["view_lengths"], cnn_channels=32,
                         fusion_hidden=500, kg_dim=0, dropout=config["dropout"])
        dataset = module.MultiViewDataset(cache["views"], cache["labels"])
    else:
        model = load("models.mmfewshotsids.multimodal_ids").MultiModalIDS(
            **checkpoint["model_params"], A_tensor=checkpoint["adjacency"].to(device))
        dataset = cache["subset"]
    model.load_state_dict(checkpoint["state_dict"]); model.to(device)
    labels, scores = graph_scores(model, DataLoader(dataset, batch_size=config["batch_size"], shuffle=False), config["model"], device)
    config.update(name=request.get("name"), output_root=request["output_root"])
    output, runtime = paths(config)
    config.update(output=str(output), runtime=str(runtime), evaluated_from=str(source))
    dump(output / "config.json", config)
    import shutil
    shutil.copy2(source / "weights/model.pt", output / "weights/model.pt")
    shutil.copy2(source / "preprocessing.joblib", output / "preprocessing.joblib")
    shutil.copy2(source / "metadata/data.json", output / "metadata/data.json")
    export(config, labels, scores, cache["ids"], cache["attacks"])
    dump(output / "status.json", {"stage": "complete", "finished_at": now(), "evaluated_from": str(source)})
