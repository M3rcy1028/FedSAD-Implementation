"""Keep paper results separate from generated outputs and runtime state."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]


def now():
    return datetime.now(ZoneInfo("Asia/Seoul")).isoformat()


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, default=str) + "\n")
    os.replace(temp, path)


def writable_output(path):
    path = Path(path).expanduser().resolve()
    paper = (ROOT / "results").resolve()
    if path == paper or paper in path.parents:
        raise ValueError("results/ contains paper artifacts; choose an outputs/ destination")
    return path


def paths(config):
    run_id = config.get("name") or datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d_%H%M%S_%f")
    if Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("Run name must be a single directory name")
    output = writable_output(Path(config["output_root"]) / config["dataset"] / config["model"] / run_id)
    runtime = writable_output(ROOT / "runtime" / config["dataset"] / config["model"] / run_id)
    if output.exists() or runtime.exists():
        raise FileExistsError(f"Run already exists: {output}")
    for folder in [output / "weights", output / "figures", output / "tables", output / "metadata", runtime]:
        folder.mkdir(parents=True)
    return output, runtime
