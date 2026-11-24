"""Read architecture metadata while preserving legacy latent-64 checkpoints."""

import json
from pathlib import Path


def resolve_latent_dim(weights_path, requested=None):
    if requested is not None:
        value = requested
    else:
        metadata = Path(weights_path).with_name("model_config.json")
        value = json.loads(metadata.read_text()).get("latent_dim", 64) if metadata.is_file() else 64
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("Checkpoint latent_dim must be a positive integer")
    return value
