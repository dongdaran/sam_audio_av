"""Atomic, configuration-checked evaluation checkpoints."""

import json
import os
from pathlib import Path


def load_progress(path, identity):
    path = Path(path)
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    if data["identity"] != identity:
        raise ValueError(f"Evaluation settings or dataset changed; move {path} before restarting")
    return data["rows"]


def save_progress(path, identity, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump({"identity": identity, "rows": rows}, stream, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
