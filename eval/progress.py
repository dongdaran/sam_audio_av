"""Atomic, configuration-checked evaluation checkpoints."""

import json
import os
from pathlib import Path


def load_progress(path, identity, audio_dir=None):
    path = Path(path)
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    if data["identity"] != identity:
        raise ValueError(f"Evaluation settings or dataset changed; move {path} before restarting")
    rows = data["rows"]
    if audio_dir is not None:
        for index in range(len(rows)):
            if not all((Path(audio_dir) / f"{index:06d}_{kind}.wav").is_file()
                       for kind in ("target", "residual")):
                return rows[:index]
    return rows


def save_audio(path, waveform, sample_rate):
    import soundfile as sf

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".wav.tmp")
    sf.write(temporary, waveform, sample_rate, format="WAV", subtype="FLOAT")
    os.replace(temporary, path)


def save_progress(path, identity, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump({"identity": identity, "rows": rows}, stream, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
