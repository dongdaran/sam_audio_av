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
    save_json(path, {"identity": identity, "rows": rows})


def save_manifest(path, dataset, identity):
    samples = []
    for index, item in enumerate(dataset.dataset):
        samples.append({
            "sample_index": index,
            "video_id": item["video_id"],
            "source_dataset": item["source_dataset"],
            "description": item["description"],
            "spans": item["spans"],
        })
    save_json(path, {"identity": identity, "samples": samples})


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(data, stream, allow_nan=False, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
