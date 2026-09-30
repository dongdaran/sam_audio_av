"""Run with python eval/test_progress.py; no GPU required."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from progress import load_progress, save_progress, save_audio


with TemporaryDirectory() as directory:
    path = Path(directory) / "progress.json"
    identity = {"dataset": "fixed", "candidates": 8}
    assert load_progress(path, identity) == []
    rows = [{"ImageBind": i / 10} for i in range(7)]
    save_progress(path, identity, rows[:5])
    resumed = load_progress(path, identity)
    assert list(range(len(resumed), 7)) == [5, 6]
    with patch("progress.os.replace", side_effect=OSError("interrupted write")):
        try:
            save_progress(path, identity, rows)
        except OSError:
            pass
    assert load_progress(path, identity) == rows[:5]
    save_progress(path, identity, resumed + rows[5:])
    assert load_progress(path, identity) == rows
    import numpy as np
    import soundfile as sf

    audio_dir = Path(directory) / "audio"
    assert load_progress(path, identity, audio_dir) == []
    waveform = np.array([0.0, 0.5, -0.5, 1.25], dtype=np.float32)
    for kind in ("target", "residual"):
        output = audio_dir / f"000000_{kind}.wav"
        save_audio(output, waveform, 48000)
        restored, rate = sf.read(output, dtype="float32")
        assert rate == 48000
        np.testing.assert_array_equal(restored, waveform)
    assert load_progress(path, identity, audio_dir) == rows[:1]
    (audio_dir / "000000_residual.wav").unlink()
    assert load_progress(path, identity, audio_dir) == []
    try:
        load_progress(path, {**identity, "candidates": 1})
    except ValueError:
        pass
    else:
        raise AssertionError("Changed evaluation settings must not reuse results")
print("Progress saving, resume, and interrupted-write recovery: OK")
