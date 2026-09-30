"""Run with python eval/test_progress.py; no GPU required."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from progress import load_progress, save_progress


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
    try:
        load_progress(path, {**identity, "candidates": 1})
    except ValueError:
        pass
    else:
        raise AssertionError("Changed evaluation settings must not reuse results")
print("Progress saving, resume, and interrupted-write recovery: OK")
