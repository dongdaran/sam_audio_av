"""Run with python eval/test_decode_batch.py; no model or GPU required."""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call


def test_decode_batch():
    source = Path(__file__).resolve().parents[1] / "sam_audio/model/model.py"
    tree = ast.parse(source.read_text())
    separate = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "separate")
    assignment = next(
        n for n in separate.body if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "wavs" for t in n.targets)
    )
    chunks = [object() for _ in range(16)]
    decoded = [object() for _ in chunks]
    features = Mock()
    features.reshape.return_value.split.return_value = chunks
    decoder = Mock(side_effect=decoded)
    torch = Mock()
    namespace = dict(
        B=8, C=4, T=10, generated_features=features, torch=torch,
        self=SimpleNamespace(audio_codec=SimpleNamespace(decode=decoder)),
    )
    exec(compile(ast.Module(body=[assignment], type_ignores=[]), str(source), "exec"), namespace)
    features.reshape.assert_called_once_with(16, 4, 10)
    features.reshape.return_value.split.assert_called_once_with(1)
    assert decoder.call_args_list == [call(chunk) for chunk in chunks]
    torch.cat.assert_called_once_with(decoded, dim=0)
    torch.cat.return_value.view.assert_called_once_with(8, 2, -1)


if __name__ == "__main__":
    test_decode_batch()
    print("Sequential decoding and output ordering: OK")
