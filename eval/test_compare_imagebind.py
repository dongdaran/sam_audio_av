"""Run: python eval/test_compare_imagebind.py. No model weights or GPU required."""
from contextlib import nullcontext, redirect_stdout
import io
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

# Keep production imports simple; replace GPU/model modules only inside this test.
mock_torch = Mock()
with patch.dict(sys.modules, {'torch': mock_torch, 'torch.nn': mock_torch.nn,
                             'torch.nn.functional': mock_torch.nn.functional,
                             'datasets': Mock(), 'dataset.sam_audio_bench': Mock(),
                             'metrics.imagebind': Mock(), 'sam_audio.processor': Mock()}):
    import compare_imagebind as compare


def expect_error(function, text):
    try:
        function()
    except ValueError as error:
        assert text in str(error), str(error)
    else:
        raise AssertionError(f'Expected error containing: {text}')


with TemporaryDirectory() as directory:
    root = Path(directory)
    args = SimpleNamespace(setting=list(compare.SETTINGS), cache_path=root,
                           results_dir=root / 'results', metadata=root / 'test.parquet',
                           output_dir=root / 'comparison', verify_only=False,
                           device='cpu', processor_config=None, imagebind_checkpoint=None, atol=1e-4)
    metadata = []
    for setting in args.setting:
        samples = []
        for index in range(2):
            sample = dict(sample_index=index, video_id=f'{setting}-{index}', source_dataset='source',
                          description='sound', spans=[[0.0, 1.0]])
            samples.append(sample)
            video_path = f'source/{sample["video_id"]}.mp4'
            path = root / 'sam_audio_bench' / video_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'path-only fixture')
            metadata.append(dict(sample, start_offset=0.0, end_offset=1.0,
                                 video_path=video_path, video_is_full_length=True))
            wav = args.results_dir / setting / 'audio' / f'{index:06d}_target.wav'
            wav.parent.mkdir(parents=True, exist_ok=True)
            sf.write(wav, np.zeros(48000), 48000)
        identity = dict(setting=setting, checkpoint='test-config')
        compare.save_json(args.results_dir / f'{setting}.progress.json',
                          dict(identity=identity, rows=[{'ImageBind': .2}, {'ImageBind': .4}]))
        compare.save_json(args.results_dir / setting / 'manifest.json',
                          dict(identity=identity, samples=samples[::-1]))
    # Neither manifest nor parquet row order defines sample_index.
    metadata.reverse()
    pq.write_table(pa.Table.from_pylist(metadata), args.metadata)
    groups = compare.load_inputs(args)
    for group in groups.values():
        assert [r['sample_index'] for r in group['rows']] == [0, 1]
        assert [r['sample_index'] for r in compare.select_extremes(group['rows'])] == [1, 0]
        assert all(metadata[r['metadata_index']]['video_id'] == r['video_id'] for r in group['rows'])
    pq.write_table(pa.Table.from_pylist(metadata + [metadata[0]]), args.metadata)
    expect_error(lambda: compare.load_inputs(args), 'expected one metadata match')
    pq.write_table(pa.Table.from_pylist(metadata), args.metadata)

    rows = [dict(type=kind, source_dataset=source, IB_original=value, IB_target=value + .1, IB_gain=.1)
            for kind, source, value in [('SFX', 'a', 0), ('SFX', 'a', .2), ('speech', 'b', .7)]]
    summary = compare.summarize(rows)
    assert np.isclose(summary['overall']['IB_original'], .3)
    assert summary['overall']['count'] == 3
    assert summary['by_source_dataset']['a']['count'] == 2
    assert np.isclose(summary['by_type']['speech']['IB_gain'], .1)

    # Exercise main's verification gate and output statuses without loading any models.
    class TestDataset:
        def __init__(self, cache_path, processor, **kwargs):
            self.dataset = kwargs['dataset']

        def __len__(self):
            return len(self.dataset)

        def _get_path(self, video_id, source_dataset, start, end):
            return str(root / 'sam_audio_bench' / source_dataset / f'{video_id}.mp4'), True

    table = Mock()
    table.select.side_effect = lambda indices: [metadata[i] for i in indices]
    mocked_dataset = SimpleNamespace(from_parquet=lambda path: table)
    mocked_processor = SimpleNamespace(from_pretrained=lambda path: object())
    calls = []

    def scoring(row, dataset, metric, original=False):
        calls.append(original)
        return (.1 if original else row['IB_target']), 48000

    with patch.multiple(compare, ImageBind=Mock(),
                        Dataset=mocked_dataset, SAMAudioProcessor=mocked_processor,
                        SAMAudioBench=TestDataset,
                        torch=SimpleNamespace(device=lambda name: name, __version__='test'),
                        compute_score=scoring, create=True), redirect_stdout(io.StringIO()):
        compare.main(args)
        assert calls == [False] * 6 + [True] * 6
        verification = json.loads((args.output_dir / 'verification.json').read_text())
        assert verification['status'] == 'passed' and len(verification['samples']) == 6
        assert json.loads((args.output_dir / 'summary.json').read_text())['overall']['count'] == 6
        for setting in args.setting:
            result = json.loads((args.output_dir / f'{setting}.json').read_text())
            assert result['status'] == 'complete'
            assert all(np.isclose(r['IB_gain'], r['IB_target'] - .1) for r in result['samples'])
        args.output_dir = root / 'failed'
        bad_score = Mock(return_value=(-.9, 48000))
        with patch.object(compare, 'compute_score', bad_score):
            expect_error(lambda: compare.main(args), 'Target verification FAILED')
        assert bad_score.call_count == 6
        assert all(not call.kwargs.get('original', False) for call in bad_score.call_args_list)
        assert json.loads((args.output_dir / 'verification.json').read_text())['status'] == 'failed'
        assert not (args.output_dir / 'sfx-visual.json').exists()

print('PASS: metadata matching, extremes, weighted means, six-sample gate, failure stops original scoring')

# Check waveform selection/padding with NumPy standing in for the tensor operations.
batch = SimpleNamespace(wav_sizes=[5], audios=np.arange(5.0).reshape(1, 1, 5), masked_video=[object()])
dataset = Mock()
dataset.__getitem__ = Mock(return_value=object())
dataset.collate.return_value = batch
dataset.collate_fn = SimpleNamespace(audio_hop_length=4, audio_sampling_rate=48000)
row = dict(setting='sfx-visual', sample_index=0, sample_rate=48000, target_num_samples=8)
metric = Mock(return_value={'ImageBind': [.25]})
tensor_ops = SimpleNamespace(isfinite=np.isfinite, from_numpy=np.asarray, inference_mode=nullcontext)
with patch.multiple(compare, torch=tensor_ops, F=SimpleNamespace(pad=np.pad)):
    value, length = compare.compute_score(row, dataset, metric, original=True)
    assert value == .25 and length == 5
    np.testing.assert_array_equal(metric.call_args.kwargs['target_wavs'][0], [0, 1, 2, 3, 4, 0, 0, 0])
    assert metric.call_args.kwargs['videos'] is batch.masked_video
    row['target_num_samples'] = 7
    expect_error(lambda: compare.compute_score(row, dataset, metric, original=True), 'rate/length')
print('PASS: original padding, shared video input, target length mismatch (mock tensor backend)')
