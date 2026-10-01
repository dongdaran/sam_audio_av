"""Compare original/target ImageBind scores; see the Compare section in README.md."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics
import sys

import pyarrow.parquet as pq
import soundfile as sf
import torch
import torch.nn.functional as F
from datasets import Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dataset.sam_audio_bench import SAMAudioBench
from metrics.imagebind import ImageBind
from progress import save_json
from sam_audio.processor import SAMAudioProcessor

SETTINGS = {'sfx-visual': 'SFX', 'speaker-visual': 'speech', 'instr-wild-visual': 'instr'}
METRICS = ('IB_original', 'IB_target', 'IB_gain')


def sample_key(row):
    return (row['video_id'], row['source_dataset'], row['description'],
            tuple(tuple(span) for span in row['spans']))


def select_extremes(rows):
    """Highest/lowest saved target score per setting; ties use sample_index."""
    return [min(rows, key=lambda r: (-r['IB_target'], r['sample_index'])),
            min(rows, key=lambda r: (r['IB_target'], r['sample_index']))]


def summarize(rows):
    def mean(group):
        return {'count': len(group), **{key: statistics.mean(r[key] for r in group) for key in METRICS}}
    result = {'overall': mean(rows)}
    for field in ('source_dataset', 'type'):
        result[f'by_{field}'] = {key: mean([r for r in rows if r[field] == key])
                               for key in sorted({r[field] for r in rows})}
    return result


def load_inputs(args):
    columns = ['video_id', 'source_dataset', 'description', 'spans', 'start_offset',
               'end_offset', 'video_path', 'video_is_full_length']
    metadata = pq.read_table(args.metadata, columns=columns).to_pylist()
    lookup = defaultdict(list)
    for index, item in enumerate(metadata):
        lookup[sample_key(item)].append((index, item))
    groups = {}
    for setting in args.setting:
        progress = json.loads((args.results_dir / f'{setting}.progress.json').read_text())
        manifest = json.loads((args.results_dir / setting / 'manifest.json').read_text())
        if progress['identity'] != manifest['identity'] or progress['identity']['setting'] != setting:
            raise ValueError(f'{setting}: progress/manifest identity mismatch')
        samples = {r['sample_index']: r for r in manifest['samples']}
        if (not samples or len(samples) != len(manifest['samples']) or
                set(samples) != set(range(len(progress['rows'])))):
            raise ValueError(f'{setting}: duplicate/missing indices or incomplete progress')
        rows = []
        for index, metric in enumerate(progress['rows']):
            sample = samples[index]
            score = metric['ImageBind']
            if isinstance(score, bool) or not isinstance(score, (float, int)) or not math.isfinite(score):
                raise ValueError(f'{setting}/{index}: invalid target score')
            matches = lookup[sample_key(sample)]
            if len(matches) != 1:
                raise ValueError(f'{setting}/{index}: expected one metadata match, got {len(matches)}')
            metadata_index, item = matches[0]
            video = args.cache_path / 'sam_audio_bench' / item['video_path']
            if not video.is_file():
                raise FileNotFoundError(video)
            target = args.results_dir / setting / 'audio' / f'{index:06d}_target.wav'
            info = sf.info(target)
            if info.channels != 1 or info.frames <= 0:
                raise ValueError(f'{target}: expected nonempty mono WAV')
            rows.append(dict(sample, setting=setting, type=SETTINGS[setting], IB_target=score,
                             metadata_index=metadata_index, start_offset=item['start_offset'],
                             end_offset=item['end_offset'], video_path=item['video_path'],
                             target_path=str(target), target_num_samples=info.frames,
                             sample_rate=info.samplerate))
        groups[setting] = {'identity': progress['identity'], 'rows': rows}
    return groups


def compute_score(row, dataset, metric, original=False):
    """Use the same loader, processor, and ImageBind metric as main.py."""
    index = row['sample_index']
    batch = dataset.collate([dataset[index]])
    processor = dataset.collate_fn
    input_length = int(batch.wav_sizes[0])
    hop = processor.audio_hop_length
    expected_length = ((input_length + hop - 1) // hop) * hop
    if row['sample_rate'] != processor.audio_sampling_rate or row['target_num_samples'] != expected_length:
        raise ValueError(f'{row["setting"]}/{index}: target rate/length does not match processor: '
                         f'saved={row["sample_rate"]}Hz/{row["target_num_samples"]}, '
                         f'expected={processor.audio_sampling_rate}Hz/{expected_length}')
    if original:
        # Target decoding rounds up to a codec frame. Equal durations keep metric video sampling identical.
        waveform = F.pad(batch.audios[0, 0, :input_length], (0, expected_length - input_length))
    else:
        audio, _ = sf.read(row['target_path'], dtype='float32')
        waveform = torch.from_numpy(audio)
    if not torch.isfinite(waveform).all():
        raise ValueError(f'{row["setting"]}/{index}: non-finite audio')
    with torch.inference_mode():
        value = metric(target_wavs=[waveform], videos=batch.masked_video,
                       target_wavs_sample_rate=processor.audio_sampling_rate)['ImageBind'][0]
    if not math.isfinite(value):
        raise ValueError(f'{row["setting"]}/{index}: non-finite ImageBind result')
    return value, input_length


def main(args):
    # 1. Match saved results to local metadata and choose target-score extremes.
    groups = load_inputs(args)
    for setting, group in groups.items():
        print(f'{setting}: {len(group["rows"])} samples; target verification:', flush=True)
        for tag, row in zip(('highest', 'lowest'), select_extremes(group['rows']), strict=True):
            print(f'  {tag}: {row["sample_index"]:06d} {row["video_id"]} '
                  f'IB_target={row["IB_target"]:.9f}', flush=True)
    # 2. Load ImageBind and the existing processor/dataset. No SAM separation model.
    device = args.device
    print(f'Loading ImageBind on {device}; SAM separation model is not loaded.', flush=True)
    metric = ImageBind(checkpoint=args.imagebind_checkpoint, device=torch.device(device))
    table = Dataset.from_parquet(str(args.metadata))
    datasets, processors = {}, {}
    for setting, group in groups.items():
        checkpoint = args.processor_config or group['identity']['checkpoint']
        if checkpoint not in processors:
            processors[checkpoint] = SAMAudioProcessor.from_pretrained(checkpoint)
        processor = processors[checkpoint]
        data = table.select([row['metadata_index'] for row in group['rows']])
        dataset = SAMAudioBench(str(args.cache_path), processor, span=False, visual=True, dataset=data)
        if len(dataset) != len(group['rows']):
            raise ValueError(f'{setting}: loader skipped samples; check video paths')
        for row, item in zip(group['rows'], dataset.dataset, strict=True):
            path, full_length = dataset._get_path(item['video_id'], item['source_dataset'],
                                                  item['start_offset'], item['end_offset'])
            expected = args.cache_path / 'sam_audio_bench' / row['video_path']
            if Path(path).resolve() != expected.resolve() or full_length != item['video_is_full_length']:
                raise ValueError(f'{setting}/{row["sample_index"]}: loader/video metadata mismatch')
        datasets[setting] = dataset

    # 3. Verify each setting's highest/lowest saved target score before comparing originals.
    run = {'metric': 'ImageBind', 'pipeline_version': 1, 'device': device,
           'torch_version': torch.__version__, 'atol': args.atol,
           'imagebind_checkpoint': args.imagebind_checkpoint or 'imagebind_huge pretrained default',
           'processor_config': args.processor_config, 'metadata': str(args.metadata),
           'audio_alignment': 'original zero-padded to saved target codec-frame length',
           'sources': {s: g['identity'] for s, g in groups.items()}}
    records = []
    verification_path = args.output_dir / 'verification.json'
    selected = [row for group in groups.values() for row in select_extremes(group['rows'])]
    save_json(verification_path, {'run': run, 'status': 'running', 'samples': records})
    for number, row in enumerate(selected, 1):
        print(f'[verify {number}/{len(selected)}] {row["setting"]}/{row["sample_index"]:06d} '
              f'{row["video_id"]} {row["description"]}', flush=True)
        try:
            value, input_length = compute_score(row, datasets[row['setting']], metric)
        except Exception as error:
            save_json(verification_path, {'run': run, 'status': 'error', 'samples': records,
                                          'error': f'{row["setting"]}/{row["sample_index"]}: {error}'})
            raise
        difference = abs(value - row['IB_target'])
        records.append({**row, 'IB_target_recomputed': value, 'absolute_error': difference,
                        'passed': difference <= args.atol, 'input_num_samples': input_length})
        print(f'  saved={row["IB_target"]:.9f}, recomputed={value:.9f}, '
              f'abs_error={difference:.9g}, {"PASS" if difference <= args.atol else "FAIL"}', flush=True)
        save_json(verification_path, {'run': run, 'status': 'running', 'samples': records})
    passed = all(r['passed'] for r in records)
    save_json(verification_path, {'run': run, 'status': 'passed' if passed else 'failed', 'samples': records})
    if not passed:
        raise ValueError(f'Target verification FAILED (atol={args.atol:g}); inspect verification.json. '
                         'Original scoring was not started. Check preprocessing and model versions.')
    if args.verify_only:
        print(f'Verification passed: {verification_path}', flush=True)
        return

    # 4. Score original audio, retain saved target scores, and aggregate paired differences.
    all_rows = []
    for setting, group in groups.items():
        completed = []
        output = args.output_dir / f'{setting}.json'
        payload = {'identity': group['identity'], 'run': run, 'status': 'running', 'samples': completed}
        save_json(output, payload)
        for number, row in enumerate(group['rows'], 1):
            print(f'[original {setting} {number}/{len(group["rows"])}] '
                  f'{row["sample_index"]:06d} {row["video_id"]}', flush=True)
            try:
                value, input_length = compute_score(row, datasets[setting], metric, original=True)
            except Exception as error:
                save_json(output, {**payload, 'status': 'error', 'error': str(error)})
                raise
            result = {**row, 'IB_original': value, 'IB_gain': row['IB_target'] - value,
                      'input_num_samples': input_length,
                      'original_padding_samples': row['target_num_samples'] - input_length}
            completed.append(result)
            save_json(output, payload)
            print(f'  original={value:.6f}, target={row["IB_target"]:.6f}, gain={result["IB_gain"]:+.6f}', flush=True)
        save_json(output, {**payload, 'status': 'complete', 'summary': summarize(completed)})
        all_rows.extend(completed)
    summary = summarize(all_rows)
    save_json(args.output_dir / 'summary.json', {'run': run, 'status': 'complete', **summary})
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--setting', nargs='+', choices=SETTINGS, default=list(SETTINGS))
    parser.add_argument('--cache-path', type=Path, default=ROOT, help='Parent of sam_audio_bench/')
    parser.add_argument('--metadata', type=Path, default=ROOT / 'sam_audio_bench/data/test.parquet')
    parser.add_argument('--results-dir', type=Path, default=ROOT / 'results')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results/ib_comparison')
    parser.add_argument('--device', default='cuda', help='GPU device, e.g. cuda or cuda:0')
    parser.add_argument('--imagebind-checkpoint', help='Optional local ImageBind checkpoint')
    parser.add_argument('--processor-config', help='Matching SAM checkpoint ID or directory with config.json only')
    parser.add_argument('--atol', type=float, default=1e-4, help='Absolute target verification tolerance')
    parser.add_argument('--verify-only', action='store_true', help='Verify target extremes, then stop')
    args = parser.parse_args()
    if not math.isfinite(args.atol) or args.atol < 0:
        parser.error('--atol must be finite and nonnegative')
    if len(set(args.setting)) != len(args.setting):
        parser.error('--setting must not contain duplicates')
    main(args)
