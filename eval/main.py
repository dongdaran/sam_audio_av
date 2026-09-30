# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved\n

import argparse
import json
import os

import pandas as pd
import torch
import torchaudio
import torch.distributed as dist
from dataset import SETTINGS, make_dataset
# from metrics import CLAP, Aesthetic, ImageBind, Judge
from metrics import ImageBind
from progress import load_progress, save_progress, save_audio
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from sam_audio import SAMAudio, SAMAudioProcessor


def gather_and_average_results(results, world_size):
    if world_size == 1:
        return json.loads(results.mean().to_json())

    # 1. Gather all dictionaries to all ranks
    all_results = [None for _ in range(world_size)]
    dist.all_gather_object(
        all_results, {"sum": results.sum().to_json(), "count": len(results)}
    )

    summed = {}
    counts = 0

    for res in all_results:
        for k, v in json.loads(res["sum"]).items():
            if k not in summed:
                summed[k] = 0.0
            summed[k] += v
        counts += res["count"]

    # 3. Compute average for keys that appeared at least once
    averaged = {k: summed[k] / counts for k in summed}

    return averaged


def main(
    settings: list[str],
    cache_path: str,
    batch_size: int,
    checkpoint_path: str,
    num_workers: int = 4,
    reranking_candidates: int = 8,
):
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    if world_size != 1:
        raise ValueError("Resumable evaluation uses one process; run python, not torchrun")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if world_size > 1:
        torch.distributed.init_process_group(backend="nccl")
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)

    print("loading SAM...")
    model = SAMAudio.from_pretrained(
        checkpoint_path, text_ranker=None, span_predictor=None
    )
    print("moving SAM to GPU...")
    metric_device = torch.device("cuda:1") if torch.cuda.device_count() > 1 else device
    model = model.eval()
    for name, module in model.named_children():
        module.to(metric_device if name in {"vision_encoder", "visual_ranker"} else device)
    model.vision_encoder.batch_size = 8
    print("loading processor...")
    processor = SAMAudioProcessor.from_pretrained(checkpoint_path)

    # judge_metric = Judge(device=device)
    # aes_metric = Aesthetic(device=device)
    # clap_metric = CLAP(device=device)
    print("loading ImageBind...")
    metric_device = torch.device("cuda:1") if world_size == 1 and torch.cuda.device_count() > 1 else device
    if model.visual_ranker is not None:
        model.visual_ranker.to(metric_device)
    imagebind_metric = ImageBind(device=metric_device)

    print("all models loaded")
    for setting in settings:
        print(f"Evaluating: {setting}")
        dset = make_dataset(setting, cache_path=cache_path, collate_fn=processor)
        progress_path = f"results/{setting}.progress.json"
        identity = {
            "setting": setting, "checkpoint": checkpoint_path,
            "candidates": reranking_candidates,
            "dataset": dset.dataset._fingerprint if hasattr(dset, "dataset") else str(len(dset)),
            "cache_path": os.path.abspath(cache_path),
        }
        audio_dir = f"results/{setting}/audio"
        rows = load_progress(progress_path, identity, audio_dir=audio_dir)
        if len(rows) > len(dset):
            raise ValueError("Saved progress exceeds dataset length")
        print(f"Resuming {setting}: {len(rows)}/{len(dset)} samples complete")
        saved_count = len(rows)
        sampler = None
        if world_size > 1:
            sampler = DistributedSampler(dset)

        dl = DataLoader(
            Subset(dset, range(len(rows), len(dset))),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=dset.collate,
            num_workers=num_workers,
            sampler=sampler,
        )

        # all_metrics = [
        #     judge_metric,
        #     aes_metric,
        #     clap_metric,
        # ]

        all_metrics = []

        if dset.visual:
            all_metrics.append(imagebind_metric)

        with torch.inference_mode():
            for batch in tqdm(dl, disable=rank > 1):
                batch = batch.to(device, video_device="cpu")
                result = model.separate(
                    batch, reranking_candidates=reranking_candidates
                )
                for offset, (target, residual) in enumerate(zip(result.target, result.residual, strict=True)):
                    sample_index = len(rows) + offset
                    for kind, waveform in (("target", target), ("residual", residual)):
                        save_audio(
                            f"{audio_dir}/{sample_index:06d}_{kind}.wav",
                            waveform.detach().float().cpu().numpy().reshape(-1),
                            model.sample_rate,
                        )
                mets = {}
                for metric in all_metrics:
                    input_wavs = model.unbatch(batch.audios.squeeze(1), batch.wav_sizes)

                    mets.update(
                        metric(
                            target_wavs=result.target,
                            target_wavs_sample_rate=model.sample_rate,
                            descriptions=batch.descriptions,
                            input_wavs=input_wavs,
                            videos=batch.masked_video,
                        )
                    )

                batch_rows = pd.DataFrame.from_dict(mets).to_dict("records")
                if len(batch_rows) != len(result.target):
                    raise ValueError("Each evaluated sample must have metric results")
                for row in batch_rows:
                    rows.append(row)
                    if len(rows) - saved_count == 5:
                        save_progress(progress_path, identity, rows)
                        saved_count = len(rows)
                del result, batch

        save_progress(progress_path, identity, rows)
        if not rows:
            print(f"No available samples for {setting}")
            continue
        df = pd.DataFrame(rows)
        averaged_results = gather_and_average_results(df, world_size)
        if rank == 0:
            results_dict = {k: f"{v:.3f}" for k, v in averaged_results.items()}
            print(json.dumps(results_dict, indent=4))
            os.makedirs("results", exist_ok=True)
            outfile = f"results/{setting}.json"
            with open(outfile, "w") as fout:
                print(json.dumps(results_dict), file=fout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--setting",
        "-s",
        choices=SETTINGS.keys(),
        help=f"Which setting to evaluate.  Choices: {SETTINGS.keys()}",
        default=["instr-pro"],
        nargs="+",
    )
    parser.add_argument(
        "--cache-path",
        type=str,
        default=os.path.expanduser("~/.cache/sam_audio"),
        help="Where to cache downloaded datasets",
    )
    parser.add_argument(
        "--checkpoint-path", "-p", type=str, default="facebook/sam-audio-large"
    )
    parser.add_argument("--batch-size", "-b", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--num-workers", "-w", type=int, default=4, help="Number of workers"
    )
    parser.add_argument("--candidates", "-c", type=int, default=8)
    opt = parser.parse_args()
    main(
        settings=opt.setting,
        cache_path=opt.cache_path,
        batch_size=opt.batch_size,
        checkpoint_path=opt.checkpoint_path,
        num_workers=opt.num_workers,
        reranking_candidates=opt.candidates,
    )
