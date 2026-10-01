# Evaluation

This directory contains the evaluation code to reproduce the results from the SAM-Audio paper. The evaluation framework supports multiple datasets, prompting modes (text-only, span, visual), and metrics.

## Setup

Before running evaluation, ensure you have:

1. Installed the SAM-Audio package and its dependencies
2. Authenticated with Hugging Face to access the model checkpoints (see main [README](../README.md))

## Quick Start

Run evaluation on the default setting (instr-pro):

```bash
python main.py
```

You can also use multiple GPUs to speed up evaluation:

```bash
torchrun --nproc_per_node=<ngpus> python main.py
```

Evaluate on a specific setting:

```bash
python main.py --setting sfx
```

Evaluate on multiple settings:

```bash
python main.py --setting sfx speech music
```

## Available Evaluation Settings

Run `python main.py --help` to see all available settings

## Command Line Options

```bash
python main.py [OPTIONS]
```

### Options:

- `-s, --setting` - Which setting(s) to evaluate (default: `instr-pro`)
  - Choices: See available settings above
  - Can specify multiple settings: `--setting sfx speech music`

- `--cache-path` - Where to cache downloaded datasets (default: `~/.cache/sam_audio`)

- `-p, --checkpoint-path` - Model checkpoint to evaluate (default: `facebook/sam-audio-1b`)
  - Can use local path or Hugging Face model ID

- `-b, --batch-size` - Batch size for evaluation (default: `1`)

- `-w, --num-workers` - Number of data loading workers (default: `4`)

- `-c, --candidates` - Number of reranking candidates (default: `8`)

## Evaluation Metrics

The evaluation framework computes the following metrics:

- **Judge** - SAM Audio Judge quality assessment metric
- **Aesthetic** - Aesthetic quality metric
- **CLAP** - Audio-text alignment metric (CLAP similarity)
- **ImageBind** - Audio-video alignment metric (for visual settings only)

## Output

Results are saved to the `results/` directory as JSON files, one per setting:

```
results/
├── sfx.json
├── speech.json
└── music.json
```

Each JSON file contains the averaged metric scores across all samples in that setting.

Example output:
```json
{
    "JudgeOverall": "4.386",
    "JudgeFaithfulness": "4.708",
    "JudgeRecall": "4.934",
    "JudgePrecision": "4.451",
    "ContentEnjoyment": "5.296",
    "ContentUsefulness": "6.903",
    "ProductionComplexity": "4.301",
    "ProductionQuality": "7.100",
    "CLAPSimilarity": "0.271"
}
```

## Compare: 원본 오디오 vs. target ImageBind

`eval/compare_imagebind.py`는 **원본 혼합 오디오와 마스크 영상의 ImageBind 점수**를 계산하고,
이미 저장된 target 점수와 비교합니다. SAM Audio 분리 모델을 다시 실행하거나 target WAV를 재생성하지 않습니다.
기존 `eval/metrics/imagebind.py`, `SAMAudioBench`, `SAMAudioProcessor`를 재사용합니다.

| 결과 필드 | 의미 |
|---|---|
| `IB_original` | 원본 혼합 오디오 ↔ 마스크 영상 |
| `IB_target` | 기존 `*.progress.json`에 저장된 target 오디오 ↔ 마스크 영상 점수 |
| `IB_gain` | `IB_target - IB_original`; 양수이면 분리 후 IB 유사도 증가 |

평가 영상은 **배경을 검게 만든 마스크 영상**입니다. HTML의 초록색 강조 영상이나 축소된 미리보기 MP4로 평가하지 않습니다.
평가 구간도 기존 클립의 `start_offset`~`end_offset`이며, `spans` 구간만 따로 자르지 않습니다.
잘려 저장된 원본 MP4는 기존 로더와 동일하게 파일 전체를 사용합니다.

기존 target은 codec frame 길이로 올림하여 저장됩니다. 두 오디오의 길이에 따라 ImageBind의 영상 프레임 선택이
달라지는 것을 막기 위해 원본 끝에 0을 채워 target 길이에 맞춥니다(한 codec frame 미만).
실제 원본 길이는 `input_num_samples`, 추가 길이는 `original_padding_samples`에 기록합니다.
target 길이가 processor 설정과 맞지 않으면 임의로 자르지 않고 오류로 중단합니다.

### 실행 환경과 필요한 파일

아래 명령은 **프로젝트 루트에서, 기존 `eval/main.py`를 실행하던 Python 환경**으로 실행합니다.
PyTorch/Torchaudio, TorchCodec, ImageBind와 SAM Audio의 import 의존성이 필요합니다.
이 저장소의 현재 로컬 `.venv`는 미디어 작업용이므로 GPU 평가 의존성이 모두 설치되어 있지는 않습니다.

- `results/<setting>.progress.json`: 샘플별 기존 target 점수
- `results/<setting>/manifest.json`: 점수와 샘플의 대응
- `results/<setting>/audio/*_target.wav`: 재현 검증용 저장된 target
- `sam_audio_bench/data/test.parquet`: 로컬 메타데이터와 원본 마스크
- `sam_audio_bench/<source_dataset>/*.mp4`: 원본 영상·오디오

기본 설정은 `sfx-visual`, `speaker-visual`, `instr-wild-visual` 전체입니다.
현재 결과는 각각 41, 41, 49개이며, 기본 전체 비교는 131개입니다.
manifest의 `sample_index`로 점수를 연결하고, video ID·source_dataset·설명·spans로
Parquet 행을 유일하게 찾아 파일 순서가 바뀌어도 샘플을 혼동하지 않습니다.

### 실행 방법

**6개 검증 → 전체 원본 점수 계산 → 집계**를 한 번에 실행하는 권장 명령입니다.
기본 실행 장치는 `cuda`이며 로그도 파일에 남겨 오류를 확인할 수 있습니다.

```bash
set -o pipefail
python -u eval/compare_imagebind.py --device cuda 2>&1 | tee compare_imagebind.log
```

검증만 먼저 실행하고 싶다면 다음 명령을 사용합니다. 이후 전체 실행 시에는 검증도 다시 수행합니다.

```bash
python -u eval/compare_imagebind.py --device cuda --verify-only
```

검증 대상은 **각 유형 안에서 기존 target IB가 가장 높은 1개·낮은 1개**입니다.
동점이면 작은 `sample_index`를 선택합니다. 현재 결과의 검증 대상은 다음과 같습니다.

| 유형 | 최고 sample_index | 최저 sample_index |
|---|---:|---:|
| SFX | 1 | 21 |
| speech (`speaker-visual`) | 13 | 33 |
| instr (`instr-wild-visual`) | 9 | 46 |

각 샘플의 기존 점수·재계산 점수·절대 오차·PASS/FAIL을 출력합니다.
기본 절대 오차 허용값은 `--atol 0.0001`이며, **6개가 모두 통과해야 원본 평가를 시작**합니다.
실패하면 `verification.json`과 로그를 확인해 모델·전처리 조건을 먼저 점검하세요.
이 검증은 모든 샘플의 target 점수가 동일함을 보증하는 전수 재평가는 아닙니다.

경로가 다른 서버에서는 필요한 경로를 지정할 수 있습니다.
`--cache-path`는 `sam_audio_bench/` 자체가 아니라 **그 상위 디렉터리**입니다.

```bash
python -u eval/compare_imagebind.py \
  --device cuda:0 \
  --cache-path /workspace/sam_audio_av \
  --metadata /workspace/sam_audio_av/sam_audio_bench/data/test.parquet \
  --results-dir /workspace/sam_audio_av/results \
  --output-dir /workspace/sam_audio_av/results/ib_comparison
```

- `--setting sfx-visual`: 특정 유형만 실행합니다. 이 경우 검증은 해당 유형의 2개입니다.
- `--imagebind-checkpoint /path/to/imagebind_huge.pth`: 기존 평가와 동일한 로컬 ImageBind 가중치를 사용합니다. 생략하면 기존 metric과 같은 기본 pretrained 모델을 사용합니다.
- `--processor-config /path/to/sam-config`: 기존 SAM checkpoint와 동일한 `config.json`이 있는 디렉터리입니다. 생략하면 manifest의 checkpoint ID에서 processor 설정만 읽습니다. SAM 분리 모델 가중치는 로드하지 않습니다.
- `--device cuda:1`: 사용할 GPU를 지정합니다. CPU 자동 대체는 하지 않습니다.

### 결과 파일과 HTML

기존 평가 결과는 수정하지 않고 아래에 별도로 저장합니다.

```text
results/ib_comparison/
├── verification.json          # 검증한 target 6개의 기존/재계산 점수와 오차
├── sfx-visual.json            # 샘플별 원본·target·차이, 유형 집계
├── speaker-visual.json
├── instr-wild-visual.json
└── summary.json               # 전체, source_dataset별, 유형별 집계
```

샘플 결과에는 `sample_index`, `video_id`, `source_dataset`, 설명, 구간과 세 점수가 포함됩니다.
평균은 그룹별 샘플 산술평균이며, 전체는 유형별 평균을 단순평균하지 않습니다.
계산 도중에도 완료한 샘플을 저장하고 `status`를 기록합니다. **자동 이어하기는 없으며 재실행하면 처음부터 계산합니다.**
오류·실행 중 파일을 완료 결과로 해석하면 안 됩니다. 기존 `summary.json`이 남아 있다면 각 파일의 `run`과 `status`도 확인하세요.

결과를 기존 HTML에 반영하려면 다음을 실행합니다. 기존 미리보기 영상을 재사용하므로 재인코딩하지 않습니다.

```bash
python scripts/build_results_report.py --html-only
```

미리보기 파일이 없는 서버에서는 `--html-only` 없이 실행하면 영상까지 생성합니다.
다른 서버에서 점수만 계산했다면 `results/ib_comparison/`을 이 프로젝트의 같은 위치로 복사한 뒤 HTML을 갱신해도 됩니다.
HTML은 기본 위치의 세 유형 결과가 모두 완료되고 검증을 통과했을 때만 원본·target·차이를 함께 표시합니다.
상·하위 5개 순위는 계속 **기존 target IB 기준**입니다. 오디오·영상 동기 재생도 유지됩니다.

모델 없이 실행하는 데이터 처리 테스트:

```bash
python eval/test_compare_imagebind.py
```
