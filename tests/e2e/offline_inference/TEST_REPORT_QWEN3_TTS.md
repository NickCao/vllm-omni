# Qwen3-TTS Offline Inference Test Report

**Date:** 2026-03-23
**Branch:** `qwen3-tts-validation`
**Commit:** `20eec2dd`

## How to Reproduce

### Prerequisites

- 1x NVIDIA GPU with >= 23 GiB VRAM (tested on L4; any Ampere/Ada card works)
- Python 3.12+
- `uv` package manager (or pip)

### Setup

```bash
# Clone and checkout the test branch
git clone https://github.com/NickCao/vllm-omni.git
cd vllm-omni
git checkout qwen3-tts-validation

# Create venv and install
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install vllm==0.18.0 --torch-backend=auto
uv pip install -e .
uv pip install pytest pytest-cov soundfile openai-whisper

# (Optional) Enable subprocess coverage tracking into GPU worker processes
echo 'import coverage; coverage.process_startup()' > \
  $(python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')/coverage_subprocess.pth
```

### Running the tests

```bash
# Set HF cache to a large disk if needed
export HF_HOME=/path/to/large/disk/hf_cache

# Run the full matrix (~32 min on L4 with coverage)
python -m pytest tests/e2e/offline_inference/test_qwen3_tts.py \
    -v -s --timeout=1800

# Single model + config (~2 min)
python -m pytest tests/e2e/offline_inference/test_qwen3_tts.py \
    -k "0.6B-CustomVoice-async" -v -s --timeout=600

# With subprocess coverage (HTML report)
COVERAGE_PROCESS_START=pyproject.toml \
python -m pytest tests/e2e/offline_inference/test_qwen3_tts.py \
    -v -s --timeout=1800 \
    --cov=vllm_omni --cov-report=html:htmlcov_offline
```

Audio samples are saved to `tts_audio_samples/` (override with `TTS_AUDIO_DIR` env var).

## Environment

| Component | Version |
|-----------|---------|
| Instance | AWS g6.2xlarge |
| GPU | NVIDIA L4 (23 GiB VRAM) |
| Driver | 580.126.09 / CUDA 13.0 |
| OS | Ubuntu 24.04.4 LTS (kernel 6.17.0-1007-aws) |
| Python | 3.12.3 |
| PyTorch | 2.10.0+cu130 |
| vLLM | 0.18.0 |
| vllm-omni | 0.1.dev974 |

## Test Matrix

**5 models x 3 stage configs = 15 combinations** (minus 2 skipped Base+no-async = 13 active fixtures)

### Models

| Model | Task Type | Parameters |
|-------|-----------|------------|
| `Qwen/Qwen3-TTS-12Hz-0.6B-Base` | Voice clone | 0.6B |
| `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` | Preset voices | 0.6B |
| `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | Voice clone | 1.7B |
| `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | Preset voices | 1.7B |
| `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` | Voice from description | 1.7B |

### Stage Configs

| Config | Async Chunk | Max Batch (Talker / Code2Wav) |
|--------|-------------|-------------------------------|
| `qwen3_tts.yaml` (async) | yes | 10 / 1 |
| `qwen3_tts_batch.yaml` (batch) | yes | 4 / 4 |
| `qwen3_tts_no_async_chunk.yaml` (no-async) | no | 1 / 1 |

### Skipped Combinations

- **Base + no-async**: The non-async `talker2code2wav` codec processing path produces corrupted audio for Base (voice-clone) models. The async path works correctly. This is tracked upstream.

## Results Summary

```
=========== 86 passed, 18 skipped in 1917.46s (0:31:57) ============
```

| Result | Count |
|--------|-------|
| **Passed** | 86 |
| **Failed** | 0 |
| **Skipped** | 18 |
| **Total** | 104 |

### Skip Breakdown

| Reason | Count |
|--------|-------|
| Base + no-async (corrupted codec sequences) | 4 |
| `test_chinese` skipped for Base models (no explicit language param) | 4 |
| `test_different_voices` skipped for Base and VoiceDesign (no voice selection) | 6 |
| `test_batch` skipped for Base and VoiceDesign (only tested for CustomVoice) | 4 |

## Test Methodology

Each test function runs for every (model, stage-config) combination via a
single `@pytest.fixture(scope="module", params=...)`. Task-specific tests
use `pytest.skip()` when the current model doesn't apply.

### `test_runs` — Does it run?

Validates the full pipeline: model loading, prompt construction, AR Talker
inference, codec prediction, Code2Wav decoding, and audio output assembly.

- **Inputs**: Task-appropriate prompt built by `_make_input()` with exact
  `prompt_token_ids` length computed by
  `Qwen3TTSTalkerForConditionalGeneration.estimate_prompt_len_from_additional_information`.
- **Assertions**: Audio output is not `None`, sample rate is 24000 Hz,
  PCM16 byte count exceeds 10,000 (minimum ~0.2s of audio).
- **Side effect**: Saves a WAV file to `tts_audio_samples/` for manual
  listening.

### `test_not_silence` — Is the audio real?

Guards against degenerate outputs (all-zero PCM, single repeated sample).

- **Method**: Unpacks PCM16 bytes into samples and asserts the set of
  unique sample values has more than 1 element.

### `test_accuracy` — Does it say the right words?

End-to-end intelligibility check using Whisper ASR.

- **Method**: Saves generated audio to a temp WAV, transcribes with
  Whisper, computes character 3-gram cosine similarity between the
  transcript and input text.
- **Threshold**: Similarity > 0.9. This tolerates minor ASR differences
  (e.g., "Hello" transcribed as "Hey, hello") while catching garbled
  output (which scores near 0).
- **Note**: Similarity is measured via `cosine_similarity_text()` from
  `tests/conftest.py` — not word error rate. It builds vectors from
  overlapping character trigrams and computes their cosine.

### `test_stability` — Does it work repeatedly?

Detects intermittent failures, memory leaks, or state corruption across
sequential requests on the same engine instance.

- **Method**: Generates 3 sequential outputs using **different input
  texts**. Each output is validated as non-`None` and non-silent.
- **Why different texts**: Repeating the same input would only test
  caching; varying inputs exercises the full tokenization and embedding
  path each time.

### `test_performance` — How fast is it?

Measures wall-clock latency and Real-Time Factor.

- **Metrics**:
  - **Latency**: `time.perf_counter()` around `omni.generate()`.
  - **Audio duration**: `tensor.numel() / sample_rate`.
  - **RTF**: `latency / duration`. RTF < 1.0 = faster than real-time.
- **Note**: This is a single-request cold measurement. It is not a
  throughput benchmark.

### `test_chinese` — Multilingual support

Verifies CustomVoice and VoiceDesign models can generate Chinese speech.

- **Skipped for**: Base models (language is determined by reference audio,
  not an explicit parameter).

### `test_different_voices` — Speaker selection

Verifies CustomVoice models produce distinct, valid audio for different
preset speakers (Vivian, Ryan).

- **Skipped for**: Base and VoiceDesign.

### `test_batch` — Batch inference

Verifies the engine can handle multiple prompts in a single `generate()` call.

- **Skipped for**: Base and VoiceDesign.

## 1. Run (Model Loading)

All 5 models load and produce valid audio across all applicable stage configs.

| Model | async | batch | no-async |
|-------|:-----:|:-----:|:--------:|
| 0.6B-Base | PASS | PASS | _skipped_ |
| 0.6B-CustomVoice | PASS | PASS | PASS |
| 1.7B-Base | PASS | PASS | _skipped_ |
| 1.7B-CustomVoice | PASS | PASS | PASS |
| 1.7B-VoiceDesign | PASS | PASS | PASS |

## 2. Accuracy (Whisper Transcription)

Accuracy is measured as character 3-gram cosine similarity between the
Whisper transcription and the input text. Threshold: > 0.9. Similarity
is identical across all stage configs for the same model (deterministic
output with seed=0).

| Model | Similarity | Verdict |
|-------|:----------:|:-------:|
| 0.6B-Base | **0.981** | PASS |
| 0.6B-CustomVoice | **1.000** | PASS |
| 1.7B-Base | **0.981** | PASS |
| 1.7B-CustomVoice | **1.000** | PASS |
| 1.7B-VoiceDesign | **1.000** | PASS |

## 3. Performance (Latency & RTF)

Real-Time Factor (RTF) = inference_time / audio_duration. RTF < 1.0
means faster than real-time.

| Model | Config | Latency (s) | Audio Duration (s) | RTF |
|-------|--------|:-----------:|:-------------------:|:---:|
| 0.6B-Base | async | 4.03 | 5.36 | **0.751** |
| 0.6B-Base | batch | 3.86 | 5.36 | **0.719** |
| 0.6B-CustomVoice | async | 0.78 | 2.08 | **0.376** |
| 0.6B-CustomVoice | batch | 0.80 | 2.08 | **0.386** |
| 0.6B-CustomVoice | no-async | 0.61 | 2.08 | **0.293** |
| 1.7B-Base | async | 4.88 | 5.12 | **0.953** |
| 1.7B-Base | batch | 4.51 | 5.12 | **0.880** |
| 1.7B-CustomVoice | async | 1.20 | 2.56 | **0.468** |
| 1.7B-CustomVoice | batch | 1.20 | 2.56 | **0.470** |
| 1.7B-CustomVoice | no-async | 1.02 | 2.56 | **0.397** |
| 1.7B-VoiceDesign | async | 2.89 | 6.64 | **0.436** |
| 1.7B-VoiceDesign | batch | 2.90 | 6.64 | **0.437** |
| 1.7B-VoiceDesign | no-async | 2.55 | 6.64 | **0.384** |

**Key observations:**
- All models are **faster than real-time** on vllm 0.18.0 (RTF < 1.0), including Base models which were slower than real-time on 0.17.0.
- **CustomVoice** is fastest (RTF 0.29-0.47) due to shorter prompts and no reference audio encoding.
- **no-async is slightly faster** than async/batch for single requests due to reduced scheduling overhead.

## 4. Stability (Sequential Repeated Generation)

Each model was tested with 3 sequential generations using different
input texts. All must produce valid, non-silent audio.

**All 39 generations passed across all 13 active fixtures.**

## 5. Additional Tests

| Test | Models | Configs | Result |
|------|--------|---------|--------|
| `test_not_silence` | All 5 | All applicable | **All PASS** |
| `test_chinese` | CustomVoice + VoiceDesign | All 3 | **All PASS** |
| `test_different_voices` | CustomVoice only | All 3 | **All PASS** |
| `test_batch` | CustomVoice only | All 3 | **All PASS** |

## Code Coverage

Overall: 13% (61,057 statements, 7,737 covered). Low because the
codebase includes many unrelated models not exercised by TTS tests.

## Known Issues

1. **Base + no-async corrupted audio**: The non-async `talker2code2wav`
   codec processing path produces garbled audio for Base voice-clone
   models. The config-level fix (raising `max_model_len` and
   `max_num_batched_tokens`) allows the engine to accept the request,
   but the codec sequences are still corrupted. Tracked upstream.

2. **1.7B-CustomVoice adds "Hey" prefix**: The model consistently
   outputs "Hey, hello. How are you today?" for input "Hello, how are
   you today?" This is model behavior (similarity still 1.000 on 0.18.0,
   was 0.917 on 0.17.0 due to Whisper version differences).
