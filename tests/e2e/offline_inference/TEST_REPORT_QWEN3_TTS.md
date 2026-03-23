# Qwen3-TTS Offline Inference Test Report

**Date:** 2026-03-19
**Branch:** `qwen3-tts-validation`
**Commit:** `20501e60`

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
uv pip install vllm==0.17.0 --torch-backend=auto
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

# Run the full matrix (~35 min on L4 with coverage, ~24 min without)
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
| vLLM | 0.17.0 |
| vllm-omni | 0.1.dev918 |

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

- **Base + no-async**: The Base model with reference audio produces `prompt_token_ids` > 100K tokens, exceeding the no-async config's `max_model_len=32768` for the Code2Wav stage. This crashes the engine.

## Results Summary

```
=========== 86 passed, 18 skipped, 0 failed in 2087.63s (0:34:47) ============
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
| `test_chinese` skipped for Base models (no explicit language param) | 4 |
| `test_different_voices` skipped for Base and VoiceDesign (no voice selection) | 8 |
| `test_batch` skipped for Base and VoiceDesign (only tested for CustomVoice) | 6 |

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
  Whisper `small` model, computes character 3-gram cosine similarity
  between the transcript and input text.
- **Threshold**: Similarity > 0.9. This tolerates minor ASR differences
  (e.g., "Hello" transcribed as "Hey, hello") while catching garbled
  output (which scores near 0).
- **Note**: Similarity is measured via `cosine_similarity_text()` from
  `tests/conftest.py` — not word error rate. It builds TF-IDF-like
  vectors from overlapping character trigrams and computes their cosine.

### `test_stability` — Does it work repeatedly?

Detects intermittent failures, memory leaks, or state corruption across
sequential requests on the same engine instance.

- **Method**: Generates 3 sequential outputs using **different input
  texts** ("The weather forecast calls for clear skies tomorrow.",
  "Please remember to submit your report by Friday.",
  "The quarterly results exceeded expectations this year.").
  Each output is validated as non-`None` and non-silent.
- **Why different texts**: Repeating the same input would only test
  caching; varying inputs exercises the full tokenization and embedding
  path each time.
- **GPU memory**: Monitored between fixture teardowns — must return to
  baseline (< 5% VRAM usage) confirming no leak.

### `test_performance` — How fast is it?

Measures wall-clock latency and Real-Time Factor.

- **Metrics**:
  - **Latency**: `time.perf_counter()` around `omni.generate()`.
  - **Audio duration**: `tensor.numel() / sample_rate`.
  - **RTF**: `latency / duration`. RTF < 1.0 = faster than real-time.
- **Note**: This is a single-request cold measurement (includes first
  `torch.compile` if applicable). It is not a throughput benchmark.

### `test_chinese` — Multilingual support

Verifies CustomVoice and VoiceDesign models can generate Chinese speech.

- **Skipped for**: Base models (language is determined by reference audio,
  not an explicit parameter).
- **Method**: Generates with `language="Chinese"` and Chinese input text,
  asserts non-silent audio output.

### `test_different_voices` — Speaker selection

Verifies CustomVoice models produce distinct, valid audio for different
preset speakers.

- **Skipped for**: Base (no presets) and VoiceDesign (voice from
  description, not presets).
- **Speakers tested**: Vivian, Ryan.
- **Assertions**: Both produce non-`None`, non-silent audio.

### `test_batch` — Batch inference

Verifies the engine can handle multiple prompts in a single `generate()`
call.

- **Skipped for**: Base and VoiceDesign (tested only for CustomVoice to
  keep runtime reasonable).
- **Method**: Passes 2 prompts (`"Hello world."`, `"Batch test."`) in one
  call, asserts at least 1 audio output is produced.

## 1. Run (Model Loading)

All 5 models load and produce valid audio across all applicable stage configs.

| Model | async | batch | no-async |
|-------|:-----:|:-----:|:--------:|
| 0.6B-Base | PASS | PASS | _skipped_ |
| 0.6B-CustomVoice | PASS | PASS | PASS |
| 1.7B-Base | PASS | PASS | _skipped_ |
| 1.7B-CustomVoice | PASS | PASS | PASS |
| 1.7B-VoiceDesign | PASS | PASS | PASS |

**Engine initialization times:** 80-102s per fixture instance (first load ~102s due to cold HF cache, subsequent ~80-88s).

## 2. Accuracy (Whisper Transcription)

Accuracy is measured as character 3-gram cosine similarity between the Whisper transcription and the input text. Threshold: > 0.9.

| Model | Input Text | Whisper Transcript | Similarity | Verdict |
|-------|-----------|-------------------|------------|---------|
| 0.6B-Base | "Good one. Okay, fine, I'm just gonna leave this sock monkey here. Goodbye." | "Oh, good one. Okay, fine. I'm just gonna leave this sock monkey here. Goodbye." | **0.981** | PASS |
| 0.6B-CustomVoice | "Hello, how are you today?" | "Hello, how are you today?" | **1.000** | PASS |
| 1.7B-Base | "Good one. Okay, fine, I'm just gonna leave this sock monkey here. Goodbye." | "Ooh, good one. Okay, fine. I'm just gonna leave this sock monkey here. Goodbye!" | **0.974** | PASS |
| 1.7B-CustomVoice | "Hello, how are you today?" | "Hey, hello. How are you today?" | **0.917** | PASS |
| 1.7B-VoiceDesign | "It's in the top drawer... wait, it's empty? No way, that's impossible!" | "It's in the top drawer. Wait, it's empty? No way, that's impossible." | **1.000** | PASS |

**Key observations:**
- Similarity is **identical across all stage configs** for the same model, confirming deterministic output (seed=0).
- 1.7B-CustomVoice consistently prepends "Hey" to the output — this is model behavior, not a bug.
- Base models add interjections ("Oh", "Ooh") at the start — characteristic of voice-clone TTS.

## 3. Performance (Latency & RTF)

Real-Time Factor (RTF) = inference_time / audio_duration. RTF < 1.0 means faster than real-time.

| Model | Config | Latency (s) | Audio Duration (s) | RTF | Faster than real-time? |
|-------|--------|:-----------:|:-------------------:|:---:|:----------------------:|
| 0.6B-Base | async | 5.12 | 4.64 | **1.103** | No |
| 0.6B-Base | batch | 4.73 | 4.64 | **1.020** | No |
| 0.6B-CustomVoice | async | 1.34 | 2.08 | **0.646** | Yes |
| 0.6B-CustomVoice | batch | 1.38 | 2.08 | **0.663** | Yes |
| 0.6B-CustomVoice | no-async | 1.29 | 2.08 | **0.621** | Yes |
| 1.7B-Base | async | 5.67 | 4.56 | **1.242** | No |
| 1.7B-Base | batch | 5.66 | 4.56 | **1.240** | No |
| 1.7B-CustomVoice | async | 2.03 | 2.56 | **0.791** | Yes |
| 1.7B-CustomVoice | batch | 2.00 | 2.56 | **0.781** | Yes |
| 1.7B-CustomVoice | no-async | 1.88 | 2.56 | **0.733** | Yes |
| 1.7B-VoiceDesign | async | 4.99 | 6.64 | **0.752** | Yes |
| 1.7B-VoiceDesign | batch | 4.94 | 6.64 | **0.744** | Yes |
| 1.7B-VoiceDesign | no-async | 4.73 | 6.64 | **0.712** | Yes |

**Key observations:**
- **CustomVoice** models are fastest (RTF 0.62-0.79) due to shorter prompts and no reference audio encoding.
- **Base (voice clone)** models are slower than real-time (RTF 1.0-1.2) due to reference audio codec processing.
- **VoiceDesign** achieves good RTF (0.71-0.75) despite longer output audio (~6.6s).
- **no-async is slightly faster** than async/batch for the same model — likely due to reduced scheduling overhead for single requests.
- The L4 GPU triggers `Not enough SMs to use max_autotune_gemm mode` for 1.7B models, which may limit throughput.

## 4. Stability (Sequential Repeated Generation)

Each model was tested with 3 sequential generations using different input texts. All must produce valid, non-silent audio.

| Model | Config | Run 1 | Run 2 | Run 3 | Verdict |
|-------|--------|:-----:|:-----:|:-----:|:-------:|
| 0.6B-Base | async | PASS | PASS | PASS | PASS |
| 0.6B-Base | batch | PASS | PASS | PASS | PASS |
| 0.6B-CustomVoice | async | PASS | PASS | PASS | PASS |
| 0.6B-CustomVoice | batch | PASS | PASS | PASS | PASS |
| 0.6B-CustomVoice | no-async | PASS | PASS | PASS | PASS |
| 1.7B-Base | async | PASS | PASS | PASS | PASS |
| 1.7B-Base | batch | PASS | PASS | PASS | PASS |
| 1.7B-CustomVoice | async | PASS | PASS | PASS | PASS |
| 1.7B-CustomVoice | batch | PASS | PASS | PASS | PASS |
| 1.7B-CustomVoice | no-async | PASS | PASS | PASS | PASS |
| 1.7B-VoiceDesign | async | PASS | PASS | PASS | PASS |
| 1.7B-VoiceDesign | batch | PASS | PASS | PASS | PASS |
| 1.7B-VoiceDesign | no-async | PASS | PASS | PASS | PASS |

**No failures across 39 generations.** GPU memory is cleanly released between fixture teardowns (returns to 0.5 GiB / 22.5 GiB).

## 5. Additional Tests

| Test | Models | Configs | Result |
|------|--------|---------|--------|
| `test_not_silence` (PCM has non-trivial samples) | All 5 | All applicable | **All PASS** |
| `test_chinese` (Chinese language generation) | CustomVoice + VoiceDesign | All 3 | **All PASS** |
| `test_different_voices` (Vivian and Ryan) | CustomVoice only | All 3 | **All PASS** |
| `test_batch` (2 prompts in one call) | CustomVoice only | All 3 | **All PASS** |

## Code Coverage

With subprocess coverage tracking enabled (`.pth` file for `multiprocessing.spawn`):

### TTS Model Coverage

| Module | Statements | Missed | Coverage |
|--------|:----------:|:------:|:--------:|
| `configuration_qwen3_tts.py` | 141 | 11 | **92%** |
| `cuda_graph_decoder_wrapper.py` | 88 | 12 | **86%** |
| `tokenizer_12hz/modeling_*.py` | 531 | 61 | **89%** |
| `qwen3_tts_code_predictor_vllm.py` | 225 | 50 | **78%** |
| `qwen3_tts_code2wav.py` | 204 | 47 | **77%** |
| `qwen3_tts_talker.py` | 968 | 345 | **64%** |
| `qwen3_tts_tokenizer.py` | 166 | 98 | **41%** |

### Engine / Worker Coverage

| Module | Statements | Missed | Coverage |
|--------|:----------:|:------:|:--------:|
| `gpu_ar_worker.py` | 51 | 4 | **92%** |
| `gpu_generation_worker.py` | 51 | 4 | **92%** |
| `gpu_ar_model_runner.py` | 295 | 108 | **63%** |
| `gpu_generation_model_runner.py` | 293 | 110 | **62%** |
| `gpu_model_runner.py` | 652 | 239 | **63%** |
| `gpu_memory_utils.py` | 64 | 25 | **61%** |

### Overall

| Metric | Value |
|--------|-------|
| Total statements | 58,454 |
| Covered | 7,175 |
| **Overall coverage** | **12%** |

Overall coverage is low because the codebase includes many unrelated models (Bagel, CosyVoice3, Fish Speech, Qwen2.5-Omni, etc.) that are not exercised by TTS tests.

## Audio Samples

13 WAV files saved for manual listening:

| File | Model | Config |
|------|-------|--------|
| `0.6B-Base-async.wav` | 0.6B-Base | async |
| `0.6B-Base-batch.wav` | 0.6B-Base | batch |
| `0.6B-CustomVoice-async.wav` | 0.6B-CustomVoice | async |
| `0.6B-CustomVoice-batch.wav` | 0.6B-CustomVoice | batch |
| `0.6B-CustomVoice-no-async.wav` | 0.6B-CustomVoice | no-async |
| `1.7B-Base-async.wav` | 1.7B-Base | async |
| `1.7B-Base-batch.wav` | 1.7B-Base | batch |
| `1.7B-CustomVoice-async.wav` | 1.7B-CustomVoice | async |
| `1.7B-CustomVoice-batch.wav` | 1.7B-CustomVoice | batch |
| `1.7B-CustomVoice-no-async.wav` | 1.7B-CustomVoice | no-async |
| `1.7B-VoiceDesign-async.wav` | 1.7B-VoiceDesign | async |
| `1.7B-VoiceDesign-batch.wav` | 1.7B-VoiceDesign | batch |
| `1.7B-VoiceDesign-no-async.wav` | 1.7B-VoiceDesign | no-async |

## Known Issues

1. **Base + no-async crash**: Base models with reference audio produce `prompt_token_ids` > 100K tokens, exceeding `max_model_len=32768` in the no-async Code2Wav stage config. These combinations are skipped.

2. **Bogus weight loading times for Code2Wav**: Stage 1 reports absurd loading times (3000-5000s) because the elapsed-time counter accumulates across engine instantiations rather than resetting. Actual load is ~0.001s.

3. **1.7B-CustomVoice adds "Hey" prefix**: The model consistently outputs "Hey, hello. How are you today?" for input "Hello, how are you today?" This is model behavior (similarity still 0.917 > 0.9 threshold).

4. **Base models slower than real-time on L4**: RTF 1.0-1.2 due to reference audio codec processing and limited SMs on L4 GPU.

5. **Offline streaming not supported via sync Omni API**: `Omni._set_final_only_for_llm_stages()` forces `FINAL_ONLY` output, suppressing intermediate chunks. Streaming is only available via `AsyncOmni` or the WebSocket online endpoint.

## Recommendations

1. **Fix the no-async max_model_len**: Increase `max_model_len` in `qwen3_tts_no_async_chunk.yaml` stage 1 to accommodate Base model reference audio embeddings (> 100K tokens).

2. **Fix the weight loading timer bug**: The Code2Wav stage's loading time counter should reset per engine instantiation.

3. **Consider exposing `py_generator` without `close()`**: The current `Omni.generate(py_generator=True)` calls `self.close()` after the generator is consumed, making it unsuitable for test fixtures that reuse the engine. A streaming mode that doesn't close the engine would enable offline streaming tests.

## Appendix: Why Base Models Are Slower Than CustomVoice

The RTF difference (Base 1.0-1.2 vs CustomVoice 0.62-0.79) comes from four additive costs unique to Base (voice-clone) models, not just GPU limitations.

### Cost 1: Reference Audio Codec Encoding (one-time, ~1-2s)

Base models encode the reference audio through a full SpeechTokenizer neural codec (Mimi-based architecture with RVQ) to produce `ref_code` of shape `[T_ref, 16]`, where `T_ref ≈ audio_duration × 12` frames. For the test's ~7s reference clip this is ~84 frames × 16 quantizer layers.

- **Where**: `Qwen3TTSTalkerForConditionalGeneration._encode_ref_audio_to_code()` (`qwen3_tts_talker.py:1136`)
- **CustomVoice equivalent**: a single dictionary lookup (`spk_id_map[speaker.lower()]`)

### Cost 2: Speaker Embedding Extraction (one-time)

Base models run the reference audio through an ECAPA-TDNN speaker encoder: mel spectrogram (128 bins) → TimeDelayNet blocks → Res2Net with squeeze-excitation → attentive statistics pooling → linear projection → speaker embedding vector.

- **Where**: `Qwen3TTSTalkerForConditionalGeneration._extract_speaker_embedding()` (`qwen3_tts_talker.py:1069`)
- **CustomVoice equivalent**: a learned embedding looked up by speaker index

### Cost 3: Larger Code2Wav Chunks (per-chunk, throughout streaming)

The `ref_code` from Cost 1 is prepended to **every** streaming chunk in Code2Wav decoding (kept via `.get()`, not `.pop()`):

| | Tokens per chunk |
|---|---|
| **CustomVoice** | 16 × (25 context + 25 chunk) = **800** |
| **Base** | 16 × (84 ref + 25 context + 25 chunk) = **2,144** (~2.7x) |

This 2.7x per-chunk overhead compounds across ~8-10 chunks for a typical utterance.

- **Where**: `talker2code2wav_async_chunk()` (`qwen3_tts.py:181-191`)

### Cost 4: Larger Talker Prompt (least significant)

Base ICL prompts include reference transcript tokens and `ref_code` embeddings (~93-139 tokens vs ~25-30 for CustomVoice). This adds negligible prefill time on GPU.

### Quantitative Breakdown (0.6B models, L4 GPU)

| | CustomVoice | Base | Ratio |
|---|---|---|---|
| Latency | 1.34s | 5.12s | 3.8x |
| Audio duration | 2.08s | 4.64s | 2.2x |
| RTF | 0.646 | 1.103 | 1.7x |
| Cost per second of audio | 0.64 s/s | 1.10 s/s | 1.7x |

The 3.8x raw latency difference decomposes into: ~2.2x from longer test text (more AR decode steps), ~1-2s fixed cost from codec + speaker encoding, and ~1.7x per-second overhead from the amplified Code2Wav chunk size.
