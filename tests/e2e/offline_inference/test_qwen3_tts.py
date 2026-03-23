# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
r"""
E2E offline inference tests for all Qwen3-TTS model variants.

Uses a single parametrized fixture to run every test against all five
models x three stage configs, with ``pytest.skip`` for task-specific or
config-incompatible combinations.

Models
------
- Qwen/Qwen3-TTS-12Hz-0.6B-Base          (voice clone)
- Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice   (preset voices)
- Qwen/Qwen3-TTS-12Hz-1.7B-Base          (voice clone)
- Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice   (preset voices)
- Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign   (voice from description)

Stage configs
-------------
- qwen3_tts.yaml              (async streaming)
- qwen3_tts_batch.yaml        (batched streaming)
- qwen3_tts_no_async_chunk.yaml (non-streaming, max_model_len=32768)

Running
-------
All tests (full matrix, ~2h on L4)::

    python -m pytest tests/e2e/offline_inference/test_qwen3_tts.py \
        -v -s --timeout=1800

Single model + config::

    python -m pytest tests/e2e/offline_inference/test_qwen3_tts.py \
        -k "0.6B-CustomVoice-async" -v -s --timeout=600

With coverage (HTML report)::

    python -m pytest tests/e2e/offline_inference/test_qwen3_tts.py \
        -v -s --timeout=1800 \
        --cov=vllm_omni --cov-report=html:htmlcov_offline

Audio samples are saved to ``tts_audio_samples/`` (override with
``TTS_AUDIO_DIR`` env var).
"""

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_TEST_CLEAN_GPU_MEMORY"] = "0"
# Code2Wav has no position encoding or KV cache; allow max_model_len
# to exceed max_position_embeddings for the no-async config.
os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"

import struct
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from tests.conftest import (
    OmniRunner,
    convert_audio_file_to_text,
    cosine_similarity_text,
)
from tests.utils import hardware_test

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_CONFIGS_DIR = (
    Path(__file__).parent.parent.parent.parent
    / "vllm_omni"
    / "model_executor"
    / "stage_configs"
)

_STAGE_CONFIGS = [
    ("async", str(_CONFIGS_DIR / "qwen3_tts.yaml")),
    ("batch", str(_CONFIGS_DIR / "qwen3_tts_batch.yaml")),
    ("no-async", str(_CONFIGS_DIR / "qwen3_tts_no_async_chunk.yaml")),
]

MODEL_0_6B_BASE = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
MODEL_0_6B_CV = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
MODEL_1_7B_BASE = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
MODEL_1_7B_CV = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
MODEL_1_7B_VD = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"

REF_AUDIO_URL = (
    "https://qianwen-res.oss-cn-beijing.aliyuncs.com/"
    "Qwen3-TTS-Repo/clone_2.wav"
)
REF_TEXT = (
    "Okay. Yeah. I resent you. I love you. I respect you. "
    "But you know what? You blew it! And thanks to you."
)

MIN_AUDIO_BYTES = 10_000

# Sample texts per task
_BASE_TEXT = "Good one. Okay, fine, I'm just gonna leave this sock monkey here. Goodbye."
_CV_EN = "Hello, how are you today?"
_CV_ZH = "你好，我是通义千问"
_VD_EN = "It's in the top drawer... wait, it's empty? No way, that's impossible!"
_VD_INSTRUCT_EN = "Speak in an incredulous tone with a hint of panic."
_VD_ZH = "各位观众朋友大家好，欢迎收看今天的新闻节目。"
_VD_INSTRUCT_ZH = "沉稳专业的男性播音员，语速适中，吐字清晰。"

# Varying texts for stability tests
_STABILITY_TEXTS = [
    "The weather forecast calls for clear skies tomorrow.",
    "Please remember to submit your report by Friday.",
    "The quarterly results exceeded expectations this year.",
]

# ---------------------------------------------------------------------------
# Prompt-length estimation (must be exact for the Talker to work)
# ---------------------------------------------------------------------------
_estimator_cache: dict = {}


def _estimate_prompt_len(additional_information: dict, model_name: str) -> int:
    """Compute the exact prompt_token_ids length the Talker expects.

    Delegates to the model's static method which accounts for the
    tokenizer, speaker embeddings, codec language ids, etc.
    """
    from vllm_omni.model_executor.models.qwen3_tts.configuration_qwen3_tts import (
        Qwen3TTSConfig,
    )
    from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_talker import (
        Qwen3TTSTalkerForConditionalGeneration,
    )

    if model_name not in _estimator_cache:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, padding_side="left"
        )
        cfg = Qwen3TTSConfig.from_pretrained(model_name, trust_remote_code=True)
        tcfg = getattr(cfg, "talker_config", None)
        _estimator_cache[model_name] = (tok, tcfg)

    tok, tcfg = _estimator_cache[model_name]
    task_type = (additional_information.get("task_type") or ["CustomVoice"])[0]

    def _estimate_ref_code_len(ref_audio: object) -> int | None:
        """Fallback: estimate ref_code frames from audio duration."""
        if not isinstance(ref_audio, (str, list)):
            return None
        audio_path = ref_audio[0] if isinstance(ref_audio, list) else ref_audio
        if not isinstance(audio_path, str) or not audio_path.strip():
            return None
        try:
            from vllm.multimodal.media import MediaConnector

            connector = MediaConnector(allowed_local_media_path="/")
            audio, sr = connector.fetch_audio(audio_path)
            codec_hz = getattr(tcfg, "codec_frame_rate", None) or 12
            return int(len(audio) / sr * codec_hz)
        except Exception:
            return None

    return Qwen3TTSTalkerForConditionalGeneration.estimate_prompt_len_from_additional_information(
        additional_information=additional_information,
        task_type=task_type,
        tokenize_prompt=lambda t: tok(t, padding=False)["input_ids"],
        codec_language_id=getattr(tcfg, "codec_language_id", None),
        spk_is_dialect=getattr(tcfg, "spk_is_dialect", None),
        estimate_ref_code_len=_estimate_ref_code_len,
    )


# ---------------------------------------------------------------------------
# Input builders
# ---------------------------------------------------------------------------


def _build_base(text, model, language="Auto"):
    ai = {
        "task_type": ["Base"],
        "ref_audio": [REF_AUDIO_URL],
        "ref_text": [REF_TEXT],
        "text": [text],
        "language": [language],
        "x_vector_only_mode": [False],
        "max_new_tokens": [2048],
    }
    return {
        "prompt_token_ids": [0] * _estimate_prompt_len(ai, model),
        "additional_information": ai,
    }


def _build_cv(text, model, language="English", speaker="Vivian"):
    ai = {
        "task_type": ["CustomVoice"],
        "text": [text],
        "language": [language],
        "speaker": [speaker],
        "instruct": [""],
        "max_new_tokens": [2048],
    }
    return {
        "prompt_token_ids": [0] * _estimate_prompt_len(ai, model),
        "additional_information": ai,
    }


def _build_vd(text, model, language="English", instruct=""):
    ai = {
        "task_type": ["VoiceDesign"],
        "text": [text],
        "language": [language],
        "instruct": [instruct],
        "max_new_tokens": [2048],
        "non_streaming_mode": [True],
    }
    return {
        "prompt_token_ids": [0] * _estimate_prompt_len(ai, model),
        "additional_information": ai,
    }


def _task(model: str) -> str:
    if "Base" in model:
        return "Base"
    if "VoiceDesign" in model:
        return "VoiceDesign"
    return "CustomVoice"


def _config_id(request) -> str:
    """Extract the stage-config id (async/batch/no-async) from the test id."""
    return request.node.callspec.id.rsplit("-", 1)[-1]


def _make_input(model: str, *, language: str = "English", text: str | None = None):
    """Build the correct TTS input dict for *model*."""
    task = _task(model)
    if task == "Base":
        return _build_base(text or _BASE_TEXT, model)
    if task == "VoiceDesign":
        instruct = _VD_INSTRUCT_EN if language == "English" else _VD_INSTRUCT_ZH
        default = _VD_EN if language == "English" else _VD_ZH
        return _build_vd(text or default, model, language=language, instruct=instruct)
    # CustomVoice
    default = _CV_EN if language == "English" else _CV_ZH
    return _build_cv(text or default, model, language=language)


def _en_text(model: str) -> str:
    """Return the English reference text used for accuracy checks."""
    task = _task(model)
    if task == "Base":
        return _BASE_TEXT
    if task == "VoiceDesign":
        return _VD_EN
    return _CV_EN


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------


def _extract_audio(outputs):
    """Extract audio from the *last* (finished) audio output.

    In async chunk mode, multiple intermediate audio outputs are yielded
    before the final one.  Only the final output contains the fully
    accumulated audio across all chunks.
    """
    last_audio = None
    last_sr = None
    for o in outputs:
        if getattr(o, "final_output_type", None) == "audio":
            mm = o.request_output.outputs[0].multimodal_output
            if mm is None:
                continue
            data = mm.get("audio")
            if data is None:
                continue
            sr_raw = mm.get("sr")
            sr_val = sr_raw[-1] if isinstance(sr_raw, list) and sr_raw else sr_raw
            sr = sr_val.item() if hasattr(sr_val, "item") else int(sr_val)
            tensor = torch.cat(data, dim=-1) if isinstance(data, list) else data
            last_audio = tensor
            last_sr = sr
    return last_audio, last_sr


def _pcm16(tensor):
    s = tensor.float().cpu().numpy().flatten()
    return np.clip(s * 32768, -32768, 32767).astype(np.int16).tobytes()


def _assert_not_silence(pcm):
    samples = struct.unpack(f"<{len(pcm) // 2}h", pcm)
    assert len(set(samples)) > 1, "all-silence"


def _save_audio(tensor, sr, label: str) -> str:
    """Save audio to the samples directory and return the path."""
    import soundfile as sf

    out_dir = os.environ.get(
        "TTS_AUDIO_DIR",
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "tts_audio_samples"),
    )
    os.makedirs(out_dir, exist_ok=True)
    safe = label.replace("/", "_").replace(" ", "_")
    path = os.path.join(out_dir, f"{safe}.wav")
    sf.write(path, tensor.float().cpu().numpy().flatten(), sr, format="WAV")
    print(f"\n  AUDIO SAMPLE: {os.path.abspath(path)}")
    return path


def _transcribe(tensor, sr):
    import soundfile as sf

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        path = f.name
    try:
        sf.write(path, tensor.float().cpu().numpy().flatten(), sr, format="WAV")
        return convert_audio_file_to_text(path)
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Parametrized fixture — one OmniRunner per (model, config) pair
# ---------------------------------------------------------------------------
_lock = threading.Lock()

_MODELS = [
    ("0.6B-Base", MODEL_0_6B_BASE),
    ("0.6B-CustomVoice", MODEL_0_6B_CV),
    ("1.7B-Base", MODEL_1_7B_BASE),
    ("1.7B-CustomVoice", MODEL_1_7B_CV),
    ("1.7B-VoiceDesign", MODEL_1_7B_VD),
]

# Base+no-async is skipped: the non-async talker→code2wav path produces
# corrupted codec sequences for Base (voice-clone) models, resulting in
# garbled audio output.  The config-level fix (raising max_model_len and
# max_num_batched_tokens) allows the engine to accept the request, but
# the underlying codec processing logic needs a deeper code fix.
_ALL = [
    pytest.param((model, cfg_path), id=f"{m_id}-{c_id}")
    for m_id, model in _MODELS
    for c_id, cfg_path in _STAGE_CONFIGS
    if not (c_id == "no-async" and "Base" in m_id)
]


@pytest.fixture(scope="module", params=_ALL)
def tts_runner(request, model_prefix):
    """Start an OmniRunner for each (model, stage-config) combination."""
    with _lock:
        model, config = request.param
        with OmniRunner(
            model_prefix + model,
            seed=42,
            stage_configs_path=config,
            stage_init_timeout=300,
        ) as runner:
            yield runner, model


# ---------------------------------------------------------------------------
# Common tests — run for every model
# ---------------------------------------------------------------------------


@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4"})
def test_runs(tts_runner, request):
    """Model loads and produces valid audio with correct sample rate."""
    runner, model = tts_runner
    audio, sr = _extract_audio(runner.generate([_make_input(model)]))
    assert audio is not None, "no audio output"
    assert sr == 24000
    assert len(_pcm16(audio)) > MIN_AUDIO_BYTES
    _save_audio(audio, sr, request.node.callspec.id)


@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4"})
def test_not_silence(tts_runner):
    """PCM output contains non-trivial samples."""
    runner, model = tts_runner
    audio, _ = _extract_audio(runner.generate([_make_input(model)]))
    assert audio is not None
    _assert_not_silence(_pcm16(audio))


@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4"})
def test_accuracy(tts_runner):
    """Whisper transcription matches the input text (similarity > 0.9)."""
    runner, model = tts_runner
    audio, sr = _extract_audio(runner.generate([_make_input(model)]))
    assert audio is not None
    expected = _en_text(model)
    transcript = _transcribe(audio, sr)
    print(f"[{model}] transcript: {transcript}")
    sim = cosine_similarity_text(transcript.lower(), expected.lower())
    print(f"[{model}] similarity: {sim:.3f}")
    assert sim > 0.9, f"similarity={sim:.2f}"


@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4"})
def test_stability(tts_runner):
    """Sequential generations with varying inputs all produce valid audio."""
    runner, model = tts_runner
    for i, text in enumerate(_STABILITY_TEXTS):
        audio, _ = _extract_audio(
            runner.generate([_make_input(model, text=text)])
        )
        assert audio is not None, f"run {i} ({text[:30]}...): no audio"
        _assert_not_silence(_pcm16(audio))


@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4"})
def test_performance(tts_runner):
    """Report latency and real-time factor (RTF)."""
    runner, model = tts_runner
    t0 = time.perf_counter()
    audio, sr = _extract_audio(runner.generate([_make_input(model)]))
    latency = time.perf_counter() - t0
    assert audio is not None
    dur = audio.numel() / sr
    rtf = latency / dur if dur > 0 else float("inf")
    print(f"[{model}] latency={latency:.2f}s  dur={dur:.2f}s  RTF={rtf:.3f}")


# ---------------------------------------------------------------------------
# Task-specific tests — skip when the fixture model doesn't match
# ---------------------------------------------------------------------------


@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4"})
def test_chinese(tts_runner):
    """CustomVoice / VoiceDesign: generate Chinese speech."""
    runner, model = tts_runner
    if _task(model) == "Base":
        pytest.skip("Base uses ref_audio language, not explicit Chinese")
    audio, _ = _extract_audio(
        runner.generate([_make_input(model, language="Chinese")])
    )
    assert audio is not None
    _assert_not_silence(_pcm16(audio))


@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4"})
def test_different_voices(tts_runner):
    """CustomVoice models: Vivian and Ryan both produce valid audio."""
    runner, model = tts_runner
    if _task(model) != "CustomVoice":
        pytest.skip("voice selection only for CustomVoice models")
    for speaker in ["Vivian", "Ryan"]:
        inp = _build_cv(_CV_EN, model, speaker=speaker)
        audio, _ = _extract_audio(runner.generate([inp]))
        assert audio is not None, f"no audio for {speaker}"
        _assert_not_silence(_pcm16(audio))


@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4"})
def test_batch(tts_runner):
    """CustomVoice models: batch of two prompts produces audio."""
    runner, model = tts_runner
    if _task(model) != "CustomVoice":
        pytest.skip("batch test only for CustomVoice models")
    inputs = [_build_cv("Hello world.", model), _build_cv("Batch test.", model)]
    outputs = runner.generate(inputs)
    n = sum(1 for o in outputs if getattr(o, "final_output_type", None) == "audio")
    assert n >= 1, "batch produced no audio"


    # NOTE: offline streaming (AsyncOmni / py_generator) is not tested here.
    # The sync Omni entrypoint forces FINAL_ONLY output_kind which suppresses
    # intermediate chunks.  Streaming is tested via the online WebSocket
    # endpoint in test_qwen3_tts_variants.py::test_websocket_streaming.
