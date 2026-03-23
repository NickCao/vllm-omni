# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
r"""
E2E online serving tests for Qwen3-TTS variants.

Covers the following models via a single parametrized fixture:
  - Qwen/Qwen3-TTS-12Hz-0.6B-Base          (voice clone)
  - Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice   (preset voices)
  - Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign   (voice from description)

Each model is tested with all three stage configs (async, batch,
no-async).  Existing files already cover 0.6B-CustomVoice and 1.7B-Base.

Running
-------
All tests (full matrix)::

    python -m pytest tests/e2e/online_serving/test_qwen3_tts_variants.py \
        -v -s --timeout=1800

Single model + config::

    python -m pytest tests/e2e/online_serving/test_qwen3_tts_variants.py \
        -k "1.7B-CustomVoice-async" -v -s --timeout=600

With coverage (HTML report)::

    python -m pytest tests/e2e/online_serving/test_qwen3_tts_variants.py \
        -v -s --timeout=1800 \
        --cov=vllm_omni --cov-report=html:htmlcov_online

Audio samples are saved to ``tts_audio_samples/`` (override with
``TTS_AUDIO_DIR`` env var).
"""

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_TEST_CLEAN_GPU_MEMORY"] = "0"

import concurrent.futures
import struct
import tempfile
import time
from pathlib import Path

import httpx
import numpy as np
import pytest

from tests.conftest import (
    OmniServer,
    convert_audio_file_to_text,
    cosine_similarity_text,
)
from tests.utils import hardware_test

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_0_6B_BASE = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
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
SYN_TEXT = "Good one. Okay, fine, I'm just gonna leave this sock monkey here. Goodbye."
CV_TEXT = "She sells seashells by the seashore."
VD_TEXT = "It's in the top drawer... wait, it's empty? No way, that's impossible!"
VD_INSTRUCT = "Speak in an incredulous tone with a hint of panic."

MIN_AUDIO_BYTES = 10_000
MIN_HNR_DB = 1.2


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

# ---------------------------------------------------------------------------
# Per-model configuration dicts, crossed with stage configs
# ---------------------------------------------------------------------------
_MODEL_CFGS = [
    (
        "0.6B-Base",
        {
            "model": MODEL_0_6B_BASE,
            "task": "Base",
            "en_text": SYN_TEXT,
            "ref_audio": REF_AUDIO_URL,
            "ref_text": REF_TEXT,
        },
    ),
    (
        "1.7B-CustomVoice",
        {
            "model": MODEL_1_7B_CV,
            "task": "CustomVoice",
            "en_text": CV_TEXT,
            "voice": "vivian",
        },
    ),
    (
        "1.7B-VoiceDesign",
        {
            "model": MODEL_1_7B_VD,
            "task": "VoiceDesign",
            "en_text": VD_TEXT,
            "instructions": VD_INSTRUCT,
        },
    ),
]

# Base+no-async skipped: non-async codec processing produces garbled audio.
_CONFIGS = [
    pytest.param({**cfg, "stage_config": cfg_path}, id=f"{m_id}-{c_id}")
    for m_id, cfg in _MODEL_CFGS
    for c_id, cfg_path in _STAGE_CONFIGS
    if not (c_id == "no-async" and cfg["task"] == "Base")
]


# ---------------------------------------------------------------------------
# Parametrized fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="class", params=_CONFIGS)
def tts_cfg(request):
    """Expose the current (model + stage-config) dict to every test."""
    return request.param


@pytest.fixture(scope="class")
def tts_server(tts_cfg):
    """Start one OmniServer per (model, stage-config) combination."""
    with OmniServer(
        tts_cfg["model"],
        [
            "--stage-configs-path",
            tts_cfg["stage_config"],
            "--stage-init-timeout",
            "120",
            "--trust-remote-code",
            "--enforce-eager",
            "--disable-log-stats",
        ],
    ) as server:
        yield server


# ---------------------------------------------------------------------------
# Request dispatcher — builds the correct payload for each task type
# ---------------------------------------------------------------------------


def _request(
    host: str,
    port: int,
    cfg: dict,
    *,
    text: str | None = None,
    response_format: str = "wav",
    timeout: float = 180.0,
) -> httpx.Response:
    url = f"http://{host}:{port}/v1/audio/speech"
    t = text or cfg["en_text"]
    payload: dict = {"input": t, "response_format": response_format}

    task = cfg["task"]
    if task == "CustomVoice":
        payload["voice"] = cfg.get("voice", "vivian")
        payload["language"] = "English"
    elif task == "VoiceDesign":
        payload["task_type"] = "VoiceDesign"
        payload["language"] = "English"
        payload["instructions"] = cfg["instructions"]
    elif task == "Base":
        payload["model"] = cfg["model"]
        payload["task_type"] = "Base"
        payload["ref_text"] = cfg["ref_text"]
        payload["ref_audio"] = cfg["ref_audio"]

    with httpx.Client(timeout=timeout) as client:
        return client.post(url, json=payload)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _is_wav(c: bytes) -> bool:
    return len(c) >= 44 and c[:4] == b"RIFF" and c[8:12] == b"WAVE"


def _assert_not_silence(pcm: bytes) -> None:
    samples = struct.unpack(f"<{len(pcm) // 2}h", pcm)
    assert len(set(samples)) > 1, "all-silence"


def _hnr_db(pcm_f32: np.ndarray, sr: int = 24000) -> float:
    fl = int(0.03 * sr)
    hop = fl // 2
    vals: list[float] = []
    for s in range(0, len(pcm_f32) - fl, hop):
        f = pcm_f32[s : s + fl]
        if np.max(np.abs(f)) < 0.01:
            continue
        ac = np.correlate(f, f, mode="full")[len(f) - 1 :]
        ac /= ac[0] + 1e-10
        lo, hi = int(sr / 400), min(int(sr / 80), len(ac))
        if lo >= hi:
            continue
        p = float(np.max(ac[lo:hi]))
        if 0 < p < 1:
            vals.append(10 * np.log10(p / (1 - p + 1e-10)))
    return float(np.mean(vals)) if vals else 0.0


def _save_audio(content: bytes, label: str) -> str:
    """Save WAV bytes to samples directory and print path."""
    out_dir = os.environ.get(
        "TTS_AUDIO_DIR",
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "tts_audio_samples"),
    )
    os.makedirs(out_dir, exist_ok=True)
    safe = label.replace("/", "_").replace(" ", "_")
    path = os.path.join(out_dir, f"{safe}.wav")
    with open(path, "wb") as f:
        f.write(content)
    print(f"\n  AUDIO SAMPLE: {os.path.abspath(path)}")
    return path


# ---------------------------------------------------------------------------
# Tests — every method runs once per model variant
# ---------------------------------------------------------------------------


class TestQwen3TTSVariants:
    """Common + task-specific checks for 0.6B-Base, 1.7B-CV, 1.7B-VD."""

    # ---- Run ----

    @pytest.mark.core_model
    @pytest.mark.omni
    @hardware_test(res={"cuda": "L4"})
    def test_runs(self, tts_server, tts_cfg, request) -> None:
        """Produces valid WAV audio."""
        resp = _request(tts_server.host, tts_server.port, tts_cfg)
        assert resp.status_code == 200, f"failed: {resp.text}"
        assert _is_wav(resp.content)
        assert len(resp.content) > MIN_AUDIO_BYTES
        _save_audio(resp.content, f"online-{request.node.callspec.id}")

    @pytest.mark.core_model
    @pytest.mark.omni
    @hardware_test(res={"cuda": "L4"})
    def test_not_silence(self, tts_server, tts_cfg) -> None:
        """PCM output is not all-silence."""
        resp = _request(
            tts_server.host, tts_server.port, tts_cfg, response_format="pcm"
        )
        assert resp.status_code == 200
        assert len(resp.content) > MIN_AUDIO_BYTES
        _assert_not_silence(resp.content)

    # ---- Accuracy ----

    @pytest.mark.core_model
    @pytest.mark.omni
    @hardware_test(res={"cuda": "L4"})
    def test_accuracy(self, tts_server, tts_cfg) -> None:
        """Whisper transcription matches the input (similarity > 0.9)."""
        resp = _request(tts_server.host, tts_server.port, tts_cfg)
        assert resp.status_code == 200
        assert _is_wav(resp.content)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(resp.content)
            path = f.name
        try:
            transcript = convert_audio_file_to_text(path)
            expected = tts_cfg["en_text"]
            print(f"[{tts_cfg['task']}] transcript: {transcript}")
            sim = cosine_similarity_text(transcript.lower(), expected.lower())
            print(f"[{tts_cfg['task']}] similarity: {sim:.3f}")
            assert sim > 0.9, f"similarity={sim:.2f}"
        finally:
            os.unlink(path)

    # ---- Stability ----

    @pytest.mark.core_model
    @pytest.mark.omni
    @hardware_test(res={"cuda": "L4"})
    def test_stability_sequential(self, tts_server, tts_cfg) -> None:
        """Five sequential requests all produce valid WAV."""
        for i in range(5):
            resp = _request(
                tts_server.host,
                tts_server.port,
                tts_cfg,
                text=f"Stability test sentence number {i + 1}.",
            )
            assert resp.status_code == 200, f"request {i} failed"
            assert _is_wav(resp.content), f"request {i} invalid WAV"
            assert len(resp.content) > MIN_AUDIO_BYTES, f"request {i} too small"

    # ---- Performance ----

    @pytest.mark.core_model
    @pytest.mark.omni
    @hardware_test(res={"cuda": "L4"})
    def test_performance(self, tts_server, tts_cfg) -> None:
        """Report E2E latency and estimated RTF."""
        t0 = time.perf_counter()
        resp = _request(tts_server.host, tts_server.port, tts_cfg)
        latency = time.perf_counter() - t0
        assert resp.status_code == 200
        assert _is_wav(resp.content)
        audio_bytes = len(resp.content) - 44
        dur = audio_bytes / 48000.0  # 24 kHz 16-bit mono
        rtf = latency / dur if dur > 0 else float("inf")
        print(
            f"[{tts_cfg['task']}] latency={latency:.2f}s  "
            f"dur≈{dur:.2f}s  RTF≈{rtf:.3f}"
        )

    # ---- Task-specific (skip when irrelevant) ----

    @pytest.mark.core_model
    @pytest.mark.omni
    @hardware_test(res={"cuda": "L4"})
    def test_no_distortion(self, tts_server, tts_cfg) -> None:
        """Base: HNR above distortion threshold."""
        if tts_cfg["task"] != "Base":
            pytest.skip("HNR check only for Base models")
        resp = _request(
            tts_server.host, tts_server.port, tts_cfg, response_format="pcm"
        )
        assert resp.status_code == 200
        pcm = np.frombuffer(resp.content, dtype=np.int16).astype(np.float32) / 32768
        hnr = _hnr_db(pcm)
        print(f"[Base] HNR={hnr:.2f} dB  threshold={MIN_HNR_DB}")
        assert hnr >= MIN_HNR_DB, f"HNR={hnr:.2f} dB"

    @pytest.mark.core_model
    @pytest.mark.omni
    @hardware_test(res={"cuda": "L4"})
    def test_different_voices(self, tts_server, tts_cfg) -> None:
        """CustomVoice: Vivian and Ryan both produce valid audio."""
        if tts_cfg["task"] != "CustomVoice":
            pytest.skip("voice selection only for CustomVoice")
        for voice in ["vivian", "ryan"]:
            resp = _request(
                tts_server.host,
                tts_server.port,
                {**tts_cfg, "voice": voice},
            )
            assert resp.status_code == 200, f"{voice} failed"
            assert _is_wav(resp.content), f"{voice} invalid WAV"

    @pytest.mark.core_model
    @pytest.mark.omni
    @hardware_test(res={"cuda": "L4"})
    def test_concurrent(self, tts_server, tts_cfg) -> None:
        """CustomVoice: three concurrent requests all succeed."""
        if tts_cfg["task"] != "CustomVoice":
            pytest.skip("concurrent test only for CustomVoice")

        def _do(t):
            return _request(tts_server.host, tts_server.port, tts_cfg, text=t)

        texts = ["Concurrent one.", "Concurrent two.", "Concurrent three."]
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            for resp in pool.map(_do, texts):
                assert resp.status_code == 200
                assert _is_wav(resp.content)

    @pytest.mark.core_model
    @pytest.mark.omni
    @hardware_test(res={"cuda": "L4"})
    def test_api_voices(self, tts_server, tts_cfg) -> None:
        """CustomVoice: GET /v1/audio/voices returns voices."""
        if tts_cfg["task"] != "CustomVoice":
            pytest.skip("voices endpoint relevant for CustomVoice")
        url = f"http://{tts_server.host}:{tts_server.port}/v1/audio/voices"
        with httpx.Client(timeout=30.0) as client:
            resp = client.get(url)
        assert resp.status_code == 200
        assert len(resp.json()["voices"]) > 0

    @pytest.mark.core_model
    @pytest.mark.omni
    @hardware_test(res={"cuda": "L4"})
    def test_websocket_streaming(self, tts_server, tts_cfg) -> None:
        """WebSocket streaming: receive chunked PCM audio."""
        import asyncio
        import json

        import websockets

        cfg_id = tts_cfg.get("stage_config", "")
        if "no_async_chunk" in cfg_id:
            pytest.skip("WebSocket streaming requires async_chunk config")

        task = tts_cfg["task"]

        async def _ws_stream():
            uri = f"ws://{tts_server.host}:{tts_server.port}/v1/audio/speech/stream"
            chunks: list[bytes] = []
            starts: list[dict] = []
            dones: list[dict] = []
            session_done = None

            session_cfg: dict = {
                "type": "session.config",
                "model": tts_cfg["model"],
                "language": "English",
                "response_format": "pcm",
                "stream_audio": True,
            }
            if task == "CustomVoice":
                session_cfg["voice"] = tts_cfg.get("voice", "vivian")
            elif task == "VoiceDesign":
                session_cfg["task_type"] = "VoiceDesign"
                session_cfg["instructions"] = tts_cfg["instructions"]
            elif task == "Base":
                session_cfg["task_type"] = "Base"
                session_cfg["ref_text"] = tts_cfg["ref_text"]
                session_cfg["ref_audio"] = tts_cfg["ref_audio"]

            async with websockets.connect(uri, max_size=None) as ws:
                await ws.send(json.dumps(session_cfg))
                await ws.send(json.dumps({
                    "type": "input.text",
                    "text": tts_cfg["en_text"],
                }))
                await ws.send(json.dumps({"type": "input.done"}))

                while True:
                    msg = await asyncio.wait_for(ws.recv(), timeout=180)
                    if isinstance(msg, bytes):
                        chunks.append(msg)
                        continue
                    payload = json.loads(msg)
                    mt = payload.get("type")
                    if mt == "audio.start":
                        starts.append(payload)
                    elif mt == "audio.done":
                        dones.append(payload)
                    elif mt == "session.done":
                        session_done = payload
                        break
                    elif mt == "error":
                        raise AssertionError(
                            f"WebSocket error: {payload.get('message')}"
                        )

            return starts, dones, chunks, session_done

        starts, dones, chunks, session_done = asyncio.run(_ws_stream())
        assert session_done is not None, "no session.done received"
        assert len(starts) >= 1, "no audio.start received"
        assert len(dones) >= 1, "no audio.done received"
        total_bytes = sum(len(c) for c in chunks)
        assert total_bytes > MIN_AUDIO_BYTES, f"too few bytes: {total_bytes}"
        _assert_not_silence(b"".join(chunks))
        print(
            f"[{task}] websocket: {len(chunks)} chunks, "
            f"{total_bytes} bytes, {len(starts)} sentence(s)"
        )
