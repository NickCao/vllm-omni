#!/usr/bin/env python3
r"""
Voice-to-Voice Changer Demo
============================

Real-time voice cloning with streaming audio playback: speak into your
mic and hear it back in someone else's voice as it generates.

Flow: mic input -> Whisper transcription -> Qwen3-TTS voice clone -> streaming audio output

Prerequisites
-------------
1. Start the vllm-omni server with a Base (voice-clone) model::

    vllm serve Qwen/Qwen3-TTS-12Hz-1.7B-Base --omni \
        --stage-configs-path vllm_omni/model_executor/stage_configs/qwen3_tts.yaml \
        --trust-remote-code --port 8000

2. Install demo dependencies::

    pip install gradio websockets soundfile numpy openai-whisper

3. Run the demo::

    python examples/demos/voice_changer/app.py [--server-url localhost:8000]
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import queue
import struct
import threading
import time
from typing import Generator

import gradio as gr
import numpy as np
import soundfile as sf
import websockets
import whisper

SAMPLE_RATE = 24000

_whisper_model = None


def _get_whisper():
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = whisper.load_model("base")
    return _whisper_model


def _prep_audio(audio: tuple[int, np.ndarray]) -> tuple[np.ndarray, int]:
    sr, data = audio
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data, sr


def _numpy_to_wav_data_uri(audio: np.ndarray, sr: int) -> str:
    # Clamp to [-1, 1] to prevent clipping warnings in the TTS tokenizer
    audio = np.clip(audio, -1.0, 1.0)
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:audio/wav;base64,{b64}"


def _transcribe(audio: np.ndarray, sr: int) -> str:
    audio = audio.astype(np.float32)
    if np.max(np.abs(audio)) > 1.0:
        audio = audio / 32768.0
    if sr != 16000:
        duration = len(audio) / sr
        n = int(duration * 16000)
        indices = np.linspace(0, len(audio) - 1, n)
        audio = np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)
    result = _get_whisper().transcribe(audio, fp16=False)
    return result["text"].strip()


def _auto_transcribe_ref(ref_audio: tuple[int, np.ndarray] | None) -> str:
    if ref_audio is None:
        return ""
    data, sr = _prep_audio(ref_audio)
    if len(data) == 0:
        return ""
    return _transcribe(data, sr)


def _pcm_to_wav_bytes(pcm: bytes, sr: int = SAMPLE_RATE) -> bytes:
    """Convert raw PCM16 bytes to a WAV byte buffer for Gradio streaming."""
    samples = np.array(struct.unpack(f"<{len(pcm) // 2}h", pcm), dtype=np.float32) / 32768.0
    buf = io.BytesIO()
    sf.write(buf, samples, sr, format="WAV")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Streaming WebSocket receiver — runs in a thread, pushes chunks to a queue
# ---------------------------------------------------------------------------

_SENTINEL = None  # signals end of stream


def _ws_receiver_thread(
    server_url: str,
    ref_audio_uri: str,
    ref_text: str,
    text: str,
    language: str,
    chunk_queue: queue.Queue,
    stats: dict,
):
    """Background thread: connect to WebSocket, push PCM bytes to queue."""

    async def _run():
        uri = f"ws://{server_url}/v1/audio/speech/stream"
        config = {
            "type": "session.config",
            "task_type": "Base",
            "ref_audio": ref_audio_uri,
            "ref_text": ref_text,
            "language": language,
            "response_format": "pcm",
            "stream_audio": True,
        }

        t_start = time.perf_counter()
        t_first = None

        try:
            async with websockets.connect(uri, max_size=None) as ws:
                await ws.send(json.dumps(config))
                await ws.send(json.dumps({"type": "input.text", "text": text}))
                await ws.send(json.dumps({"type": "input.done"}))

                while True:
                    msg = await asyncio.wait_for(ws.recv(), timeout=300)
                    if isinstance(msg, bytes):
                        if t_first is None:
                            t_first = time.perf_counter()
                            stats["ttfa_ms"] = (t_first - t_start) * 1000
                        stats["num_chunks"] += 1
                        chunk_queue.put(msg)
                    else:
                        payload = json.loads(msg)
                        if payload.get("type") == "session.done":
                            break
                        if payload.get("type") == "error":
                            chunk_queue.put(RuntimeError(payload.get("message", "server error")))
                            return
        except Exception as e:
            chunk_queue.put(e)
            return
        finally:
            stats["total_ms"] = (time.perf_counter() - t_start) * 1000
            chunk_queue.put(_SENTINEL)

    asyncio.run(_run())


# Bytes per second at 24kHz 16-bit mono
_BYTES_PER_SEC = SAMPLE_RATE * 2
# Buffer a longer first chunk to build playback headroom, then use
# shorter chunks for lower latency on subsequent yields.
_FIRST_CHUNK_BYTES = _BYTES_PER_SEC * 5  # 5 seconds
_NEXT_CHUNK_BYTES = _BYTES_PER_SEC  # 1 second


def _streaming_clone(
    server_url: str,
    ref_audio_uri: str,
    ref_text: str,
    text: str,
    language: str,
) -> Generator[bytes, None, None]:
    """Yield WAV byte chunks (>= 1s each) as audio streams from the server."""
    chunk_queue: queue.Queue = queue.Queue()
    stats = {"ttfa_ms": 0, "total_ms": 0, "num_chunks": 0}

    t = threading.Thread(
        target=_ws_receiver_thread,
        args=(server_url, ref_audio_uri, ref_text, text, language, chunk_queue, stats),
        daemon=True,
    )
    t.start()

    pcm_buffer = b""
    first = True
    while True:
        item = chunk_queue.get()
        if item is _SENTINEL:
            if pcm_buffer:
                yield _pcm_to_wav_bytes(pcm_buffer)
            break
        if isinstance(item, Exception):
            raise item
        pcm_buffer += item
        threshold = _FIRST_CHUNK_BYTES if first else _NEXT_CHUNK_BYTES
        if len(pcm_buffer) >= threshold:
            yield _pcm_to_wav_bytes(pcm_buffer)
            pcm_buffer = b""
            first = False

    t.join(timeout=5)


# ---------------------------------------------------------------------------
# Gradio callbacks — split into prepare (regular) + stream (generator)
# ---------------------------------------------------------------------------


def _prepare_v2v(
    ref_audio: tuple[int, np.ndarray] | None,
    ref_text: str,
    voice_input: tuple[int, np.ndarray] | None,
    language: str,
    server_url: str,
) -> tuple[str, str, str, str, str]:
    """Validate inputs, transcribe mic, return (ref_uri, ref_text, transcript, language, server_url).

    Stores results in gr.State components for the streaming step.
    """
    if ref_audio is None:
        raise gr.Error("Upload or record a reference voice first.")
    if not ref_text.strip():
        raise gr.Error("Reference transcript is empty. Upload audio and wait for auto-transcription.")
    if voice_input is None:
        raise gr.Error("Record your voice input first.")

    ref_np, sr_ref = _prep_audio(ref_audio)
    ref_uri = _numpy_to_wav_data_uri(ref_np, sr_ref)

    voice_np, sr_in = _prep_audio(voice_input)
    transcript = _transcribe(voice_np, sr_in)
    if not transcript:
        raise gr.Error("Could not transcribe your voice. Try speaking more clearly.")

    return ref_uri, ref_text, transcript, language, server_url


def _stream_v2v(ref_uri, ref_text, transcript, language, server_url):
    """Generator: yield (sr, chunk) for streaming Audio output."""
    for chunk in _streaming_clone(server_url, ref_uri, ref_text, transcript, language):
        yield chunk


def _prepare_t2v(
    ref_audio: tuple[int, np.ndarray] | None,
    ref_text: str,
    text: str,
    language: str,
    server_url: str,
) -> tuple[str, str, str, str, str]:
    """Validate inputs, return (ref_uri, ref_text, text, language, server_url)."""
    if ref_audio is None:
        raise gr.Error("Upload or record a reference voice first.")
    if not ref_text.strip():
        raise gr.Error("Reference transcript is empty. Upload audio and wait for auto-transcription.")
    if not text.strip():
        raise gr.Error("Enter the text you want to say.")

    ref_np, sr_ref = _prep_audio(ref_audio)
    ref_uri = _numpy_to_wav_data_uri(ref_np, sr_ref)

    return ref_uri, ref_text, text, language, server_url


def _stream_t2v(ref_uri, ref_text, text, language, server_url):
    """Generator: yield (sr, chunk) for streaming Audio output."""
    for chunk in _streaming_clone(server_url, ref_uri, ref_text, text, language):
        yield chunk


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------


def build_ui(server_url: str) -> gr.Blocks:
    with gr.Blocks(title="Voice Changer") as app:
        gr.Markdown(
            "# Voice Changer\n"
            "Speak into your mic and hear it back in someone else's voice.\n\n"
            "**Step 1:** Upload a reference voice clip (transcript is auto-detected).  \n"
            "**Step 2:** Use either tab below to clone via mic or text.  \n"
            "Audio streams progressively as it generates."
        )

        server_state = gr.State(server_url)

        with gr.Group():
            gr.Markdown("### Reference Voice (the voice to clone)")
            with gr.Row():
                ref_audio = gr.Audio(
                    label="Reference audio (5-15s recommended)",
                    sources=["upload", "microphone"],
                    type="numpy",
                )
                ref_text = gr.Textbox(
                    label="Reference transcript (auto-detected, editable)",
                    placeholder="Upload audio — transcript will appear here automatically...",
                    lines=3,
                    interactive=True,
                )

            ref_audio.change(
                fn=_auto_transcribe_ref,
                inputs=[ref_audio],
                outputs=[ref_text],
            )

        with gr.Tabs():
            with gr.TabItem("Voice to Voice"):
                gr.Markdown("Record yourself speaking, then hear it in the cloned voice.")
                voice_input = gr.Audio(
                    label="Your voice (press record and speak)",
                    sources=["microphone"],
                    type="numpy",
                )
                language_v2v = gr.Dropdown(
                    choices=["Auto", "English", "Chinese"],
                    value="Auto",
                    label="Language",
                )
                btn_v2v = gr.Button("Clone Voice", variant="primary", size="lg")
                transcript_box = gr.Textbox(
                    label="Detected transcript (from your voice)",
                    interactive=False,
                )
                output_v2v = gr.Audio(
                    label="Cloned output",
                    type="numpy",
                    streaming=True,
                    autoplay=True,
                )

                # Hidden states to pass data between prepare and stream steps
                v2v_ref_uri = gr.State()
                v2v_ref_text = gr.State()
                v2v_transcript = gr.State()
                v2v_language = gr.State()
                v2v_server = gr.State()

                # Step 1: validate + transcribe (regular function)
                # Step 2: stream audio (generator, only outputs to Audio)
                btn_v2v.click(
                    fn=_prepare_v2v,
                    inputs=[ref_audio, ref_text, voice_input, language_v2v, server_state],
                    outputs=[v2v_ref_uri, v2v_ref_text, v2v_transcript, v2v_language, v2v_server],
                ).then(
                    fn=lambda t: t,
                    inputs=[v2v_transcript],
                    outputs=[transcript_box],
                ).then(
                    fn=_stream_v2v,
                    inputs=[v2v_ref_uri, v2v_ref_text, v2v_transcript, v2v_language, v2v_server],
                    outputs=[output_v2v],
                )

            with gr.TabItem("Text to Voice"):
                gr.Markdown("Type what you want to say in the cloned voice.")
                text_input = gr.Textbox(
                    label="Text to synthesize",
                    placeholder="Type what you want to say...",
                    lines=3,
                )
                language_t2v = gr.Dropdown(
                    choices=["Auto", "English", "Chinese"],
                    value="Auto",
                    label="Language",
                )
                btn_t2v = gr.Button("Clone Voice", variant="primary", size="lg")
                output_t2v = gr.Audio(
                    label="Cloned output",
                    type="numpy",
                    streaming=True,
                    autoplay=True,
                )

                t2v_ref_uri = gr.State()
                t2v_ref_text = gr.State()
                t2v_text = gr.State()
                t2v_language = gr.State()
                t2v_server = gr.State()

                btn_t2v.click(
                    fn=_prepare_t2v,
                    inputs=[ref_audio, ref_text, text_input, language_t2v, server_state],
                    outputs=[t2v_ref_uri, t2v_ref_text, t2v_text, t2v_language, t2v_server],
                ).then(
                    fn=_stream_t2v,
                    inputs=[t2v_ref_uri, t2v_ref_text, t2v_text, t2v_language, t2v_server],
                    outputs=[output_t2v],
                )

        with gr.Accordion("Example reference audio", open=False):
            gr.Markdown(
                "Try the official Qwen3-TTS reference:\n"
                "- Download: https://qianwen-res.oss-cn-beijing.aliyuncs.com/"
                "Qwen3-TTS-Repo/clone_2.wav\n"
                "- Upload it above — the transcript will be auto-detected."
            )

    return app


def main():
    parser = argparse.ArgumentParser(description="Voice-to-Voice Changer Demo")
    parser.add_argument(
        "--server-url",
        default="localhost:8000",
        help="vllm-omni server host:port (default: localhost:8000)",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Gradio bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=3000, help="Gradio UI port (default: 3000)")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link")
    args = parser.parse_args()

    print("Loading Whisper model...", flush=True)
    _get_whisper()
    print("Whisper ready.", flush=True)

    app = build_ui(args.server_url)
    print(f"Launching on {args.host}:{args.port}", flush=True)
    app.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
