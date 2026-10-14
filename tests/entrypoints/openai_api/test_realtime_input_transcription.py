# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import asyncio
import io
import json
import wave
from types import SimpleNamespace

import pybase64 as base64
import pytest
from openai.types import realtime as types
from vllm.entrypoints.speech_to_text.transcription.protocol import (
    TranscriptionRequest,
    TranscriptionResponse,
)

from vllm_omni.entrypoints.duplex import openai as realtime_openai
from vllm_omni.entrypoints.duplex.openai import OpenAIRealtimeHandler
from vllm_omni.entrypoints.openai.realtime.connection import (
    SAMPLE_RATE_HZ,
    OpenAIFullDuplexConnection,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _FakeWebSocket:
    def __init__(self) -> None:
        self.messages: list[str] = []

    async def send_text(self, message: str) -> None:
        self.messages.append(message)


class _FakeTranscriptionServing:
    def __init__(self, result: object) -> None:
        self.result = result
        self.calls: list[tuple[bytes, TranscriptionRequest, object]] = []

    async def create_transcription(
        self, audio_data: bytes, request: TranscriptionRequest, raw_request: object = None
    ) -> object:
        self.calls.append((audio_data, request, raw_request))
        return self.result


def _make_connection(
    websocket: _FakeWebSocket,
    transcription_serving: object | None = None,
) -> OpenAIFullDuplexConnection:
    engine = SimpleNamespace(model_config=SimpleNamespace(max_model_len=100))
    chat_handler = SimpleNamespace(renderer=SimpleNamespace(get_tokenizer=lambda: object()))
    return OpenAIFullDuplexConnection(
        websocket=websocket,
        engine=engine,
        model_name="test-model",
        chat_handler=chat_handler,
        transcription_serving=transcription_serving,
    )


def _session_update_event() -> types.SessionUpdateEvent:
    session = types.RealtimeSessionCreateRequest.model_validate(
        {
            "type": "realtime",
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": SAMPLE_RATE_HZ},
                    "transcription": {"language": "en", "prompt": "greeting style"},
                }
            },
        }
    )
    return types.SessionUpdateEvent(type="session.update", session=session)


def _sent_events(websocket: _FakeWebSocket) -> list[dict]:
    return [json.loads(message) for message in websocket.messages]


class _HandlerState:
    def __init__(self) -> None:
        self.engine_client = SimpleNamespace(model_config=object())
        self.openai_serving_models = object()


def test_realtime_handler_builds_and_caches_transcription_serving(monkeypatch) -> None:
    handler = OpenAIRealtimeHandler()
    state = _HandlerState()
    constructed: list[dict[str, object]] = []

    class _FakeServing:
        def __init__(self, *args: object, **kwargs: object) -> None:
            constructed.append({"args": args, "kwargs": kwargs})

    monkeypatch.setattr(realtime_openai, "get_model_cls", lambda _model_config: object())
    monkeypatch.setattr(realtime_openai, "supports_transcription", lambda _model_cls: True)
    monkeypatch.setattr(realtime_openai, "OpenAIServingTranscription", _FakeServing)

    serving = handler._resolve_transcription_serving(state)

    assert isinstance(serving, _FakeServing)
    assert len(constructed) == 1
    assert constructed[0]["args"] == (state.engine_client, state.openai_serving_models)
    assert constructed[0]["kwargs"] == {"request_logger": None}
    assert handler._resolve_transcription_serving(state) is serving
    assert len(constructed) == 1


def test_realtime_handler_skips_transcription_for_unsupported_model(monkeypatch) -> None:
    handler = OpenAIRealtimeHandler()
    state = _HandlerState()
    monkeypatch.setattr(realtime_openai, "get_model_cls", lambda _model_config: object())
    monkeypatch.setattr(realtime_openai, "supports_transcription", lambda _model_cls: False)

    assert handler._resolve_transcription_serving(state) is None


def test_realtime_handler_fails_closed_on_model_lookup_error(monkeypatch) -> None:
    handler = OpenAIRealtimeHandler()
    state = _HandlerState()

    def _get_model_cls(_model_config: object) -> object:
        raise RuntimeError("model lookup failed")

    monkeypatch.setattr(realtime_openai, "get_model_cls", _get_model_cls)

    assert handler._resolve_transcription_serving(state) is None


@pytest.mark.asyncio
async def test_input_transcription_reuses_transcription_serving() -> None:
    websocket = _FakeWebSocket()
    serving = _FakeTranscriptionServing(
        TranscriptionResponse(text="hello world", usage={"type": "duration", "seconds": 1})
    )
    connection = _make_connection(websocket, transcription_serving=serving)

    await connection._handle_session_update(_session_update_event())
    audio = b"\x00\x00\x00\x40\x00\x80"
    connection.session.input_audio_buffer.extend(audio)
    item = await connection._commit_audio_buffer()
    await asyncio.gather(*connection._transcription_tasks)

    assert item is not None
    assert len(serving.calls) == 1
    wav_audio, request, raw_request = serving.calls[0]
    assert raw_request is None
    with wave.open(io.BytesIO(wav_audio)) as wav:
        assert wav.getframerate() == SAMPLE_RATE_HZ
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.readframes(len(audio) // 2) == audio
    assert request.model == "test-model"
    assert request.language == "en"
    assert request.prompt == "greeting style"
    assert request.response_format == "json"
    assert "file" not in json.loads(request.model_dump_json(exclude_unset=True))
    assert base64.b64decode(item.content[0].audio) == audio

    events = _sent_events(websocket)
    completed = next(
        event for event in events if event["type"] == "conversation.item.input_audio_transcription.completed"
    )
    assert completed["item_id"] == item.id
    assert completed["content_index"] == 0
    assert completed["transcript"] == "hello world"


@pytest.mark.asyncio
async def test_input_transcription_failed_event() -> None:
    websocket = _FakeWebSocket()
    serving = _FakeTranscriptionServing(SimpleNamespace(error=SimpleNamespace(message="transcription engine failed")))
    connection = _make_connection(websocket, transcription_serving=serving)

    await connection._handle_session_update(_session_update_event())
    connection.session.input_audio_buffer.extend(b"\x00\x00\x00\x40")
    await connection._commit_audio_buffer()
    await asyncio.gather(*connection._transcription_tasks)

    events = _sent_events(websocket)
    failed = next(event for event in events if event["type"] == "conversation.item.input_audio_transcription.failed")
    assert failed["error"]["type"] == "transcription_error"
    assert failed["error"]["code"] == "transcription_failed"


@pytest.mark.asyncio
async def test_input_transcription_is_ignored_without_model_support() -> None:
    websocket = _FakeWebSocket()
    connection = _make_connection(websocket, transcription_serving=None)

    await connection._handle_session_update(_session_update_event())

    events = _sent_events(websocket)
    assert events[-1]["type"] == "session.updated"
    assert connection.session.config.audio.input.transcription is None

    connection.session.input_audio_buffer.extend(b"\x00\x00\x00\x40")
    await connection._commit_audio_buffer()
    await asyncio.sleep(0)
    assert not connection._transcription_tasks
    assert not [event for event in events if event["type"].startswith("conversation.item.input_audio_transcription")]
