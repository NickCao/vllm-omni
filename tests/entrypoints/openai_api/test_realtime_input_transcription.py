# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import asyncio
import io
import json
import wave
from types import SimpleNamespace
from typing import Any

import pybase64 as base64
import pytest
from openai.types import realtime as types
from pydantic import TypeAdapter
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
from vllm_omni.model_executor.models.qwen3_omni.pipeline import (
    QWEN3_OMNI_PIPELINE,
    QWEN3_OMNI_THINKER_ONLY_PIPELINE,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_SERVER_EVENT_ADAPTER: TypeAdapter[types.RealtimeServerEvent] = TypeAdapter(types.RealtimeServerEvent)


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
                    "transcription": {
                        "model": "whisper-1",
                        "language": "en",
                        "prompt": "greeting style",
                        "delay": "low",
                        "keywords": ["vLLM"],
                        "languages": ["en", "fr"],
                    },
                }
            },
        }
    )
    return types.SessionUpdateEvent(type="session.update", session=session)


def _sent_events(websocket: _FakeWebSocket) -> list[dict]:
    return [json.loads(message) for message in websocket.messages]


class _HandlerState:
    def __init__(self, engine_client: object | None = None) -> None:
        self.engine_client = engine_client or SimpleNamespace(model_config=object())
        self.openai_serving_models = object()


class _FakeStandardServing:
    def __init__(self, engine_client: Any, models: object, *, request_logger: object) -> None:
        self.engine_client = engine_client
        self.models = models
        self.request_logger = request_logger


def _make_handler(state: _HandlerState) -> OpenAIRealtimeHandler:
    handler = OpenAIRealtimeHandler.from_engine_client(state.engine_client)
    handler._openai_serving_models = state.openai_serving_models
    return handler


def test_realtime_handler_builds_and_caches_transcription_serving(monkeypatch) -> None:
    state = _HandlerState()
    handler = _make_handler(state)
    outer_model_cls = object()

    monkeypatch.setattr(realtime_openai, "get_model_cls", lambda _model_config: outer_model_cls)
    monkeypatch.setattr(realtime_openai, "supports_transcription", lambda _model_cls: True)
    monkeypatch.setattr(realtime_openai, "OpenAIServingTranscription", _FakeStandardServing)

    serving = handler._transcription_serving

    assert isinstance(serving, _FakeStandardServing)
    assert serving.engine_client is state.engine_client
    assert serving.models is state.openai_serving_models
    assert serving.request_logger is None
    assert handler._transcription_serving is serving


@pytest.mark.asyncio
async def test_realtime_handler_uses_comprehension_stage_transcription_model(monkeypatch) -> None:
    thinker_config = object()
    transcription_model_config = object()
    transcription_model_cls = object()

    class _FakeEngine:
        def __init__(self) -> None:
            self.model_config = object()
            self.stage_configs = (
                SimpleNamespace(
                    is_comprehension=True,
                    hf_config_name="thinker_config",
                    transcription_model_arch="StageTranscriber",
                ),
            )
            self.stage_vllm_configs = (
                SimpleNamespace(
                    model_config=SimpleNamespace(hf_config=SimpleNamespace(thinker_config=thinker_config)),
                    with_hf_config=lambda hf_config, architectures: SimpleNamespace(
                        model_config=transcription_model_config,
                    ),
                ),
            )
            self.generate_calls: list[dict[str, Any]] = []

        async def generate(self, *args: object, **kwargs: object):
            self.generate_calls.append({"args": args, "kwargs": kwargs})
            yield object()

    engine = _FakeEngine()
    state = _HandlerState(engine)
    handler = _make_handler(state)

    monkeypatch.setattr(
        realtime_openai,
        "get_model_cls",
        lambda model_config: transcription_model_cls if model_config is transcription_model_config else object(),
    )
    monkeypatch.setattr(
        realtime_openai,
        "supports_transcription",
        lambda model_cls: model_cls is transcription_model_cls,
    )
    monkeypatch.setattr(realtime_openai, "OpenAIServingTranscription", _FakeStandardServing)

    serving = handler._transcription_serving

    assert isinstance(serving, _FakeStandardServing)
    assert serving.engine_client.model_config is transcription_model_config
    async for _ in serving.engine_client.generate("prompt", request_id="request"):
        pass
    assert engine.generate_calls[0]["kwargs"]["output_modalities"] == ["text"]


def test_realtime_handler_skips_transcription_for_unsupported_model(monkeypatch) -> None:
    state = _HandlerState()
    handler = _make_handler(state)
    monkeypatch.setattr(realtime_openai, "get_model_cls", lambda _model_config: object())
    monkeypatch.setattr(realtime_openai, "supports_transcription", lambda _model_cls: False)
    monkeypatch.setattr(realtime_openai, "_resolve_comprehension_transcription_model_config", lambda *_args: None)

    assert handler._transcription_serving is None


def test_realtime_handler_fails_closed_on_model_lookup_error(monkeypatch) -> None:
    state = _HandlerState()
    handler = _make_handler(state)

    def _get_model_cls(_model_config: object) -> object:
        raise RuntimeError("model lookup failed")

    monkeypatch.setattr(realtime_openai, "get_model_cls", _get_model_cls)

    assert handler._transcription_serving is None


@pytest.mark.parametrize(
    "pipeline",
    (QWEN3_OMNI_PIPELINE, QWEN3_OMNI_THINKER_ONLY_PIPELINE),
)
def test_qwen_pipelines_declare_thinker_transcription_arch(pipeline) -> None:
    stage = pipeline.stages[0]

    assert stage.owns_tokenizer
    assert stage.transcription_model_arch == "Qwen3OmniMoeThinkerForConditionalGeneration"


def test_comprehension_stage_resolver_uses_declared_model_arch(monkeypatch) -> None:
    outer_model_config = object()
    thinker_config = object()
    transcription_model_config = object()
    transcription_model_cls = object()
    with_hf_config_calls: list[tuple[object, list[str]]] = []

    def _with_hf_config(hf_config: object, architectures: list[str]) -> object:
        with_hf_config_calls.append((hf_config, architectures))
        return SimpleNamespace(model_config=transcription_model_config)

    stage_vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(thinker_config=thinker_config)),
        with_hf_config=_with_hf_config,
    )
    engine = SimpleNamespace(
        model_config=outer_model_config,
        stage_configs=(
            SimpleNamespace(
                is_comprehension=True,
                hf_config_name="thinker_config",
                transcription_model_arch="StageTranscriber",
            ),
        ),
        stage_vllm_configs=(stage_vllm_config,),
    )

    monkeypatch.setattr(realtime_openai, "get_model_cls", lambda _model_config: transcription_model_cls)
    monkeypatch.setattr(
        realtime_openai, "supports_transcription", lambda model_cls: model_cls is transcription_model_cls
    )

    result = realtime_openai._resolve_comprehension_transcription_model_config(engine)

    assert result is transcription_model_config
    assert with_hf_config_calls == [(thinker_config, ["StageTranscriber"])]


def test_comprehension_stage_resolver_rejects_unsupported_declared_arch(monkeypatch) -> None:
    transcription_model_config = object()
    stage_vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(thinker_config=object())),
        with_hf_config=lambda *_args, **_kwargs: SimpleNamespace(model_config=transcription_model_config),
    )
    engine = SimpleNamespace(
        stage_configs=(
            SimpleNamespace(
                stage_id=0,
                is_comprehension=True,
                hf_config_name="thinker_config",
                transcription_model_arch="UnsupportedTranscriber",
            ),
        ),
        stage_vllm_configs=(stage_vllm_config,),
    )

    monkeypatch.setattr(realtime_openai, "get_model_cls", lambda _model_config: object())
    monkeypatch.setattr(realtime_openai, "supports_transcription", lambda _model_cls: False)

    assert realtime_openai._resolve_comprehension_transcription_model_config(engine) is None


def test_comprehension_stage_resolver_skips_stages_without_transcription_arch(monkeypatch) -> None:
    thinker_config = object()
    transcription_model_config = object()
    transcription_model_cls = object()
    stage_vllm_configs = (
        SimpleNamespace(model_config=SimpleNamespace(hf_config=SimpleNamespace(thinker_config=object()))),
        SimpleNamespace(
            model_config=SimpleNamespace(hf_config=SimpleNamespace(thinker_config=thinker_config)),
            with_hf_config=lambda *_args, **_kwargs: SimpleNamespace(model_config=transcription_model_config),
        ),
    )
    engine = SimpleNamespace(
        stage_configs=(
            SimpleNamespace(stage_id=0, is_comprehension=True, hf_config_name="thinker_config"),
            SimpleNamespace(
                stage_id=1,
                is_comprehension=True,
                hf_config_name="thinker_config",
                transcription_model_arch="StageTranscriber",
            ),
        ),
        stage_vllm_configs=stage_vllm_configs,
    )

    monkeypatch.setattr(realtime_openai, "get_model_cls", lambda _model_config: transcription_model_cls)
    monkeypatch.setattr(
        realtime_openai, "supports_transcription", lambda model_cls: model_cls is transcription_model_cls
    )

    assert realtime_openai._resolve_comprehension_transcription_model_config(engine) is transcription_model_config


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
    assert request.file.filename == "realtime-input.wav"
    assert base64.b64decode(item.content[0].audio) == audio

    transcription = connection.session.config.audio.input.transcription
    assert transcription is not None
    assert transcription.model == "test-model"
    assert transcription.language == "en"
    assert transcription.prompt == "greeting style"
    assert transcription.delay is None
    assert transcription.keywords is None
    assert transcription.languages is None
    assert item.content[0].transcript == "hello world"

    events = _sent_events(websocket)
    completed = next(
        event for event in events if event["type"] == "conversation.item.input_audio_transcription.completed"
    )
    _SERVER_EVENT_ADAPTER.validate_python(completed)
    assert completed["item_id"] == item.id
    assert completed["content_index"] == 0
    assert completed["transcript"] == "hello world"
    assert completed["usage"] == {"type": "duration", "seconds": 1}


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
    _SERVER_EVENT_ADAPTER.validate_python(failed)
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
