# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import base64
import io
import json
import wave
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from openai.types import realtime as types
from vllm.sampling_params import RequestOutputKind, SamplingParams

from tests.helpers.serving_chat import build_serving_chat
from vllm_omni.entrypoints.openai.realtime.connection import (
    SAMPLE_RATE_HZ,
    OpenAIFullDuplexConnection,
    _ResolvedResponse,
)
from vllm_omni.entrypoints.openai.realtime.session import (
    ActiveResponse,
    AudioFullDuplexSessionState,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _FakeRenderer:
    def get_tokenizer(self) -> object:
        return object()

    def __init__(self) -> None:
        self.raw_prompt_lengths: list[int] = []
        self.tokenization_params: list[Any] = []
        self.engine_inputs: list[dict[str, Any]] = []
        self.conversations: list[list[dict[str, Any]]] = []

    async def render_chat_async(
        self,
        conversations: list[list[dict[str, Any]]],
        _chat_params: Any,
        tok_params: Any,
    ):
        self.conversations.append(conversations[0])
        raw_prompt_length = 2 * len(conversations[0])
        self.raw_prompt_lengths.append(raw_prompt_length)
        self.tokenization_params.append(tok_params)
        # Model-side expansion makes the rendered prompt much longer than its text tokens.
        engine_input = {"prompt_token_ids": [0] * (10 + 20 * raw_prompt_length)}
        self.engine_inputs.append(engine_input)
        return conversations, [engine_input]


class _FakeWebSocket:
    def __init__(self) -> None:
        self.messages: list[str] = []

    async def send_text(self, message: str) -> None:
        self.messages.append(message)


def _make_connection(
    *,
    max_model_len: int = 100,
    websocket: Any = None,
    engine: Any = None,
    chat_handler: Any = None,
) -> tuple[OpenAIFullDuplexConnection, _FakeRenderer]:
    renderer = _FakeRenderer()
    model_config = SimpleNamespace(max_model_len=max_model_len, multimodal_config=None)

    async def preprocess_chat(request: Any, messages: list[dict[str, Any]], **kwargs: Any):
        tok_params = request.build_tok_params(model_config)
        return await renderer.render_chat_async(
            [messages],
            kwargs.get("default_template_kwargs"),
            tok_params,
        )

    connection = OpenAIFullDuplexConnection(
        websocket=websocket,
        engine=engine if engine is not None else SimpleNamespace(model_config=model_config),
        model_name="test-model",
        chat_handler=chat_handler
        if chat_handler is not None
        else SimpleNamespace(
            renderer=renderer,
            chat_template=None,
            chat_template_content_format="auto",
            _effective_chat_template_kwargs=lambda _request: {},
            _preprocess_chat=preprocess_chat,
            _fix_minicpmo45_audio_stream_output_kinds=lambda params, _modalities: params,
        ),
    )
    return connection, renderer


def _message_item(
    item_id: str,
    text: str | None = None,
) -> types.RealtimeConversationItemUserMessage:
    return types.RealtimeConversationItemUserMessage(
        id=item_id,
        type="message",
        role="user",
        content=[{"type": "input_text", "text": text or item_id}],
    )


@pytest.mark.asyncio
async def test_preflight_truncates_by_rendered_token_count() -> None:
    connection, renderer = _make_connection()

    first = _message_item("first")
    second = _message_item("second")
    items = [first, second]
    response = _ResolvedResponse(
        input=items,
        instructions=None,
        modalities=["text"],
        max_output_tokens="inf",
        tools=None,
        tool_choice="none",
        metadata=None,
    )

    prompt_items = await connection._truncate_prompt_items(response)
    engine_input = await connection._build_full_prompt(
        tools=response.tools,
        instructions=response.instructions,
        items=prompt_items,
    )

    assert renderer.raw_prompt_lengths == [4, 2, 2]
    assert [len(prompt["prompt_token_ids"]) for prompt in renderer.engine_inputs] == [90, 50, 50]
    assert all(params.max_total_tokens == 100 for params in renderer.tokenization_params)
    assert all(params.max_output_tokens == 0 for params in renderer.tokenization_params)
    assert items == [first, second]
    assert prompt_items == [second]
    assert engine_input is renderer.engine_inputs[-1]


@pytest.mark.asyncio
async def test_persistent_model_context_cursor_reuses_truncated_window() -> None:
    connection, renderer = _make_connection()
    first = _message_item("first")
    second = _message_item("second")
    connection.session.insert_item(first)
    connection.session.insert_item(second)
    response = _ResolvedResponse(
        input=None,
        instructions=None,
        modalities=["text"],
        max_output_tokens="inf",
        tools=None,
        tool_choice="none",
        metadata=None,
    )

    prompt_items = await connection._truncate_prompt_items(response)
    await connection._build_full_prompt(
        tools=response.tools,
        instructions=response.instructions,
        items=prompt_items,
    )
    connection.session.commit_model_context_items(prompt_items)
    renderer.raw_prompt_lengths.clear()

    prompt_items = await connection._truncate_prompt_items(response)

    assert renderer.raw_prompt_lengths == [2]
    assert prompt_items == [second]
    assert connection.session.model_context_first_item_id == "second"
    assert not connection.session.model_context_cursor_at_end


@pytest.mark.asyncio
async def test_empty_model_context_allows_a_later_appended_item() -> None:
    connection, renderer = _make_connection(max_model_len=20)
    response = _ResolvedResponse(
        input=None,
        instructions=None,
        modalities=["text"],
        max_output_tokens="inf",
        tools=None,
        tool_choice="none",
        metadata=None,
    )

    prompt_items = await connection._truncate_prompt_items(response)
    connection.session.commit_model_context_items(prompt_items)
    connection.session.insert_item(_message_item("appended"))

    assert renderer.raw_prompt_lengths == [0]
    assert prompt_items == []
    assert [item.id for item in connection.session.model_context_items()] == ["appended"]
    assert connection.session.model_context_first_item_id == "appended"
    assert not connection.session.model_context_cursor_at_end


@pytest.mark.asyncio
async def test_truncation_to_empty_selects_no_items_when_overhead_fits() -> None:
    connection, renderer = _make_connection(max_model_len=20)
    connection.session.insert_item(_message_item("item"))
    response = _ResolvedResponse(
        input=None,
        instructions=None,
        modalities=["text"],
        max_output_tokens="inf",
        tools=None,
        tool_choice="none",
        metadata=None,
    )

    prompt_items = await connection._truncate_prompt_items(response)
    connection.session.commit_model_context_items(prompt_items)
    connection.session.insert_item(_message_item("appended"))

    assert renderer.raw_prompt_lengths == [2, 0]
    assert prompt_items == []
    assert [item.id for item in connection.session.model_context_items()] == ["appended"]


def test_model_context_cursor_moves_only_forward() -> None:
    session = AudioFullDuplexSessionState()

    session.insert_item(_message_item("first"))
    session.insert_item(_message_item("second"))
    session.insert_item(_message_item("third"))
    assert session.model_context_first_item_id == "first"
    assert session.model_context_items() == session.items

    session.commit_model_context_items(session.items[1:])
    session.insert_item(_message_item("before-second"), previous_item_id="first")
    assert session.model_context_first_item_id == "second"
    assert [item.id for item in session.model_context_items()] == ["second", "third"]

    session.insert_item(_message_item("after-second"), previous_item_id="second")
    assert session.model_context_first_item_id == "second"
    assert [item.id for item in session.model_context_items()] == [
        "second",
        "after-second",
        "third",
    ]

    session.remove_item("second")
    assert session.model_context_first_item_id == "after-second"
    assert [item.id for item in session.model_context_items()] == ["after-second", "third"]

    session.commit_model_context_items([])
    session.insert_item(_message_item("fourth"))
    session.insert_item(_message_item("root"), previous_item_id="root")
    assert session.model_context_first_item_id == "fourth"
    assert not session.model_context_cursor_at_end
    assert [item.id for item in session.model_context_items()] == ["fourth"]


@pytest.mark.asyncio
async def test_prompt_truncation_bisects_the_model_context_cursor() -> None:
    connection, renderer = _make_connection()
    items = [
        SimpleNamespace(
            id=f"item_{index}",
            type="message",
            role="user",
            content=[SimpleNamespace(type="input_text", text=str(index))],
        )
        for index in range(8)
    ]
    response = _ResolvedResponse(
        input=items,
        instructions=None,
        modalities=["text"],
        max_output_tokens="inf",
        tools=None,
        tool_choice="none",
        metadata=None,
    )

    prompt_items = await connection._truncate_prompt_items(response)
    await connection._build_full_prompt(
        tools=response.tools,
        instructions=response.instructions,
        items=prompt_items,
    )

    assert renderer.raw_prompt_lengths == [16, 8, 4, 2, 2]
    assert prompt_items == [items[-1]]


@pytest.mark.asyncio
async def test_audio_prompt_uses_standard_chat_audio_content() -> None:
    audio = b"\x00\x00\x00\x40"
    connection, renderer = _make_connection(websocket=_FakeWebSocket())
    connection.session.input_audio_buffer.extend(audio)
    item = await connection._commit_audio_buffer()

    assert item is not None
    assert base64.b64decode(item.content[0].audio) == audio

    await connection._build_full_prompt(items=[item])

    audio_content = renderer.conversations[0][0]["content"][0]["input_audio"]
    assert audio_content["format"] == "wav"
    with wave.open(io.BytesIO(base64.b64decode(audio_content["data"]))) as wav:
        assert wav.getframerate() == SAMPLE_RATE_HZ
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.readframes(2) == audio


@pytest.mark.asyncio
async def test_generated_audio_is_resampled_to_realtime_rate() -> None:
    source_rate = 32_000
    waveform = np.sin(2 * np.pi * 440 * np.arange(source_rate) / source_rate).astype(np.float32)
    outputs: list[Any] = []
    for chunk in np.split(waveform, 2):
        multimodal_output = {"audio": chunk, "sample_rate": source_rate}
        outputs.append(
            SimpleNamespace(
                final_output_type="audio",
                multimodal_output=multimodal_output,
                outputs=[SimpleNamespace(multimodal_output=multimodal_output)],
            )
        )

    async def generate(**_kwargs):
        for output in outputs:
            yield output

    model_config = SimpleNamespace(max_model_len=100, multimodal_config=None)
    engine = SimpleNamespace(
        model_config=model_config,
        default_sampling_params_list=[],
        generate=generate,
    )
    websocket = _FakeWebSocket()
    connection, _ = _make_connection(websocket=websocket, engine=engine)
    response = _ResolvedResponse(
        input=[],
        instructions=None,
        modalities=["audio"],
        max_output_tokens="inf",
        tools=None,
        tool_choice="none",
        metadata=None,
    )
    active = ActiveResponse(response_id="resp_test", request_id="req_test")
    connection.session.active_response = active

    await connection._run_response(active.response_id, response, {"prompt_token_ids": []})

    events = [json.loads(message) for message in websocket.messages]
    audio_deltas = [event for event in events if event["type"] == "response.output_audio.delta"]
    pcm = b"".join(base64.b64decode(event["delta"]) for event in audio_deltas)
    assert active.item_id is not None
    assert len(pcm) == SAMPLE_RATE_HZ * 2
    assert connection.session.item_duration_ms[active.item_id] == pytest.approx(1000)


@pytest.mark.asyncio
async def test_minicpmo45_realtime_keeps_thinker_final_only() -> None:
    model_arch = "MiniCPMO45OmniForConditionalGeneration"
    stages = [
        SimpleNamespace(engine_args=SimpleNamespace(model_arch=model_arch, model_stage="llm")),
        SimpleNamespace(engine_args=SimpleNamespace(model_arch=model_arch, model_stage="tts")),
    ]
    submitted_params: list[list[SamplingParams]] = []

    async def generate(**kwargs):
        submitted_params.append(kwargs["sampling_params_list"])
        yield SimpleNamespace(final_output_type="text", outputs=[])

    model_config = SimpleNamespace(max_model_len=100, multimodal_config=None)
    chat_handler = build_serving_chat()
    engine = chat_handler.engine_client
    engine.model_config = model_config
    engine.stage_configs = stages
    engine.default_sampling_params_list = [SamplingParams(), SamplingParams()]
    engine.generate = generate
    websocket = _FakeWebSocket()
    connection, _ = _make_connection(
        websocket=websocket,
        engine=engine,
        chat_handler=chat_handler,
    )
    response = _ResolvedResponse(
        input=[],
        instructions=None,
        modalities=["audio"],
        max_output_tokens="inf",
        tools=None,
        tool_choice="none",
        metadata=None,
    )
    active = ActiveResponse(response_id="resp_test", request_id="req_test")
    connection.session.active_response = active

    await connection._run_response(active.response_id, response, {"prompt_token_ids": []})

    assert len(submitted_params) == 1
    assert [params.output_kind for params in submitted_params[0]] == [
        RequestOutputKind.FINAL_ONLY,
        RequestOutputKind.DELTA,
    ]


@pytest.mark.parametrize(
    ("audio_end_ms", "expected"),
    [(0, ""), (250, "0"), (500, "01"), (999, "012"), (1000, "0123")],
)
def test_truncate_transcript_uses_item_audio_duration(audio_end_ms: float, expected: str) -> None:
    connection, _ = _make_connection()
    connection._tokenizer = SimpleNamespace(decode=lambda ids, **_kwargs: "".join(map(str, ids)))
    connection.session.item_duration_ms["item"] = 1000
    connection.session.item_token_ids["item"] = [0, 1, 2, 3]

    assert connection._truncate_transcript("item", audio_end_ms) == expected


def test_truncate_transcript_falls_back_without_tokens_or_duration() -> None:
    connection, _ = _make_connection()
    connection._tokenizer = SimpleNamespace(decode=lambda ids, **_kwargs: "".join(map(str, ids)))
    connection.session.item_token_ids["missing_duration"] = [0, 1, 2, 3]
    connection.session.item_duration_ms["missing_tokens"] = 1000

    assert connection._truncate_transcript("missing_duration", 500) == ""
    assert connection._truncate_transcript("missing_tokens", 500) == ""


@pytest.mark.asyncio
async def test_rejected_response_create_does_not_cancel_active_response() -> None:
    websocket = _FakeWebSocket()
    connection, _ = _make_connection(max_model_len=5, websocket=websocket)
    active_response = SimpleNamespace(response_id="active", request_id="active-request")
    connection.session.active_response = active_response

    await connection._handle_response_create(SimpleNamespace(event_id="evt_create", response=None))

    assert connection.session.active_response is active_response
    assert not connection._response_cancel_event.is_set()
    assert len(websocket.messages) == 1
    assert "exceeds the model's input token limit" in websocket.messages[0]
