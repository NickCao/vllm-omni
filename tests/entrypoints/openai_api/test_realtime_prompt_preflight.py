# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import base64
import io
import wave
from types import SimpleNamespace
from typing import Any

import pytest

from vllm_omni.entrypoints.openai.realtime.connection import (
    SAMPLE_RATE_HZ,
    OpenAIFullDuplexConnection,
    _ResolvedResponse,
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
    *, max_model_len: int = 100, websocket: Any = None
) -> tuple[OpenAIFullDuplexConnection, _FakeRenderer]:
    renderer = _FakeRenderer()
    model_config = SimpleNamespace(max_model_len=max_model_len, multimodal_config=None)

    async def preprocess_chat(request: Any, messages: list[dict[str, Any]], **kwargs: Any):
        tok_params = kwargs.get("tok_params")
        if tok_params is None:
            tok_params = request.build_tok_params(model_config)
        return await renderer.render_chat_async(
            [messages],
            kwargs.get("default_template_kwargs"),
            tok_params,
        )

    connection = OpenAIFullDuplexConnection(
        websocket=websocket,
        engine=SimpleNamespace(model_config=model_config),
        model_name="test-model",
        chat_handler=SimpleNamespace(
            renderer=renderer,
            chat_template=None,
            chat_template_content_format="auto",
            _effective_chat_template_kwargs=lambda _request: {},
            _preprocess_chat=preprocess_chat,
        ),
    )
    return connection, renderer


@pytest.mark.asyncio
async def test_preflight_truncates_by_rendered_token_count() -> None:
    connection, renderer = _make_connection()

    first = SimpleNamespace(
        id="first",
        type="message",
        role="user",
        content=[SimpleNamespace(type="input_text", text="first")],
    )
    second = SimpleNamespace(
        id="second",
        type="message",
        role="user",
        content=[SimpleNamespace(type="input_text", text="second")],
    )
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

    engine_input = await connection._prepare_engine_input_with_auto_truncation(response)

    assert renderer.raw_prompt_lengths == [4, 2]
    assert [len(prompt["prompt_token_ids"]) for prompt in renderer.engine_inputs] == [90, 50]
    assert all(params.max_total_tokens is None for params in renderer.tokenization_params)
    assert all(params.max_output_tokens == 0 for params in renderer.tokenization_params)
    assert items == [second]
    assert engine_input is renderer.engine_inputs[-1]


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
