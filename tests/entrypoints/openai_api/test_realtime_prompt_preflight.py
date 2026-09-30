# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from vllm_omni.entrypoints.openai.realtime.connection import OpenAIFullDuplexConnection, _ResolvedResponse

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@dataclass
class _FakeTokenizationParams:
    max_total_tokens: int | None = 128
    max_output_tokens: int | None = 32


class _FakeTokenizer:
    def apply_chat_template(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        del kwargs
        return "|".join(message["content"] for message in messages)

    @staticmethod
    def encode(text: str, *, add_special_tokens: bool) -> list[int]:
        del add_special_tokens
        return [0] * (2 * len(text.split("|")) if text else 0)


class _FakeRenderer:
    default_chat_tok_params = _FakeTokenizationParams()

    def __init__(self) -> None:
        self.raw_prompt_lengths: list[int] = []
        self.tokenization_params: list[_FakeTokenizationParams] = []
        self.engine_inputs: list[dict[str, Any]] = []

    async def render_cmpl_async(self, prompts: list[Any], tok_params: _FakeTokenizationParams):
        raw_prompt_length = len(prompts[0]["prompt_token_ids"])
        self.raw_prompt_lengths.append(raw_prompt_length)
        self.tokenization_params.append(tok_params)
        # Model-side expansion makes the rendered prompt much longer than its text tokens.
        engine_input = {"prompt_token_ids": [0] * (10 + 20 * raw_prompt_length)}
        self.engine_inputs.append(engine_input)
        return [engine_input]


class _FakeWebSocket:
    def __init__(self) -> None:
        self.messages: list[str] = []

    async def send_text(self, message: str) -> None:
        self.messages.append(message)


def _make_connection(
    *, max_model_len: int = 100, websocket: Any = None
) -> tuple[OpenAIFullDuplexConnection, _FakeRenderer]:
    renderer = _FakeRenderer()
    connection = OpenAIFullDuplexConnection(
        websocket=websocket,
        engine=SimpleNamespace(model_config=SimpleNamespace(max_model_len=max_model_len), renderer=renderer),
        model_name="test-model",
        tokenizer=_FakeTokenizer(),
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
    assert all(params.max_output_tokens == 32 for params in renderer.tokenization_params)
    assert items == [second]
    assert engine_input is renderer.engine_inputs[-1]


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
