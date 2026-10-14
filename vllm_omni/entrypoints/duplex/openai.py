# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from functools import cached_property
from typing import Any

from fastapi import WebSocket
from vllm.entrypoints.speech_to_text.transcription.serving import OpenAIServingTranscription
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model_cls
from vllm.model_executor.models import ModelRegistry, supports_transcription
from vllm.utils import random_uuid

from vllm_omni.engine.duplex.turn_detection import SileroVADBackendProvider
from vllm_omni.entrypoints.duplex.warmup import wait_for_duplex_warmup

logger = init_logger(__name__)

# Keep OpenAI connection imports in the handlers: openai/__init__.py imports
# api_server, which imports this module.


async def reject_realtime_websocket(websocket: WebSocket, message: str) -> None:
    await websocket.accept()
    await websocket.send_json(
        {
            "event_id": f"evt_{random_uuid()}",
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "code": "unsupported_model",
                "message": message,
                "param": "model",
                "event_id": None,
            },
        }
    )
    await websocket.close(code=1008)


async def _reject_unavailable_realtime_websocket(websocket: WebSocket) -> None:
    await websocket.accept()
    await websocket.send_json({"type": "error", "error": "Realtime API is not available", "code": "unsupported"})
    await websocket.close()


class OpenAIRealtimeHandler:
    """App-scoped OpenAI Realtime handler with one shared VAD backend provider."""

    def __init__(
        self,
        *,
        model_path: str | None = None,
        engine_client: Any = None,
    ) -> None:
        self._vad_backend_provider = SileroVADBackendProvider(model_path=model_path)
        self._engine_client: Any = engine_client
        self._openai_serving_models: Any = None

    @classmethod
    def from_engine_client(cls, engine_client: Any) -> OpenAIRealtimeHandler:
        backend_engine = getattr(engine_client, "engine", engine_client)
        duplex_session_config = getattr(backend_engine, "duplex_session_config", None)
        return cls(
            model_path=getattr(duplex_session_config, "server_vad_model_path", None),
            engine_client=engine_client,
        )

    @cached_property
    def _transcription_serving(self) -> OpenAIServingTranscription | None:
        """Reuse the standard transcription serving stack for Realtime input transcription."""
        engine_client = self._engine_client
        model_config = getattr(engine_client, "model_config", None)
        if engine_client is None or model_config is None or self._openai_serving_models is None:
            return None
        try:
            model_cls = get_model_cls(model_config)
        except Exception:
            logger.debug("Realtime input transcription is unavailable for the loaded model", exc_info=True)
            return None

        if not supports_transcription(model_cls):
            model_cls = _resolve_comprehension_transcription_model_cls(engine_client, model_config)
            if model_cls is None or not supports_transcription(model_cls):
                return None

        try:
            transcription_engine_client = (
                _TextOutputEngineClient(engine_client)
                if getattr(engine_client, "stage_configs", None)
                else engine_client
            )
            return _RealtimeTranscriptionServing(
                transcription_engine_client,
                self._openai_serving_models,
                model_cls=model_cls,
            )
        except Exception:
            logger.debug("Realtime input transcription is unavailable for the loaded model", exc_info=True)
            return None

    async def handle_websocket(self, websocket: WebSocket) -> None:
        state = websocket.app.state
        if (
            getattr(state, "diffusion_engine", None) is not None
            or getattr(state, "engine_client", None) is None
            or getattr(state, "openai_serving_chat", None) is None
        ):
            await _reject_unavailable_realtime_websocket(websocket)
            return

        self._openai_serving_models = state.openai_serving_models
        model_name = state.openai_serving_models.base_model_paths[0].name
        # Some OpenAI-compatible clients always send ``model=`` even when the
        # caller leaves model selection to this single-model Omni server.
        requested_model = websocket.query_params.get("model")
        if requested_model and requested_model != model_name:
            await reject_realtime_websocket(websocket, f"Model '{requested_model}' is not available")
            return

        await wait_for_duplex_warmup(websocket)

        from vllm_omni.entrypoints.openai.realtime.connection import OpenAIFullDuplexConnection

        connection = OpenAIFullDuplexConnection(
            websocket=websocket,
            engine=state.engine_client,
            model_name=model_name,
            chat_handler=state.openai_serving_chat,
            tool_call_parser=getattr(state.args, "tool_call_parser", None),
            enable_auto_tool_choice=getattr(state.args, "enable_auto_tool_choice", False),
            vad_backend_provider=self._vad_backend_provider,
            transcription_serving=self._transcription_serving,
        )
        await connection.handle_connection()


class _TextOutputEngineClient:
    """Force staged transcription requests to stop at a text-output stage."""

    def __init__(self, engine_client: Any) -> None:
        self._engine_client = engine_client

    def __getattr__(self, name: str) -> Any:
        return getattr(self._engine_client, name)

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("output_modalities", ["text"])
        return self._engine_client.generate(*args, **kwargs)


class _RealtimeTranscriptionServing(OpenAIServingTranscription):
    """Use a stage-declared transcription implementation with standard serving."""

    def __init__(self, engine_client: Any, models: Any, *, model_cls: type[Any]) -> None:
        self._transcription_model_cls = model_cls
        super().__init__(engine_client, models, request_logger=None)

    @property
    def model_cls(self) -> type[Any]:
        return self._transcription_model_cls


def _resolve_declared_model_cls(model_config: Any, model_arch: str) -> type[Any]:
    registry = getattr(model_config, "registry", ModelRegistry)
    return registry.resolve_model_cls([model_arch], model_config)[0]


def _resolve_comprehension_transcription_model_cls(
    engine_client: Any,
    model_config: Any,
) -> type[Any] | None:
    backend_engine = getattr(engine_client, "engine", engine_client)
    stage_configs = getattr(backend_engine, "stage_configs", ())
    stage_vllm_configs = getattr(backend_engine, "stage_vllm_configs", ())
    for stage_config, stage_vllm_config in zip(stage_configs, stage_vllm_configs):
        if stage_vllm_config is None or not getattr(stage_config, "is_comprehension", False):
            continue
        model_arch = getattr(stage_config, "transcription_model_arch", None)
        if model_arch is None:
            return None
        stage_model_config = getattr(stage_vllm_config, "model_config", model_config)
        return _resolve_declared_model_cls(stage_model_config, model_arch)
    return None
