# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from functools import cached_property
from typing import Any

from fastapi import WebSocket
from vllm.entrypoints.speech_to_text.transcription.serving import OpenAIServingTranscription
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model_cls
from vllm.model_executor.models import supports_transcription
from vllm.utils import random_uuid

from vllm_omni.engine.duplex.turn_detection import SileroVADBackendProvider, SmartTurnBackendProvider
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
    """App-scoped OpenAI Realtime handler with shared VAD backend providers."""

    def __init__(
        self,
        *,
        server_model_path: str | None = None,
        semantic_model_path: str | None = None,
        engine_client: Any = None,
    ) -> None:
        self._vad_backend_provider = SileroVADBackendProvider(model_path=server_model_path)
        self._semantic_vad_backend_provider = SmartTurnBackendProvider(model_path=semantic_model_path)
        self._engine_client: Any = engine_client
        self._openai_serving_models: Any = None

    @classmethod
    def from_engine_client(cls, engine_client: Any) -> OpenAIRealtimeHandler:
        backend_engine = getattr(engine_client, "engine", engine_client)
        duplex_session_config = getattr(backend_engine, "duplex_session_config", None)
        return cls(
            server_model_path=getattr(duplex_session_config, "server_vad_model_path", None),
            semantic_model_path=getattr(duplex_session_config, "semantic_vad_model_path", None),
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
            stage_model_config = _resolve_comprehension_transcription_model_config(engine_client)
            if stage_model_config is not None:
                return OpenAIServingTranscription(
                    _TextOutputEngineClient(engine_client, stage_model_config),
                    self._openai_serving_models,
                    request_logger=None,
                )

            if not _model_config_supports_transcription(model_config):
                return None

            transcription_engine_client = (
                _TextOutputEngineClient(engine_client)
                if getattr(engine_client, "stage_configs", None)
                else engine_client
            )
            return OpenAIServingTranscription(
                transcription_engine_client,
                self._openai_serving_models,
                request_logger=None,
            )
        except Exception:
            logger.warning("Realtime input transcription is unavailable for the loaded model", exc_info=True)
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
            vad_backend_provider=self._vad_backend_provider,
            semantic_vad_backend_provider=self._semantic_vad_backend_provider,
            transcription_serving=self._transcription_serving,
        )
        await connection.handle_connection()


class _TextOutputEngineClient:
    """Force staged transcription requests to stop at a text-output stage."""

    def __init__(self, engine_client: Any, model_config: Any = None) -> None:
        self._engine_client = engine_client
        self._model_config = model_config

    @property
    def model_config(self) -> Any:
        if self._model_config is not None:
            return self._model_config
        return self._engine_client.model_config

    def __getattr__(self, name: str) -> Any:
        return getattr(self._engine_client, name)

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("output_modalities", ["text"])
        return self._engine_client.generate(*args, **kwargs)


def _model_config_supports_transcription(model_config: Any) -> bool:
    try:
        return supports_transcription(get_model_cls(model_config))
    except Exception:
        logger.debug("Could not inspect a model config for transcription support", exc_info=True)
        return False


def _resolve_comprehension_transcription_model_config(engine_client: Any) -> Any | None:
    backend_engine = getattr(engine_client, "engine", engine_client)
    stage_configs = getattr(backend_engine, "stage_configs", ())
    stage_vllm_configs = getattr(backend_engine, "stage_vllm_configs", ())
    for stage_config, stage_vllm_config in zip(stage_configs, stage_vllm_configs):
        if stage_vllm_config is None or not getattr(stage_config, "is_comprehension", False):
            continue
        model_arch = getattr(stage_config, "transcription_model_arch", None)
        if model_arch is None:
            continue
        stage_model_config = getattr(stage_vllm_config, "model_config", None)
        if stage_model_config is None:
            continue
        hf_config_name = getattr(stage_config, "hf_config_name", None)
        hf_config = (
            getattr(stage_model_config.hf_config, hf_config_name, None)
            if hf_config_name is not None
            else stage_model_config.hf_config
        )
        if hf_config is None:
            continue
        transcription_model_config = stage_vllm_config.with_hf_config(
            hf_config,
            architectures=[model_arch],
        ).model_config
        if _model_config_supports_transcription(transcription_model_config):
            return transcription_model_config
        logger.warning(
            "Realtime transcription architecture %s declared by comprehension stage %s does not support transcription",
            model_arch,
            getattr(stage_config, "stage_id", "unknown"),
        )
    return None
