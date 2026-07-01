"""Shared serialization helpers for omni engine request payloads."""

from __future__ import annotations

from typing import Any

from vllm.logger import init_logger

from vllm_omni.distributed.omni_connectors.utils.serialization import (
    OmniMsgpackDecoder,
    OmniMsgpackEncoder,
)

logger = init_logger(__name__)

_encoder = OmniMsgpackEncoder()
_decoder = OmniMsgpackDecoder()


def serialize_additional_information(
    raw_info: dict[str, Any] | bytes | None,
    *,
    log_prefix: str | None = None,
) -> bytes | None:
    """Serialize omni request metadata for EngineCore transport."""
    if raw_info is None:
        return None
    if isinstance(raw_info, bytes):
        return raw_info
    return _encoder.encode(raw_info)


def deserialize_additional_information(
    payload: dict | bytes | None,
) -> dict:
    """Deserialize an *additional_information* payload into a plain dict."""
    if payload is None:
        return {}
    if isinstance(payload, dict):
        return payload
    return _decoder.decode(payload)
