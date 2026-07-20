# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
OpenAI-compatible base models for /v1/audio/speech, generated from the
OpenAI OpenAPI specification (openai/openai-openapi on GitHub).

To regenerate:
    python scripts/generate_openai_speech_models.py

These models define the OpenAI-standard fields only.  vllm-omni extensions
(voice cloning, Qwen3-TTS params, etc.) are layered on via subclassing in
``audio.py``.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# CreateSpeechRequest  (generated from ``CreateSpeechRequest`` schema)
# ---------------------------------------------------------------------------


class CreateSpeechRequest(BaseModel):
    """OpenAI ``/v1/audio/speech`` request body.

    Ref: https://platform.openai.com/docs/api-reference/audio/createSpeech
    """

    input: str = Field(
        ...,
        description="The text to generate audio for.",
        max_length=4096,
    )
    model: str = Field(
        ...,
        description="TTS model identifier.",
    )
    voice: str = Field(
        ...,
        description="The voice to use when generating the audio.",
    )
    instructions: str | None = Field(
        default=None,
        description="Control the voice of your generated audio with additional instructions.",
        max_length=4096,
    )
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = Field(
        default="mp3",
        description="The format to audio in.",
    )
    speed: float = Field(
        default=1.0,
        ge=0.25,
        le=4.0,
        description="The speed of the generated audio.",
    )
    stream_format: Literal["sse", "audio"] | None = Field(
        default="audio",
        description="The format to stream the audio in.",
    )


# ---------------------------------------------------------------------------
# SSE streaming events  (generated from Speech* event schemas)
# ---------------------------------------------------------------------------


class SpeechUsage(BaseModel):
    """Token usage reported on ``speech.audio.done``."""

    input_tokens: int = Field(description="Number of input tokens in the prompt.")
    output_tokens: int = Field(description="Number of output tokens generated.")
    total_tokens: int = Field(description="Total number of tokens used (input + output).")


class SpeechAudioDeltaEvent(BaseModel):
    """``speech.audio.delta`` SSE event -- one chunk of base64-encoded audio."""

    type: Literal["speech.audio.delta"] = "speech.audio.delta"
    audio: str = Field(description="Base64-encoded audio data chunk.")
    # vllm-omni extension: include the response format so clients know the
    # encoding without out-of-band state.
    response_format: str | None = Field(
        default=None,
        description="Audio encoding of this chunk (vllm-omni extension).",
    )


class SpeechAudioDoneEvent(BaseModel):
    """``speech.audio.done`` SSE event -- signals end of audio stream."""

    type: Literal["speech.audio.done"] = "speech.audio.done"
    usage: SpeechUsage | dict[str, Any] | None = Field(
        default=None,
        description="Token usage statistics for the request.",
    )


class SpeechAudioErrorEvent(BaseModel):
    """``speech.audio.error`` SSE event -- not in OpenAI spec but emitted
    by vllm-omni on stream failure."""

    type: Literal["speech.audio.error"] = "speech.audio.error"
    error: dict[str, Any] = Field(
        description="OpenAI-compatible error payload.",
    )
