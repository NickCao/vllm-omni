from __future__ import annotations

import json
import os
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
from safetensors import safe_open
from transformers.utils.hub import cached_file
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.platforms import current_platform

from vllm_omni.model_executor.models.output_templates import OmniOutput

from .tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import (
    Qwen3TTSTokenizerV2Config,
)
from .tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
    Qwen3TTSTokenizerV2Decoder,
)

logger = init_logger(__name__)


class Qwen3TTSCode2Wav(nn.Module):
    """Stage-1 code2wav model for Qwen3-TTS (GenerationModelRunner).
    Consumes frame-aligned codec tokens from input_ids and decodes waveform
    via the SpeechTokenizer decoder directly (bypassing HF wrapper overhead)."""

    input_modalities = "audio"

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        self.model_path = vllm_config.model_config.model

        self.have_multimodal_outputs = True
        self.has_preprocess = False
        self.has_postprocess = False
        self.enable_update_additional_information = True
        self.requires_raw_input_tokens = True
        self._logged_codec_stats = False

        # Load speech tokenizer config and construct the decoder as a
        # proper sub-module so it is visible to vLLM's profiling.
        cfg_path = cached_file(self.model_path, "speech_tokenizer/config.json")
        if cfg_path is None:
            raise ValueError(f"{self.model_path}/speech_tokenizer/config.json not found")
        self._speech_tokenizer_dir = os.path.dirname(cfg_path)

        with open(cfg_path) as f:
            raw_config = json.load(f)
        tok_config = Qwen3TTSTokenizerV2Config(**raw_config)
        dec_config = tok_config.decoder_config

        self._num_quantizers = int(dec_config.num_quantizers)
        self._output_sample_rate = int(tok_config.output_sample_rate)

        self.decoder = Qwen3TTSTokenizerV2Decoder._from_config(dec_config)
        self.decoder.eval()
        self._total_upsample = int(self.decoder.total_upsample)

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        # This stage ignores token embeddings. Keep a stable dummy embedding for vLLM runner.
        if input_ids.numel() == 0:
            return torch.empty((0, 1), device=input_ids.device, dtype=torch.float32)
        return torch.zeros((input_ids.shape[0], 1), device=input_ids.device, dtype=torch.float32)

    def compute_logits(self, hidden_states: torch.Tensor | OmniOutput, sampling_metadata: Any = None) -> None:
        return None

    def _split_request_ids(self, ids: torch.Tensor, seq_token_counts: list[int] | None = None) -> list[torch.Tensor]:
        """Split concatenated input_ids into per-request segments.

        Uses seq_token_counts (injected by the runner via model_kwargs) when
        available, falling back to forward-context ubatch_slices when
        micro-batching is active. Returns [ids] for single-request batches.
        """
        if seq_token_counts is not None and len(seq_token_counts) > 1:
            boundaries = [0]
            for count in seq_token_counts:
                boundaries.append(boundaries[-1] + count)
            n = ids.numel()
            return [ids[boundaries[i] : min(boundaries[i + 1], n)] for i in range(len(seq_token_counts))]
        if is_forward_context_available():
            slices = get_forward_context().ubatch_slices
            if slices is not None and len(slices) > 1 and not any(hasattr(s, "token_slice") for s in slices):
                boundaries = [0]
                for s in slices:
                    boundaries.append(boundaries[-1] + s)
                return [ids[boundaries[i] : boundaries[i + 1]] for i in range(len(boundaries) - 1)]
        return [ids]

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        runtime_additional_information: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> OmniOutput:
        """Decode codec codes into audio waveform.

        input_ids layout per request: [codec_context_frames, *flat_codes]
        where flat_codes is codebook-major [q*F].

        Bypasses the HF Qwen3TTSTokenizer.decode() wrapper and calls the
        decoder.chunked_decode() directly to avoid GPU->CPU->GPU round-trips.
        Length management is done here instead of relying on HF's padding=-1
        sentinel logic.
        """
        decoder = self.decoder
        q = self._num_quantizers
        upsample = self._total_upsample
        sr_val = self._output_sample_rate
        sr_tensor = torch.tensor(sr_val, dtype=torch.int32)
        empty = torch.zeros((0,), dtype=torch.float32)

        if input_ids is None or input_ids.numel() == 0:
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={"model_outputs": [empty], "sr": [sr_tensor]},
            )

        ids = input_ids.reshape(-1).to(dtype=torch.long)
        request_ids_list = self._split_request_ids(ids, kwargs.get("seq_token_counts"))

        parsed: list[tuple[int, int]] = []
        valid_codes_qf: list[torch.Tensor] = []
        valid_indices: list[int] = []
        left_context_size = [0] * len(request_ids_list)
        if runtime_additional_information is not None:
            for i, info in enumerate(runtime_additional_information):
                if i >= len(left_context_size):
                    break
                if "left_context_size" in info:
                    # left_context_size may come through serialization as an int, [int], or tensor([int]).
                    value = info["left_context_size"]
                    if isinstance(value, list):
                        value = value[0] if value else 0
                    if isinstance(value, torch.Tensor):
                        value = value.reshape(-1)[0].item() if value.numel() > 0 else 0
                    left_context_size[i] = int(value)
        for i, req_ids in enumerate(request_ids_list):
            if req_ids.numel() < 1:
                parsed.append((0, 0))
                continue
            ctx_frames = left_context_size[i]
            flat = req_ids
            n = flat.numel()
            if n == 0 or n % q != 0:
                if n > 0:
                    logger.warning(
                        "Code2Wav input_ids length %d not divisible by num_quantizers %d; skipping malformed request.",
                        n,
                        q,
                    )
                parsed.append((0, 0))
                continue
            frames = n // q
            # [q*F] -> [Q, F] for direct decoder call (decoder expects [B, Q, F])
            codes_qf = flat.reshape(q, frames)
            parsed.append((ctx_frames, frames))
            valid_codes_qf.append(codes_qf)
            valid_indices.append(i)

        num_req = len(request_ids_list)
        if not valid_codes_qf:
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={
                    "model_outputs": [empty] * num_req,
                    "sr": [sr_tensor] * num_req,
                },
            )

        if not self._logged_codec_stats:
            self._logged_codec_stats = True
            try:
                c = valid_codes_qf[0]
                logger.info(
                    "Code2Wav codec: frames=%d q=%d uniq=%d range=[%d,%d] batch=%d",
                    c.shape[1],
                    q,
                    int(torch.unique(c).numel()),
                    int(c.min().item()),
                    int(c.max().item()),
                    len(valid_codes_qf),
                )
            except Exception:
                pass

        # Decode directly via decoder.chunked_decode(), staying entirely on GPU.
        # Each request decoded individually with CUDA graph replay at bs=1.
        wav_tensors: list[torch.Tensor] = []
        for codes_qf in valid_codes_qf:
            codes_bqf = codes_qf.unsqueeze(0)  # [1, Q, F]
            wav = decoder.chunked_decode(codes_bqf)  # [1, 1, wav_len]
            wav_tensors.append(wav.squeeze(0).squeeze(0))  # [wav_len]

        audios: list[torch.Tensor] = [empty] * num_req
        srs = [sr_tensor] * num_req

        for j, idx in enumerate(valid_indices):
            ctx_frames, actual_frames = parsed[idx]
            wav = wav_tensors[j]
            # Drop the ref_code prefix from the decoded waveform, keeping only newly generated audio.
            if ctx_frames <= 0:
                expected_len = actual_frames * upsample
                if wav.shape[0] > expected_len:
                    wav = wav[:expected_len]
            else:
                cut = int(ctx_frames / max(actual_frames, 1) * wav.shape[0])
                if cut >= wav.shape[0]:
                    logger.warning(
                        "Context trim %d >= decoded length %d; returning empty audio.",
                        cut,
                        wav.shape[0],
                    )
                    continue
                wav = wav[cut:]
            if wav.shape[0] > 0:
                audios[idx] = wav.to(dtype=torch.float32).reshape(-1)

        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={"model_outputs": audios, "sr": srs},
        )

    def make_omni_output(self, model_outputs: torch.Tensor | OmniOutput, **kwargs: Any) -> OmniOutput:
        if isinstance(model_outputs, OmniOutput):
            return model_outputs

        if not (isinstance(model_outputs, tuple) and len(model_outputs) == 2):
            raise TypeError(f"Qwen3TTSCode2Wav expected (audio_tensor, sr) outputs, got {type(model_outputs)}")

        audio_tensor, sr = model_outputs
        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={
                "model_outputs": audio_tensor,
                "sr": sr,
            },
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Load speech tokenizer decoder weights from safetensors.
        # Only decoder.* keys are needed; encoder.* keys are skipped.
        safetensors_path = os.path.join(self._speech_tokenizer_dir, "model.safetensors")
        if not os.path.isfile(safetensors_path):
            raise FileNotFoundError(
                f"Speech tokenizer weights not found at {safetensors_path}. "
                "All Qwen3-TTS checkpoints store decoder weights as a single "
                "speech_tokenizer/model.safetensors file."
            )
        device = self.vllm_config.device_config.device

        # The safetensors file contains both encoder and decoder weights.
        # Use safe_open to load only decoder keys, avoiding ~325 MiB of
        # encoder weights ever touching GPU memory.
        state_dict = {}
        with safe_open(safetensors_path, framework="pt", device=str(device)) as f:
            for key in f.keys():
                if key.startswith("decoder."):
                    state_dict[key.removeprefix("decoder.")] = f.get_tensor(key).to(dtype=torch.float32)
        self.decoder.load_state_dict(state_dict, strict=True)
        loaded_names = {"decoder." + k for k in state_dict}
        del state_dict
        logger.info("Code2Wav decoder weights loaded from %s", safetensors_path)

        # Precompute SnakeBeta exp caches (benefits both Triton and eager).
        if hasattr(self.decoder, "precompute_snake_caches"):
            self.decoder.precompute_snake_caches()

        # Enable CUDA graph capture with memory budget enforcement.
        if hasattr(self.decoder, "enable_cudagraph") and device.type == "cuda":
            try:
                chunk_frames = 0
                left_frames = 0

                model_cfg = getattr(self.vllm_config, "model_config", None)
                connector_cfg = getattr(model_cfg, "stage_connector_config", None)
                extra_cfg = (
                    connector_cfg.get("extra", connector_cfg)
                    if isinstance(connector_cfg, dict)
                    else getattr(connector_cfg, "extra", None)
                )
                if isinstance(extra_cfg, dict):
                    chunk_frames = int(extra_cfg.get("codec_chunk_frames") or 0)
                    left_frames = int(extra_cfg.get("codec_left_context_frames") or 0)

                # Compute minimum free GPU memory to preserve for other
                # stages sharing this device.  CUDA graph capture stops
                # when free memory drops below this threshold.
                min_free_bytes: int | None = None
                gpu_util = self.vllm_config.cache_config.gpu_memory_utilization
                if gpu_util < 1:
                    free, total_gpu = current_platform.mem_get_info(device)
                    budget = int(total_gpu * gpu_util)
                    min_free_bytes = total_gpu - budget
                    logger.info(
                        "Code2Wav memory budget: %.0f MiB, "
                        "min free to preserve: %.0f MiB "
                        "(gpu_util=%.2f, free=%.0f MiB, total=%.0f MiB)",
                        budget / 1024**2,
                        min_free_bytes / 1024**2,
                        gpu_util,
                        free / 1024**2,
                        total_gpu / 1024**2,
                    )

                self.decoder.enable_cudagraph(
                    device=device,
                    codec_chunk_frames=chunk_frames,
                    codec_left_context_frames=left_frames,
                    min_free_bytes=min_free_bytes,
                )
                logger.info("Code2Wav decoder CUDA Graph enabled")
            except Exception:
                logger.warning(
                    "Failed to enable CUDA Graph for Code2Wav decoder",
                    exc_info=True,
                )

        return loaded_names
