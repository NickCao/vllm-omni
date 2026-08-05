# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Diffusers backend adapter for vLLM-Omni.

Provides a black-box wrapper around any 🤗 Diffusers pipeline, enabling
vLLM-Omni to directly serve Diffusers models with near-zero per-model code.

The adapter delegates full pipeline execution to diffusers' ``__call__()``.
It does NOT support:
- CFG parallel (diffusers handles CFG via guidance_scale internally)
- Sequence parallel (requires model-specific attention surgery)
- TeaCache / Cache-DiT (requires hooking into transformer blocks)
- Step-wise execution (continuous batching)
"""

import inspect
import logging
from typing import Any

import torch
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from torch import nn

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.models.diffusers_adapter.call_delegation_mixin import DiffusersCallDelegationMixin
from vllm_omni.diffusion.models.diffusers_adapter.pipeline_utils import BasePipelineUtils, get_pipeline_utils
from vllm_omni.diffusion.models.diffusers_adapter.quantization_utils import (
    apply_diffusers_quantization_config,
    convert_diffusers_quantization_config,
    ensure_supported_diffusers_quantization,
)
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import DiffusionPipelineProfilerMixin

logger = logging.getLogger(__name__)

_DIFFUSERS_CONFIG_LOAD_KWARGS = {
    "cache_dir",
    "dduf_entries",
    "force_download",
    "local_dir",
    "local_dir_use_symlinks",
    "local_files_only",
    "proxies",
    "revision",
    "subfolder",
    "token",
    "user_agent",
}


class DiffusersAdapterPipeline(nn.Module, DiffusionPipelineProfilerMixin, DiffusersCallDelegationMixin):
    """Black-box adapter that delegates full pipeline execution to a diffusers pipeline.

    Usage::

        adapter = DiffusersAdapterPipeline(od_config=od_config)
        adapter.load_weights()  # calls DiffusionPipeline.from_pretrained()
        output = adapter.forward(req)

    Step-wise execution is explicitly rejected — diffusers encapsulates the
    full denoising loop internally. Use native pipelines for continuous
    batching mode.
    """

    supports_request_batch = False
    supports_step_execution: bool = False

    def __init__(self, *, od_config: OmniDiffusionConfig, device: torch.device | None = None):
        super().__init__()
        self._pipeline: DiffusionPipeline
        self._accept_call_kwargs: set[str] | None = None  # None to accept all kwargs
        self.od_config = od_config
        self.device = device
        self._capabilities: dict[str, Any] = {}
        self._pipeline_utils: BasePipelineUtils = BasePipelineUtils()
        self._raise_unsupported_features()

        self.setup_diffusion_pipeline_profiler(
            enable_diffusion_pipeline_profiler=od_config.enable_diffusion_pipeline_profiler,
            profiler_targets=["forward"],
        )
        if od_config.enable_diffusion_pipeline_profiler:
            logger.info("Profiling enabled for DiffusersAdapterPipeline. Only 'forward' is supported.")

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def load_weights(self) -> None:
        """Load the diffusers pipeline via ``DiffusionPipeline.from_pretrained()``."""

        model_id = self.od_config.model
        dtype = self.od_config.dtype

        load_kwargs = {
            "torch_dtype": dtype,
            **self.od_config.diffusers_load_kwargs,
        }
        convert_diffusers_quantization_config(load_kwargs)

        pipeline_class = self.od_config.diffusers_pipeline_cls
        pipeline_class_name = pipeline_class.__name__ if pipeline_class is not None else None
        self._pipeline_utils = get_pipeline_utils(pipeline_class_name)
        self._pipeline_utils.update_load_kwargs(self.od_config, load_kwargs)
        component_names = (
            self._load_diffusers_component_names(model_id, load_kwargs)
            if self.od_config.quantization_config is not None and "quantization_config" not in load_kwargs
            else {}
        )
        apply_diffusers_quantization_config(self.od_config, load_kwargs, component_names)
        logger.debug(f"Loading diffusers pipeline with kwargs: {load_kwargs}")

        self._pipeline = DiffusionPipeline.from_pretrained(model_id, **load_kwargs)
        self._pipeline_utils.apply_post_load_updates(self._pipeline, self.od_config)

        self._pipeline.to(self.device)

        # Cache __call__kwargs signature introspection for later input validation
        self._accept_call_kwargs = set(inspect.signature(self._pipeline.__call__).parameters.keys())

        # CPU offloading
        if self.od_config.enable_layerwise_offload:
            self._pipeline.enable_sequential_cpu_offload()
        elif self.od_config.enable_cpu_offload:
            self._pipeline.enable_model_cpu_offload()

        # VAE slicing and tiling: try-catch because not all models have VAE
        if self.od_config.vae_use_slicing:
            try:
                self._pipeline.enable_vae_slicing()
            except Exception as e:
                logger.warning(
                    f"Failed to enable VAE slicing for diffusers pipeline {self._pipeline.__class__.__name__}: {e}"
                )
        if self.od_config.vae_use_tiling:
            try:
                self._pipeline.enable_vae_tiling()
            except Exception as e:
                logger.warning(
                    f"Failed to enable VAE tiling for diffusers pipeline {self._pipeline.__class__.__name__}: {e}"
                )

        # Attention backend
        self._set_attention_backend()

    # ------------------------------------------------------------------
    # Step-wise execution — explicitly rejected
    # ------------------------------------------------------------------

    def prepare_encode(self, **_: Any) -> Any:
        raise NotImplementedError(
            "Step-wise execution is not yet supported with the diffusers backend. "
            "Use a native pipeline for continuous batching mode."
        )

    def denoise_step(self, **_: Any) -> torch.Tensor | None:
        raise NotImplementedError(
            "Step-wise execution is not yet supported with the diffusers backend. "
            "Use a native pipeline for continuous batching mode."
        )

    def step_scheduler(self, **_: Any) -> None:
        raise NotImplementedError(
            "Step-wise execution is not yet supported with the diffusers backend. "
            "Use a native pipeline for continuous batching mode."
        )

    def post_decode(self, **_: Any) -> Any:
        raise NotImplementedError(
            "Step-wise execution is not yet supported with the diffusers backend. "
            "Use a native pipeline for continuous batching mode."
        )

    # ------------------------------------------------------------------
    # Validation guards
    # ------------------------------------------------------------------

    def _raise_unsupported_features(self) -> None:
        """Raise an error for incompatible feature switches."""
        pc = self.od_config.parallel_config
        if pc.cfg_parallel_size > 1:
            raise NotImplementedError(
                "CFG parallel is not supported with the diffusers backend. "
                "Diffusers handles CFG internally via guidance_scale."
            )
        if pc.sequence_parallel_size is not None and pc.sequence_parallel_size > 1:
            raise NotImplementedError(
                "Sequence parallel is not supported with the diffusers backend. "
                "It requires model-specific attention surgery."
            )
        if self.od_config.cache_backend not in ("none", None):
            raise NotImplementedError(
                f"Cache backend '{self.od_config.cache_backend}' is not supported "
                "with the diffusers backend. TeaCache/Cache-DiT require hooking "
                "into individual transformer blocks."
            )
        if self.od_config.enforce_eager:
            raise NotImplementedError(
                "Eager execution is not supported with the diffusers backend. "
                "Use a native pipeline for continuous batching mode."
            )
        if (
            self.od_config.quantization_config is not None
            and "quantization_config" not in self.od_config.diffusers_load_kwargs
        ):
            ensure_supported_diffusers_quantization(self.od_config.quantization_config)
            if self.od_config.diffusers_load_kwargs.get("dduf_file"):
                raise NotImplementedError(
                    "Diffusers backend quantization conversion does not support "
                    "diffusers_load_kwargs.dduf_file yet. The preflight component "
                    "discovery would need to mirror Diffusers' DDUF config loading. "
                    "Use diffusers_load_kwargs.quantization_config for a native "
                    "Diffusers quantization config, or omit dduf_file for vLLM-Omni "
                    "quantization conversion."
                )

    def _load_diffusers_component_names(
        self,
        model_id: str,
        load_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        config_load_kwargs = {k: load_kwargs[k] for k in _DIFFUSERS_CONFIG_LOAD_KWARGS if k in load_kwargs}
        pipeline_config = DiffusionPipeline.load_config(model_id, **config_load_kwargs)
        return {
            name: value
            for name, value in pipeline_config.items()
            if (
                isinstance(value, list)
                and len(value) > 0
                and value[0] is not None
                and (name not in load_kwargs or load_kwargs[name] is not None)
            )
        }
