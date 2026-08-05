# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Diffusers backend adapter for vLLM-Omni.

Loads any model with modular block definitions in diffusers as a
``ModularPipeline`` and drives the denoising loop one step at a time,
enabling step-wise execution (continuous batching) and component-level
CPU offload via ``SupportsComponentDiscovery``.

Does not support CFG parallel, sequence parallel, or
TeaCache / Cache-DiT.
"""

from __future__ import annotations

import copy
import inspect
import logging
import re
from typing import TYPE_CHECKING, Any, ClassVar, cast

import torch
from diffusers import ModelMixin
from diffusers.modular_pipelines import (
    LoopSequentialPipelineBlocks,
    ModularPipeline,
    PipelineState,
)
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from torch import nn
from transformers import PreTrainedModel

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.models.diffusers_adapter.pipeline_utils import BasePipelineUtils, get_pipeline_utils
from vllm_omni.diffusion.models.diffusers_adapter.quantization_utils import (
    apply_diffusers_quantization_config,
    convert_diffusers_quantization_config,
    ensure_supported_diffusers_quantization,
)
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import DiffusionPipelineProfilerMixin
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.inputs.data import OmniPromptType, OmniTextPrompt
from vllm_omni.platforms import current_omni_platform

if TYPE_CHECKING:
    from collections.abc import Sequence

    from diffusers.modular_pipelines import BlockState

    from vllm_omni.diffusion.worker.input_batch import InputBatch
    from vllm_omni.diffusion.worker.utils import StepRequestState

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


class DiffusersAdapterPipeline(nn.Module, DiffusionPipelineProfilerMixin):
    """Adapter that wraps a ``ModularPipeline`` for vLLM-Omni serving.

    Loads the model via diffusers' modular-pipeline framework, partitions
    its blocks into pre-denoise / denoise-loop / post-denoise stages, and
    implements the ``SupportsStepExecution`` protocol by calling
    ``loop_step()`` one iteration at a time.

    Component roles (DiT, encoder, VAE) are explicitly discovered and
    declared via ``SupportsComponentDiscovery`` for offload support.
    """

    supports_request_batch = False
    supports_step_execution: bool = True

    _dit_modules: ClassVar[list[str]] = []
    _encoder_modules: ClassVar[list[str]] = []
    _vae_modules: ClassVar[list[str]] = []
    _resident_modules: ClassVar[list[str]] = []

    def __init__(self, *, od_config: OmniDiffusionConfig, device: torch.device | None = None):
        super().__init__()
        self._pipeline: ModularPipeline
        self._accept_call_kwargs: set[str] | None = None  # None to accept all kwargs
        self.od_config = od_config
        self.device = device
        self._capabilities: dict[str, Any] = {}
        self._pipeline_utils: BasePipelineUtils = BasePipelineUtils()
        self._raise_unsupported_features()

        self._pre_denoise_block_names: list[str] = []
        self._denoise_prep_blocks: list[tuple[str, Any]] = []
        self._denoise_block: LoopSequentialPipelineBlocks | None = None
        self._denoise_post_blocks: list[tuple[str, Any]] = []
        self._post_denoise_block_names: list[str] = []

        self.setup_diffusion_pipeline_profiler(
            enable_diffusion_pipeline_profiler=od_config.enable_diffusion_pipeline_profiler,
            profiler_targets=["forward"],
        )

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def load_weights(self) -> None:
        """Load the model as a ``ModularPipeline`` and partition its blocks."""
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

        pipe = ModularPipeline.from_pretrained(model_id, **load_kwargs)
        pipe.load_components(**load_kwargs)
        self._pipeline_utils.apply_post_load_updates(pipe, self.od_config)
        pipe.to(self.device)

        self._pipeline = pipe
        self._partition_blocks(pipe)
        self._discover_components(pipe)

        logger.info(
            "Loaded modular pipeline %s with step-wise execution enabled.",
            pipe.__class__.__name__,
        )

        # Cache __call__kwargs signature introspection for later input validation
        self._accept_call_kwargs = set(inspect.signature(pipe.__call__).parameters.keys())

        # CPU offloading is handled by vllm-omni's offload framework
        # (sequential_backend / distributed_layerwise_backend) using the
        # component roles declared in _dit_modules / _encoder_modules /
        # _vae_modules, not by diffusers' own offload methods.

        # VAE slicing and tiling: try-catch because not all models have VAE
        if self.od_config.vae_use_slicing:
            try:
                pipe.enable_vae_slicing()
            except Exception as e:
                logger.warning("Failed to enable VAE slicing: %s", e)
        if self.od_config.vae_use_tiling:
            try:
                pipe.enable_vae_tiling()
            except Exception as e:
                logger.warning("Failed to enable VAE tiling: %s", e)

        # Attention backend
        self._set_attention_backend()

    def _partition_blocks(self, pipe: ModularPipeline) -> None:
        """Partition blocks into stages for step-wise execution.

        Splits at two levels:
        1. Top-level: blocks before/after ``denoise`` (text_encoder, decode, etc.)
        2. Inside ``denoise``: resolves the default workflow via
           ``get_execution_blocks()``, then splits around the
           ``LoopSequentialPipelineBlocks`` (the actual denoise loop).
        """
        blocks = pipe.blocks
        sub_blocks = blocks.sub_blocks

        if "denoise" not in sub_blocks:
            raise ValueError(
                f"ModularPipeline {pipe.__class__.__name__} has no 'denoise' sub-block. "
                f"Available sub-blocks: {list(sub_blocks.keys())}"
            )

        # Resolve the denoise ConditionalPipelineBlocks to a flat sequence
        denoise_top = sub_blocks["denoise"]
        resolved = denoise_top.get_execution_blocks()
        resolved_subs = resolved.sub_blocks

        loop_name = None
        for name, block in resolved_subs.items():
            if isinstance(block, LoopSequentialPipelineBlocks):
                loop_name = name
                break

        if loop_name is None:
            raise TypeError(
                f"No LoopSequentialPipelineBlocks found inside 'denoise' block. "
                f"Resolved sub-blocks: {list(resolved_subs.keys())}"
            )

        # Top-level split
        pre_top, post_top = [], []
        before_denoise = True
        for name in sub_blocks:
            if name == "denoise":
                before_denoise = False
                continue
            (pre_top if before_denoise else post_top).append(name)

        # Denoise-internal split around the loop
        denoise_prep = []
        denoise_post = []
        before_loop = True
        for name, block in resolved_subs.items():
            if name == loop_name:
                before_loop = False
                continue
            (denoise_prep if before_loop else denoise_post).append((name, block))

        self._pre_denoise_block_names = pre_top
        self._denoise_prep_blocks = denoise_prep
        self._denoise_block = resolved_subs[loop_name]
        self._denoise_post_blocks = denoise_post
        self._post_denoise_block_names = post_top
        logger.info(
            "Partitioned blocks: top_pre=%s, denoise_prep=%s, loop=%s, denoise_post=%s, top_post=%s",
            pre_top,
            [n for n, _ in denoise_prep],
            loop_name,
            [n for n, _ in denoise_post],
            post_top,
        )

    def _discover_components(self, pipe: ModularPipeline) -> None:
        """Populate ``_dit_modules``, ``_encoder_modules``, ``_vae_modules``.

        Components are categorized by their base class:
        - ``transformers.PreTrainedModel`` subclasses are encoders.
        - ``diffusers.ModelMixin`` subclasses with "Autoencoder" or "VQ"
          in the class name are VAEs.
        - All other ``diffusers.ModelMixin`` subclasses are DiT modules.
        """
        dit: list[str] = []
        enc: list[str] = []
        vae: list[str] = []

        for name in pipe._component_specs:
            component = getattr(pipe, name, None)
            if component is None or not isinstance(component, nn.Module):
                continue
            path = f"_pipeline.{name}"
            if isinstance(component, PreTrainedModel):
                enc.append(path)
            elif isinstance(component, ModelMixin):
                cls_name = type(component).__name__
                if "Autoencoder" in cls_name or "VQ" in cls_name:
                    vae.append(path)
                else:
                    dit.append(path)

        self._dit_modules = dit
        self._encoder_modules = enc
        self._vae_modules = vae
        logger.info(
            "Discovered components: dit=%s, encoder=%s, vae=%s",
            dit,
            enc,
            vae,
        )

    # ------------------------------------------------------------------
    # Step-wise execution
    # ------------------------------------------------------------------

    def prepare_encode(
        self,
        state: StepRequestState,
        **kwargs: Any,
    ) -> StepRequestState:
        """Run all blocks up to (but not including) the denoise loop.

        This includes top-level pre-denoise blocks (text_encoder,
        vae_encoder) and the denoise block's input-prep sub-blocks
        (text_inputs, prepare_latents, set_timesteps, rope_inputs).
        """
        pipe = self._pipeline
        top_blocks = pipe.blocks.sub_blocks

        prompt, negative_prompt = self._extract_step_prompts(state)
        sampling = state.sampling

        call_kwargs: dict[str, Any] = {
            "prompt": prompt,
            "height": sampling.height,
            "width": sampling.width,
            "num_inference_steps": sampling.num_inference_steps or 50,
            "guidance_scale": sampling.guidance_scale if sampling.guidance_scale_provided else None,
        }
        if negative_prompt is not None:
            call_kwargs["negative_prompt"] = negative_prompt
        if sampling.seed is not None:
            call_kwargs["generator"] = torch.Generator(
                device=sampling.generator_device,
            ).manual_seed(sampling.seed)

        call_kwargs = {k: v for k, v in call_kwargs.items() if v is not None}

        # Populate PipelineState: start with declared defaults for inputs
        # that have them (num_images_per_prompt=1, num_inference_steps=50,
        # etc.), then override with user-provided kwargs.
        pipeline_state = PipelineState()
        for param in pipe.blocks.inputs:
            if param.name is not None and param.default is not None:
                pipeline_state.set(param.name, param.default, param.kwargs_type)
        for key, value in call_kwargs.items():
            pipeline_state.set(key, value)

        # Run pre-denoise top-level blocks (text_encoder, vae_encoder, etc.)
        for block_name in self._pre_denoise_block_names:
            block = top_blocks[block_name]
            pipe, pipeline_state = block(pipe, pipeline_state)

        # Run the denoise block's input-prep sub-blocks, stopping before
        # the LoopSequentialPipelineBlocks (the actual denoise loop).
        for name, block in self._denoise_prep_blocks:
            pipe, pipeline_state = block(pipe, pipeline_state)

        denoise_block_state = self._denoise_block.get_block_state(pipeline_state)

        state.latents = denoise_block_state.latents
        state.timesteps = denoise_block_state.timesteps
        state.step_index = 0
        state.scheduler = copy.deepcopy(pipe.scheduler)

        if hasattr(denoise_block_state, "prompt_embeds"):
            state.prompt_embeds = denoise_block_state.prompt_embeds
        if hasattr(denoise_block_state, "prompt_embeds_mask"):
            state.prompt_embeds_mask = denoise_block_state.prompt_embeds_mask
        if hasattr(denoise_block_state, "negative_prompt_embeds"):
            state.negative_prompt_embeds = denoise_block_state.negative_prompt_embeds
        if hasattr(denoise_block_state, "negative_prompt_embeds_mask"):
            state.negative_prompt_embeds_mask = denoise_block_state.negative_prompt_embeds_mask
        if hasattr(denoise_block_state, "img_shapes"):
            state.img_shapes = denoise_block_state.img_shapes
        if hasattr(denoise_block_state, "txt_seq_lens"):
            state.txt_seq_lens = denoise_block_state.txt_seq_lens

        state.extra["_block_state_attrs"] = {
            k: v for k, v in denoise_block_state.__dict__.items() if k not in ("latents", "timesteps")
        }
        state.extra["_pipeline_state"] = pipeline_state

        return state

    def denoise_step(
        self,
        input_batch: InputBatch,
        *,
        states: Sequence[StepRequestState] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | None:
        """Run one iteration of the denoise loop via ``loop_step``."""
        pipe = self._pipeline
        denoise = self._denoise_block

        state = states[0] if states else None
        if state is None:
            return None

        from diffusers.modular_pipelines import BlockState

        saved_attrs = dict(state.extra.get("_block_state_attrs", {}))
        saved_attrs["latents"] = input_batch.latents
        saved_attrs["timesteps"] = state.timesteps
        saved_attrs.setdefault("num_inference_steps", len(state.timesteps))
        saved_attrs.setdefault("additional_cond_kwargs", {})
        block_state = BlockState(**saved_attrs)

        i = state.step_index
        t = state.timesteps[i]

        pipe, block_state = denoise.loop_step(pipe, block_state, i=i, t=t)

        state.extra["_updated_latents"] = block_state.latents
        return getattr(block_state, "noise_pred", block_state.latents)

    def step_scheduler(
        self,
        state: StepRequestState,
        noise_pred: torch.Tensor | None,
        **kwargs: Any,
    ) -> None:
        """Advance step state after ``denoise_step``.

        The scheduler already ran inside ``loop_step``, so we sync the
        updated latents and advance the step index.
        """
        updated_latents = state.extra.pop("_updated_latents", None)
        if updated_latents is not None:
            state.latents = updated_latents
        state.step_index += 1

    def post_decode(
        self,
        state: StepRequestState,
        **kwargs: Any,
    ) -> DiffusionOutput:
        """Run post-loop denoise blocks and post-denoise top-level blocks."""
        pipe = self._pipeline
        top_blocks = pipe.blocks.sub_blocks

        pipeline_state: PipelineState = state.extra.pop("_pipeline_state")
        self._denoise_block.set_block_state(pipeline_state, self._rebuild_block_state(state))

        # Run denoise-internal post-loop blocks (e.g. after_denoise)
        for _name, block in self._denoise_post_blocks:
            pipe, pipeline_state = block(pipe, pipeline_state)

        # Run top-level post-denoise blocks (e.g. decode)
        for block_name in self._post_denoise_block_names:
            block = top_blocks[block_name]
            pipe, pipeline_state = block(pipe, pipeline_state)

        for key in ("images", "frames", "audios"):
            val = pipeline_state.get(key)
            if val is not None:
                return DiffusionOutput(output=val)

        return DiffusionOutput(output=pipeline_state.get("latents"))

    def _rebuild_block_state(self, state: StepRequestState) -> BlockState:
        """Reconstruct a ``BlockState`` from ``StepRequestState`` for decode."""
        from diffusers.modular_pipelines import BlockState

        attrs = state.extra.get("_block_state_attrs", {})
        return BlockState(latents=state.latents, **attrs)

    def _extract_step_prompts(self, state: StepRequestState) -> tuple[str, str | None]:
        """Extract prompt and negative prompt from a StepRequestState."""
        prompt = ""
        negative_prompt = None
        if state.prompt is not None:
            if isinstance(state.prompt, str):
                prompt = state.prompt
            elif isinstance(state.prompt, dict):
                prompt = state.prompt.get("prompt", "")
                negative_prompt = state.prompt.get("negative_prompt")
        return prompt, negative_prompt

    # ------------------------------------------------------------------
    # Forward pass (black-box __call__ delegation)
    # ------------------------------------------------------------------

    def forward(self, req: DiffusionRequestBatch) -> DiffusionOutput:
        """Full delegation to ``ModularPipeline.__call__()``."""
        kwargs = self._build_call_kwargs(req)
        logger.debug("Calling diffusers pipeline with kwargs: %s", kwargs)

        with torch.inference_mode():
            output = self._pipeline(**kwargs)  # pyright: ignore[reportCallIssue]

        return self._wrap_output(output)

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

    # ------------------------------------------------------------------
    # Wrap settings, inputs, and outputs
    # ------------------------------------------------------------------

    def _set_attention_backend(self) -> None:
        """Set the attention backend.

        Roughly follow the logic in vllm_omni/diffusion/attention/backends/utils/fa.py,
        But also consider the available attention backends in diffusers.
        (See: https://huggingface.co/docs/diffusers/optimization/attention_backends)
        """
        if not hasattr(self._pipeline, "transformer"):
            logging.info("No transformer found in diffusers pipeline. Skipping attention backend setting.")
            return

        default_spec = self.od_config.diffusion_attention_config.default
        attention_backend_config = default_spec.backend if default_spec is not None else None
        attention_backend_attempts: list[str] = []
        match attention_backend_config:
            case "FLASH_ATTN" | None:
                if current_omni_platform.is_rocm():
                    attention_backend_attempts.append("aiter")
                elif current_omni_platform.is_xpu():
                    attention_backend_attempts.append("_native_xla")
                elif current_omni_platform.is_musa():
                    logger.warning(
                        "Unknown diffusers attention backend option for MUSA platform. Falling back to SDPA."
                    )
                    attention_backend_attempts.append("native")
                else:
                    attention_backend_attempts.extend(
                        [
                            "_flash_3_hub",
                            "_flash_3_varlen_hub",
                            "_flash_3",
                            "_flash_varlen_3",
                            "flash_hub",
                            "flash_varlen_hub",
                            "flash",
                            "flash_varlen",
                            "_native_flash",
                        ]
                    )
            case "SAGE_ATTN":
                attention_backend_attempts.extend(["sage_hub", "sage", "sage", "sage_varlen"])
            case "ASCEND":
                attention_backend_attempts.append("_native_npu")
            case "TORCH_SDPA":
                attention_backend_attempts.append("native")
            case _:
                logger.warning(f"Invalid attention backend: {attention_backend_config}. Falling back to SDPA.")
                attention_backend_attempts.append("native")

        attempt_errors: list[str] = []
        set_backend: str | None = None
        for backend in attention_backend_attempts:
            try:
                self._pipeline.transformer.set_attention_backend(backend)
                set_backend = backend
                break
            except Exception as e:
                attempt_errors.append(str(e))

        # If all attempts fail, fallback to SDPA and warn the user about the failures
        if len(attempt_errors) == len(attention_backend_attempts):
            self._pipeline.transformer.set_attention_backend("native")
            logger.warning(
                f"Failed to set attention backend '{attention_backend_config}' for "
                f"diffusers pipeline {self._pipeline.__class__.__name__}. "
                "Falling back to SDPA. "
                f"The following attempts were made: {dict(zip(attention_backend_attempts, attempt_errors))}"
            )
            return

        # If some attempts fail, only warn the user about the failures
        logger.info(
            f"Set diffusers attention backend to '{set_backend}', adapted from "
            f"user config value '{attention_backend_config}'."
        )
        if len(attempt_errors) > 0:
            logger.warning(
                f"The following failed attempts were made before choosing this diffusers backend: "
                f"{dict(zip(attention_backend_attempts, attempt_errors))}"
            )

    def _build_call_kwargs(self, req: DiffusionRequestBatch) -> dict[str, Any]:
        """Translate a ``DiffusionRequestBatch`` into diffusers ``__call__`` kwargs."""
        sampling = req.sampling_params
        input_kwargs = self._extract_input(req.prompts)

        self._pipeline_utils.validate_runtime_sampling_params(sampling)

        # Merge user-provided call kwargs from stage/CLI defaults.
        # Load time defaults -> input kwargs (prompts, neg prompts, images...) -> request-time sampling params
        kwargs: dict[str, Any] = {}

        # Load time defaults
        for key, value in self.od_config.diffusers_call_kwargs.items():
            if self._accept_call_kwargs is None or key in self._accept_call_kwargs:
                kwargs[key] = value
            else:
                logger.warning(
                    f"Skipping unsupported diffusers pipeline __call__ argument `{key}` from "
                    f"diffusers_call_kwargs. Check out the documentation of {self._pipeline.__class__.__name__}."
                )

        # Input kwargs
        for key, value in input_kwargs.items():
            if self._accept_call_kwargs is None or key in self._accept_call_kwargs:
                kwargs[key] = value
            else:
                logger.warning(
                    f"Skipping unsupported diffusers pipeline __call__ argument `{key}` from prompt input."
                    f"Check out the documentation of {self._pipeline.__class__.__name__}."
                )

        # Request-time sampling params
        for key, value in sampling.__dict__.items():
            if value is None:
                continue
            if self._accept_call_kwargs is None or key in self._accept_call_kwargs:
                kwargs[key] = value

        # Special format fields in sampling params
        if output_type := sampling.output_type or self.od_config.output_type:
            kwargs["output_type"] = output_type

        if (num_outputs_per_prompt := sampling.num_outputs_per_prompt) > 0:
            # In diffusers, they are num_images_per_prompt, num_videos_per_prompt, etc.
            for key in self._accept_call_kwargs or ():
                if re.match(r"num_[a-z]+_per_prompt", key):
                    kwargs[key] = num_outputs_per_prompt

        if sampling.generator is not None:
            kwargs["generator"] = sampling.generator
        elif sampling.seed is not None:
            kwargs["generator"] = torch.Generator(device=sampling.generator_device).manual_seed(sampling.seed)
        else:
            kwargs["generator"] = torch.Generator(device=sampling.generator_device)

        logger.info(
            "Calling diffusers pipeline with kwargs: %s", DiffusersAdapterPipeline._summarize_call_kwargs_value(kwargs)
        )

        return kwargs

    def _extract_input(self, prompt_obj: list[OmniPromptType]) -> dict[str, Any]:
        """Extract the text prompts and negative prompts from a list of prompt objects."""
        if len(prompt_obj) == 1:
            if isinstance(prompt_obj[0], str):
                return {"prompt": prompt_obj[0]}
            else:
                obj = cast(OmniTextPrompt, prompt_obj[0])
                negative_prompt = obj.get("negative_prompt")
                multi_modal_data = obj.get("multi_modal_data") or {}
                kwargs = {
                    "prompt": obj.get("prompt", ""),
                    **multi_modal_data,
                }
                if negative_prompt is not None:
                    kwargs["negative_prompt"] = negative_prompt
                return kwargs

        # Check the first element for the presence of multimodal data.
        # The following elements should have the same multimodal data fields, or none has multimodal data.
        multi_modal_data_fields: list[str] = []
        if isinstance(prompt_obj[0], dict) and (multi_modal_data := prompt_obj[0].get("multi_modal_data")):
            multi_modal_data_fields = list(multi_modal_data.keys())
        if multi_modal_data_fields:
            for i, prompt in enumerate(prompt_obj):
                assert isinstance(prompt, dict) and (multi_modal_data := prompt.get("multi_modal_data")) is not None, (
                    "When there are multiple prompts and the first prompt has multimodal data, "
                    f"each prompt should also contain the same multimodal data fields, but prompt {i} does not."
                )
                for key in multi_modal_data_fields:
                    assert key in multi_modal_data, (
                        "When there are multiple prompts and the first prompt has multimodal data, each prompt should "
                        f"also contain the same multimodal data fields, but prompt {i} does not contain {key}."
                    )
                    assert not isinstance(multi_modal_data.get(key), list), (
                        f"When there are multiple prompts and each prompt has multiple {key} data, this input pattern "
                        "is ambiguous as diffusers accepts flattened lists of text prompts and multimodal data. "
                        f"To use multiple {key} data, please use one single prompt instead."
                    )

        input_kwargs = {"prompt": [], **{key: [] for key in multi_modal_data_fields}}

        # Negative prompt rule:
        # - If any OmniTextPrompt has a negative prompt, or diffusers_call_kwargs has `negative_prompt`,
        #     enforce a negative prompt input (list[str]) -> `kwargs_should_contain_negative_prompt=true`
        #     (Because the negative prompt must be str, list[str], or None. It cannot be list[str|None])
        # -   Further in this case, try:
        # -   1. negative prompt in this OmniTextPrompt (typed dict)
        # -   2. fallback negative prompt from diffusers_call_kwargs (single str or the i-th item in list[str])
        # -   3. empty string ""
        # - Otherwise, `kwargs_should_contain_negative_prompt=False`. Do not add "negative_prompt" key to input_kwargs.
        has_negative_prompt = any(isinstance(p, dict) and p.get("negative_prompt") is not None for p in prompt_obj)
        fallback_negative_prompt = self.od_config.diffusers_call_kwargs.get("negative_prompt")
        kwargs_should_contain_negative_prompt = has_negative_prompt or fallback_negative_prompt is not None
        if kwargs_should_contain_negative_prompt:
            input_kwargs["negative_prompt"] = []

        for i, prompt in enumerate(prompt_obj):
            this_fallback_negative_prompt = ""
            if isinstance(fallback_negative_prompt, str):
                this_fallback_negative_prompt = fallback_negative_prompt
            elif isinstance(fallback_negative_prompt, list):
                try:
                    this_fallback_negative_prompt = fallback_negative_prompt[i]
                except IndexError:
                    raise ValueError(
                        "The fallback negative_prompt in diffusers_call_kwargs is a list, but its length "
                        f"({len(fallback_negative_prompt)}) is less than the number of prompts ({len(prompt_obj)}). "
                        "Please provide a list with the same length as the number of prompts."
                    )

            if isinstance(prompt, str):
                input_kwargs["prompt"].append(prompt)
                if kwargs_should_contain_negative_prompt:
                    input_kwargs["negative_prompt"].append(this_fallback_negative_prompt)
                for key in multi_modal_data_fields:
                    input_kwargs[key].append(None)
            else:
                obj = cast(OmniTextPrompt, prompt)
                input_kwargs["prompt"].append(obj.get("prompt", ""))

                if kwargs_should_contain_negative_prompt:
                    negative_prompt: str = obj.get("negative_prompt", this_fallback_negative_prompt)
                    input_kwargs["negative_prompt"].append(negative_prompt)

                multi_modal_data = obj.get("multi_modal_data") or {}
                for key in multi_modal_data_fields:
                    input_kwargs[key].append(multi_modal_data.get(key))
        return input_kwargs

    def _wrap_output(self, output: Any) -> DiffusionOutput:
        """Convert diffusers pipeline output to ``DiffusionOutput``.

        Diffusers output types:
        - ``ImagePipelineOutput(images=...)`` -- text2img, img2img
        - ``VideoPipelineOutput(frames=...)`` -- text2vid, img2vid
        """
        from vllm_omni.diffusion.data import DiffusionOutput

        if hasattr(output, "images"):
            # Preserve diffusers image format (`output_type`)
            return DiffusionOutput(output=output.images)

        if hasattr(output, "frames"):
            # Preserve diffusers video format (`output_type`)
            return DiffusionOutput(output=output.frames)

        if hasattr(output, "audios"):
            return DiffusionOutput(output=output.audios)

        return DiffusionOutput(output=output)

    @staticmethod
    def _summarize_call_kwargs_value(value: Any) -> Any:
        """Return a sanitized summary of diffusers call kwargs for logging."""
        if value is None or isinstance(value, (bool, int, float)):
            return value
        if isinstance(value, str):
            if len(value) < 20:
                return value
            return f"{value[:10]}...{value[-10:]}"
        if isinstance(value, torch.Tensor):
            return f"Tensor with shape {tuple(value.shape)}, dtype {value.dtype}, device {value.device}"
        if isinstance(value, torch.Generator):
            return f"Generator with seed {value.initial_seed()} on device {value.device}"
        if isinstance(value, dict):
            return {k: DiffusersAdapterPipeline._summarize_call_kwargs_value(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            if len(value) > 10:
                return f"{value.__class__.__name__} of length {len(value)}"
            return value.__class__([DiffusersAdapterPipeline._summarize_call_kwargs_value(v) for v in value])
        shape = getattr(value, "shape", None)
        size = getattr(value, "size", None)
        if shape is not None:
            return {"type": type(value).__name__, "shape": tuple(shape)}
        if size is not None and not callable(size):
            return {"type": type(value).__name__, "size": size}
        return f"<{type(value).__name__}>"
