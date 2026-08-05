# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared black-box ``__call__`` delegation logic for diffusers-backed adapters.

Used by both ``DiffusersAdapterPipeline`` (classic ``DiffusionPipeline``)
and ``ModularDiffusersAdapterPipeline`` (``ModularPipeline``) -- the two
differ only in how they load the pipeline, not in how requests are
translated to kwargs or how output is wrapped.
"""

import logging
import re
from typing import Any, cast

import torch

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.inputs.data import OmniPromptType, OmniTextPrompt
from vllm_omni.platforms import current_omni_platform

logger = logging.getLogger(__name__)


class DiffusersCallDelegationMixin:
    """Mixin providing ``forward()`` and its supporting helpers.

    Classes using this mixin must set, before ``forward()`` is called:
    - ``self._pipeline``: object with a diffusers-style ``__call__``
    - ``self.od_config``: ``OmniDiffusionConfig``
    - ``self._pipeline_utils``: ``BasePipelineUtils``
    - ``self._accept_call_kwargs``: ``set[str] | None`` (``None`` accepts all)
    """

    def forward(self, req: DiffusionRequestBatch) -> DiffusionOutput:
        """Full delegation to the wrapped pipeline's ``__call__()``."""
        kwargs = self._build_call_kwargs(req)
        logger.debug("Calling diffusers pipeline with kwargs: %s", kwargs)

        with torch.inference_mode():
            output = self._pipeline(**kwargs)  # pyright: ignore[reportCallIssue]

        return self._wrap_output(output)

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

        logger.info("Calling diffusers pipeline with kwargs: %s", self._summarize_call_kwargs_value(kwargs))

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
            return {k: DiffusersCallDelegationMixin._summarize_call_kwargs_value(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            if len(value) > 10:
                return f"{value.__class__.__name__} of length {len(value)}"
            return value.__class__([DiffusersCallDelegationMixin._summarize_call_kwargs_value(v) for v in value])
        shape = getattr(value, "shape", None)
        size = getattr(value, "size", None)
        if shape is not None:
            return {"type": type(value).__name__, "shape": tuple(shape)}
        if size is not None and not callable(size):
            return {"type": type(value).__name__, "size": size}
        return f"<{type(value).__name__}>"
