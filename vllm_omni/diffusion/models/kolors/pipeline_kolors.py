# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from diffusers KolorsPipeline
# Copyright 2025 Stability AI, Kwai-Kolors Team and The HuggingFace Team.
import inspect
import json
import logging
import os
from collections.abc import Iterable

import torch
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.image_processor import VaeImageProcessor
from diffusers.schedulers import EulerDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor
from torch import nn
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.model_executor.model_loader.weight_utils import (
    download_weights_from_hf_specific,
)

logger = logging.getLogger(__name__)


def get_kolors_image_post_process_func(
    od_config: OmniDiffusionConfig,
):
    """Create a post-processing function for Kolors image output.

    Converts raw VAE-decoded tensors to PIL images using VaeImageProcessor.
    """
    if od_config.output_type == "latent":
        return lambda x: x
    model_name = od_config.model
    if os.path.exists(model_name):
        model_path = model_name
    else:
        model_path = download_weights_from_hf_specific(model_name, None, ["*"])
    vae_config_path = os.path.join(model_path, "vae/config.json")
    with open(vae_config_path) as f:
        vae_config = json.load(f)
        vae_scale_factor = (
            2 ** (len(vae_config["block_out_channels"]) - 1)
            if "block_out_channels" in vae_config
            else 8
        )

    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)

    def post_process_func(
        images: torch.Tensor,
    ):
        return image_processor.postprocess(images)

    return post_process_func


def retrieve_timesteps(
    scheduler,
    num_inference_steps: int | None = None,
    device: str | torch.device | None = None,
    timesteps: list[int] | None = None,
    sigmas: list[float] | None = None,
    **kwargs,
):
    """Retrieve timesteps from the scheduler after calling set_timesteps.

    Handles custom timesteps and sigmas.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError(
            "Only one of `timesteps` or `sigmas` can be passed."
        )
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s "
                "`set_timesteps` does not support custom timestep schedules."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s "
                "`set_timesteps` does not support custom sigmas schedules."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class KolorsPipeline(nn.Module, CFGParallelMixin):
    """Pipeline for text-to-image generation using Kolors.

    Kolors is a large-scale text-to-image model based on latent diffusion,
    using ChatGLM3 as the text encoder and an SDXL-style UNet2D architecture.

    Compatible with Kwai-Kolors/Kolors-diffusers (model_index._class_name="KolorsPipeline").
    """

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.od_config = od_config
        self.device = get_local_device()
        model = od_config.model
        local_files_only = os.path.exists(model)
        dtype = od_config.dtype

        # UNet weights loaded through the weight loading system
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=od_config.model,
                subfolder="unet",
                revision=None,
                prefix="unet.",
                fall_back_to_pt=True,
            )
        ]

        # Load scheduler
        self.scheduler = EulerDiscreteScheduler.from_pretrained(
            model,
            subfolder="scheduler",
            local_files_only=local_files_only,
        )

        # Load text encoder and tokenizer (ChatGLM3)
        from diffusers.pipelines.kolors.text_encoder import ChatGLMModel
        from diffusers.pipelines.kolors.tokenizer import ChatGLMTokenizer

        self.tokenizer = ChatGLMTokenizer.from_pretrained(
            model,
            subfolder="tokenizer",
            local_files_only=local_files_only,
        )
        # Some repos (e.g. Kwai-Kolors/Kolors-diffusers) only ship fp16
        # variant files. Try default first, fall back to variant="fp16".
        self.text_encoder = self._from_pretrained_with_variant_fallback(
            ChatGLMModel,
            model,
            subfolder="text_encoder",
            local_files_only=local_files_only,
            torch_dtype=dtype,
        ).to(self.device)

        # Create UNet from config; actual weights loaded via weights_sources
        unet_config = UNet2DConditionModel.load_config(
            model, subfolder="unet", local_files_only=local_files_only
        )
        self.unet = UNet2DConditionModel.from_config(unet_config)

        # Load VAE
        self.vae = self._from_pretrained_with_variant_fallback(
            AutoencoderKL,
            model,
            subfolder="vae",
            local_files_only=local_files_only,
            torch_dtype=dtype,
        ).to(self.device)

        # Configuration
        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1)
            if getattr(self, "vae", None)
            else 8
        )
        self.image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor
        )
        self.default_sample_size = (
            self.unet.config.sample_size
            if hasattr(self.unet.config, "sample_size")
            else 128
        )
        self.force_zeros_for_empty_prompt = False
        self.output_type = od_config.output_type

    @staticmethod
    def _from_pretrained_with_variant_fallback(cls, model, **kwargs):
        """Load a pretrained model, falling back to variant='fp16' on failure.

        Some model repos (e.g. Kwai-Kolors/Kolors-diffusers) only ship
        fp16 variant weight files. This helper tries the default variant
        first, then retries with variant='fp16'.
        """
        try:
            return cls.from_pretrained(model, **kwargs)
        except (OSError, ValueError):
            return cls.from_pretrained(model, variant="fp16", **kwargs)

    def encode_prompt(
        self,
        prompt,
        negative_prompt=None,
        do_classifier_free_guidance: bool = True,
        max_sequence_length: int = 256,
    ):
        """Encode text prompts into embeddings using ChatGLM3.

        Returns:
            Tuple of (prompt_embeds, negative_prompt_embeds,
                      pooled_prompt_embeds, negative_pooled_prompt_embeds).
        """
        device = self.device

        if isinstance(prompt, str):
            prompt = [prompt]
        batch_size = len(prompt)

        # Encode positive prompt
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        output = self.text_encoder(
            input_ids=text_inputs["input_ids"],
            attention_mask=text_inputs["attention_mask"],
            position_ids=text_inputs["position_ids"],
            output_hidden_states=True,
        )

        # [max_seq_len, batch, hidden_size] -> [batch, max_seq_len, hidden_size]
        prompt_embeds = output.hidden_states[-2].permute(1, 0, 2).clone()
        # [max_seq_len, batch, hidden_size] -> [batch, hidden_size]
        pooled_prompt_embeds = output.hidden_states[-1][-1, :, :].clone()

        # Encode negative prompt
        negative_prompt_embeds = None
        negative_pooled_prompt_embeds = None

        zero_out_negative_prompt = (
            negative_prompt is None and self.force_zeros_for_empty_prompt
        )

        if (
            do_classifier_free_guidance
            and negative_prompt_embeds is None
            and zero_out_negative_prompt
        ):
            negative_prompt_embeds = torch.zeros_like(prompt_embeds)
            negative_pooled_prompt_embeds = torch.zeros_like(
                pooled_prompt_embeds
            )
        elif do_classifier_free_guidance:
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt` has batch size {len(negative_prompt)}, "
                    f"but `prompt` has batch size {batch_size}."
                )
            else:
                uncond_tokens = negative_prompt

            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_sequence_length,
                truncation=True,
                return_tensors="pt",
            ).to(device)
            output = self.text_encoder(
                input_ids=uncond_input["input_ids"],
                attention_mask=uncond_input["attention_mask"],
                position_ids=uncond_input["position_ids"],
                output_hidden_states=True,
            )

            negative_prompt_embeds = (
                output.hidden_states[-2].permute(1, 0, 2).clone()
            )
            negative_pooled_prompt_embeds = (
                output.hidden_states[-1][-1, :, :].clone()
            )

        return (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        )

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        """Prepare random noise latents."""
        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )
        if latents is None:
            latents = randn_tensor(
                shape, generator=generator, device=device, dtype=dtype
            )
        else:
            latents = latents.to(device)

        # Scale initial noise by the standard deviation required by scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def _get_add_time_ids(
        self,
        original_size,
        crops_coords_top_left,
        target_size,
        dtype,
        text_encoder_projection_dim=None,
    ):
        """Compute SDXL-style micro-conditioning time IDs."""
        add_time_ids = list(
            original_size + crops_coords_top_left + target_size
        )
        passed_add_embed_dim = (
            self.unet.config.addition_time_embed_dim * len(add_time_ids)
            + text_encoder_projection_dim
        )
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length "
                f"{expected_add_embed_dim}, but a vector of "
                f"{passed_add_embed_dim} was created."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        return add_time_ids

    def get_guidance_scale_embedding(
        self,
        w: torch.Tensor,
        embedding_dim: int = 512,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Compute guidance scale embedding for distilled models."""
        assert len(w.shape) == 1
        w = w * 1000.0
        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
        emb = w.to(dtype)[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1))
        return emb

    def predict_noise(self, *args, **kwargs):
        """Forward pass through UNet to predict noise.

        Overrides CFGParallelMixin.predict_noise which defaults to
        self.transformer. Kolors uses a UNet instead.
        """
        return self.unet(*args, **kwargs)[0]

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return (
            self._guidance_scale > 1
            and self.unet.config.time_cond_proj_dim is None
        )

    def diffuse(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        negative_prompt_embeds: torch.Tensor | None,
        negative_pooled_prompt_embeds: torch.Tensor | None,
        add_time_ids: torch.Tensor,
        negative_add_time_ids: torch.Tensor | None,
        do_true_cfg: bool,
        guidance_scale: float,
        timestep_cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Denoising loop with optional classifier-free guidance.

        Uses CFGParallelMixin's predict_noise_maybe_with_cfg for automatic
        CFG parallel handling.
        """
        for _, t in enumerate(timesteps):
            # Scale latents for this timestep
            scaled_latents = self.scheduler.scale_model_input(latents, t)

            # Build positive (conditional) kwargs
            positive_kwargs = {
                "sample": scaled_latents,
                "timestep": t,
                "encoder_hidden_states": prompt_embeds,
                "timestep_cond": timestep_cond,
                "added_cond_kwargs": {
                    "text_embeds": pooled_prompt_embeds,
                    "time_ids": add_time_ids,
                },
                "return_dict": False,
            }

            if do_true_cfg:
                negative_kwargs = {
                    "sample": scaled_latents,
                    "timestep": t,
                    "encoder_hidden_states": negative_prompt_embeds,
                    "timestep_cond": timestep_cond,
                    "added_cond_kwargs": {
                        "text_embeds": negative_pooled_prompt_embeds,
                        "time_ids": negative_add_time_ids,
                    },
                    "return_dict": False,
                }
            else:
                negative_kwargs = None

            # Predict noise with automatic CFG parallel handling
            noise_pred = self.predict_noise_maybe_with_cfg(
                do_true_cfg,
                guidance_scale,
                positive_kwargs,
                negative_kwargs,
            )

            # Scheduler step with automatic CFG sync
            latents = self.scheduler_step_maybe_with_cfg(
                noise_pred, t, latents, do_true_cfg
            )

        return latents

    def forward(
        self,
        req: OmniDiffusionRequest,
        prompt: str | list[str] = "",
        negative_prompt: str | list[str] = "",
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        sigmas: list[float] | None = None,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        max_sequence_length: int = 256,
    ) -> DiffusionOutput:
        """Full pipeline: encode prompt -> prepare latents -> diffuse -> decode.

        Args:
            req: The diffusion request containing prompts and sampling params.
        """
        # Extract from request
        prompt = [
            p if isinstance(p, str) else (p.get("prompt") or "")
            for p in req.prompts
        ] or prompt
        negative_prompt = [
            "" if isinstance(p, str) else (p.get("negative_prompt") or "")
            for p in req.prompts
        ] or negative_prompt

        height = (
            req.sampling_params.height
            or self.default_sample_size * self.vae_scale_factor
        )
        width = (
            req.sampling_params.width
            or self.default_sample_size * self.vae_scale_factor
        )
        num_inference_steps = (
            req.sampling_params.num_inference_steps or num_inference_steps
        )
        self._guidance_scale = req.sampling_params.guidance_scale
        sigmas = req.sampling_params.sigmas or sigmas
        max_sequence_length = (
            req.sampling_params.max_sequence_length or max_sequence_length
        )
        generator = req.sampling_params.generator or generator
        num_images_per_prompt = (
            req.sampling_params.num_outputs_per_prompt
            if req.sampling_params.num_outputs_per_prompt > 0
            else 1
        )

        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = 1

        do_cfg = self.do_classifier_free_guidance

        # 1. Encode prompts
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt if do_cfg else None,
            do_classifier_free_guidance=do_cfg,
            max_sequence_length=max_sequence_length,
        )

        # 2. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, self.device, sigmas=sigmas
        )

        # 3. Prepare latents
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            self.device,
            generator,
            latents,
        )

        # 4. Prepare SDXL micro-conditioning
        original_size = (height, width)
        target_size = (height, width)
        crops_coords_top_left = (0, 0)
        text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])

        add_time_ids = self._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        ).to(self.device)
        add_time_ids = add_time_ids.repeat(
            batch_size * num_images_per_prompt, 1
        )
        negative_add_time_ids = add_time_ids

        # 5. Optionally get guidance scale embedding (for distilled models)
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(
                self.guidance_scale - 1
            ).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor,
                embedding_dim=self.unet.config.time_cond_proj_dim,
            ).to(device=self.device, dtype=latents.dtype)

        # 6. Denoising loop
        latents = self.diffuse(
            latents=latents,
            timesteps=timesteps,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_prompt_embeds=(
                negative_prompt_embeds if do_cfg else None
            ),
            negative_pooled_prompt_embeds=(
                negative_pooled_prompt_embeds if do_cfg else None
            ),
            add_time_ids=add_time_ids,
            negative_add_time_ids=(
                negative_add_time_ids if do_cfg else None
            ),
            do_true_cfg=do_cfg,
            guidance_scale=self.guidance_scale,
            timestep_cond=timestep_cond,
        )

        # 7. Decode latents
        if self.output_type == "latent":
            image = latents
        else:
            latents = latents.to(self.vae.dtype)
            latents = latents / self.vae.config.scaling_factor
            image = self.vae.decode(latents, return_dict=False)[0]

        return DiffusionOutput(output=image)

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)
