import os
import tempfile
from pathlib import Path

import torch
from diffusers.models.autoencoders.autoencoder_kl_flux2 import AutoencoderKLFlux2
from diffusers.models.transformers.transformer_flux2 import Flux2Transformer2DModel
from diffusers.pipelines.flux2.pipeline_flux2_klein import Flux2KleinPipeline
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from transformers import AutoConfig, AutoModel, AutoTokenizer

TINY_MODEL_DIR = os.path.join(tempfile.gettempdir(), "vllm-omni-tiny-models")
TINY_CONFIGS_DIR = Path(__file__).parent / "tiny_configs"


def _get_tiny_model_path(name: str) -> str:
    path = os.path.join(TINY_MODEL_DIR, name)
    os.makedirs(path, exist_ok=True)
    return path


def tiny_flux2_klein_builder() -> str:
    """Build a tiny Flux2Klein model from vendored configs."""
    model_id = "black-forest-labs/FLUX.2-klein-4B"
    model_dir = _get_tiny_model_path("Flux2KleinPipeline")
    cfg = TINY_CONFIGS_DIR / "Flux2KleinPipeline"

    pipe = Flux2KleinPipeline(
        scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler"),
        vae=AutoencoderKLFlux2.from_config(cfg / "vae"),
        # NOTE: For now we need 28 layers because of hardcoded stuff in the model :(
        text_encoder=AutoModel.from_config(AutoConfig.from_pretrained(cfg / "text_encoder")),
        tokenizer=AutoTokenizer.from_pretrained(model_id, subfolder="tokenizer"),
        # NOTE: For now we need at least 2 layers for the transformer
        # due to hardcoded hacks in CacheDiT for Flux2Klein specifically.
        transformer=Flux2Transformer2DModel.from_config(cfg / "transformer"),
    )
    # Need dtypes to be consistent; for now we just put it on bfloat16
    pipe.to(torch.bfloat16).save_pretrained(model_dir)
    return model_dir
