import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch
from diffusers import ModelMixin
from diffusers.pipelines.pipeline_loading_utils import _get_pipeline_class, simple_get_class_obj
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from transformers import AutoConfig, PreTrainedModel

TINY_MODEL_DIR = os.path.join(tempfile.gettempdir(), "vllm-omni-tiny-models")
TINY_CONFIGS_DIR = Path(__file__).parent / "tiny_configs"


def _get_tiny_model_path(name: str) -> str:
    path = os.path.join(TINY_MODEL_DIR, name)
    os.makedirs(path, exist_ok=True)
    return path


def _diffusers_from_config(cls, pretrained_model_name_or_path, **kwargs):
    """Replacement for ModelMixin.from_pretrained that initializes random weights."""
    subfolder = kwargs.get("subfolder")
    load_kwargs = {"subfolder": subfolder} if subfolder else {}
    config = cls.load_config(pretrained_model_name_or_path, **load_kwargs)
    return cls.from_config(config)


def _transformers_from_config(cls, pretrained_model_name_or_path, **kwargs):
    """Replacement for PreTrainedModel.from_pretrained that initializes random weights."""
    subfolder = kwargs.get("subfolder")
    load_kwargs = {"subfolder": subfolder} if subfolder else {}
    config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **load_kwargs)
    return cls(config)


def build_tiny_from_configs(pipeline_name: str, model_id: str) -> str:
    """Build a tiny model from vendored configs with random weights.

    Mirrors the component loading loop in DiffusionPipeline.from_pretrained,
    but monkeypatches from_pretrained on ModelMixin and PreTrainedModel to
    initialize with random weights instead of loading checkpoint files.
    Components with vendored configs use those; others load configs from
    the upstream HF model.
    """
    model_dir = _get_tiny_model_path(pipeline_name)
    config_dir = TINY_CONFIGS_DIR / pipeline_name

    config_dict = DiffusionPipeline.load_config(config_dir)
    pipeline_cls = _get_pipeline_class(DiffusionPipeline, config=config_dict)

    init_dict, _, _ = pipeline_cls.extract_init_dict(config_dict)

    # Pop non-component entries (optional pipeline kwargs like is_distilled),
    # same as DiffusionPipeline.from_pretrained lines 345-350
    _, optional_kwargs = DiffusionPipeline._get_signature_keys(pipeline_cls)
    init_kwargs = {k: init_dict.pop(k) for k in optional_kwargs if k in init_dict}

    with (
        patch.object(ModelMixin, "from_pretrained", classmethod(_diffusers_from_config)),
        patch.object(PreTrainedModel, "from_pretrained", classmethod(_transformers_from_config)),
    ):
        for name, (library_name, class_name) in init_dict.items():
            cls = simple_get_class_obj(library_name, class_name)

            # Use vendored config dir if available, otherwise upstream model
            if (config_dir / name).exists():
                init_kwargs[name] = cls.from_pretrained(config_dir / name)
            else:
                init_kwargs[name] = cls.from_pretrained(model_id, subfolder=name)

    pipe = pipeline_cls(**init_kwargs)
    pipe.to(torch.bfloat16).save_pretrained(model_dir)
    return model_dir


def tiny_flux2_klein_builder() -> str:
    """Build a tiny Flux2Klein model from vendored configs."""
    return build_tiny_from_configs("Flux2KleinPipeline", "black-forest-labs/FLUX.2-klein-4B")


def tiny_ltx2_builder() -> str:
    """Build a tiny LTX2 model from vendored configs."""
    return build_tiny_from_configs("LTX2Pipeline", "Lightricks/LTX-2")
