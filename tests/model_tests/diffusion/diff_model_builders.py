import os
import tempfile
from pathlib import Path

import torch
from diffusers import pipelines
from diffusers.pipelines.pipeline_loading_utils import (
    ALL_IMPORTABLE_CLASSES,
    _get_pipeline_class,
    get_class_obj_and_candidates,
)
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from transformers import AutoConfig

TINY_MODEL_DIR = os.path.join(tempfile.gettempdir(), "vllm-omni-tiny-models")
TINY_CONFIGS_DIR = Path(__file__).parent / "tiny_configs"

# Parallel to LOADABLE_CLASSES: for each base class, how to init from config
# with random weights. None means always load from the upstream model.
INIT_FROM_CONFIG_METHOD = {
    "ModelMixin": lambda cls, path: cls.from_config(cls.load_config(path)),
    "SchedulerMixin": lambda cls, path: cls.from_config(cls.load_config(path)),
    "BaseGuidance": lambda cls, path: cls.from_config(cls.load_config(path)),
    "PreTrainedModel": lambda cls, path: cls(AutoConfig.from_pretrained(path)),
    # Tokenizers/processors need real files, not random init
    "PreTrainedTokenizer": None,
    "PreTrainedTokenizerFast": None,
    "ProcessorMixin": None,
    "ImageProcessingMixin": None,
    "FeatureExtractionMixin": None,
}


def _get_tiny_model_path(name: str) -> str:
    path = os.path.join(TINY_MODEL_DIR, name)
    os.makedirs(path, exist_ok=True)
    return path


def build_tiny_from_configs(pipeline_name: str, model_id: str) -> str:
    """Build a tiny model from vendored configs with random weights.

    Mirrors the component loading loop in DiffusionPipeline.from_pretrained,
    but uses from_config (random init) instead of loading pretrained weights.
    Components without vendored configs are loaded from the upstream HF model.
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

    for name, (library_name, class_name) in init_dict.items():
        is_pipeline_module = hasattr(pipelines, library_name)

        cls, class_candidates = get_class_obj_and_candidates(
            library_name,
            class_name,
            ALL_IMPORTABLE_CLASSES,
            pipelines,
            is_pipeline_module,
        )

        # Find the matching base class, same way load_sub_model does
        init_fn = None
        for base_name, base_cls in class_candidates.items():
            if base_cls is not None and issubclass(cls, base_cls):
                init_fn = INIT_FROM_CONFIG_METHOD.get(base_name)
                break

        comp_dir = config_dir / name

        if init_fn and comp_dir.exists():
            init_kwargs[name] = init_fn(cls, comp_dir)
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
