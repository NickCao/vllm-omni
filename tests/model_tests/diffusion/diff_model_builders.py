from functools import partial
from pathlib import Path

from tests.helpers.tiny_model import build_tiny_from_configs

TINY_CONFIGS_DIR = Path(__file__).parent / "tiny_configs"


def _shrink_dit_rope_config(
    config: dict,
    num_layers: int = 2,
    num_single_layers: int | None = None,
    target_head_dim: int = 32,
    target_heads: int = 4,
    default_axes_dims_rope: list[int] | None = None,
    joint_attention_dim: int | None = None,
) -> dict:
    config["num_layers"] = num_layers
    if num_single_layers is not None:
        config["num_single_layers"] = num_single_layers
    if joint_attention_dim is not None:
        config["joint_attention_dim"] = joint_attention_dim
    axes_dims_rope = config.get("axes_dims_rope", default_axes_dims_rope)
    factor = config["attention_head_dim"] / target_head_dim
    config["attention_head_dim"] = target_head_dim
    config["num_attention_heads"] = target_heads
    config["axes_dims_rope"] = [int(d / factor) for d in axes_dims_rope]
    return config


def tiny_flux2_klein_builder() -> str:
    """Build a tiny Flux2Klein model from vendored configs."""
    return build_tiny_from_configs(
        "Flux2KleinPipeline", "black-forest-labs/FLUX.2-klein-4B", TINY_CONFIGS_DIR / "Flux2KleinPipeline"
    )


def tiny_ltx2_builder() -> str:
    """Build a tiny LTX2 model from vendored configs."""
    return build_tiny_from_configs("LTX2Pipeline", "Lightricks/LTX-2", TINY_CONFIGS_DIR / "LTX2Pipeline")


def _shrink_hunyuan_video_text_encoder_config(config: dict, hidden_size: int = 64) -> dict:
    old_head_dim = config["hidden_size"] / config["num_attention_heads"]
    config["num_hidden_layers"] = 2
    config["intermediate_size"] = 64
    config["hidden_size"] = hidden_size
    config["num_attention_heads"] = 2
    config["num_key_value_heads"] = 2
    config["layer_types"] = config["layer_types"][:2]
    factor = old_head_dim / (hidden_size / 2)
    mrope_section = config["rope_scaling"]["mrope_section"]
    config["rope_scaling"]["mrope_section"] = [round(d / factor) for d in mrope_section]
    return config


def _shrink_hunyuan_video_transformer_config(config: dict, text_embed_dim: int = 64) -> dict:
    config["num_layers"] = 2
    config["num_refiner_layers"] = 2
    config["text_embed_dim"] = text_embed_dim
    factor = config["attention_head_dim"] / 32
    config["attention_head_dim"] = 32
    config["num_attention_heads"] = 4
    config["rope_axes_dim"] = [int(d / factor) for d in config["rope_axes_dim"]]
    return config


def tiny_hunyuan_video_15_builder() -> str:
    return build_tiny_from_configs(
        "HunyuanVideo15Pipeline",
        "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v",
        transform={
            "text_encoder": _shrink_hunyuan_video_text_encoder_config,
            "text_encoder_2": _shrink_t5_text_encoder_config,
            "transformer": _shrink_hunyuan_video_transformer_config,
        },
    )


def _shrink_qwen_text_encoder_config(config: dict, hidden_size: int = 64) -> dict:
    text_config = config["text_config"]
    old_head_dim = text_config["hidden_size"] / text_config["num_attention_heads"]
    text_config["num_hidden_layers"] = 2
    text_config["intermediate_size"] = 64
    text_config["hidden_size"] = hidden_size
    text_config["num_attention_heads"] = 2
    text_config["num_key_value_heads"] = 2
    text_config["layer_types"] = text_config["layer_types"][:2]
    factor = old_head_dim / (hidden_size / 2)
    mrope_section = text_config["rope_scaling"]["mrope_section"]
    text_config["rope_scaling"]["mrope_section"] = [round(d / factor) for d in mrope_section]
    config["num_hidden_layers"] = 2
    config["intermediate_size"] = 64
    config["hidden_size"] = hidden_size
    config["num_attention_heads"] = 2
    config["num_key_value_heads"] = 2
    config["vision_config"]["depth"] = 2
    config["vision_config"]["intermediate_size"] = 64
    config["vision_config"]["fullatt_block_indexes"] = [0, 1]
    config["vision_config"]["out_hidden_size"] = hidden_size
    return config


def _shrink_qwen_transformer_config(config: dict, joint_attention_dim: int = 64) -> dict:
    return _shrink_dit_rope_config(config, joint_attention_dim=joint_attention_dim)


def tiny_qwen_image_builder() -> str:
    return build_tiny_from_configs(
        "QwenImagePipeline",
        "Qwen/Qwen-Image",
        transform={"text_encoder": _shrink_qwen_text_encoder_config, "transformer": _shrink_qwen_transformer_config},
    )


def tiny_qwen_image_edit_builder() -> str:
    return build_tiny_from_configs(
        "QwenImageEditPipeline",
        "Qwen/Qwen-Image-Edit",
        transform={"text_encoder": _shrink_qwen_text_encoder_config, "transformer": _shrink_qwen_transformer_config},
    )


def tiny_qwen_image_edit_plus_builder() -> str:
    return build_tiny_from_configs(
        "QwenImageEditPlusPipeline",
        "Qwen/Qwen-Image-Edit-2511",
        transform={"text_encoder": _shrink_qwen_text_encoder_config, "transformer": _shrink_qwen_transformer_config},
    )


def _shrink_longcat_text_encoder_config(config: dict, hidden_size: int = 64) -> dict:
    # LongCat ships an older (pre-text_config-nesting) Qwen2.5-VL config layout,
    # so unlike _shrink_qwen_text_encoder_config, fields live at the top level.
    old_head_dim = config["hidden_size"] / config["num_attention_heads"]
    config["num_hidden_layers"] = 2
    config["intermediate_size"] = 64
    config["hidden_size"] = hidden_size
    config["num_attention_heads"] = 2
    config["num_key_value_heads"] = 2
    factor = old_head_dim / (hidden_size / 2)
    mrope_section = config["rope_scaling"]["mrope_section"]
    config["rope_scaling"]["mrope_section"] = [round(d / factor) for d in mrope_section]
    config["vision_config"]["depth"] = 2
    config["vision_config"]["intermediate_size"] = 64
    config["vision_config"]["fullatt_block_indexes"] = [0, 1]
    config["vision_config"]["out_hidden_size"] = hidden_size
    return config


def _shrink_longcat_transformer_config(config: dict, joint_attention_dim: int = 64) -> dict:
    config = _shrink_dit_rope_config(
        config, num_single_layers=2, default_axes_dims_rope=[16, 56, 56], joint_attention_dim=joint_attention_dim
    )
    config["pooled_projection_dim"] = joint_attention_dim
    return config


def tiny_longcat_image_builder() -> str:
    return build_tiny_from_configs(
        "LongCatImagePipeline",
        "meituan-longcat/LongCat-Image",
        transform={
            "text_encoder": _shrink_longcat_text_encoder_config,
            "transformer": _shrink_longcat_transformer_config,
        },
    )


def tiny_longcat_image_edit_builder() -> str:
    return build_tiny_from_configs(
        "LongCatImageEditPipeline",
        "meituan-longcat/LongCat-Image-Edit",
        transform={
            "text_encoder": _shrink_longcat_text_encoder_config,
            "transformer": _shrink_longcat_transformer_config,
        },
    )


def _shrink_clip_text_encoder_config(config: dict) -> dict:
    config["num_hidden_layers"] = 2
    return config


def _shrink_t5_text_encoder_config(config: dict) -> dict:
    config["num_layers"] = 2
    config["num_decoder_layers"] = 2
    config["d_ff"] = 64
    config["num_heads"] = 4
    return config


def tiny_flux_builder() -> str:
    return build_tiny_from_configs(
        "FluxPipeline",
        "black-forest-labs/FLUX.1-schnell",
        transform={
            "text_encoder": _shrink_clip_text_encoder_config,
            "text_encoder_2": _shrink_t5_text_encoder_config,
            "transformer": partial(_shrink_dit_rope_config, num_single_layers=2, default_axes_dims_rope=[16, 56, 56]),
        },
    )


def tiny_flux_kontext_builder() -> str:
    return build_tiny_from_configs(
        "FluxKontextPipeline",
        "black-forest-labs/FLUX.1-Kontext-dev",
        transform={
            "text_encoder": _shrink_clip_text_encoder_config,
            "text_encoder_2": _shrink_t5_text_encoder_config,
            "transformer": partial(_shrink_dit_rope_config, num_single_layers=2, default_axes_dims_rope=[16, 56, 56]),
        },
    )


def tiny_flux2_builder() -> str:
    def shrink_text_encoder(config: dict) -> dict:
        # The real checkpoint ships language_model.lm_head.weight as its own
        # tensor even though tie_word_embeddings defaults to True; keep it
        # untied here too so transformers doesn't drop it on save.
        config["tie_word_embeddings"] = False
        config["text_config"]["num_hidden_layers"] = 31
        config["text_config"]["intermediate_size"] = 64
        config["text_config"]["num_attention_heads"] = 4
        config["text_config"]["head_dim"] = 32
        config["text_config"]["num_key_value_heads"] = 2
        config["vision_config"]["num_hidden_layers"] = 2
        config["vision_config"]["intermediate_size"] = 64
        config["vision_config"]["num_attention_heads"] = 4
        config["vision_config"]["head_dim"] = 32
        return config

    return build_tiny_from_configs(
        "Flux2Pipeline",
        "black-forest-labs/FLUX.2-dev",
        transform={
            "text_encoder": shrink_text_encoder,
            "transformer": partial(_shrink_dit_rope_config, num_single_layers=2),
        },
    )


def _shrink_ovis_text_encoder_config(config: dict, hidden_size: int = 64) -> dict:
    config["num_hidden_layers"] = 2
    config["intermediate_size"] = 64
    config["hidden_size"] = hidden_size
    config["num_attention_heads"] = 2
    config["num_key_value_heads"] = 2
    config["head_dim"] = hidden_size // config["num_attention_heads"]
    config["layer_types"] = config["layer_types"][:2]
    return config


def tiny_ovis_image_builder() -> str:
    return build_tiny_from_configs(
        "OvisImagePipeline",
        "AIDC-AI/Ovis-Image-7B",
        transform={
            "text_encoder": _shrink_ovis_text_encoder_config,
            "transformer": partial(_shrink_dit_rope_config, num_single_layers=2, joint_attention_dim=64),
        },
    )


def _shrink_glm_vision_language_encoder_config(config: dict) -> dict:
    # vllm_omni's GlmImagePipeline never loads this component (only the diffusers-native
    # class needs it to build a valid pipeline); shrink it purely to keep the tiny model
    # small on disk, without worrying about internal shape invariants.
    text_config = config["text_config"]
    old_head_dim = text_config["hidden_size"] / text_config["num_attention_heads"]
    text_config["num_hidden_layers"] = 2
    text_config["hidden_size"] = 64
    text_config["intermediate_size"] = 64
    text_config["num_attention_heads"] = 2
    text_config["num_key_value_heads"] = 2
    factor = old_head_dim / (64 / 2)
    mrope_section = text_config["rope_parameters"]["mrope_section"]
    text_config["rope_parameters"]["mrope_section"] = [round(d / factor) for d in mrope_section]
    config["vision_config"]["depth"] = 2
    config["vision_config"]["hidden_size"] = 64
    config["vision_config"]["intermediate_size"] = 64
    config["vision_config"]["num_heads"] = 2
    return config


def _shrink_glm_image_transformer_config(config: dict) -> dict:
    config["num_layers"] = 2
    config["attention_head_dim"] = 32
    config["num_attention_heads"] = 4
    return config


def tiny_glm_image_builder() -> str:
    return build_tiny_from_configs(
        "GlmImagePipeline",
        "zai-org/GLM-Image",
        transform={
            "text_encoder": _shrink_t5_text_encoder_config,
            "vision_language_encoder": _shrink_glm_vision_language_encoder_config,
            "transformer": _shrink_glm_image_transformer_config,
        },
    )
