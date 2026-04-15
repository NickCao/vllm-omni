# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Kokoro TTS Pipeline for vLLM-Omni diffusion engine.

Single-stage pipeline: text -> G2P -> BERT encode -> duration/prosody
prediction -> ISTFTNet decode -> 24kHz audio waveform.

Uses request-mode execution (all steps in one forward() call).
Based on https://github.com/hexgrad/kokoro (Apache-2.0).
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from typing import ClassVar

import torch
from torch import nn
from vllm.logger import init_logger

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.models.interface import SupportAudioOutput
from vllm_omni.diffusion.models.kokoro.istftnet import Decoder
from vllm_omni.diffusion.models.kokoro.modules import (
    AlbertConfig,
    CustomAlbert,
    ProsodyPredictor,
    TextEncoder,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest

logger = init_logger(__name__)

# Default voice to use when none is specified in the request.
_DEFAULT_VOICE = "af_heart"

# Maximum phoneme length (BERT context window minus BOS/EOS tokens).
_MAX_PHONEME_LEN = 510

# Audio sample rate produced by Kokoro.
_SAMPLE_RATE = 24000

# Mapping of language aliases accepted in voice names.
_LANG_ALIASES = {
    "en-us": "a",
    "en-gb": "b",
    "es": "e",
    "fr-fr": "f",
    "hi": "h",
    "it": "i",
    "pt-br": "p",
    "ja": "j",
    "zh": "z",
}


def get_kokoro_post_process_func(od_config: OmniDiffusionConfig):
    """Post-processing: convert audio tensor to numpy for WAV encoding."""

    def post_process_func(audio: torch.Tensor, output_type: str = "np"):
        if output_type == "pt":
            return audio
        return audio.cpu().float().numpy()

    return post_process_func


class KokoroPipeline(nn.Module, SupportAudioOutput):
    """Kokoro text-to-speech pipeline for the vLLM-Omni diffusion engine.

    Wraps the full Kokoro-82M inference flow (BERT + prosody prediction +
    ISTFTNet vocoder) into a single ``forward()`` call that accepts an
    ``OmniDiffusionRequest`` and returns a ``DiffusionOutput`` with audio.
    """

    support_audio_output: ClassVar[bool] = True

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__()
        self.od_config = od_config
        self.device = get_local_device()
        self.model_path = od_config.model

        # Resolve model path (HF hub ID -> local cache).
        if not os.path.isdir(self.model_path):
            from huggingface_hub import snapshot_download

            self.model_path = snapshot_download(self.model_path)

        # Load config.json.
        config_path = os.path.join(self.model_path, "config.json")
        with open(config_path) as f:
            config = json.load(f)

        self.vocab: dict[str, int] = config["vocab"]
        self.sample_rate = _SAMPLE_RATE

        # Build submodules.
        self.bert = CustomAlbert(AlbertConfig(vocab_size=config["n_token"], **config["plbert"]))
        self.bert_encoder = nn.Linear(self.bert.config.hidden_size, config["hidden_dim"])
        self.context_length: int = self.bert.config.max_position_embeddings

        self.predictor = ProsodyPredictor(
            style_dim=config["style_dim"],
            d_hid=config["hidden_dim"],
            nlayers=config["n_layer"],
            max_dur=config["max_dur"],
            dropout=config["dropout"],
        )
        self.text_encoder = TextEncoder(
            channels=config["hidden_dim"],
            kernel_size=config["text_encoder_kernel_size"],
            depth=config["n_layer"],
            n_symbols=config["n_token"],
        )
        self.decoder = Decoder(
            dim_in=config["hidden_dim"],
            style_dim=config["style_dim"],
            dim_out=config["n_mels"],
            **config["istftnet"],
        )

        # Lazily initialised G2P (so the import cost is deferred).
        self._g2p = None
        self._g2p_lang: str | None = None

        # Voice embedding cache: voice_name -> FloatTensor[256].
        self._voices: dict[str, torch.Tensor] = {}

        # Detect available model weight file.
        self._weight_filename = self._find_weight_file()

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def _find_weight_file(self) -> str:
        """Find the .pth weight file in the model directory."""
        for fname in os.listdir(self.model_path):
            if fname.endswith(".pth"):
                return fname
        raise FileNotFoundError(f"No .pth weight file found in {self.model_path}")

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights from model directory.

        Kokoro uses a custom ``.pth`` format with nested state dicts,
        not standard HF safetensors.  We consume the iterator (loader
        contract) and then load from the ``.pth`` file directly.
        """
        # Consume the iterator (required by the loader contract).
        for _ in weights:
            pass

        device = self.device
        weight_path = os.path.join(self.model_path, self._weight_filename)
        state_dicts = torch.load(weight_path, map_location=device, mmap=True, weights_only=True)

        for key in list(state_dicts.keys()):
            state_dict = state_dicts.pop(key)
            if not hasattr(self, key):
                logger.warning("Skipping unknown weight key: %s", key)
                del state_dict
                continue
            try:
                getattr(self, key).load_state_dict(state_dict)
            except RuntimeError:
                # Handle potential module prefix mismatch.
                cleaned = {k.removeprefix("module."): v for k, v in state_dict.items()}
                getattr(self, key).load_state_dict(cleaned, strict=False)
            del state_dict

        del state_dicts

        # Bake weight_norm parametrizations: remove the stored weight_orig
        # and weight_g, keeping only the computed weight tensor.  Saves ~40%
        # of conv parameter memory at inference time.
        from torch.nn.utils import parametrize

        for module in self.modules():
            if parametrize.is_parametrized(module, "weight"):
                parametrize.remove_parametrizations(module, "weight")

        self.eval()
        logger.info("Kokoro pipeline loaded on %s", device)

        # Return None to skip the loader's strict weight check.  The check
        # captures parameter names before loading (with weight_norm
        # parametrization names like ``parametrizations.weight.original0``),
        # but after baking those names no longer exist, causing a false
        # "not loaded" error.
        return None

    # ------------------------------------------------------------------
    # G2P (grapheme-to-phoneme)
    # ------------------------------------------------------------------

    def _ensure_g2p(self, lang_code: str = "a"):
        """Lazily initialise the G2P module for the given language."""
        lang_code = _LANG_ALIASES.get(lang_code.lower(), lang_code.lower())
        if self._g2p is not None and self._g2p_lang == lang_code:
            return

        if lang_code in ("a", "b"):
            from misaki import en, espeak

            try:
                fallback = espeak.EspeakFallback(british=lang_code == "b")
            except Exception:
                logger.warning("espeak-ng not available; OOD words will be skipped")
                fallback = None
            self._g2p = en.G2P(trf=False, british=lang_code == "b", fallback=fallback, unk="")
        elif lang_code == "j":
            from misaki import ja

            self._g2p = ja.JAG2P()
        elif lang_code == "z":
            from misaki import zh

            self._g2p = zh.ZHG2P()
        else:
            from misaki import espeak

            _ESPEAK_LANGS = {"e": "es", "f": "fr-fr", "h": "hi", "i": "it", "p": "pt-br"}
            language = _ESPEAK_LANGS.get(lang_code, lang_code)
            self._g2p = espeak.EspeakG2P(language=language)

        self._g2p_lang = lang_code

    def _text_to_phonemes(self, text: str, lang_code: str = "a") -> str:
        """Convert text to IPA phoneme string via misaki G2P."""
        self._ensure_g2p(lang_code)

        if self._g2p_lang in ("a", "b"):
            # English G2P returns (text, tokens).
            _, tokens = self._g2p(text)
            ps = "".join(t.phonemes + (" " if t.whitespace else "") for t in tokens).strip()
        else:
            # Non-English G2P returns (phonemes, _).
            ps, _ = self._g2p(text)

        return ps

    # ------------------------------------------------------------------
    # Voice loading
    # ------------------------------------------------------------------

    def _load_voice(self, voice_name: str) -> torch.Tensor:
        """Load a voice embedding tensor, caching the result."""
        if voice_name in self._voices:
            return self._voices[voice_name]

        voice_path = os.path.join(self.model_path, "voices", f"{voice_name}.pt")
        if not os.path.isfile(voice_path):
            # Try downloading from the HF repo.
            try:
                from huggingface_hub import hf_hub_download

                repo_id = self.od_config.model
                voice_path = hf_hub_download(repo_id=repo_id, filename=f"voices/{voice_name}.pt")
            except Exception:
                logger.warning(
                    "Voice '%s' not found; falling back to '%s'",
                    voice_name,
                    _DEFAULT_VOICE,
                )
                if voice_name != _DEFAULT_VOICE:
                    return self._load_voice(_DEFAULT_VOICE)
                raise FileNotFoundError(f"Default voice '{_DEFAULT_VOICE}' not found")

        pack = torch.load(voice_path, map_location=self.device, weights_only=True)
        self._voices[voice_name] = pack
        return pack

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def _generate(
        self,
        phonemes: str,
        ref_s: torch.Tensor,
        speed: float = 1.0,
    ) -> torch.Tensor:
        """Run the Kokoro model: phonemes + style -> waveform."""
        device = self.device

        # Map phonemes to token IDs.
        input_ids = [i for i in (self.vocab.get(p) for p in phonemes) if i is not None]
        if not input_ids:
            return torch.zeros(0, device=device)

        assert len(input_ids) + 2 <= self.context_length, (
            f"Input too long: {len(input_ids) + 2} > {self.context_length}"
        )

        input_ids = torch.LongTensor([[0, *input_ids, 0]]).to(device)
        ref_s = ref_s.to(device)
        if ref_s.ndim == 1:
            ref_s = ref_s.unsqueeze(0)

        input_lengths = torch.full(
            (input_ids.shape[0],),
            input_ids.shape[-1],
            device=device,
            dtype=torch.long,
        )

        text_mask = torch.arange(input_lengths.max(), device=device).unsqueeze(0).expand(input_lengths.shape[0], -1)
        text_mask = torch.gt(text_mask + 1, input_lengths.unsqueeze(1))

        bert_dur = self.bert(input_ids, attention_mask=(~text_mask).int())
        d_en = self.bert_encoder(bert_dur).transpose(-1, -2)

        s = ref_s[:, 128:]
        d = self.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        x, _ = self.predictor.lstm(d)
        duration = self.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1) / speed
        pred_dur = torch.round(duration).clamp(min=1).long().squeeze()
        if pred_dur.ndim == 0:
            pred_dur = pred_dur.unsqueeze(0)

        indices = torch.repeat_interleave(torch.arange(input_ids.shape[1], device=device), pred_dur)
        pred_aln_trg = torch.zeros((input_ids.shape[1], indices.shape[0]), device=device)
        pred_aln_trg[indices, torch.arange(indices.shape[0], device=device)] = 1
        pred_aln_trg = pred_aln_trg.unsqueeze(0)

        en = d.transpose(-1, -2) @ pred_aln_trg
        F0_pred, N_pred = self.predictor.F0Ntrain(en, s)

        t_en = self.text_encoder(input_ids, input_lengths, text_mask)
        asr = t_en @ pred_aln_trg

        audio = self.decoder(asr, F0_pred, N_pred, ref_s[:, :128]).squeeze()
        return audio

    # ------------------------------------------------------------------
    # Pipeline interface
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def forward(self, req: OmniDiffusionRequest) -> DiffusionOutput:
        """Generate speech audio from text.

        Args:
            req: Diffusion request containing text prompt(s).

        Returns:
            DiffusionOutput with audio tensor in .output
        """
        # Extract text from request.
        prompt = req.prompts[0] if req.prompts else ""
        if isinstance(prompt, dict):
            text = prompt.get("input", prompt.get("text", prompt.get("prompt", str(prompt))))
        else:
            text = str(prompt)

        if not text or text == "dummy run":
            # Handle warm-up / empty requests gracefully.
            dummy = torch.zeros(1, _SAMPLE_RATE, device=self.device)
            return DiffusionOutput(output=dummy)

        # Determine voice.
        voice_name = _DEFAULT_VOICE
        if isinstance(prompt, dict):
            voice_name = prompt.get("voice", voice_name)
        if hasattr(req.sampling_params, "extra_args") and req.sampling_params.extra_args:
            voice_name = req.sampling_params.extra_args.get("voice", voice_name)

        # Determine speed.
        speed = 1.0
        if hasattr(req.sampling_params, "extra_args") and req.sampling_params.extra_args:
            speed = float(req.sampling_params.extra_args.get("speed", speed))

        # Infer language from voice name prefix (first character).
        lang_code = voice_name[0] if voice_name else "a"

        # G2P: text -> phonemes.
        try:
            phonemes = self._text_to_phonemes(text, lang_code)
        except Exception as e:
            logger.error("G2P failed: %s", e)
            return DiffusionOutput(error=f"G2P conversion failed: {e}")

        if not phonemes:
            return DiffusionOutput(error="G2P produced empty phoneme string")

        # Truncate if too long.
        if len(phonemes) > _MAX_PHONEME_LEN:
            logger.warning("Truncating phonemes from %d to %d", len(phonemes), _MAX_PHONEME_LEN)
            phonemes = phonemes[:_MAX_PHONEME_LEN]

        # Load voice embedding.
        try:
            voice_pack = self._load_voice(voice_name)
        except Exception as e:
            return DiffusionOutput(error=f"Failed to load voice '{voice_name}': {e}")

        # Select the style vector for this phoneme length.
        # Voice packs are tensors of shape [max_len, 256], indexed by phoneme count.
        ref_s = voice_pack[min(len(phonemes) - 1, voice_pack.shape[0] - 1)]

        # Generate audio.
        audio = self._generate(phonemes, ref_s, speed)

        if audio.numel() == 0:
            return DiffusionOutput(error="Model produced empty audio")

        return DiffusionOutput(output=audio)
