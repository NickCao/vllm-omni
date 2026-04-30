# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared sinusoidal position embedding for timestep conditioning.

Used by Qwen3-TTS, Qwen2.5-Omni, CosyVoice3, and Ming-Flash-Omni
for diffusion timestep encoding in DiT/CFM architectures.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusPositionEmbedding(nn.Module):
    """Sinusoidal position embedding for scalar timesteps.

    Maps scalar timestep values to ``dim``-dimensional embeddings using
    the standard log-spaced frequency formula from DDPM/DiT.

    Args:
        dim: Output embedding dimension (must be even).
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor, scale: float = 1000.0) -> torch.Tensor:
        """
        Args:
            x: (N,) scalar timesteps.
            scale: Frequency scaling factor.

        Returns:
            (N, dim) sinusoidal embeddings, cast to the input dtype.
        """
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb.to(x.dtype)
