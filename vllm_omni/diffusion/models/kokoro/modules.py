# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/yl4579/StyleTTS2/blob/main/models.py
# and https://github.com/hexgrad/kokoro

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm

from vllm_omni.diffusion.models.kokoro.istftnet import AdainResBlk1d


class LinearNorm(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, bias: bool = True, w_init_gain: str = "linear"):
        super().__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)
        nn.init.xavier_uniform_(self.linear_layer.weight, gain=nn.init.calculate_gain(w_init_gain))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)


class LayerNorm(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)


class TextEncoder(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        depth: int,
        n_symbols: int,
        actv: nn.Module = nn.LeakyReLU(0.2),
    ):
        super().__init__()
        self.embedding = nn.Embedding(n_symbols, channels)
        padding = (kernel_size - 1) // 2
        self.cnn = nn.ModuleList()
        for _ in range(depth):
            self.cnn.append(
                nn.Sequential(
                    weight_norm(nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding)),
                    LayerNorm(channels),
                    actv,
                    nn.Dropout(0.2),
                )
            )
        self.lstm = nn.LSTM(channels, channels // 2, 1, batch_first=True, bidirectional=True)

    def forward(
        self,
        x: torch.Tensor,
        input_lengths: torch.Tensor,
        m: torch.Tensor,
    ) -> torch.Tensor:
        x = self.embedding(x)  # [B, T, emb]
        x = x.transpose(1, 2)  # [B, emb, T]
        m = m.unsqueeze(1)
        x.masked_fill_(m, 0.0)
        for c in self.cnn:
            x = c(x)
            x.masked_fill_(m, 0.0)
        x = x.transpose(1, 2)  # [B, T, chn]
        lengths = input_lengths if input_lengths.device == torch.device("cpu") else input_lengths.to("cpu")
        x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        x = x.transpose(-1, -2)
        x_pad = torch.zeros([x.shape[0], x.shape[1], m.shape[-1]], device=x.device)
        x_pad[:, :, : x.shape[-1]] = x
        x = x_pad
        x.masked_fill_(m, 0.0)
        return x


class AdaLayerNorm(nn.Module):
    def __init__(self, style_dim: int, channels: int, eps: float = 1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.fc = nn.Linear(style_dim, channels * 2)

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        x = x.transpose(-1, -2)
        x = x.transpose(1, -1)
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        gamma, beta = gamma.transpose(1, -1), beta.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), eps=self.eps)
        x = (1 + gamma) * x + beta
        return x.transpose(1, -1).transpose(-1, -2)


class DurationEncoder(nn.Module):
    def __init__(self, sty_dim: int, d_model: int, nlayers: int, dropout: float = 0.1):
        super().__init__()
        self.lstms = nn.ModuleList()
        for _ in range(nlayers):
            self.lstms.append(
                nn.LSTM(d_model + sty_dim, d_model // 2, num_layers=1, batch_first=True, bidirectional=True)
            )
            self.lstms.append(AdaLayerNorm(sty_dim, d_model))
        self.dropout = dropout
        self.d_model = d_model
        self.sty_dim = sty_dim

    def forward(
        self,
        x: torch.Tensor,
        style: torch.Tensor,
        text_lengths: torch.Tensor,
        m: torch.Tensor,
    ) -> torch.Tensor:
        masks = m
        x = x.permute(2, 0, 1)
        s = style.expand(x.shape[0], x.shape[1], -1)
        x = torch.cat([x, s], axis=-1)
        x.masked_fill_(masks.unsqueeze(-1).transpose(0, 1), 0.0)
        x = x.transpose(0, 1)
        x = x.transpose(-1, -2)
        for block in self.lstms:
            if isinstance(block, AdaLayerNorm):
                x = block(x.transpose(-1, -2), style).transpose(-1, -2)
                x = torch.cat([x, s.permute(1, 2, 0)], axis=1)
                x.masked_fill_(masks.unsqueeze(-1).transpose(-1, -2), 0.0)
            else:
                lengths = text_lengths if text_lengths.device == torch.device("cpu") else text_lengths.to("cpu")
                x = x.transpose(-1, -2)
                x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
                block.flatten_parameters()
                x, _ = block(x)
                x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
                x = F.dropout(x, p=self.dropout, training=False)
                x = x.transpose(-1, -2)
                x_pad = torch.zeros([x.shape[0], x.shape[1], m.shape[-1]], device=x.device)
                x_pad[:, :, : x.shape[-1]] = x
                x = x_pad
        return x.transpose(-1, -2)


class ProsodyPredictor(nn.Module):
    def __init__(
        self,
        style_dim: int,
        d_hid: int,
        nlayers: int,
        max_dur: int = 50,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.text_encoder = DurationEncoder(sty_dim=style_dim, d_model=d_hid, nlayers=nlayers, dropout=dropout)
        self.lstm = nn.LSTM(d_hid + style_dim, d_hid // 2, 1, batch_first=True, bidirectional=True)
        self.duration_proj = LinearNorm(d_hid, max_dur)
        self.shared = nn.LSTM(d_hid + style_dim, d_hid // 2, 1, batch_first=True, bidirectional=True)
        self.F0 = nn.ModuleList()
        self.F0.append(AdainResBlk1d(d_hid, d_hid, style_dim, dropout_p=dropout))
        self.F0.append(AdainResBlk1d(d_hid, d_hid // 2, style_dim, upsample=True, dropout_p=dropout))
        self.F0.append(AdainResBlk1d(d_hid // 2, d_hid // 2, style_dim, dropout_p=dropout))
        self.N = nn.ModuleList()
        self.N.append(AdainResBlk1d(d_hid, d_hid, style_dim, dropout_p=dropout))
        self.N.append(AdainResBlk1d(d_hid, d_hid // 2, style_dim, upsample=True, dropout_p=dropout))
        self.N.append(AdainResBlk1d(d_hid // 2, d_hid // 2, style_dim, dropout_p=dropout))
        self.F0_proj = nn.Conv1d(d_hid // 2, 1, 1, 1, 0)
        self.N_proj = nn.Conv1d(d_hid // 2, 1, 1, 1, 0)

    def F0Ntrain(self, x: torch.Tensor, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x, _ = self.shared(x.transpose(-1, -2))
        F0 = x.transpose(-1, -2)
        for block in self.F0:
            F0 = block(F0, s)
        F0 = self.F0_proj(F0)
        N = x.transpose(-1, -2)
        for block in self.N:
            N = block(N, s)
        N = self.N_proj(N)
        return F0.squeeze(1), N.squeeze(1)


# ---------------------------------------------------------------------------
# Standalone ALBERT (PL-BERT) -- replaces transformers.AlbertModel to avoid
# pulling in the entire HuggingFace transformers library (~150-300 MB RSS).
# ALBERT shares all transformer layer weights, so 12 "layers" use one set
# of parameters.
#
# Module naming mirrors HuggingFace's AlbertModel exactly so that
# state_dict keys from the upstream .pth checkpoint load without remapping.
# ---------------------------------------------------------------------------


class _AlbertEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.embedding_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.embedding_size)
        self.token_type_embeddings = nn.Embedding(2, config.embedding_size)
        self.LayerNorm = nn.LayerNorm(config.embedding_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).unsqueeze(0), persistent=False
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_length = input_ids.shape[1]
        position_ids = self.position_ids[:, :seq_length]
        embeddings = self.word_embeddings(input_ids)
        embeddings = embeddings + self.position_embeddings(position_ids)
        embeddings = embeddings + self.token_type_embeddings(torch.zeros_like(input_ids))
        return self.dropout(self.LayerNorm(embeddings))


class _AlbertAttention(nn.Module):
    """Self-attention with post-LN residual (matches HF albert attention key names)."""

    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.head_dim
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, _ = hidden_states.shape
        q = self.query(hidden_states).view(B, T, self.num_attention_heads, self.head_dim).transpose(1, 2)
        k = self.key(hidden_states).view(B, T, self.num_attention_heads, self.head_dim).transpose(1, 2)
        v = self.value(hidden_states).view(B, T, self.num_attention_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            scores = scores + attention_mask
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(B, T, self.all_head_size)
        projected = self.dense(context)
        return self.LayerNorm(self.dropout(projected) + hidden_states)


class _AlbertLayer(nn.Module):
    """Single ALBERT transformer layer.

    Attribute names match HF: ``attention``, ``ffn``, ``ffn_output``,
    ``full_layer_layer_norm`` so that state_dict keys align exactly.
    """

    def __init__(self, config):
        super().__init__()
        self.attention = _AlbertAttention(config)
        self.ffn = nn.Linear(config.hidden_size, config.intermediate_size)
        self.ffn_output = nn.Linear(config.intermediate_size, config.hidden_size)
        self.full_layer_layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        attention_output = self.attention(hidden_states, attention_mask)
        ffn_out = F.gelu(self.ffn(attention_output))
        ffn_out = self.ffn_output(ffn_out)
        return self.full_layer_layer_norm(self.dropout(ffn_out) + attention_output)


class _AlbertLayerGroup(nn.Module):
    """Layer group containing one shared layer (matches HF nesting)."""

    def __init__(self, config):
        super().__init__()
        self.albert_layers = nn.ModuleList([_AlbertLayer(config)])


class _AlbertEncoder(nn.Module):
    """ALBERT encoder: optional embedding projection + shared layer groups.

    Key names: ``embedding_hidden_mapping_in``, ``albert_layer_groups``.
    """

    def __init__(self, config):
        super().__init__()
        if config.embedding_size != config.hidden_size:
            self.embedding_hidden_mapping_in = nn.Linear(config.embedding_size, config.hidden_size)
        else:
            self.embedding_hidden_mapping_in = None
        self.albert_layer_groups = nn.ModuleList([_AlbertLayerGroup(config)])
        self.num_hidden_layers = config.num_hidden_layers

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        if self.embedding_hidden_mapping_in is not None:
            hidden_states = self.embedding_hidden_mapping_in(hidden_states)
        layer = self.albert_layer_groups[0].albert_layers[0]
        for _ in range(self.num_hidden_layers):
            hidden_states = layer(hidden_states, attention_mask)
        return hidden_states


class AlbertConfig:
    """Minimal ALBERT config matching HF's AlbertConfig fields used by Kokoro."""

    def __init__(
        self,
        vocab_size: int = 30000,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        intermediate_size: int = 2048,
        max_position_embeddings: int = 512,
        num_hidden_layers: int = 12,
        embedding_size: int | None = None,
        hidden_dropout_prob: float | None = None,
        dropout: float = 0.1,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_hidden_layers = num_hidden_layers
        self.embedding_size = embedding_size if embedding_size is not None else 128
        self.hidden_dropout_prob = hidden_dropout_prob if hidden_dropout_prob is not None else dropout


class CustomAlbert(nn.Module):
    """Standalone ALBERT encoder that returns ``last_hidden_state`` directly.

    Module structure mirrors ``transformers.AlbertModel`` so that
    checkpoint state_dict keys produced by HuggingFace load without
    any key remapping.  Unused sub-modules (``pooler``) are omitted;
    their keys are silently skipped during ``load_state_dict``.
    """

    def __init__(self, config: AlbertConfig):
        super().__init__()
        self.config = config
        self.embeddings = _AlbertEmbeddings(config)
        self.encoder = _AlbertEncoder(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = self.embeddings(input_ids)
        # Expand attention_mask to [B, 1, 1, T] for broadcast with scores.
        extended_mask = None
        if attention_mask is not None:
            extended_mask = (1.0 - attention_mask[:, None, None, :].float()) * torch.finfo(hidden_states.dtype).min
        return self.encoder(hidden_states, extended_mask)
