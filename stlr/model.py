from __future__ import annotations

from einops import rearrange, einsum
import torch.nn as nn
import torch
import torch.nn.functional as F

from typing import Annotated
from dataclasses import dataclass


N_EMBED = 1024
N_ATTN_CONTEXT = 1024
N_HIDDEN_DIM = 1024
N_VOCAB = 40478


@dataclass
class STLRConfig:
    n_embed: int = N_EMBED
    n_attn_context: int = N_ATTN_CONTEXT
    n_hidden_dim: int = N_HIDDEN_DIM
    attn_cover: int = 2


# https://arxiv.org/abs/1706.03762
# https://github.com/lucidrains/x-transformers (MIT License)
class Attention(nn.Module):
    dim: int
    value_dim: int

    k_dim_head: int
    v_dim_head: int
    num_heads: int

    w_q: nn.Linear
    w_k: nn.Linear
    w_v: nn.Linear
    w_out: nn.Linear

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        k_dim_head: int = 64,
        v_dim_head: int = 64,
        value_dim: int | None = None,
    ) -> None:
        super().__init__()

        # calculate the dimensions
        if value_dim is None:
            value_dim = dim

        q_dim = k_dim = k_dim_head * num_heads
        v_dim = v_dim_head * num_heads

        self.dim = dim
        self.value_dim = value_dim

        self.k_dim_head = k_dim_head
        self.v_dim_head = v_dim_head

        self.num_heads = num_heads

        self.w_q = nn.Linear(dim, q_dim, bias=False)
        self.w_k = nn.Linear(dim, k_dim, bias=False)
        self.w_v = nn.Linear(value_dim, v_dim, bias=False)
        self.w_out = nn.Linear(v_dim, value_dim, bias=False)

    def forward(
        self,
        Q: Annotated[torch.Tensor, 'B', 'T', 'K'],
        K: Annotated[torch.Tensor, 'B', 'T', 'K'],
        V: Annotated[torch.Tensor, 'B', 'T', 'V'],
    ) -> Annotated[torch.Tensor, 'B', 'T', 'V']:
        q_i = rearrange(self.w_q(Q), 'B T (H k) -> B H T k', h=self.num_heads)
        k_i = rearrange(self.w_k(K), 'B T (H k) -> B H T k', h=self.num_heads)
        v_i = rearrange(self.w_k(V), 'B T (H k) -> B H T v', h=self.num_heads)

        # get dotporduct similarity
        s_qk = einsum('B H i k, B H j k -> B H i j', q_i, k_i) / (q_i.shape[-1]) ** 0.5
        # why softmax over the key dim?
        attn: Annotated[torch.Tensor, 'B', 'H', 'T', 'T'] = F.softmax(s_qk, dim=-1)

        vals = einsum('B H T i, B H i v -> B H T v', attn, v_i)
        out = self.w_out(rearrange(vals, 'B H T v ->  B T (H v)'))
        return out


# https://arxiv.org/abs/1706.03762
class AttentionLayer(nn.Module):
    attention: Attention
    norm_attn: nn.LayerNorm
    feed_forward: nn.Linear
    norm_ff: nn.LayerNorm

    in_features: int
    out_features: int
    skip: bool

    def __init__(
        self, in_features: int = 1024, out_features: int = 512, skip: bool = False
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.skip = skip
        if skip:
            assert in_features == out_features

        self.attention = Attention(dim=in_features)
        self.norm_attn = nn.LayerNorm(out_features)
        self.feed_forward = nn.Linear(out_features, out_features, bias=True)
        self.norm_ff = nn.LayerNorm(out_features)

    def forward(
        self, x: Annotated[torch.Tensor, 'B', 'T', 'I']
    ) -> Annotated[torch.Tensor, 'B', 'T', 'O']:
        # TODO fix this
        attn = self.attention(x, x, x)
        if self.skip:
            attn = x + attn
        x = self.norm_attn(attn)
        return self.norm_ff(x + self.feed_forward(x))


class ShortTransformer(nn.Module):
    n_layers: int
    layers: nn.ModuleList

    feature_dim: int
    embed_dim: int

    def __init__(
        self, n_layers: int = 6, embed_dim: int = 1024, feature_dim: int = 32
    ) -> None:
        super().__init__()

        self.feature_dim = feature_dim
        self.embed_dim = embed_dim

        assert n_layers >= 2
        self.n_layers = n_layers

        first_layer = AttentionLayer(embed_dim, embed_dim, skip=True)
        self.layers = nn.ModuleList([first_layer])

        prev_dim = embed_dim
        for layer in range(1, n_layers):
            out_dim = embed_dim // 2 if layer != n_layers - 1 else feature_dim
            self.layers.append(AttentionLayer(prev_dim, out_dim, skip=False))
            prev_dim = out_dim

    def forward(
        self, x: Annotated[torch.Tensor, 'B', 'T', 'E']
    ) -> Annotated[torch.Tensor, 'B', 'T', 'F']:
        for layer in self.layers:
            x = layer(x)
        return x


class RNNLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        x: Annotated[torch.Tensor, 'B', 'H'],
        state: Annotated[torch.Tensor, 'B', 'H'],
    ) -> tuple[Annotated[torch.Tensor, 'B', 'H'], Annotated[torch.Tensor, 'B', 'H']]:
        return x, state


class LongRNN(nn.Module):
    feature_dim: int
    context_dim: int
    hidden_dim: int
    vocab_size: int

    n_layers: int
    layers: nn.ModuleList

    def __init__(
        self,
        feature_dim: int = 32,
        context_dim: int = 256,
        n_layers: int = 8,
        hidden_dim: int = 1024,
        vocab_size: int = N_VOCAB,
    ) -> None:
        super().__init__()

        self.feature_dim = feature_dim
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        self.n_layers = n_layers

        self.to_hidden = nn.Linear(feature_dim * context_dim, hidden_dim, bias=False)
        self.in_norm = nn.LayerNorm(hidden_dim)

        self.layers = nn.ModuleList([RNNLayer() for _ in range(n_layers)])

        self.to_dist = nn.Linear(hidden_dim, vocab_size, bias=True)

    def forward(
        self,
        x: Annotated[torch.Tensor, 'B', 'T', 'F'],
        state: Annotated[torch.Tensor, 'B', 'H', 'L'],
    ) -> tuple[
        Annotated[torch.Tensor, 'B', 'V'], Annotated[torch.Tensor, 'B', 'H', 'L']
    ]:
        assert x.shape[-1] == self.feature_dim
        assert x.shape[-2] == self.context_dim

        hidden = self.to_hidden(rearrange(x, 'B T F -> B (T F)'))
        hidden = self.in_norm(hidden)

        for id, layer in enumerate(self.layers):
            hidden, state[:, :, id] = layer(hidden, state[:, :, id])

        dist = self.to_dist(hidden)
        dist = F.softmax(dist, dim=-1)
        return dist, state


class STLR(nn.Module):
    config: STLRConfig

    def __init__(self, config: STLRConfig | None = None) -> None:
        if config is None:
            config = STLRConfig()
        self.config = config
