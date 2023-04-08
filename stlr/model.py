from __future__ import annotations

from einops import rearrange, einsum
import torch.nn as nn
import torch
import torch.nn.functional as F

from typing import Annotated
from dataclasses import dataclass


EMBED_DIM = 1024
VOCAB_DIM = 40478  # GPT2 vocab size

TRANSFORMER_N_LAYERS = 8
TRANSFOMER_OUT_FEATURES = 64

RNN_HIDDEN_DIM = 4096
RNN_N_LAYERS = 12

SHORT_CONTEXT = 256
MAX_CONTEXT = 32768


@dataclass(frozen=True)
class STLRConfig:
    short_context: int = SHORT_CONTEXT
    max_context: int = MAX_CONTEXT

    embed_dim: int = EMBED_DIM
    vocab_dim: int = VOCAB_DIM

    transformer_n_layers: int = TRANSFORMER_N_LAYERS
    transformer_out_features: int = TRANSFOMER_OUT_FEATURES

    rnn_hidden_dim: int = RNN_HIDDEN_DIM
    rnn_n_layers: int = RNN_N_LAYERS


DEFAULT_CONFIG = STLRConfig()


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
        Q: Annotated[torch.Tensor, '*B', 'T', 'K'],
        K: Annotated[torch.Tensor, '*B', 'T', 'K'],
        V: Annotated[torch.Tensor, '*B', 'T', 'V'],
    ) -> Annotated[torch.Tensor, '*B', 'T', 'V']:
        q_i = rearrange(self.w_q(Q), '... B T (H k) -> ... B H T k', h=self.num_heads)
        k_i = rearrange(self.w_k(K), '... B T (H k) -> ... B H T k', h=self.num_heads)
        v_i = rearrange(self.w_k(V), '... B T (H k) -> ... B H T v', h=self.num_heads)

        # get dotporduct similarity
        s_qk = einsum('... B H i k, ... B H j k -> ... B H i j', q_i, k_i)
        s_qk = s_qk / (q_i.shape[-1]) ** 0.5
        # why softmax over the key dim?
        attn: Annotated[torch.Tensor, '*B', 'H', 'T', 'T'] = F.softmax(s_qk, dim=-1)

        vals = einsum('... B H T i, ... B H i v -> ... B H T v', attn, v_i)
        out = self.w_out(rearrange(vals, '... B H T v ->  ... B T (H v)'))
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
        self, in_features: int = EMBED_DIM, out_features: int = 512, skip: bool = False
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
        self, x: Annotated[torch.Tensor, '*B', 'T', 'I']
    ) -> Annotated[torch.Tensor, '*B', 'T', 'O']:
        # TODO fix this
        attn = self.attention(x, x, x)
        if self.skip:
            attn = x + attn
        x = self.norm_attn(attn)
        return self.norm_ff(x + self.feed_forward(x))


class ShortContextTransformer(nn.Module):
    n_layers: int
    layers: nn.ModuleList

    feature_dim: int
    embed_dim: int

    def __init__(
        self,
        n_layers: int = TRANSFORMER_N_LAYERS,
        embed_dim: int = EMBED_DIM,
        feature_dim: int = TRANSFOMER_OUT_FEATURES,
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
            out_dim = max(out_dim, feature_dim)
            self.layers.append(AttentionLayer(prev_dim, out_dim, skip=False))
            prev_dim = out_dim

    def forward(
        self, x: Annotated[torch.Tensor, '*B', 'T', 'E']
    ) -> Annotated[torch.Tensor, '*B', 'T', 'F']:
        for layer in self.layers:
            x = layer(x)
        return x


class RNNLayer(nn.Module):
    hidden_dim: int

    def __init__(self, hidden_dim: int = RNN_HIDDEN_DIM) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim

        self.w_mix = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.u_mix = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.mix_norm = nn.LayerNorm(hidden_dim)

        self.w_out = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: Annotated[torch.Tensor, '*B', 'H'],
        state: Annotated[torch.Tensor, '*B', 'H'],
    ) -> tuple[Annotated[torch.Tensor, '*B', 'H'], Annotated[torch.Tensor, '*B', 'H']]:
        ds = self.w_mix(x) + self.u_mix(state)
        state = self.mix_norm(state + ds)

        out = self.w_out(state)
        out = self.out_norm(out)

        return out, state


class LongContextRNN(nn.Module):
    feature_dim: int
    context_dim: int
    hidden_dim: int
    vocab_dim: int

    n_layers: int
    layers: nn.ModuleList

    def __init__(
        self,
        feature_dim: int = TRANSFOMER_OUT_FEATURES,
        context_dim: int = SHORT_CONTEXT,
        n_layers: int = RNN_N_LAYERS,
        hidden_dim: int = RNN_HIDDEN_DIM,
        vocab_dim: int = VOCAB_DIM,
    ) -> None:
        super().__init__()

        self.feature_dim = feature_dim
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.vocab_dim = vocab_dim

        self.n_layers = n_layers

        self.to_hidden = nn.Linear(feature_dim * context_dim, hidden_dim, bias=False)
        self.in_norm = nn.LayerNorm(hidden_dim)

        self.layers = nn.ModuleList([RNNLayer() for _ in range(n_layers)])

        self.to_dist = nn.Linear(hidden_dim, vocab_dim, bias=True)

    def forward(
        self,
        x: Annotated[torch.Tensor, '*B', 'T', 'F'],
        state: Annotated[torch.Tensor, '*B', 'H', 'L'],
    ) -> tuple[
        Annotated[torch.Tensor, '*B', 'V'], Annotated[torch.Tensor, '*B', 'H', 'L']
    ]:
        assert x.shape[-1] == self.feature_dim
        assert x.shape[-2] == self.context_dim

        hidden = self.to_hidden(rearrange(x, '... B T F -> ... B (T F)'))
        hidden = self.in_norm(hidden)

        for id, layer in enumerate(self.layers):
            hidden, state[:, :, id, :] = layer(hidden, state[:, :, id, :])

        dist = self.to_dist(hidden)
        dist = F.softmax(dist, dim=-1)
        return dist, state


class STLR(nn.Module):
    config: STLRConfig

    short_transformer: ShortContextTransformer
    long_rnn: LongContextRNN

    def __init__(self, config: STLRConfig = DEFAULT_CONFIG) -> None:
        super().__init__()

        self.config = config

        self.short_transformer = ShortContextTransformer(
            n_layers=config.transformer_n_layers,
            embed_dim=config.embed_dim,
            feature_dim=config.transformer_out_features,
        )
        self.long_rnn = LongContextRNN(
            feature_dim=config.transformer_out_features,
            context_dim=config.short_context,
            n_layers=config.rnn_n_layers,
            hidden_dim=config.rnn_hidden_dim,
            vocab_dim=config.vocab_dim,
        )

    def forward(
        self,
        x: Annotated[torch.Tensor, '*B', 'M', 'E'],
    ) -> Annotated[torch.Tensor, '*B', 'V']:
        T = self.config.short_context
        B = x.shape[:-2]
        M = x.shape[-2]
        assert M <= self.config.max_context
        assert M % T == 0  # TODO left pad to multiple of T (independent across batch)

        x_batched = rearrange(x, '... B (N T) E -> ... B N T E', T=T)
        N = x_batched.shape[-2]

        # run short transformers in parallel
        x_rnn: Annotated[torch.Tensor, '*B', 'N', 'T', 'F']
        x_rnn = self.short_transformer(x_batched)

        # initialize state
        state_shape = (*B, self.long_rnn.hidden_dim, self.long_rnn.n_layers)
        state = torch.zeros(*state_shape)

        for i in range(N):
            dist, state = self.long_rnn(x_rnn[..., i, :, :], state)

        return dist


def main() -> int:
    n_params = sum(p.numel() for p in STLR().parameters() if p.requires_grad)
    print(f'Number of parameters: {n_params:,}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
