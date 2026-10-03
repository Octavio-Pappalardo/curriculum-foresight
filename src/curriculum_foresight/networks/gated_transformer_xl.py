from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen.initializers import constant

from curriculum_foresight.networks.relative_attention import RelativeMultiHeadAttention


class GRUGate(nn.Module):
    d_input: int
    bg: float = 0.0

    @nn.compact
    def __call__(self, x: jax.Array, y: jax.Array) -> jax.Array:
        r = jax.nn.sigmoid(nn.Dense(self.d_input, use_bias=False)(y) + nn.Dense(self.d_input, use_bias=False)(x))
        z = jax.nn.sigmoid(
            nn.Dense(self.d_input, use_bias=False)(y)
            + nn.Dense(self.d_input, use_bias=False)(x)
            - self.param("gating_bias", constant(self.bg), (self.d_input,))
        )
        h = jnp.tanh(nn.Dense(self.d_input, use_bias=False)(y) + nn.Dense(self.d_input, use_bias=False)(r * x))
        return (1.0 - z) * x + z * h


class GatedTransformerBlock(nn.Module):
    num_heads: int
    out_features: int
    qkv_features: int
    gating_bias: float = 0.0

    def setup(self):
        self.attention = RelativeMultiHeadAttention(
            num_heads=self.num_heads, qkv_features=self.qkv_features, out_features=self.out_features
        )
        self.ln1 = nn.LayerNorm()
        self.dense1 = nn.Dense(self.out_features)
        self.dense2 = nn.Dense(self.out_features)
        self.ln2 = nn.LayerNorm()
        self.gate1 = GRUGate(self.out_features, self.gating_bias)
        self.gate2 = GRUGate(self.out_features, self.gating_bias)

    def __call__(
        self, *, values_keys: jax.Array, queries: jax.Array, pos_embed: jax.Array, mask: jax.Array
    ) -> jax.Array:
        values_keys_n = self.ln1(values_keys)
        queries_n = self.ln1(queries)
        attention = self.attention(inputs_kv=values_keys_n, inputs_q=queries_n, mask=mask, pos_embed=pos_embed)
        out_attention = self.gate1(queries, jax.nn.relu(attention))

        out_attention_n = self.ln2(out_attention)
        out = self.dense1(out_attention_n)
        out = nn.gelu(out)
        out = self.dense2(out)
        return self.gate2(out_attention, jax.nn.relu(out))


class PositionalEmbedding(nn.Module):
    dim_emb: int

    def setup(self):
        self.inv_freq = 1.0 / (10000 ** (jnp.arange(0.0, self.dim_emb, 2.0) / self.dim_emb))

    def __call__(self, pos_seq: jax.Array) -> jax.Array:
        sinusoid_inp = jnp.outer(pos_seq, self.inv_freq)
        return jnp.concatenate([jnp.sin(sinusoid_inp), jnp.cos(sinusoid_inp)], axis=-1)


class GatedTransformerXL(nn.Module):
    hidden_dim: int
    num_heads: int
    qkv_features: int
    num_layers: int
    gating_bias: float = 0.0

    def setup(self):
        self.encoder = nn.Dense(self.hidden_dim)
        self.layers = [
            GatedTransformerBlock(
                num_heads=self.num_heads,
                qkv_features=self.qkv_features,
                out_features=self.hidden_dim,
                gating_bias=self.gating_bias,
            )
            for _ in range(self.num_layers)
        ]
        self.pos_emb = PositionalEmbedding(self.hidden_dim)

    def __call__(self, memories: jax.Array, inputs: jax.Array, mask: jax.Array) -> jax.Array:
        encoded = self.encoder(inputs)
        memory_length = memories.shape[-3]
        pos_embed = self.pos_emb(jnp.arange(memory_length + 1, 0, -1))

        x = encoded
        for layer_idx, layer in enumerate(self.layers):
            memory = jnp.concatenate([memories[:, :, layer_idx], x[:, None]], axis=-2)
            x = layer(values_keys=memory, pos_embed=pos_embed, queries=x[:, None], mask=mask)
            x = x.squeeze(axis=1)
        return x

    def forward_step(self, memories: jax.Array, inputs: jax.Array, mask: jax.Array) -> tuple[jax.Array, jax.Array]:
        encoded = self.encoder(inputs)
        memory_length = memories.shape[-3]
        pos_embed = self.pos_emb(jnp.arange(memory_length + 1, 0, -1))

        memory_out = jnp.zeros((encoded.shape[0], self.num_layers, encoded.shape[-1]), dtype=encoded.dtype)
        x = encoded
        for layer_idx, layer in enumerate(self.layers):
            memory_out = memory_out.at[:, layer_idx].set(x)
            memory = jnp.concatenate([memories[:, :, layer_idx], x[:, None]], axis=-2)
            x = layer(values_keys=memory, pos_embed=pos_embed, queries=x[:, None], mask=mask)
            x = x.squeeze(axis=1)
        return x, memory_out

    def forward_sequence(self, memories: jax.Array, inputs: jax.Array, mask: jax.Array) -> jax.Array:
        encoded = self.encoder(inputs)
        key_length = encoded.shape[-2] + memories.shape[-3]
        pos_embed = self.pos_emb(jnp.arange(key_length, 0, -1))

        x = encoded
        for layer_idx, layer in enumerate(self.layers):
            memory = jnp.concatenate([jnp.take(memories, layer_idx, axis=-2), x], axis=-2)
            x = layer(values_keys=memory, pos_embed=pos_embed, queries=x, mask=mask)
        return x
