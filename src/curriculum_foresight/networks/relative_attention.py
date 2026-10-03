from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen import initializers
from flax.linen.linear import default_kernel_init


roll_vmap = jax.vmap(jnp.roll, in_axes=(-2, 0, None), out_axes=-2)


def dot_product_attention_weights(
    query: jax.Array,
    key: jax.Array,
    r_pos_embed: jax.Array,
    r_r_bias: jax.Array,
    r_w_bias: jax.Array,
    *,
    mask: jax.Array | None = None,
) -> jax.Array:
    depth = query.shape[-1]
    attn_weights = jnp.einsum("...qhd,...khd->...hqk", query + r_w_bias, key)
    attn_weights_r = jnp.einsum("...qhd,khd->...hqk", query + r_r_bias, r_pos_embed)
    attn_weights_r = roll_vmap(attn_weights_r, jnp.arange(0, query.shape[-3]) - (query.shape[-3] - 1), -1)
    attn_weights = (attn_weights + attn_weights_r) / jnp.sqrt(depth)

    if mask is not None:
        attn_weights = jnp.where(mask, attn_weights, jnp.finfo(attn_weights.dtype).min)

    return jax.nn.softmax(attn_weights, axis=-1)


def dot_product_attention(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    r_pos_embed: jax.Array,
    r_r_bias: jax.Array,
    r_w_bias: jax.Array,
    *,
    mask: jax.Array | None = None,
) -> jax.Array:
    attn_weights = dot_product_attention_weights(query, key, r_pos_embed, r_r_bias, r_w_bias, mask=mask)
    return jnp.einsum("...hqk,...khd->...qhd", attn_weights, value)


class RelativeMultiHeadAttention(nn.Module):
    num_heads: int
    qkv_features: int | None = None
    out_features: int | None = None
    use_bias: bool = True
    kernel_init: Callable[..., Any] = default_kernel_init
    bias_init: Callable[..., Any] = initializers.zeros_init()
    attention_fn: Callable[..., jax.Array] = dot_product_attention

    @nn.compact
    def __call__(
        self, *, inputs_q: jax.Array, inputs_kv: jax.Array, pos_embed: jax.Array, mask: jax.Array | None = None
    ) -> jax.Array:
        features = self.out_features or inputs_q.shape[-1]
        qkv_features = self.qkv_features or inputs_q.shape[-1]
        if qkv_features % self.num_heads != 0:
            raise ValueError(
                "qkv_features must be divisible by num_heads. "
                f"Got qkv_features={qkv_features}, num_heads={self.num_heads}."
            )
        head_dim = qkv_features // self.num_heads

        dense = functools.partial(
            nn.DenseGeneral,
            axis=-1,
            features=(self.num_heads, head_dim),
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            use_bias=self.use_bias,
        )
        query = dense(name="query")(inputs_q)
        key = dense(name="key")(inputs_kv)
        value = dense(name="value")(inputs_kv)

        dense_relpos = functools.partial(
            nn.DenseGeneral, axis=-1, features=(self.num_heads, head_dim), kernel_init=self.kernel_init, use_bias=False
        )
        r_pos_embed = dense_relpos(name="pos_embed_mat")(pos_embed)
        r_r_bias = self.param("r_r_bias", self.bias_init, (self.num_heads, head_dim))
        r_w_bias = self.param("r_w_bias", self.bias_init, (self.num_heads, head_dim))

        x = self.attention_fn(query, key, value, r_pos_embed, r_r_bias, r_w_bias, mask=mask)
        return nn.DenseGeneral(
            features=features,
            axis=(-2, -1),
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            use_bias=self.use_bias,
            name="out",
        )(x)
