from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, normal, orthogonal

CRAFTAX_SYMBOLIC_ENV_ID = "Craftax-Symbolic-v1"

CRAFTAX_HEIGHT = 9
CRAFTAX_WIDTH = 11
CRAFTAX_BLOCK_TOKENS = 37
CRAFTAX_ITEM_TOKENS = 5
CRAFTAX_ACTOR_CHANNELS = 40
CRAFTAX_EXTRA_FEATURES_DIM = 51
CRAFTAX_MAP_CHANNELS = CRAFTAX_BLOCK_TOKENS + CRAFTAX_ITEM_TOKENS + CRAFTAX_ACTOR_CHANNELS + 1
CRAFTAX_TOTAL_OBS_DIM = CRAFTAX_HEIGHT * CRAFTAX_WIDTH * CRAFTAX_MAP_CHANNELS + CRAFTAX_EXTRA_FEATURES_DIM


class DenseCategoricalEmbedding(nn.Module):
    num_categories: int
    features: int
    embedding_init: Callable[..., Any] = normal(stddev=0.02)

    @nn.compact
    def __call__(self, category_channels: jax.Array) -> jax.Array:
        category_channels = category_channels.astype(jnp.float32)
        embedding = self.param("embedding", self.embedding_init, (self.num_categories + 1, self.features))
        category_embedding = category_channels @ embedding[1:]
        null_weight = 1.0 - jnp.sum(category_channels, axis=-1, keepdims=True)
        return category_embedding + null_weight * embedding[0]


class CraftaxObservationEncoder(nn.Module):
    env_id: str
    obs_emb_dim: int

    @nn.compact
    def __call__(self, observations: jax.Array) -> jax.Array:
        if self.env_id != CRAFTAX_SYMBOLIC_ENV_ID:
            raise ValueError(f"Unsupported env_id: {self.env_id}")

        batch_shape = observations.shape[:-1]
        observations = observations.astype(jnp.float32).reshape((-1, observations.shape[-1]))
        flat_map_dim = CRAFTAX_HEIGHT * CRAFTAX_WIDTH * CRAFTAX_MAP_CHANNELS
        map_channels = observations[:, :flat_map_dim].reshape((-1, CRAFTAX_HEIGHT, CRAFTAX_WIDTH, CRAFTAX_MAP_CHANNELS))
        extra_features = observations[:, flat_map_dim:]

        block_channels = map_channels[..., :CRAFTAX_BLOCK_TOKENS]
        item_channels = map_channels[..., CRAFTAX_BLOCK_TOKENS : CRAFTAX_BLOCK_TOKENS + CRAFTAX_ITEM_TOKENS]
        actor_multihot = map_channels[
            ...,
            CRAFTAX_BLOCK_TOKENS + CRAFTAX_ITEM_TOKENS : CRAFTAX_BLOCK_TOKENS
            + CRAFTAX_ITEM_TOKENS
            + CRAFTAX_ACTOR_CHANNELS,
        ]
        visibility = map_channels[..., -1].astype(jnp.int32)
        visible_mask = visibility.astype(bool)[..., None].astype(jnp.float32)
        block_channels = block_channels * visible_mask
        item_channels = item_channels * visible_mask
        actor_multihot = actor_multihot * visible_mask

        block_embedding = DenseCategoricalEmbedding(
            num_categories=CRAFTAX_BLOCK_TOKENS, features=16, name="block_embedding"
        )(block_channels)

        actor_embedding_table = self.param("actor_embedding_table", normal(stddev=0.02), (CRAFTAX_ACTOR_CHANNELS, 16))
        actor_embedding = actor_multihot @ actor_embedding_table
        actor_is_present = jnp.any(actor_multihot > 0.0, axis=-1, keepdims=True)
        no_actor_embedding = self.param("no_actor_embedding", normal(stddev=0.02), (16,))
        actor_embedding = actor_embedding + (1.0 - actor_is_present.astype(jnp.float32)) * no_actor_embedding.reshape(
            (1, 1, 1, -1)
        )

        item_embedding = DenseCategoricalEmbedding(
            num_categories=CRAFTAX_ITEM_TOKENS, features=8, name="item_embedding"
        )(item_channels)
        visibility_embedding = DenseCategoricalEmbedding(num_categories=1, features=4, name="visibility_embedding")(
            visibility[..., None]
        )
        cell_features = jnp.concatenate(
            [block_embedding, item_embedding, actor_embedding, visibility_embedding], axis=-1
        )

        cell_features = nn.Dense(32, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(cell_features)
        cell_features = nn.gelu(cell_features)

        spatial_features = nn.Conv(
            features=32, kernel_size=(3, 3), padding="SAME", kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(cell_features)
        spatial_features = nn.gelu(spatial_features)
        spatial_features = nn.Conv(
            features=32, kernel_size=(3, 3), padding="SAME", kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(spatial_features)
        spatial_features = nn.gelu(spatial_features)
        spatial_features = spatial_features.reshape((spatial_features.shape[0], -1))

        extra_features = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(extra_features)
        extra_features = nn.gelu(extra_features)

        fused_features = jnp.concatenate([spatial_features, extra_features], axis=-1)
        fused_features = nn.Dense(self.obs_emb_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(
            fused_features
        )
        fused_features = nn.gelu(fused_features)
        return fused_features.reshape((*batch_shape, self.obs_emb_dim))


class PolicyInputEncoder(nn.Module):
    env_id: str
    obs_emb_dim: int

    def setup(self):
        self.obs_encoder = CraftaxObservationEncoder(env_id=self.env_id, obs_emb_dim=self.obs_emb_dim)

    def __call__(self, inputs: Mapping[str, jax.Array]) -> jax.Array:
        return self.obs_encoder(inputs["observation"])
