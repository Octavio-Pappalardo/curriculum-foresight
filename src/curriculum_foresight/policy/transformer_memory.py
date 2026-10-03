from __future__ import annotations

import jax
import jax.numpy as jnp

from curriculum_foresight.main.config import PolicyConfig


def init_transformer_memories(batch_size: int, policy_config: PolicyConfig) -> jax.Array:
    return jnp.zeros(
        (
            batch_size,
            policy_config.past_context_length,
            policy_config.num_transformer_blocks,
            policy_config.transformer_hidden_states_dim,
        ),
        dtype=jnp.float32,
    )


def init_transformer_memory_mask(batch_size: int, policy_config: PolicyConfig) -> jax.Array:
    return jnp.zeros(
        (batch_size, policy_config.num_attn_heads, 1, policy_config.past_context_length + 1), dtype=jnp.bool_
    )


def init_transformer_memory_mask_idx(batch_size: int, policy_config: PolicyConfig) -> jax.Array:
    return jnp.full((batch_size,), policy_config.past_context_length + 1, dtype=jnp.int32)


def _mark_memory_mask_index(
    memories_mask: jax.Array, memories_mask_idx: jax.Array, policy_config: PolicyConfig
) -> jax.Array:
    memories_mask_idx_one_hot = jax.nn.one_hot(
        memories_mask_idx, policy_config.past_context_length + 1, dtype=jnp.bool_
    )
    memories_mask_idx_one_hot = jnp.repeat(
        memories_mask_idx_one_hot[:, None, None, :], policy_config.num_attn_heads, axis=1
    )
    return jnp.logical_or(memories_mask, memories_mask_idx_one_hot)


def advance_transformer_memory_mask_episodic(
    memories_mask: jax.Array, memories_mask_idx: jax.Array, prev_reset_done: jax.Array, policy_config: PolicyConfig
) -> tuple[jax.Array, jax.Array]:
    next_memories_mask_idx = jnp.where(
        prev_reset_done,
        policy_config.past_context_length,
        jnp.clip(memories_mask_idx - 1, 0, policy_config.past_context_length),
    )
    next_memories_mask = jnp.where(prev_reset_done[:, None, None, None], jnp.zeros_like(memories_mask), memories_mask)
    return (_mark_memory_mask_index(next_memories_mask, next_memories_mask_idx, policy_config), next_memories_mask_idx)
