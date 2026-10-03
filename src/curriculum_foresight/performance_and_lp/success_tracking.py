from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import struct

from curriculum_foresight.main.config import PerformanceConfig


class GoalSuccessTrackingState(struct.PyTreeNode):
    completed_episode_count: jax.Array
    last_k_episode_successes: jax.Array
    last_k_write_index: jax.Array


def init_goal_success_tracking_state(batch_size: int, config: PerformanceConfig) -> GoalSuccessTrackingState:
    return GoalSuccessTrackingState(
        completed_episode_count=jnp.zeros((batch_size,), dtype=jnp.int32),
        last_k_episode_successes=jnp.zeros((batch_size, config.last_k_success_k), dtype=jnp.float32),
        last_k_write_index=jnp.zeros((batch_size,), dtype=jnp.int32),
    )


def update_goal_success_tracking(
    state: GoalSuccessTrackingState, reset_done: jax.Array, goal_success: jax.Array
) -> GoalSuccessTrackingState:
    reset_done = reset_done.astype(jnp.bool_)

    batch_indices = jnp.arange(state.last_k_episode_successes.shape[0])
    last_k_episode_successes = state.last_k_episode_successes.at[batch_indices, state.last_k_write_index].set(
        jnp.where(
            reset_done,
            goal_success.astype(jnp.float32),
            state.last_k_episode_successes[batch_indices, state.last_k_write_index],
        )
    )
    k = state.last_k_episode_successes.shape[1]

    return state.replace(
        completed_episode_count=state.completed_episode_count + reset_done.astype(jnp.int32),
        last_k_episode_successes=last_k_episode_successes,
        last_k_write_index=jnp.where(reset_done, (state.last_k_write_index + 1) % k, state.last_k_write_index),
    )
