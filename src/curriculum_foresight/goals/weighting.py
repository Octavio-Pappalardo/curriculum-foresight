from __future__ import annotations

import jax
import jax.numpy as jnp

from curriculum_foresight.goals.types import GoalSpace


def uniform_goal_weights(goal_space: GoalSpace) -> jax.Array:
    return jnp.ones((len(goal_space.task_ids),), dtype=jnp.float32)


def single_target_goal_weights(goal_space: GoalSpace, target_task_id: str, target_weight_fraction: float) -> jax.Array:
    target_index = goal_space.task_ids.index(target_task_id)
    num_goals = len(goal_space.task_ids)
    non_target_weight = (1.0 - target_weight_fraction) * num_goals / (num_goals - 1)
    return (
        jnp.full((num_goals,), non_target_weight, dtype=jnp.float32)
        .at[target_index]
        .set(target_weight_fraction * num_goals)
    )
