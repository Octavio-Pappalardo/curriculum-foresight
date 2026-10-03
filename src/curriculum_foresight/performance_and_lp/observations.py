from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import struct

from curriculum_foresight.main.config import PerformanceConfig
from curriculum_foresight.performance_and_lp.success_tracking import GoalSuccessTrackingState


class GoalPerformanceObservations(struct.PyTreeNode):
    success_rates: jax.Array
    observed_mask: jax.Array
    assigned_environment_counts: jax.Array
    target_attempt_counts: jax.Array


def init_goal_performance_observations(num_goals: int) -> GoalPerformanceObservations:
    return GoalPerformanceObservations(
        success_rates=jnp.zeros((num_goals,), dtype=jnp.float32),
        observed_mask=jnp.zeros((num_goals,), dtype=jnp.bool_),
        assigned_environment_counts=jnp.zeros((num_goals,), dtype=jnp.int32),
        target_attempt_counts=jnp.zeros((num_goals,), dtype=jnp.int32),
    )


def compute_goal_performance_observations(
    success_tracking_state: GoalSuccessTrackingState,
    environment_goal_indices: jax.Array,
    *,
    num_goals: int,
    config: PerformanceConfig,
) -> GoalPerformanceObservations:
    environment_goal_indices = jnp.asarray(environment_goal_indices, dtype=jnp.int32)
    assigned_environment_counts = jnp.bincount(environment_goal_indices, length=num_goals)

    environment_attempt_counts = jnp.minimum(success_tracking_state.completed_episode_count, config.last_k_success_k)
    environment_success_sums = jnp.sum(success_tracking_state.last_k_episode_successes, axis=1)

    target_attempt_counts = (
        jnp.zeros((num_goals,), dtype=jnp.int32).at[environment_goal_indices].add(environment_attempt_counts)
    )
    observed_mask = target_attempt_counts > 0
    success_sums = jnp.zeros((num_goals,), dtype=jnp.float32).at[environment_goal_indices].add(environment_success_sums)
    success_rates = success_sums / jnp.maximum(target_attempt_counts, 1).astype(jnp.float32)

    return GoalPerformanceObservations(
        success_rates=jnp.where(observed_mask, success_rates, 0.0).astype(jnp.float32),
        observed_mask=observed_mask,
        assigned_environment_counts=assigned_environment_counts.astype(jnp.int32),
        target_attempt_counts=target_attempt_counts.astype(jnp.int32),
    )
