from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import struct

from curriculum_foresight.performance_and_lp.observations import GoalPerformanceObservations


class PerformanceEstimatorState(struct.PyTreeNode):
    performance_estimates: jax.Array
    ever_observed_mask: jax.Array


def init_performance_estimator_state(num_goals: int, *, prior: float) -> PerformanceEstimatorState:
    return PerformanceEstimatorState(
        performance_estimates=jnp.full((num_goals,), prior, dtype=jnp.float32),
        ever_observed_mask=jnp.zeros((num_goals,), dtype=jnp.bool_),
    )


def update_performance_estimates(
    state: PerformanceEstimatorState, observations: GoalPerformanceObservations, *, ema_coefficient: float
) -> PerformanceEstimatorState:
    smoothed_estimates = state.performance_estimates + ema_coefficient * (
        observations.success_rates - state.performance_estimates
    )
    observed_estimates = jnp.where(state.ever_observed_mask, smoothed_estimates, observations.success_rates)
    return state.replace(
        performance_estimates=jnp.where(
            observations.observed_mask, observed_estimates, state.performance_estimates
        ).astype(jnp.float32),
        ever_observed_mask=state.ever_observed_mask | observations.observed_mask,
    )
