from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import struct

from curriculum_foresight.performance_and_lp.observations import GoalPerformanceObservations


class LearningProgressEstimatorState(struct.PyTreeNode):
    observation_iterations: jax.Array
    observed_success_rates: jax.Array
    history_counts: jax.Array
    signed_lp: jax.Array


def init_learning_progress_estimator_state(
    num_goals: int, *, max_history_observations: int
) -> LearningProgressEstimatorState:
    observation_iterations = jnp.zeros((num_goals, max_history_observations), dtype=jnp.int32)
    observation_iterations = observation_iterations.at[:, -1].set(-1)
    return LearningProgressEstimatorState(
        observation_iterations=observation_iterations,
        observed_success_rates=jnp.zeros((num_goals, max_history_observations), dtype=jnp.float32),
        history_counts=jnp.ones((num_goals,), dtype=jnp.int32),
        signed_lp=jnp.zeros((num_goals,), dtype=jnp.float32),
    )


def update_learning_progress_estimates(
    state: LearningProgressEstimatorState,
    observations: GoalPerformanceObservations,
    observation_iteration: jax.Array | int,
    *,
    max_lookback_curriculum_iterations: int,
) -> LearningProgressEstimatorState:
    observation_iteration = jnp.asarray(observation_iteration, dtype=jnp.int32)
    observed_mask = observations.observed_mask
    num_goals, history_capacity = state.observation_iterations.shape

    appended_iterations = jnp.concatenate(
        (state.observation_iterations[:, 1:], jnp.full((num_goals, 1), observation_iteration, dtype=jnp.int32)), axis=1
    )
    appended_success_rates = jnp.concatenate(
        (state.observed_success_rates[:, 1:], observations.success_rates[:, None]), axis=1
    )
    appended_counts = jnp.minimum(state.history_counts + 1, history_capacity)

    candidate_iterations = jnp.where(observed_mask[:, None], appended_iterations, state.observation_iterations)
    candidate_success_rates = jnp.where(observed_mask[:, None], appended_success_rates, state.observed_success_rates)
    candidate_counts = jnp.where(observed_mask, appended_counts, state.history_counts)

    positions = jnp.arange(history_capacity, dtype=jnp.int32)[None, :]
    valid_mask = positions >= history_capacity - candidate_counts[:, None]
    ages = observation_iteration - candidate_iterations
    recent_past_mask = valid_mask & (ages >= 1) & (ages <= max_lookback_curriculum_iterations)
    has_recent_past = jnp.any(recent_past_mask, axis=1)
    current_point_mask = positions == history_capacity - 1
    previous_point_mask = positions == history_capacity - 2
    fallback_rows = ~has_recent_past
    fit_mask = current_point_mask | recent_past_mask | (fallback_rows[:, None] & previous_point_mask)

    clipped_previous_iterations = jnp.where(
        fallback_rows[:, None] & previous_point_mask,
        observation_iteration - max_lookback_curriculum_iterations,
        candidate_iterations,
    )
    regression_iterations = clipped_previous_iterations.astype(jnp.float32)
    fit_counts = jnp.sum(fit_mask, axis=1).astype(jnp.float32)
    mean_iterations = jnp.sum(jnp.where(fit_mask, regression_iterations, 0.0), axis=1) / fit_counts
    mean_success_rates = jnp.sum(jnp.where(fit_mask, candidate_success_rates, 0.0), axis=1) / fit_counts
    centered_iterations = regression_iterations - mean_iterations[:, None]
    centered_success_rates = candidate_success_rates - mean_success_rates[:, None]
    slope_numerators = jnp.sum(jnp.where(fit_mask, centered_iterations * centered_success_rates, 0.0), axis=1)
    slope_denominators = jnp.sum(jnp.where(fit_mask, jnp.square(centered_iterations), 0.0), axis=1)
    fitted_lp = slope_numerators / jnp.where(slope_denominators > 0.0, slope_denominators, 1.0)

    return state.replace(
        observation_iterations=candidate_iterations,
        observed_success_rates=candidate_success_rates,
        history_counts=candidate_counts,
        signed_lp=jnp.where(observed_mask, fitted_lp, state.signed_lp).astype(jnp.float32),
    )
