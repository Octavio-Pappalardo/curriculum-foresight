from __future__ import annotations

import jax
import jax.numpy as jnp

from curriculum_foresight.main.config import CurriculumConfig


def compute_downstream_augmented_goal_values(
    goal_weights: jax.Array,
    downstream_potential: jax.Array,
    value_normalization_support: jax.Array,
    *,
    downstream_coefficient: float,
) -> jax.Array:
    downstream_coefficient = jnp.asarray(downstream_coefficient, dtype=goal_weights.dtype)
    support_size = jnp.maximum(jnp.sum(value_normalization_support), 1)
    mean_weight = jnp.sum(jnp.where(value_normalization_support, goal_weights, 0.0)) / support_size
    mean_downstream = jnp.sum(jnp.where(value_normalization_support, downstream_potential, 0.0)) / support_size

    safe_mean_weight = jnp.where(mean_weight > 0.0, mean_weight, 1.0)
    direct_value = jnp.where(mean_weight > 0.0, goal_weights / safe_mean_weight, 0.0)
    downstream_normalizer = jnp.maximum(mean_downstream, mean_weight)
    safe_downstream_normalizer = jnp.where(downstream_normalizer > 0.0, downstream_normalizer, 1.0)
    downstream_value = jnp.where(
        downstream_normalizer > 0.0, downstream_coefficient * downstream_potential / safe_downstream_normalizer, 0.0
    )
    return direct_value + downstream_value


def compute_alp_downstream_scores(
    signed_lp: jax.Array, goal_weights: jax.Array, downstream_potential: jax.Array, *, config: CurriculumConfig
) -> jax.Array:
    progress_signal = jnp.abs(signed_lp)
    calibrated_progress = progress_signal**config.learning_progress_exponent

    value_normalization_support = calibrated_progress > 0.0
    goal_values = compute_downstream_augmented_goal_values(
        goal_weights,
        downstream_potential,
        value_normalization_support,
        downstream_coefficient=config.downstream_coefficient,
    )
    return calibrated_progress * goal_values


def compute_feasibility_scores(
    progress_feasibility: jax.Array,
    performance_estimates: jax.Array,
    goal_weights: jax.Array,
    downstream_potential: jax.Array,
    *,
    use_downstream_value: bool,
    feasibility_performance_threshold: float,
    downstream_coefficient: float,
) -> jax.Array:
    eligibility = performance_estimates <= feasibility_performance_threshold
    feasibility_signal = jnp.where(eligibility, progress_feasibility, 0.0)
    if use_downstream_value:
        value_normalization_support = feasibility_signal > 0.0
        goal_values = compute_downstream_augmented_goal_values(
            goal_weights,
            downstream_potential,
            value_normalization_support,
            downstream_coefficient=downstream_coefficient,
        )
        return feasibility_signal * goal_values
    return feasibility_signal * goal_weights
