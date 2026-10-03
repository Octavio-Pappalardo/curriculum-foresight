from __future__ import annotations

import jax
import jax.numpy as jnp

from curriculum_foresight.curriculum.signals import CurriculumIterationSignals
from curriculum_foresight.main.config import CurriculumConfig, TrainConfig


def compute_score_distribution(scores: jax.Array) -> jax.Array:
    scores = jnp.asarray(scores, dtype=jnp.float32)
    positive_support = scores > 0.0
    has_positive_support = jnp.any(positive_support)
    safe_scores = jnp.where(positive_support, scores, jnp.ones_like(scores))
    logits = jnp.where(positive_support, jnp.log(safe_scores), -jnp.inf)
    normalization_logits = jnp.where(has_positive_support, logits, jnp.zeros_like(logits))
    probabilities = jax.nn.softmax(normalization_logits)
    probabilities = jnp.where(positive_support, probabilities, jnp.zeros_like(probabilities))
    uniform_distribution = jnp.full(scores.shape, 1.0 / scores.size, dtype=jnp.float32)
    return jnp.where(has_positive_support, probabilities, uniform_distribution)


def compute_final_goal_selection_distribution(
    method_scores: jax.Array, feasibility_scores: jax.Array, *, config: CurriculumConfig
) -> jax.Array:
    method_distribution = compute_score_distribution(method_scores)
    uniform_distribution = jnp.full(method_scores.shape, 1.0 / method_scores.size, dtype=jnp.float32)
    uniform_mix = jnp.asarray(config.uniform_mix, dtype=method_distribution.dtype)
    if not (config.feasibility_mix > 0):
        return (1.0 - uniform_mix) * method_distribution + uniform_mix * uniform_distribution

    feasibility_distribution = compute_score_distribution(feasibility_scores)
    feasibility_mix = jnp.asarray(config.feasibility_mix, dtype=method_distribution.dtype)
    return (
        uniform_mix * uniform_distribution
        + feasibility_mix * feasibility_distribution
        + (1.0 - uniform_mix - feasibility_mix) * method_distribution
    )


def compute_sampling_distribution(
    iteration_signals: CurriculumIterationSignals, curriculum_iteration: jax.Array | int, *, config: TrainConfig
) -> jax.Array:
    uniform_distribution = jnp.full(
        iteration_signals.method_scores.shape, 1.0 / iteration_signals.method_scores.size, dtype=jnp.float32
    )
    if config.algorithm_id in ("alp-downstream", "absolute-lp", "intermediate-difficulty"):
        adaptive_distribution = compute_final_goal_selection_distribution(
            iteration_signals.method_scores, iteration_signals.feasibility_scores, config=config.curriculum
        )
        selection_ready = curriculum_iteration >= config.learning_progress.num_warmup_curriculum_iterations
        return jnp.where(selection_ready, adaptive_distribution, uniform_distribution)
    if config.algorithm_id in ("uniform", "target-only"):
        return uniform_distribution
    raise NotImplementedError(f"Unsupported curriculum algorithm: {config.algorithm_id!r}.")


def sample_goal_indices(
    rng: jax.Array, probabilities: jax.Array, *, num_goal_draws: int
) -> tuple[jax.Array, jax.Array]:
    probabilities = jnp.asarray(probabilities, dtype=jnp.float32)
    next_rng, sample_rng = jax.random.split(rng)
    goal_draws = jax.random.choice(
        sample_rng, probabilities.shape[0], shape=(num_goal_draws,), replace=True, p=probabilities
    )
    return next_rng, goal_draws.astype(jnp.int32)


def assign_goal_draws_to_envs(goal_draws: jax.Array, *, num_envs: int) -> jax.Array:
    num_goal_draws = goal_draws.shape[0]
    num_repeats = num_envs // num_goal_draws
    return jnp.repeat(goal_draws, num_repeats)


def sample_environment_goal_indices(
    rng: jax.Array, probabilities: jax.Array, *, num_envs: int, num_goal_draws: int
) -> tuple[jax.Array, jax.Array]:
    rng, goal_draws = sample_goal_indices(rng, probabilities, num_goal_draws=num_goal_draws)
    return rng, assign_goal_draws_to_envs(goal_draws, num_envs=num_envs)
