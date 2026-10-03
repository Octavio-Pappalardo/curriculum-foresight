from __future__ import annotations

import distrax
import jax
import jax.numpy as jnp
from craftax.craftax.constants import Action
from flax import struct


class CallEligibility(struct.PyTreeNode):
    eligible_call_mask: jax.Array
    num_eligible_goals: jax.Array
    call_logit_correction: jax.Array


def prepare_call_eligibility(
    performance_estimates: jax.Array,
    *,
    competence_threshold: float,
    max_subroutine_steps: int,
    num_primitive_actions: int,
) -> CallEligibility:
    eligible_call_mask = performance_estimates >= competence_threshold
    num_eligible_goals = jnp.sum(eligible_call_mask, dtype=jnp.int32)
    safe_count = jnp.maximum(num_eligible_goals, 1).astype(jnp.float32)
    correction = jnp.log((max_subroutine_steps + 1) * safe_count / num_primitive_actions)
    correction = jnp.where(num_eligible_goals > 0, correction, 0.0)
    return CallEligibility(eligible_call_mask, num_eligible_goals, correction)


def subroutine_action_distribution(
    preference_logits: jax.Array, eligibility: CallEligibility, *, num_primitive_actions: int
) -> distrax.Categorical:
    call_logits = jnp.where(
        eligibility.eligible_call_mask,
        preference_logits[..., num_primitive_actions:] - eligibility.call_logit_correction,
        -jnp.inf,
    )
    logits = jnp.concatenate((preference_logits[..., :num_primitive_actions], call_logits), axis=-1)
    return distrax.Categorical(logits=logits)


def subroutine_distribution_statistics(
    distribution: distrax.Categorical, eligibility: CallEligibility, *, num_primitive_actions: int
) -> tuple[jax.Array, jax.Array, jax.Array]:
    entropy = distribution.entropy()
    call_probability = jnp.sum(distribution.probs[..., num_primitive_actions:], axis=-1)
    corrected_entropy = entropy - eligibility.call_logit_correction * call_probability
    return entropy, call_probability, corrected_entropy


def to_environment_actions(actions: jax.Array, *, num_primitive_actions: int) -> jax.Array:
    return jnp.where(actions < num_primitive_actions, actions, Action.NOOP.value)


def advance_subroutine_state(
    assigned_goal_indices: jax.Array,
    active_goal_indices: jax.Array,
    remaining_subroutine_steps: jax.Array,
    actions: jax.Array,
    episode_done: jax.Array,
    *,
    num_primitive_actions: int,
    max_subroutine_steps: int,
) -> tuple[jax.Array, jax.Array]:
    is_call = actions >= num_primitive_actions
    next_remaining_subroutine_steps = jnp.where(
        remaining_subroutine_steps > 0, remaining_subroutine_steps - 1, jnp.where(is_call, max_subroutine_steps, 0)
    )
    next_active_goal_indices = jnp.where(is_call, actions - num_primitive_actions, active_goal_indices)
    return_to_assigned_goal = episode_done | (next_remaining_subroutine_steps == 0)
    return (
        jnp.where(return_to_assigned_goal, assigned_goal_indices, next_active_goal_indices),
        jnp.where(episode_done, 0, next_remaining_subroutine_steps),
    )
