from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from curriculum_foresight.main.config import TrainConfig
from curriculum_foresight.networks.subroutine_actor_critic import SubroutineActorCritic
from curriculum_foresight.policy.subroutines import CallEligibility, subroutine_distribution_statistics
from curriculum_foresight.support.types import SubroutineTransition


def normalize_outer_advantages(advantages_raw: jax.Array, remaining_subroutine_steps: jax.Array) -> jax.Array:
    outer_decision_mask = remaining_subroutine_steps == 0
    normalization_count = jnp.maximum(jnp.sum(outer_decision_mask), 1)
    mean = jnp.sum(jnp.where(outer_decision_mask, advantages_raw, 0.0)) / normalization_count
    centered = jnp.where(outer_decision_mask, advantages_raw - mean, 0.0)
    std = jnp.sqrt(jnp.sum(jnp.square(centered)) / normalization_count)
    return centered / (std + 1e-8)


def _subroutine_minibatch_loss(
    params: Any,
    transitions: SubroutineTransition,
    memories: jax.Array,
    assigned_goal_embeddings: jax.Array,
    all_goal_embeddings: jax.Array,
    eligibility: CallEligibility,
    advantages: jax.Array,
    value_targets: jax.Array,
    *,
    apply_fn: Callable,
    num_primitive_actions: int,
    config: TrainConfig,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    policy = config.policy
    length = policy.subsequence_length_in_loss_calculation
    num_subsequences = transitions.obs.shape[1] // length

    memories = jax.vmap(lambda bank, indices: bank[indices])(memories, transitions.memories_indices[:, ::length])
    memories = memories.reshape((-1, *memories.shape[2:]))
    mask = transitions.memories_mask.reshape((-1, length, *transitions.memories_mask.shape[2:]))
    mask = jnp.swapaxes(mask, 1, 2)
    mask = jnp.concatenate((mask, jnp.zeros((*mask.shape[:-1], length - 1), dtype=jnp.bool_)), axis=-1)
    mask = jax.vmap(jnp.roll, in_axes=(-2, 0, None), out_axes=-2)(mask, jnp.arange(length), -1)

    transitions, advantages, value_targets = jax.tree.map(
        lambda x: x.reshape((-1, length, *x.shape[2:])), (transitions, advantages, value_targets)
    )
    assigned_goal_embeddings = jnp.repeat(assigned_goal_embeddings, repeats=num_subsequences, axis=0)
    active_goal_embeddings = all_goal_embeddings[transitions.active_goal_indices]
    distribution, predicted_values = apply_fn(
        {"params": params},
        memories,
        {"observation": transitions.obs},
        mask,
        assigned_goal_embeddings,
        active_goal_embeddings,
        transitions.remaining_subroutine_steps,
        eligibility,
        method=SubroutineActorCritic.forward_sequence,
    )
    log_prob = distribution.log_prob(transitions.action)

    clipped_values = transitions.value + (predicted_values - transitions.value).clip(-policy.clip_eps, policy.clip_eps)
    value_loss = (
        0.5
        * jnp.maximum(jnp.square(predicted_values - value_targets), jnp.square(clipped_values - value_targets)).mean()
    )

    outer_decision_mask = transitions.remaining_subroutine_steps == 0
    normalization_count = jnp.maximum(jnp.sum(outer_decision_mask), 1)
    log_ratio = jnp.where(outer_decision_mask, log_prob - transitions.log_prob, 0.0)
    ratio = jnp.exp(log_ratio)
    actor_advantages = jnp.where(outer_decision_mask, advantages, 0.0)
    actor_loss = (
        -jnp.minimum(
            ratio * actor_advantages, jnp.clip(ratio, 1.0 - policy.clip_eps, 1.0 + policy.clip_eps) * actor_advantages
        ).sum()
        / normalization_count
    )
    entropy, _, corrected_entropy = subroutine_distribution_statistics(
        distribution, eligibility, num_primitive_actions=num_primitive_actions
    )
    entropy = jnp.where(outer_decision_mask, entropy, 0.0).sum() / normalization_count
    corrected_entropy = jnp.where(outer_decision_mask, corrected_entropy, 0.0).sum() / normalization_count
    approx_kl = ((ratio - 1.0) - log_ratio).sum() / normalization_count
    clip_fraction = ((jnp.abs(ratio - 1.0) > policy.clip_eps) & outer_decision_mask).sum() / normalization_count
    total_loss = actor_loss + policy.vf_coef * value_loss - config.subroutine.reference_kl_coef * corrected_entropy
    return total_loss, {
        "ppo/value_loss": value_loss,
        "ppo/actor_loss": actor_loss,
        "ppo/entropy": entropy,
        "ppo/corrected_entropy": corrected_entropy,
        "ppo/approx_kl": approx_kl,
        "ppo/clip_fraction": clip_fraction,
    }


def subroutine_update_agent(
    rng: jax.Array,
    train_state: TrainState,
    transitions: SubroutineTransition,
    memories_for_ppo: jax.Array,
    assigned_goal_embeddings: jax.Array,
    all_goal_embeddings: jax.Array,
    eligibility: CallEligibility,
    advantages: jax.Array,
    value_targets: jax.Array,
    *,
    num_primitive_actions: int,
    config: TrainConfig,
) -> tuple[jax.Array, TrainState, dict[str, jax.Array]]:
    def update_minibatch(state, batch):
        transitions_mb, memories_mb, goals_mb, advantages_mb, targets_mb = batch
        (_, metrics), grads = jax.value_and_grad(_subroutine_minibatch_loss, has_aux=True)(
            state.params,
            transitions_mb,
            memories_mb,
            goals_mb,
            all_goal_embeddings,
            eligibility,
            advantages_mb,
            targets_mb,
            apply_fn=state.apply_fn,
            num_primitive_actions=num_primitive_actions,
            config=config,
        )
        metrics["ppo/grad_norm"] = optax.global_norm(grads)
        return state.apply_gradients(grads=grads), metrics

    def update_epoch(carry, _unused):
        rng, state = carry
        rng, shuffle_rng = jax.random.split(rng)
        permutation = jax.random.permutation(shuffle_rng, config.num_envs_per_batch)
        data = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), (transitions, memories_for_ppo, advantages, value_targets))
        batches = (data[0], data[1], assigned_goal_embeddings, data[2], data[3])
        batches = jax.tree.map(lambda x: jnp.take(x, permutation, axis=0), batches)
        batches = jax.tree.map(lambda x: x.reshape((config.policy.num_minibatches, -1, *x.shape[1:])), batches)
        state, metrics = jax.lax.scan(update_minibatch, state, batches)
        return (rng, state), metrics

    (rng, train_state), metrics = jax.lax.scan(
        update_epoch, (rng, train_state), None, length=config.policy.update_epochs
    )
    metrics = jax.tree.map(lambda x: x.mean(axis=(0, 1)), metrics)
    return rng, train_state, metrics
