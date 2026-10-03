from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import optax

from curriculum_foresight.main.config import TrainConfig
from curriculum_foresight.networks.primitive_actor_critic import PrimitiveActorCritic

batch_indices_select = jax.vmap(lambda x, y: x[y])
roll_vmap = jax.vmap(jnp.roll, in_axes=(-2, 0, None), out_axes=-2)


def primitive_update_agent(
    rng: jax.Array,
    train_state: Any,
    transitions: Any,
    memories_for_ppo: jax.Array,
    goal_embeddings: jax.Array,
    advantages: jax.Array,
    value_targets: jax.Array,
    config: TrainConfig,
) -> tuple[jax.Array, Any, dict[str, jax.Array]]:
    policy_config = config.policy
    num_minibatches = int(policy_config.num_minibatches)
    subsequence_length = policy_config.subsequence_length_in_loss_calculation

    def update_minibatch(train_state: Any, batch_info: tuple[Any, jax.Array, jax.Array, jax.Array, jax.Array]):
        def loss_fn(
            params: Any,
            transitions_mb: Any,
            memories_mb: jax.Array,
            goal_embeddings_mb: jax.Array,
            advantages_mb: jax.Array,
            targets_mb: jax.Array,
        ):
            num_subsequences = transitions_mb.obs.shape[1] // subsequence_length

            memories_mb = batch_indices_select(memories_mb, transitions_mb.memories_indices[:, ::subsequence_length])
            memories_mb = jnp.reshape(
                memories_mb, (memories_mb.shape[0] * memories_mb.shape[1], *memories_mb.shape[2:])
            )

            memories_mask = transitions_mb.memories_mask.reshape(
                (-1, subsequence_length, *transitions_mb.memories_mask.shape[2:])
            )
            memories_mask = jnp.swapaxes(memories_mask, 1, 2)
            memories_mask = jnp.concatenate(
                (memories_mask, jnp.zeros((*memories_mask.shape[:-1], subsequence_length - 1), dtype=jnp.bool_)),
                axis=-1,
            )
            memories_mask = roll_vmap(memories_mask, jnp.arange(subsequence_length), -1)

            transitions_mb, advantages_mb, targets_mb = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, (-1, subsequence_length, *x.shape[2:])),
                (transitions_mb, advantages_mb, targets_mb),
            )
            goal_embeddings_mb = jnp.repeat(goal_embeddings_mb, repeats=num_subsequences, axis=0)
            policy_inputs = {"observation": transitions_mb.obs}

            pi, predicted_values = train_state.apply_fn(
                {"params": params},
                memories_mb,
                policy_inputs,
                memories_mask,
                goal_embeddings_mb,
                method=PrimitiveActorCritic.forward_sequence,
            )
            log_prob = pi.log_prob(transitions_mb.action)

            values_pred_clipped = transitions_mb.value + (predicted_values - transitions_mb.value).clip(
                -policy_config.clip_eps, policy_config.clip_eps
            )
            value_losses = jnp.square(predicted_values - targets_mb)
            value_losses_clipped = jnp.square(values_pred_clipped - targets_mb)
            value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

            log_ratio = log_prob - transitions_mb.log_prob
            ratio = jnp.exp(log_ratio)
            actor_loss_unclipped = ratio * advantages_mb
            actor_loss_clipped = (
                jnp.clip(ratio, 1.0 - policy_config.clip_eps, 1.0 + policy_config.clip_eps) * advantages_mb
            )
            actor_loss = -jnp.minimum(actor_loss_unclipped, actor_loss_clipped).mean()

            entropy = pi.entropy().mean()
            total_loss = actor_loss + policy_config.vf_coef * value_loss - policy_config.ent_coef * entropy
            approx_kl = jnp.mean((ratio - 1.0) - log_ratio)
            clip_fraction = jnp.mean((jnp.abs(ratio - 1.0) > policy_config.clip_eps).astype(jnp.float32))
            return total_loss, (value_loss, actor_loss, entropy, approx_kl, clip_fraction)

        transitions_mb, memories_mb, goal_embeddings_mb, advantages_mb, targets_mb = batch_info
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (_total_loss, (value_loss, actor_loss, entropy, approx_kl, clip_fraction)), grads = grad_fn(
            train_state.params, transitions_mb, memories_mb, goal_embeddings_mb, advantages_mb, targets_mb
        )
        metrics = {
            "ppo/value_loss": value_loss,
            "ppo/actor_loss": actor_loss,
            "ppo/entropy": entropy,
            "ppo/approx_kl": approx_kl,
            "ppo/clip_fraction": clip_fraction,
            "ppo/grad_norm": optax.global_norm(grads),
        }
        train_state = train_state.apply_gradients(grads=grads)
        return train_state, metrics

    def update_epoch(
        update_state: tuple[jax.Array, Any, Any, jax.Array, jax.Array, jax.Array, jax.Array], _unused: Any
    ):
        rng, train_state, transitions, memories_for_ppo, goal_embeddings, advantages, value_targets = update_state
        rng, shuffle_rng = jax.random.split(rng)
        permutation = jax.random.permutation(shuffle_rng, config.num_envs_per_batch)

        intermediate = (transitions, memories_for_ppo, advantages, value_targets)
        intermediate = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), intermediate)
        minibatches = (intermediate[0], intermediate[1], goal_embeddings, intermediate[2], intermediate[3])
        minibatches = jax.tree_util.tree_map(lambda x: jnp.take(x, permutation, axis=0), minibatches)
        minibatches = jax.tree_util.tree_map(lambda x: jnp.reshape(x, (num_minibatches, -1, *x.shape[1:])), minibatches)

        train_state, metrics = jax.lax.scan(update_minibatch, train_state, minibatches)
        return (rng, train_state, transitions, memories_for_ppo, goal_embeddings, advantages, value_targets), metrics

    update_state = (rng, train_state, transitions, memories_for_ppo, goal_embeddings, advantages, value_targets)
    update_state, metrics = jax.lax.scan(update_epoch, update_state, None, policy_config.update_epochs)
    rng, train_state = update_state[:2]
    metrics = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=(0, 1)), metrics)
    return rng, train_state, metrics
