from __future__ import annotations

import jax
import jax.numpy as jnp


def compute_gae(
    rewards: jax.Array,
    values: jax.Array,
    last_values: jax.Array,
    episode_done: jax.Array,
    gamma: float,
    gae_lambda: float,
) -> tuple[jax.Array, jax.Array]:
    def scan_step(carry: tuple[jax.Array, jax.Array], transition_t: tuple[jax.Array, jax.Array, jax.Array]):
        gae, next_values = carry
        rewards_t, values_t, episode_done_t = transition_t
        episode_not_done = 1.0 - episode_done_t.astype(values_t.dtype)
        delta = rewards_t + gamma * next_values * episode_not_done - values_t
        gae = delta + gamma * gae_lambda * episode_not_done * gae
        return (gae, values_t), gae

    _, advantages = jax.lax.scan(
        scan_step, (jnp.zeros_like(last_values), last_values), (rewards, values, episode_done), reverse=True
    )
    value_targets = advantages + values
    return advantages, value_targets
