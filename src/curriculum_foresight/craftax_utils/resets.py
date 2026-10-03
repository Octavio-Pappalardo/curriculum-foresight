from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp


def reset_if_done(
    env: Any,
    reset_rng: jax.Array,
    post_action_obs: jax.Array,
    post_action_state: Any,
    done: jax.Array,
    env_params: Any = None,
) -> tuple[jax.Array, Any]:
    done = done.astype(jnp.bool_)
    reset_obs, reset_state = env.reset(reset_rng, env_params)
    next_obs = jax.lax.select(done, reset_obs, post_action_obs)
    next_state = jax.tree.map(
        lambda reset_leaf, post_leaf: jax.lax.select(done, reset_leaf, post_leaf), reset_state, post_action_state
    )
    return next_obs, next_state
