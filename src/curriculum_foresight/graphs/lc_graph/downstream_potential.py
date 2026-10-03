from __future__ import annotations

import jax
import jax.numpy as jnp

from curriculum_foresight.main.config import CurriculumConfig


def compute_downstream_potential(
    active_lc_adjacency: jax.Array, goal_weights: jax.Array, *, config: CurriculumConfig
) -> jax.Array:
    if config.max_path_length == 0:
        return jnp.zeros_like(goal_weights)

    path_discount = jnp.asarray(config.path_discount, dtype=goal_weights.dtype)
    goal_indices = jnp.arange(goal_weights.shape[0])

    def compute_for_root(root_index: jax.Array) -> jax.Array:
        root_target_mask = goal_indices != root_index
        root_adjacency = active_lc_adjacency * root_target_mask[None, :]
        exact_path_strengths = path_discount * root_adjacency[root_index]
        best_path_strengths = exact_path_strengths

        def extend_paths(carry: tuple[jax.Array, jax.Array], _unused: None) -> tuple[tuple[jax.Array, jax.Array], None]:
            exact_path_strengths, best_path_strengths = carry
            path_extensions = exact_path_strengths[:, None] * root_adjacency
            next_exact_path_strengths = path_discount * jnp.max(path_extensions, axis=0)
            next_best_path_strengths = jnp.maximum(best_path_strengths, next_exact_path_strengths)
            return (next_exact_path_strengths, next_best_path_strengths), None

        (_, best_path_strengths), _ = jax.lax.scan(
            extend_paths, (exact_path_strengths, best_path_strengths), None, length=config.max_path_length - 1
        )
        return jnp.sum(best_path_strengths * goal_weights)

    return jax.vmap(compute_for_root)(goal_indices)
