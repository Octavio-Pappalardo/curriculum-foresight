from __future__ import annotations

import jax


def create_active_lc_adjacency(base_lc_adjacency: jax.Array, non_solved_mask: jax.Array) -> jax.Array:
    return base_lc_adjacency * non_solved_mask[None, :]
