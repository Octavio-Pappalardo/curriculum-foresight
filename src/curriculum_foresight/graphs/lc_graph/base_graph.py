from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from curriculum_foresight.goals.types import GoalSpace
from curriculum_foresight.graphs.common import GRAPH_FILENAME, load_graph_manifest, validate_graph_identity
from curriculum_foresight.main.config import CurriculumConfig


def load_lc_graph(graph_dir: Path, goal_space: GoalSpace, *, learner_kind: str) -> jax.Array:
    graph_dir = Path(graph_dir).expanduser()
    manifest = load_graph_manifest(graph_dir, "lc")
    validate_graph_identity(manifest, goal_space, learner_kind)
    with np.load(graph_dir / GRAPH_FILENAME, allow_pickle=False) as graph:
        if set(graph.files) != {"adjacency"}:
            raise ValueError("LC graph NPZ must contain only 'adjacency'.")
        adjacency = graph["adjacency"]
    num_goals = len(goal_space.task_ids)
    if adjacency.shape != (num_goals, num_goals):
        raise ValueError(f"expected LC graph shape {(num_goals, num_goals)}, got {adjacency.shape}.")
    if adjacency.dtype != np.float32:
        raise ValueError("LC graph adjacency must have dtype float32.")
    if not np.isfinite(adjacency).all() or np.any(adjacency < 0) or np.any(adjacency > 1):
        raise ValueError("LC graph values must be finite and lie in [0, 1].")
    if np.any(np.diag(adjacency) != 0):
        raise ValueError("LC graph adjacency must have a zero diagonal.")
    return jnp.asarray(adjacency)


def create_base_lc_adjacency(goal_space: GoalSpace, config: CurriculumConfig, *, learner_kind: str) -> jax.Array:
    num_goals = len(goal_space.task_ids)
    if config.base_graph_source == "none":
        return jnp.zeros((num_goals, num_goals), dtype=jnp.float32)
    if config.base_graph_source in ("artifact", "permuted"):
        adjacency = load_lc_graph(config.lc_graph_dir, goal_space, learner_kind=learner_kind)
        if config.base_graph_source == "permuted":
            permutation = jax.random.permutation(jax.random.key(config.graph_permutation_seed), num_goals)
            adjacency = adjacency[permutation][:, permutation]
        return adjacency
    raise NotImplementedError(f"Unsupported base LC graph source: {config.base_graph_source!r}.")
