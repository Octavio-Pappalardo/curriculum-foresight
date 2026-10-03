from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
from flax import struct

from curriculum_foresight.graphs.feasibility_graph.graph_storage import load_feasibility_graph_artifact
from curriculum_foresight.goals.types import GoalSpace


class FeasibilityGraph(struct.PyTreeNode):
    goal_adjacency: jax.Array
    from_scratch_feasibility: jax.Array


def load_feasibility_graph(graph_dir: Path, goal_space: GoalSpace, *, learner_kind: str) -> FeasibilityGraph:
    goal_adjacency, from_scratch_feasibility = load_feasibility_graph_artifact(
        graph_dir, goal_space, learner_kind=learner_kind
    )
    return FeasibilityGraph(
        goal_adjacency=jnp.asarray(goal_adjacency), from_scratch_feasibility=jnp.asarray(from_scratch_feasibility)
    )
