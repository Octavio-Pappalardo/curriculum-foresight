from __future__ import annotations

import jax

from curriculum_foresight.graphs.feasibility_graph.runtime_graph import FeasibilityGraph


def compute_progress_feasibility(feasibility_graph: FeasibilityGraph, performance_estimates: jax.Array) -> jax.Array:
    predecessor_feasibility = feasibility_graph.goal_adjacency.T @ performance_estimates
    return feasibility_graph.from_scratch_feasibility + predecessor_feasibility
