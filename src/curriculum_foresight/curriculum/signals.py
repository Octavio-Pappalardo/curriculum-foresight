from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import struct

from curriculum_foresight.curriculum.scoring import compute_alp_downstream_scores, compute_feasibility_scores
from curriculum_foresight.graphs.feasibility_graph.progress_feasibility import compute_progress_feasibility
from curriculum_foresight.graphs.feasibility_graph.runtime_graph import FeasibilityGraph
from curriculum_foresight.graphs.lc_graph.active_graph import create_active_lc_adjacency
from curriculum_foresight.graphs.lc_graph.downstream_potential import compute_downstream_potential
from curriculum_foresight.main.config import TrainConfig


class CurriculumIterationSignals(struct.PyTreeNode):
    performance_estimates: jax.Array
    signed_lp: jax.Array
    solved_mask: jax.Array
    downstream_potential: jax.Array
    method_scores: jax.Array
    progress_feasibility: jax.Array
    feasibility_scores: jax.Array


def update_curriculum_iteration_signals(
    performance_estimates: jax.Array,
    signed_lp: jax.Array,
    base_lc_adjacency: jax.Array,
    goal_weights: jax.Array,
    *,
    config: TrainConfig,
    feasibility_graph: FeasibilityGraph | None = None,
) -> CurriculumIterationSignals:
    curriculum_config = config.curriculum
    solved_mask = performance_estimates >= curriculum_config.solved_success_threshold
    downstream_potential = jnp.zeros_like(goal_weights)
    method_scores = jnp.zeros_like(goal_weights)

    if config.algorithm_id == "alp-downstream":
        active_lc_adjacency = create_active_lc_adjacency(base_lc_adjacency, ~solved_mask)
        downstream_potential = compute_downstream_potential(active_lc_adjacency, goal_weights, config=curriculum_config)
        method_scores = compute_alp_downstream_scores(
            signed_lp, goal_weights, downstream_potential, config=curriculum_config
        )
    elif config.algorithm_id == "absolute-lp":
        calibrated_progress = jnp.abs(signed_lp) ** curriculum_config.learning_progress_exponent
        method_scores = goal_weights * calibrated_progress
    elif config.algorithm_id == "intermediate-difficulty":
        in_band = (performance_estimates >= curriculum_config.intermediate_difficulty_min) & (
            performance_estimates <= curriculum_config.intermediate_difficulty_max
        )
        method_scores = jnp.where(in_band, goal_weights, jnp.zeros_like(goal_weights))
    elif config.algorithm_id not in ("uniform", "target-only"):
        raise NotImplementedError(f"Unsupported curriculum algorithm: {config.algorithm_id!r}.")

    progress_feasibility = jnp.zeros_like(goal_weights)
    feasibility_scores = jnp.zeros_like(goal_weights)
    if curriculum_config.feasibility_mix > 0:
        progress_feasibility = compute_progress_feasibility(feasibility_graph, performance_estimates)
        feasibility_scores = compute_feasibility_scores(
            progress_feasibility,
            performance_estimates,
            goal_weights,
            downstream_potential,
            use_downstream_value=config.algorithm_id == "alp-downstream",
            feasibility_performance_threshold=(curriculum_config.feasibility_performance_threshold),
            downstream_coefficient=curriculum_config.downstream_coefficient,
        )

    return CurriculumIterationSignals(
        performance_estimates=performance_estimates,
        signed_lp=signed_lp,
        solved_mask=solved_mask,
        downstream_potential=downstream_potential,
        method_scores=method_scores,
        progress_feasibility=progress_feasibility,
        feasibility_scores=feasibility_scores,
    )
