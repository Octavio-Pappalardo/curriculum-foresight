from __future__ import annotations

from pathlib import Path

import numpy as np

from curriculum_foresight.goals.types import GoalSpace
from curriculum_foresight.graphs.common import (
    GRAPH_FILENAME,
    atomic_write_npz,
    load_graph_manifest,
    validate_graph_identity,
)


def save_feasibility_graph(graph_dir: Path, goal_adjacency: np.ndarray, from_scratch_feasibility: np.ndarray) -> None:
    atomic_write_npz(
        graph_dir / GRAPH_FILENAME, goal_adjacency=goal_adjacency, from_scratch_feasibility=from_scratch_feasibility
    )


def load_feasibility_graph_artifact(
    graph_dir: Path, goal_space: GoalSpace, *, learner_kind: str
) -> tuple[np.ndarray, np.ndarray]:
    graph_dir = Path(graph_dir).expanduser()
    manifest = load_graph_manifest(graph_dir, "feasibility")
    validate_graph_identity(manifest, goal_space, learner_kind)
    expected_members = {"goal_adjacency", "from_scratch_feasibility"}
    with np.load(graph_dir / GRAPH_FILENAME, allow_pickle=False) as graph:
        if set(graph.files) != expected_members:
            raise ValueError(
                f"Expected arrays {sorted(expected_members)} in feasibility graph.npz, got {sorted(graph.files)}."
            )
        goal_adjacency = graph["goal_adjacency"]
        from_scratch_feasibility = graph["from_scratch_feasibility"]
    _validate_feasibility_graph_arrays(goal_adjacency, from_scratch_feasibility, num_goals=len(goal_space.task_ids))
    return goal_adjacency, from_scratch_feasibility


def _validate_feasibility_graph_arrays(
    goal_adjacency: np.ndarray, from_scratch_feasibility: np.ndarray, *, num_goals: int
) -> None:
    if goal_adjacency.shape != (num_goals, num_goals):
        raise ValueError(
            f"feasibility goal adjacency must have shape {(num_goals, num_goals)}, got {goal_adjacency.shape}."
        )
    if from_scratch_feasibility.shape != (num_goals,):
        raise ValueError(
            f"from-scratch feasibility must have shape {(num_goals,)}, got {from_scratch_feasibility.shape}."
        )
    if goal_adjacency.dtype != np.float32 or from_scratch_feasibility.dtype != np.float32:
        raise ValueError("feasibility graph arrays must have dtype float32.")
    if not np.isfinite(goal_adjacency).all() or not np.isfinite(from_scratch_feasibility).all():
        raise ValueError("feasibility graph arrays must contain only finite values.")
    if (
        np.any(goal_adjacency < 0)
        or np.any(goal_adjacency > 1)
        or np.any(from_scratch_feasibility < 0)
        or np.any(from_scratch_feasibility > 1)
    ):
        raise ValueError("feasibility graph values must lie in [0, 1].")
    if np.any(np.diag(goal_adjacency) != 0):
        raise ValueError("feasibility goal adjacency must have a zero diagonal.")
    joint_mass = from_scratch_feasibility + goal_adjacency.sum(axis=0)
    if not np.allclose(joint_mass, 1.0, rtol=0.0, atol=1e-6):
        raise ValueError("From-scratch feasibility and incoming goal weights must sum to one for every target.")
