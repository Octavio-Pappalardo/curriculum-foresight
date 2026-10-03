from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import jax
import numpy as np

from curriculum_foresight.goals.types import GoalSpace
from curriculum_foresight.main.config import TrainConfig, config_to_dict


def build_run_metadata(config: TrainConfig, goal_space: GoalSpace) -> dict[str, Any]:
    return {
        "config": config_to_dict(config),
        "task_ids": list(goal_space.task_ids),
        "goal_weights": np.asarray(jax.device_get(goal_space.arrays.weights)).tolist(),
    }


def compute_target_anchored_selection_probabilities(
    ordinary_probabilities: np.ndarray, target_goal_index: int, *, num_target_anchor_draws: int, num_goal_draws: int
) -> np.ndarray:
    ordinary_probabilities = np.asarray(ordinary_probabilities, dtype=np.float32)
    anchor_fraction = np.float32(num_target_anchor_draws / num_goal_draws)
    probabilities = (np.float32(1.0) - anchor_fraction) * ordinary_probabilities
    probabilities[target_goal_index] += anchor_fraction
    return probabilities


def build_training_record(
    completed_iteration: int,
    iteration_signals: Any,
    selection_probabilities: jax.Array,
    train_metrics: Mapping[str, Any],
    *,
    config: TrainConfig,
    target_goal_index: int | None,
    cumulative_wall_clock_sec: float,
    curriculum_iteration_sec: float,
) -> dict[str, Any]:
    row = jax.device_get(
        {
            "run/curriculum_iteration": np.int64(completed_iteration),
            "run/total_env_steps": np.int64(completed_iteration * config.num_env_steps_per_curriculum_iteration),
            "time/cumulative_wall_clock_sec": np.float64(cumulative_wall_clock_sec),
            "time/curriculum_iteration_sec": np.float64(curriculum_iteration_sec),
            **train_metrics,
            "curriculum/performance_estimates": iteration_signals.performance_estimates,
            "curriculum/signed_lp": iteration_signals.signed_lp,
            "curriculum/downstream_potential": iteration_signals.downstream_potential,
            "curriculum/progress_feasibility": iteration_signals.progress_feasibility,
            "curriculum/selection_probabilities": selection_probabilities,
        }
    )
    for key in train_metrics:
        dtype = np.int32 if key == "subroutine/eligible_goal_count" else np.float32
        row[key] = np.asarray(row[key], dtype=dtype)
    if config.num_target_anchor_draws_per_batch > 0:
        row["curriculum/selection_probabilities"] = compute_target_anchored_selection_probabilities(
            row["curriculum/selection_probabilities"],
            target_goal_index,
            num_target_anchor_draws=config.num_target_anchor_draws_per_batch,
            num_goal_draws=config.num_goal_draws_per_batch,
        )
    return row


def build_evaluation_record(
    completed_iteration: int, total_env_steps: int, rollout: Any, wall_clock_sec: float
) -> dict[str, Any]:
    return jax.device_get(
        {
            "evaluation/curriculum_iterations": np.int64(completed_iteration),
            "evaluation/total_env_steps": np.int64(total_env_steps),
            "evaluation/wall_clock_sec": np.float64(wall_clock_sec),
            "evaluation/episode_successes": rollout.episode_successes,
            "evaluation/episode_lengths": rollout.episode_lengths,
        }
    )


def save_metrics_npz_atomic(path: Path, metrics: Mapping[str, np.ndarray]) -> None:
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b", prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            np.savez(temporary_file, **metrics)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


class MetricsRecorder:
    def __init__(self, run_dir: Path | None) -> None:
        self.path = None if run_dir is None else Path(run_dir) / "metrics.npz"
        self.training: dict[str, list[np.ndarray]] = {}
        self.evaluation: dict[str, list[np.ndarray]] = {}
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, training_row: Mapping[str, Any], evaluation_row: Mapping[str, Any] | None = None) -> None:
        for history, row in ((self.training, training_row), (self.evaluation, evaluation_row)):
            if row is None:
                continue
            if history and history.keys() != row.keys():
                raise ValueError("Metric record fields do not match the recorded history.")
            for key, value in row.items():
                history.setdefault(key, []).append(np.array(value, copy=True))

    def as_arrays(self) -> dict[str, np.ndarray]:
        return {
            key: np.stack(values) for history in (self.training, self.evaluation) for key, values in history.items()
        }

    def save(self) -> None:
        if self.path is not None:
            save_metrics_npz_atomic(self.path, self.as_arrays())

    def restore(self, completed_iteration: int, evaluation_iterations: Sequence[int]) -> None:
        expected_evaluations = np.asarray(
            [iteration for iteration in evaluation_iterations if iteration <= completed_iteration], dtype=np.int64
        )
        if not expected_evaluations.size or expected_evaluations[-1] != completed_iteration:
            raise ValueError("Checkpoint iteration must match a scheduled evaluation.")
        trimmed = False
        with np.load(self.path, allow_pickle=False) as snapshot:
            for history, coordinate_key, expected_coordinates, is_evaluation in (
                (self.training, "run/curriculum_iteration", np.arange(1, completed_iteration + 1), False),
                (self.evaluation, "evaluation/curriculum_iterations", expected_evaluations, True),
            ):
                coordinates = snapshot[coordinate_key]
                count = len(expected_coordinates)
                if coordinates.ndim != 1 or not np.array_equal(coordinates[:count], expected_coordinates):
                    raise ValueError(f"Metric iteration history does not match the checkpoint: {coordinate_key}.")
                for key in snapshot.files:
                    if key.startswith("evaluation/") != is_evaluation:
                        continue
                    values = snapshot[key]
                    if values.ndim == 0 or len(values) != len(coordinates):
                        raise ValueError(f"Inconsistent metric history length: {key}.")
                    history[key] = list(values[:count])
                trimmed |= len(coordinates) > count
        if trimmed:
            self.save()

    @property
    def cumulative_wall_clock_offset(self) -> float:
        times = self.training.get("time/cumulative_wall_clock_sec", [])
        return float(times[-1]) if times else 0.0
