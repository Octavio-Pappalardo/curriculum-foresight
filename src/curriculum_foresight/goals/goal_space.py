from __future__ import annotations

import hashlib
import importlib
import json
from collections.abc import Sequence
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from curriculum_foresight.goals.types import (
    CONDITION_PARAMS_WIDTH,
    GoalSpace,
    GoalSpaceArrays,
    GoalSpaceRecipe,
    SelectedGoalBatch,
    SuccessEvaluator,
    TaskSpec,
)


GOAL_SPACE_RECIPE_MODULES: dict[str, str] = {"craftax_256": "curriculum_foresight.goals.goal_spaces.craftax_256"}


def load_goal_space_recipe(goal_space_id: str) -> GoalSpaceRecipe:
    module_path = GOAL_SPACE_RECIPE_MODULES.get(goal_space_id)
    if module_path is None:
        supported = ", ".join(sorted(GOAL_SPACE_RECIPE_MODULES))
        msg = f"unknown goal_space_id {goal_space_id!r}. Supported ids: {supported}."
        raise ValueError(msg)

    module = importlib.import_module(module_path)
    recipe = module.build_recipe()
    if recipe.goal_space_id != goal_space_id:
        msg = f"recipe id mismatch: requested {goal_space_id!r}, got {recipe.goal_space_id!r}."
        raise ValueError(msg)
    return recipe


def build_goal_space(recipe: GoalSpaceRecipe) -> GoalSpace:
    if not recipe.task_specs:
        msg = "goal-space recipes must contain at least one task."
        raise ValueError(msg)

    success_evaluators: list[int] = []
    condition_params_rows: list[list[int]] = []
    seen_task_ids: set[str] = set()

    for task_spec in recipe.task_specs:
        if task_spec.task_id in seen_task_ids:
            msg = f"duplicate task_id in goal-space recipe: {task_spec.task_id!r}."
            raise ValueError(msg)
        seen_task_ids.add(task_spec.task_id)

        condition_params = _pad_condition_params(task_spec)

        success_evaluators.append(SuccessEvaluator(task_spec.success_evaluator).value)
        condition_params_rows.append(condition_params)

    unweighted_fingerprint = _compute_unweighted_fingerprint(recipe.task_specs)

    return GoalSpace(
        goal_space_id=recipe.goal_space_id,
        task_specs=recipe.task_specs,
        task_ids=tuple(task.task_id for task in recipe.task_specs),
        goal_texts=tuple(task.goal_text for task in recipe.task_specs),
        arrays=GoalSpaceArrays(
            success_evaluator=jnp.array(success_evaluators, dtype=jnp.int32),
            condition_params=jnp.array(condition_params_rows, dtype=jnp.int32),
        ),
        unweighted_goal_space_fingerprint=unweighted_fingerprint,
    )


def apply_goal_weights(goal_space: GoalSpace, weights: Sequence[float] | np.ndarray | jax.Array) -> GoalSpace:
    resolved_weights = np.asarray(weights, dtype=np.float32)
    expected_shape = (len(goal_space.task_ids),)
    if resolved_weights.shape != expected_shape:
        msg = f"goal weights must have shape {expected_shape}, got {resolved_weights.shape}."
        raise ValueError(msg)
    if (
        not np.all(np.isfinite(resolved_weights))
        or np.any(resolved_weights < 0.0)
        or not np.any(resolved_weights > 0.0)
    ):
        msg = "goal weights must be finite and nonnegative with at least one positive value."
        raise ValueError(msg)

    arrays = goal_space.arrays.replace(weights=jnp.asarray(resolved_weights, dtype=jnp.float32))
    weighted_fingerprint = _compute_weighted_fingerprint(
        goal_space.unweighted_goal_space_fingerprint, resolved_weights.tolist()
    )
    return replace(goal_space, arrays=arrays, weighted_goal_space_fingerprint=weighted_fingerprint)


def gather_selected_goal_batch(goal_space_arrays: GoalSpaceArrays, goal_indices: jax.Array) -> SelectedGoalBatch:
    return SelectedGoalBatch(
        goal_indices=goal_indices,
        success_evaluator=goal_space_arrays.success_evaluator[goal_indices],
        condition_params=goal_space_arrays.condition_params[goal_indices],
        goal_embeddings=goal_space_arrays.goal_embeddings[goal_indices],
    )


def _pad_condition_params(task_spec: TaskSpec) -> list[int]:
    if len(task_spec.condition_params) > CONDITION_PARAMS_WIDTH:
        msg = (
            f"Task {task_spec.task_id!r} has {len(task_spec.condition_params)} condition parameters; maximum is "
            f"{CONDITION_PARAMS_WIDTH}."
        )
        raise ValueError(msg)
    padded = [0] * CONDITION_PARAMS_WIDTH
    for index, value in enumerate(task_spec.condition_params):
        padded[index] = int(value)
    return padded


def _compute_unweighted_fingerprint(task_specs: Sequence[TaskSpec]) -> str:
    payload = {
        "tasks": [
            {
                "task_id": task.task_id,
                "goal_text": task.goal_text,
                "success_evaluator": SuccessEvaluator(task.success_evaluator).name,
                "params_i32": _pad_condition_params(task),
            }
            for task in task_specs
        ]
    }
    return _sha256_json(payload)


def _compute_weighted_fingerprint(unweighted_fingerprint: str, weights: Sequence[float]) -> str:
    payload = {
        "unweighted_goal_space_fingerprint": unweighted_fingerprint,
        "weights": [float(weight) for weight in weights],
    }
    return _sha256_json(payload)


def _sha256_json(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
