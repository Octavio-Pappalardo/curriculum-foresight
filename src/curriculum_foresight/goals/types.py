from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import jax
from flax import struct


CONDITION_PARAMS_WIDTH = 3


class SuccessEvaluator(IntEnum):
    INVENTORY_AT_LEAST = 1
    BLOCK_RELATION = 2
    ITEM_RELATION = 3
    MOB_RELATION = 4
    EQUIPMENT_TIER_AT_LEAST = 5
    ENCHANTMENT_PRESENT = 6
    FLOOR_AT_LEAST = 7
    ATTRIBUTE_AT_LEAST = 8
    MOB_DAMAGED = 9
    MOB_DEFEATED = 10
    FLOOR_MONSTERS_KILLED_AT_LEAST = 11
    ITEM_RELATION_ON_FLOOR = 12


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    goal_text: str
    success_evaluator: SuccessEvaluator
    condition_params: tuple[int, ...] = ()


@dataclass(frozen=True)
class GoalSpaceRecipe:
    goal_space_id: str
    task_specs: tuple[TaskSpec, ...]


class GoalSpaceArrays(struct.PyTreeNode):
    success_evaluator: jax.Array
    condition_params: jax.Array
    weights: jax.Array | None = None
    goal_embeddings: jax.Array | None = None


@dataclass(frozen=True)
class GoalSpace:
    goal_space_id: str
    task_specs: tuple[TaskSpec, ...]
    task_ids: tuple[str, ...]
    goal_texts: tuple[str, ...]
    arrays: GoalSpaceArrays
    unweighted_goal_space_fingerprint: str
    weighted_goal_space_fingerprint: str | None = None


class SelectedGoalBatch(struct.PyTreeNode):
    goal_indices: jax.Array
    success_evaluator: jax.Array
    condition_params: jax.Array
    goal_embeddings: jax.Array
