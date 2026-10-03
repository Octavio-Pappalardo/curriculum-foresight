from __future__ import annotations

from typing import Any

import jax
from flax.training.train_state import TrainState
from jax.tree_util import Partial

from curriculum_foresight.goals.types import SelectedGoalBatch
from curriculum_foresight.main.config import TrainConfig
from curriculum_foresight.main.primitive_collection_and_update import (
    primitive_collect_data_and_update_agent,
    primitive_initialize_runner_state,
)
from curriculum_foresight.performance_and_lp.observations import (
    GoalPerformanceObservations,
    compute_goal_performance_observations,
)


def primitive_train_on_goal_batch(
    rng: jax.Array,
    policy_train_state: TrainState,
    selected_goal_batch: SelectedGoalBatch,
    *,
    num_goals: int,
    env: Any,
    env_params: Any,
    config: TrainConfig,
) -> tuple[jax.Array, TrainState, GoalPerformanceObservations, dict[str, jax.Array]]:
    runner_state = primitive_initialize_runner_state(
        rng, policy_train_state, env=env, env_params=env_params, config=config
    )
    runner_state, metrics = jax.lax.scan(
        Partial(
            primitive_collect_data_and_update_agent,
            env=env,
            env_params=env_params,
            selected_goal_batch=selected_goal_batch,
            config=config,
        ),
        runner_state,
        None,
        length=config.num_policy_updates_per_curriculum_iteration,
    )
    metrics = jax.tree.map(lambda value: value.mean(axis=0), metrics)
    performance_observations = compute_goal_performance_observations(
        runner_state.success_tracking_state,
        selected_goal_batch.goal_indices,
        num_goals=num_goals,
        config=config.performance,
    )
    return runner_state.rng, runner_state.policy_train_state, performance_observations, metrics
