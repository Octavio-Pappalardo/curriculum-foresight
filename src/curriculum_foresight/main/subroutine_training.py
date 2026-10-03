from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
from jax.tree_util import Partial

from curriculum_foresight.goals.types import SelectedGoalBatch
from curriculum_foresight.main.config import TrainConfig
from curriculum_foresight.main.subroutine_collection_and_update import (
    subroutine_collect_data_and_update_agent,
    subroutine_initialize_runner_state,
)
from curriculum_foresight.performance_and_lp.observations import (
    GoalPerformanceObservations,
    compute_goal_performance_observations,
)
from curriculum_foresight.policy.subroutines import prepare_call_eligibility


def subroutine_train_on_goal_batch(
    rng: jax.Array,
    policy_train_state: TrainState,
    selected_goal_batch: SelectedGoalBatch,
    all_goal_embeddings: jax.Array,
    performance_estimates: jax.Array,
    *,
    env: Any,
    env_params: Any,
    config: TrainConfig,
) -> tuple[jax.Array, TrainState, GoalPerformanceObservations, dict[str, jax.Array]]:
    eligibility = prepare_call_eligibility(
        performance_estimates,
        competence_threshold=config.subroutine.competence_threshold,
        max_subroutine_steps=config.subroutine.max_subroutine_steps,
        num_primitive_actions=env.action_space(env_params).n,
    )
    runner_state = subroutine_initialize_runner_state(
        rng, policy_train_state, selected_goal_batch, env=env, env_params=env_params, config=config
    )
    runner_state, metrics = jax.lax.scan(
        Partial(
            subroutine_collect_data_and_update_agent,
            env=env,
            env_params=env_params,
            selected_goal_batch=selected_goal_batch,
            all_goal_embeddings=all_goal_embeddings,
            eligibility=eligibility,
            config=config,
        ),
        runner_state,
        None,
        length=config.num_policy_updates_per_curriculum_iteration,
    )
    outer_count = metrics.pop("subroutine/outer_decision_count").sum()
    call_count = metrics.pop("subroutine/outer_call_count").sum()
    subroutine_count = metrics.pop("subroutine/subroutine_step_count").sum()
    metrics = jax.tree.map(lambda x: x.mean(axis=0), metrics)
    metrics.update(
        {
            "subroutine/eligible_goal_count": eligibility.num_eligible_goals,
            "subroutine/outer_call_frequency": call_count / jnp.maximum(outer_count, 1),
            "subroutine/subroutine_step_fraction": subroutine_count / config.num_env_steps_per_curriculum_iteration,
        }
    )
    observations = compute_goal_performance_observations(
        runner_state.success_tracking_state,
        selected_goal_batch.goal_indices,
        num_goals=all_goal_embeddings.shape[0],
        config=config.performance,
    )
    return runner_state.rng, runner_state.policy_train_state, observations, metrics
