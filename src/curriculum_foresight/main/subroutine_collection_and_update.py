from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
from jax.tree_util import Partial

from curriculum_foresight.craftax_utils.resets import reset_if_done
from curriculum_foresight.goals.success import compute_goal_success
from curriculum_foresight.goals.types import SelectedGoalBatch
from curriculum_foresight.main.config import TrainConfig
from curriculum_foresight.networks.subroutine_actor_critic import SubroutineActorCritic
from curriculum_foresight.performance_and_lp.success_tracking import (
    init_goal_success_tracking_state,
    update_goal_success_tracking,
)
from curriculum_foresight.policy.advantage_estimation import compute_gae
from curriculum_foresight.policy.subroutine_ppo import normalize_outer_advantages, subroutine_update_agent
from curriculum_foresight.policy.subroutines import CallEligibility, to_environment_actions, advance_subroutine_state
from curriculum_foresight.policy.transformer_memory import (
    advance_transformer_memory_mask_episodic,
    init_transformer_memories,
    init_transformer_memory_mask,
    init_transformer_memory_mask_idx,
)
from curriculum_foresight.support.types import SubroutineRunnerState, SubroutineTransition


def subroutine_initialize_runner_state(
    rng: jax.Array,
    policy_train_state: TrainState,
    selected_goal_batch: SelectedGoalBatch,
    *,
    env: Any,
    env_params: Any,
    config: TrainConfig,
) -> SubroutineRunnerState:
    rng, reset_rng_base = jax.random.split(rng)
    reset_rngs = jax.random.split(reset_rng_base, config.num_envs_per_batch)
    prev_obs, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rngs, env_params)

    return SubroutineRunnerState(
        rng=rng,
        policy_train_state=policy_train_state,
        env_state=env_state,
        prev_obs=prev_obs,
        prev_reset_done=jnp.zeros((config.num_envs_per_batch,), dtype=jnp.bool_),
        memories=init_transformer_memories(config.num_envs_per_batch, config.policy),
        memories_mask=init_transformer_memory_mask(config.num_envs_per_batch, config.policy),
        memories_mask_idx=init_transformer_memory_mask_idx(config.num_envs_per_batch, config.policy),
        success_tracking_state=init_goal_success_tracking_state(config.num_envs_per_batch, config.performance),
        active_goal_indices=selected_goal_batch.goal_indices,
        remaining_subroutine_steps=jnp.zeros((config.num_envs_per_batch,), dtype=jnp.int32),
    )


def subroutine_step_envs(
    runner_state: SubroutineRunnerState,
    current_update_step_num: jax.Array,
    *,
    env: Any,
    env_params: Any,
    selected_goal_batch: SelectedGoalBatch,
    all_goal_embeddings: jax.Array,
    eligibility: CallEligibility,
    config: TrainConfig,
) -> tuple[SubroutineRunnerState, tuple[SubroutineTransition, jax.Array]]:
    num_primitive_actions = env.action_space(env_params).n
    memories_mask, memories_mask_idx = advance_transformer_memory_mask_episodic(
        runner_state.memories_mask, runner_state.memories_mask_idx, runner_state.prev_reset_done, config.policy
    )
    active_goal_embeddings = all_goal_embeddings[runner_state.active_goal_indices]
    rng, action_rng = jax.random.split(runner_state.rng)
    distribution, values, memories_out = runner_state.policy_train_state.apply_fn(
        {"params": runner_state.policy_train_state.params},
        runner_state.memories,
        {"observation": runner_state.prev_obs[:, None, ...]},
        memories_mask,
        selected_goal_batch.goal_embeddings,
        active_goal_embeddings,
        runner_state.remaining_subroutine_steps,
        eligibility,
        method=SubroutineActorCritic.forward_step,
    )
    action = distribution.sample(seed=action_rng)
    log_prob = distribution.log_prob(action)
    primitive_action = to_environment_actions(action, num_primitive_actions=num_primitive_actions)
    memories = jnp.roll(runner_state.memories, -1, axis=1).at[:, -1].set(memories_out)

    rng, step_rng_base, reset_rng_base = jax.random.split(rng, 3)
    step_rngs = jax.random.split(step_rng_base, config.num_envs_per_batch)
    post_action_obs, post_action_env_state, _, native_done, _ = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
        step_rngs, runner_state.env_state, primitive_action, env_params
    )
    goal_success = compute_goal_success(
        runner_state.env_state,
        post_action_env_state,
        selected_goal_batch.success_evaluator,
        selected_goal_batch.condition_params,
    )
    reward = goal_success.astype(jnp.float32)
    reset_done = native_done.astype(jnp.bool_) | goal_success
    active_goal_indices, remaining_subroutine_steps = advance_subroutine_state(
        selected_goal_batch.goal_indices,
        runner_state.active_goal_indices,
        runner_state.remaining_subroutine_steps,
        action,
        reset_done,
        num_primitive_actions=num_primitive_actions,
        max_subroutine_steps=config.subroutine.max_subroutine_steps,
    )

    reset_rngs = jax.random.split(reset_rng_base, config.num_envs_per_batch)
    next_obs, env_state = jax.vmap(reset_if_done, in_axes=(None, 0, 0, 0, 0, None))(
        env, reset_rngs, post_action_obs, post_action_env_state, reset_done, env_params
    )
    success_tracking_state = update_goal_success_tracking(
        runner_state.success_tracking_state, reset_done=reset_done, goal_success=goal_success
    )

    memories_indices = jnp.broadcast_to(
        jnp.arange(config.policy.past_context_length, dtype=jnp.int32)[None, :] + current_update_step_num,
        (config.num_envs_per_batch, config.policy.past_context_length),
    )
    transition = SubroutineTransition(
        obs=runner_state.prev_obs,
        action=action,
        episode_done=reset_done,
        reward=reward,
        value=values,
        log_prob=log_prob,
        memories_mask=memories_mask.squeeze(axis=2),
        memories_indices=memories_indices,
        active_goal_indices=runner_state.active_goal_indices,
        remaining_subroutine_steps=runner_state.remaining_subroutine_steps,
    )
    runner_state = runner_state.replace(
        rng=rng,
        env_state=env_state,
        prev_obs=next_obs,
        prev_reset_done=reset_done,
        memories=memories,
        memories_mask=memories_mask,
        memories_mask_idx=memories_mask_idx,
        success_tracking_state=success_tracking_state,
        active_goal_indices=active_goal_indices,
        remaining_subroutine_steps=remaining_subroutine_steps,
    )
    return runner_state, (transition, memories_out)


def subroutine_collect_data(
    runner_state: SubroutineRunnerState,
    num_steps: int,
    *,
    env: Any,
    env_params: Any,
    selected_goal_batch: SelectedGoalBatch,
    all_goal_embeddings: jax.Array,
    eligibility: CallEligibility,
    config: TrainConfig,
) -> tuple[SubroutineRunnerState, SubroutineTransition, jax.Array]:
    runner_state, (transitions, memories_batch) = jax.lax.scan(
        Partial(
            subroutine_step_envs,
            env=env,
            env_params=env_params,
            selected_goal_batch=selected_goal_batch,
            all_goal_embeddings=all_goal_embeddings,
            eligibility=eligibility,
            config=config,
        ),
        runner_state,
        jnp.arange(num_steps, dtype=jnp.int32),
    )
    return runner_state, transitions, memories_batch


def subroutine_compute_bootstrap_values(
    runner_state: SubroutineRunnerState,
    selected_goal_batch: SelectedGoalBatch,
    all_goal_embeddings: jax.Array,
    eligibility: CallEligibility,
    *,
    config: TrainConfig,
) -> jax.Array:
    bootstrap_mask, _ = advance_transformer_memory_mask_episodic(
        runner_state.memories_mask, runner_state.memories_mask_idx, runner_state.prev_reset_done, config.policy
    )
    _, values, _ = runner_state.policy_train_state.apply_fn(
        {"params": runner_state.policy_train_state.params},
        runner_state.memories,
        {"observation": runner_state.prev_obs[:, None, ...]},
        bootstrap_mask,
        selected_goal_batch.goal_embeddings,
        all_goal_embeddings[runner_state.active_goal_indices],
        runner_state.remaining_subroutine_steps,
        eligibility,
        method=SubroutineActorCritic.forward_step,
    )
    return values


def subroutine_collect_data_and_update_agent(
    runner_state: SubroutineRunnerState,
    _unused: Any,
    *,
    env: Any,
    env_params: Any,
    selected_goal_batch: SelectedGoalBatch,
    all_goal_embeddings: jax.Array,
    eligibility: CallEligibility,
    config: TrainConfig,
) -> tuple[SubroutineRunnerState, dict[str, jax.Array]]:
    previous_memories = runner_state.memories
    runner_state, transitions, memory_rows = subroutine_collect_data(
        runner_state,
        config.num_steps_per_update,
        env=env,
        env_params=env_params,
        selected_goal_batch=selected_goal_batch,
        all_goal_embeddings=all_goal_embeddings,
        eligibility=eligibility,
        config=config,
    )
    memories_for_ppo = jnp.concatenate((jnp.swapaxes(previous_memories, 0, 1), memory_rows), axis=0)
    last_values = subroutine_compute_bootstrap_values(
        runner_state, selected_goal_batch, all_goal_embeddings, eligibility, config=config
    )
    advantages_raw, value_targets = compute_gae(
        transitions.reward,
        transitions.value,
        last_values,
        transitions.episode_done,
        gamma=config.policy.gamma,
        gae_lambda=config.policy.gae_lambda,
    )
    advantages = normalize_outer_advantages(advantages_raw, transitions.remaining_subroutine_steps)
    rng, policy_train_state, metrics = subroutine_update_agent(
        runner_state.rng,
        runner_state.policy_train_state,
        transitions,
        memories_for_ppo,
        selected_goal_batch.goal_embeddings,
        all_goal_embeddings,
        eligibility,
        advantages,
        value_targets,
        num_primitive_actions=env.action_space(env_params).n,
        config=config,
    )
    outer_decision_mask = transitions.remaining_subroutine_steps == 0
    metrics.update(
        {
            "subroutine/outer_decision_count": jnp.sum(outer_decision_mask, dtype=jnp.int32),
            "subroutine/outer_call_count": jnp.sum(
                outer_decision_mask & (transitions.action >= env.action_space(env_params).n), dtype=jnp.int32
            ),
            "subroutine/subroutine_step_count": jnp.sum(~outer_decision_mask, dtype=jnp.int32),
        }
    )
    return runner_state.replace(rng=rng, policy_train_state=policy_train_state), metrics
