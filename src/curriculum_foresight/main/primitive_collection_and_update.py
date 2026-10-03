from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax.tree_util import Partial

from curriculum_foresight.craftax_utils.resets import reset_if_done
from curriculum_foresight.goals.success import compute_goal_success
from curriculum_foresight.goals.types import SelectedGoalBatch
from curriculum_foresight.main.config import TrainConfig
from curriculum_foresight.networks.primitive_actor_critic import PrimitiveActorCritic
from curriculum_foresight.performance_and_lp.success_tracking import (
    init_goal_success_tracking_state,
    update_goal_success_tracking,
)
from curriculum_foresight.policy.advantage_estimation import compute_gae
from curriculum_foresight.policy.transformer_memory import (
    advance_transformer_memory_mask_episodic,
    init_transformer_memories,
    init_transformer_memory_mask,
    init_transformer_memory_mask_idx,
)
from curriculum_foresight.policy.primitive_ppo import primitive_update_agent
from curriculum_foresight.support.types import RunnerState, Transition


def primitive_initialize_runner_state(
    rng: jax.Array, policy_train_state: Any, *, env: Any, env_params: Any, config: TrainConfig
) -> RunnerState:
    rng, reset_rng_base = jax.random.split(rng)
    reset_rngs = jax.random.split(reset_rng_base, config.num_envs_per_batch)
    prev_obs, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rngs, env_params)

    return RunnerState(
        rng=rng,
        policy_train_state=policy_train_state,
        env_state=env_state,
        prev_obs=prev_obs,
        prev_reset_done=jnp.zeros((config.num_envs_per_batch,), dtype=jnp.bool_),
        memories=init_transformer_memories(config.num_envs_per_batch, config.policy),
        memories_mask=init_transformer_memory_mask(config.num_envs_per_batch, config.policy),
        memories_mask_idx=init_transformer_memory_mask_idx(config.num_envs_per_batch, config.policy),
        success_tracking_state=init_goal_success_tracking_state(config.num_envs_per_batch, config.performance),
    )


def primitive_step_envs(
    runner_state: RunnerState,
    current_update_step_num: jax.Array,
    *,
    env: Any,
    env_params: Any,
    selected_goal_batch: SelectedGoalBatch,
    config: TrainConfig,
) -> tuple[RunnerState, tuple[Transition, jax.Array]]:
    memories_mask, memories_mask_idx = advance_transformer_memory_mask_episodic(
        memories_mask=runner_state.memories_mask,
        memories_mask_idx=runner_state.memories_mask_idx,
        prev_reset_done=runner_state.prev_reset_done,
        policy_config=config.policy,
    )

    rng, action_rng = jax.random.split(runner_state.rng)
    pi, values, memories_out = runner_state.policy_train_state.apply_fn(
        {"params": runner_state.policy_train_state.params},
        runner_state.memories,
        {"observation": runner_state.prev_obs[:, None, ...]},
        memories_mask,
        selected_goal_batch.goal_embeddings,
        method=PrimitiveActorCritic.forward_step,
    )
    action = pi.sample(seed=action_rng)
    log_prob = pi.log_prob(action)
    memories = jnp.roll(runner_state.memories, -1, axis=1).at[:, -1].set(memories_out)

    rng, step_rng_base, reset_rng_base = jax.random.split(rng, 3)
    step_rngs = jax.random.split(step_rng_base, config.num_envs_per_batch)
    post_action_obs, post_action_env_state, _, native_done, _ = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
        step_rngs, runner_state.env_state, action, env_params
    )
    goal_success = compute_goal_success(
        runner_state.env_state,
        post_action_env_state,
        selected_goal_batch.success_evaluator,
        selected_goal_batch.condition_params,
    )
    reward = goal_success.astype(jnp.float32)
    native_done = native_done.astype(jnp.bool_)
    reset_done = native_done | goal_success

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
    transition = Transition(
        obs=runner_state.prev_obs,
        action=action,
        episode_done=reset_done,
        reward=reward,
        value=values,
        log_prob=log_prob,
        memories_mask=memories_mask.squeeze(axis=2),
        memories_indices=memories_indices,
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
    )
    return runner_state, (transition, memories_out)


def primitive_collect_data(
    runner_state: RunnerState,
    num_steps: int,
    *,
    env: Any,
    env_params: Any,
    selected_goal_batch: SelectedGoalBatch,
    config: TrainConfig,
) -> tuple[RunnerState, Transition, jax.Array]:
    runner_state, (transitions, memories_batch) = jax.lax.scan(
        Partial(
            primitive_step_envs, env=env, env_params=env_params, selected_goal_batch=selected_goal_batch, config=config
        ),
        runner_state,
        jnp.arange(num_steps, dtype=jnp.int32),
    )
    return runner_state, transitions, memories_batch


def primitive_compute_bootstrap_values(
    runner_state: RunnerState, selected_goal_batch: SelectedGoalBatch, *, config: TrainConfig
) -> jax.Array:
    bootstrap_inputs = {"observation": runner_state.prev_obs[:, None, ...]}
    bootstrap_mask, _ = advance_transformer_memory_mask_episodic(
        memories_mask=runner_state.memories_mask,
        memories_mask_idx=runner_state.memories_mask_idx,
        prev_reset_done=runner_state.prev_reset_done,
        policy_config=config.policy,
    )

    _, last_values, _ = runner_state.policy_train_state.apply_fn(
        {"params": runner_state.policy_train_state.params},
        runner_state.memories,
        bootstrap_inputs,
        bootstrap_mask,
        selected_goal_batch.goal_embeddings,
        method=PrimitiveActorCritic.forward_step,
    )
    return last_values


def primitive_collect_data_and_update_agent(
    runner_state: RunnerState,
    _unused: Any,
    *,
    env: Any,
    env_params: Any,
    selected_goal_batch: SelectedGoalBatch,
    config: TrainConfig,
) -> tuple[RunnerState, dict[str, jax.Array]]:
    memories_previous = runner_state.memories
    runner_state, transitions, memories_batch = primitive_collect_data(
        runner_state=runner_state,
        num_steps=config.num_steps_per_update,
        env=env,
        env_params=env_params,
        selected_goal_batch=selected_goal_batch,
        config=config,
    )
    memories_for_ppo = jnp.concatenate([jnp.swapaxes(memories_previous, 0, 1), memories_batch], axis=0)
    last_values = primitive_compute_bootstrap_values(runner_state, selected_goal_batch, config=config)
    advantages_raw, value_targets = compute_gae(
        rewards=transitions.reward,
        values=transitions.value,
        last_values=last_values,
        episode_done=transitions.episode_done,
        gamma=config.policy.gamma,
        gae_lambda=config.policy.gae_lambda,
    )
    advantages = (advantages_raw - jnp.mean(advantages_raw)) / (jnp.std(advantages_raw) + 1e-8)
    rng, policy_train_state, ppo_metrics = primitive_update_agent(
        rng=runner_state.rng,
        train_state=runner_state.policy_train_state,
        transitions=transitions,
        memories_for_ppo=memories_for_ppo,
        goal_embeddings=selected_goal_batch.goal_embeddings,
        advantages=advantages,
        value_targets=value_targets,
        config=config,
    )
    runner_state = runner_state.replace(rng=rng, policy_train_state=policy_train_state)

    return runner_state, ppo_metrics
