from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from flax import struct
from jax.tree_util import Partial

from curriculum_foresight.goals.success import compute_goal_success
from curriculum_foresight.goals.types import GoalSpaceArrays
from curriculum_foresight.main.config import TrainConfig
from curriculum_foresight.networks.subroutine_actor_critic import SubroutineActorCritic
from curriculum_foresight.networks.primitive_actor_critic import PrimitiveActorCritic
from curriculum_foresight.policy.subroutines import (
    CallEligibility,
    to_environment_actions,
    prepare_call_eligibility,
    advance_subroutine_state,
)
from curriculum_foresight.policy.transformer_memory import (
    advance_transformer_memory_mask_episodic,
    init_transformer_memories,
    init_transformer_memory_mask,
    init_transformer_memory_mask_idx,
)


EVALUATION_STREAM_ID = 1


def derive_evaluation_iterations(num_curriculum_iterations: int, every_n_curriculum_iterations: int) -> tuple[int, ...]:
    scheduled = {1, num_curriculum_iterations}
    scheduled.update(range(every_n_curriculum_iterations, num_curriculum_iterations + 1, every_n_curriculum_iterations))
    return tuple(sorted(scheduled))


class EvaluationRolloutResults(struct.PyTreeNode):
    episode_successes: jax.Array
    episode_lengths: jax.Array


class _EvaluationPolicyState(struct.PyTreeNode):
    memories: jax.Array
    memories_mask: jax.Array
    memories_mask_idx: jax.Array
    prev_reset_done: jax.Array


class _SubroutineEvaluationPolicyState(struct.PyTreeNode):
    memories: jax.Array
    memories_mask: jax.Array
    memories_mask_idx: jax.Array
    prev_reset_done: jax.Array
    active_goal_indices: jax.Array
    remaining_subroutine_steps: jax.Array


class _EvaluationEpisodeState(struct.PyTreeNode):
    policy_state: _EvaluationPolicyState | _SubroutineEvaluationPolicyState
    env_state: Any
    obs: jax.Array
    rollout_key: jax.Array
    done: jax.Array
    success: jax.Array
    length: jax.Array


def derive_evaluation_root(train_seed: int) -> jax.Array:
    return jax.random.fold_in(jax.random.key(train_seed), EVALUATION_STREAM_ID)


def derive_evaluation_episode_keys(
    evaluation_root: jax.Array,
    goal_index: jax.Array | int,
    worker_index: jax.Array | int,
    episode_index: jax.Array | int,
) -> tuple[jax.Array, jax.Array]:
    goal_root = jax.random.fold_in(evaluation_root, goal_index)
    worker_root = jax.random.fold_in(goal_root, worker_index)
    episode_root = jax.random.fold_in(worker_root, episode_index)
    reset_key, rollout_key = jax.random.split(episode_root)
    return reset_key, rollout_key


def _evaluate_goal_worker(
    worker_index: jax.Array,
    *,
    policy_train_state: Any,
    evaluation_root: jax.Array,
    goal_index: jax.Array,
    goal_embedding: jax.Array,
    success_evaluator: jax.Array,
    condition_params: jax.Array,
    env: Any,
    env_params: Any,
    config: TrainConfig,
) -> EvaluationRolloutResults:
    policy_state = _EvaluationPolicyState(
        memories=init_transformer_memories(1, config.policy),
        memories_mask=init_transformer_memory_mask(1, config.policy),
        memories_mask_idx=init_transformer_memory_mask_idx(1, config.policy),
        prev_reset_done=jnp.zeros((1,), dtype=jnp.bool_),
    )

    def run_episode(
        policy_state: _EvaluationPolicyState, episode_index: jax.Array
    ) -> tuple[_EvaluationPolicyState, tuple[jax.Array, jax.Array]]:
        reset_key, rollout_key = derive_evaluation_episode_keys(
            evaluation_root, goal_index, worker_index, episode_index
        )
        obs, env_state = env.reset(reset_key, env_params)

        episode_state = _EvaluationEpisodeState(
            policy_state=policy_state,
            env_state=env_state,
            obs=obs,
            rollout_key=rollout_key,
            done=jnp.asarray(False),
            success=jnp.asarray(False),
            length=jnp.asarray(0, dtype=jnp.int32),
        )

        def episode_not_done(state: _EvaluationEpisodeState) -> jax.Array:
            return ~state.done

        def step_episode(state: _EvaluationEpisodeState) -> _EvaluationEpisodeState:
            policy_state = state.policy_state
            memories_mask, memories_mask_idx = advance_transformer_memory_mask_episodic(
                policy_state.memories_mask, policy_state.memories_mask_idx, policy_state.prev_reset_done, config.policy
            )
            policy_inputs = {"observation": state.obs[None, None, ...]}

            next_rollout_key, policy_key, transition_key = jax.random.split(state.rollout_key, 3)
            pi, _, memories_out = policy_train_state.apply_fn(
                {"params": policy_train_state.params},
                policy_state.memories,
                policy_inputs,
                memories_mask,
                goal_embedding[None, :],
                method=PrimitiveActorCritic.forward_step,
            )
            action = pi.sample(seed=policy_key)
            memories = jnp.roll(policy_state.memories, -1, axis=1).at[:, -1].set(memories_out)
            post_action_obs, post_action_env_state, _, native_done, _ = env.step(
                transition_key, state.env_state, action[0], env_params
            )
            pre_state_batched = jax.tree.map(lambda leaf: leaf[None, ...], state.env_state)
            post_state_batched = jax.tree.map(lambda leaf: leaf[None, ...], post_action_env_state)
            goal_success = compute_goal_success(
                pre_state_batched, post_state_batched, success_evaluator[None], condition_params[None, :]
            )[0]
            reset_done = goal_success | native_done.astype(jnp.bool_)
            next_policy_state = policy_state.replace(
                memories=memories,
                memories_mask=memories_mask,
                memories_mask_idx=memories_mask_idx,
                prev_reset_done=reset_done[None],
            )
            return state.replace(
                policy_state=next_policy_state,
                env_state=post_action_env_state,
                obs=post_action_obs,
                rollout_key=next_rollout_key,
                done=reset_done,
                success=goal_success,
                length=state.length + jnp.asarray(1, dtype=jnp.int32),
            )

        episode_state = jax.lax.while_loop(episode_not_done, step_episode, episode_state)
        return episode_state.policy_state, (episode_state.success, episode_state.length)

    _, (episode_successes, episode_lengths) = jax.lax.scan(
        run_episode, policy_state, jnp.arange(config.evaluation.num_episodes_per_env, dtype=jnp.int32)
    )
    return EvaluationRolloutResults(
        episode_successes=episode_successes.astype(jnp.bool_), episode_lengths=episode_lengths.astype(jnp.int32)
    )


def _evaluate_subroutine_goal_worker(
    worker_index: jax.Array,
    *,
    policy_train_state: Any,
    evaluation_root: jax.Array,
    goal_index: jax.Array,
    goal_embedding: jax.Array,
    success_evaluator: jax.Array,
    condition_params: jax.Array,
    all_goal_embeddings: jax.Array,
    eligibility: CallEligibility,
    env: Any,
    env_params: Any,
    config: TrainConfig,
) -> EvaluationRolloutResults:
    num_actions = env.action_space(env_params).n
    assigned_goal_indices = goal_index[None]
    policy_state = _SubroutineEvaluationPolicyState(
        memories=init_transformer_memories(1, config.policy),
        memories_mask=init_transformer_memory_mask(1, config.policy),
        memories_mask_idx=init_transformer_memory_mask_idx(1, config.policy),
        prev_reset_done=jnp.zeros((1,), dtype=jnp.bool_),
        active_goal_indices=assigned_goal_indices,
        remaining_subroutine_steps=jnp.zeros((1,), dtype=jnp.int32),
    )

    def run_episode(policy_state, episode_index):
        reset_key, rollout_key = derive_evaluation_episode_keys(
            evaluation_root, goal_index, worker_index, episode_index
        )
        obs, env_state = env.reset(reset_key, env_params)
        state = _EvaluationEpisodeState(
            policy_state=policy_state,
            env_state=env_state,
            obs=obs,
            rollout_key=rollout_key,
            done=jnp.asarray(False),
            success=jnp.asarray(False),
            length=jnp.asarray(0, dtype=jnp.int32),
        )

        def step_episode(state):
            context = state.policy_state
            mask, mask_idx = advance_transformer_memory_mask_episodic(
                context.memories_mask, context.memories_mask_idx, context.prev_reset_done, config.policy
            )
            next_key, policy_key, transition_key = jax.random.split(state.rollout_key, 3)
            distribution, _, memories_out = policy_train_state.apply_fn(
                {"params": policy_train_state.params},
                context.memories,
                {"observation": state.obs[None, None, ...]},
                mask,
                goal_embedding[None],
                all_goal_embeddings[context.active_goal_indices],
                context.remaining_subroutine_steps,
                eligibility,
                method=SubroutineActorCritic.forward_step,
            )
            action = distribution.sample(seed=policy_key)
            native_action = to_environment_actions(action, num_primitive_actions=num_actions)
            obs, env_state, _, native_done, _ = env.step(transition_key, state.env_state, native_action[0], env_params)
            success = compute_goal_success(
                jax.tree.map(lambda x: x[None, ...], state.env_state),
                jax.tree.map(lambda x: x[None, ...], env_state),
                success_evaluator[None],
                condition_params[None],
            )[0]
            done = success | native_done.astype(jnp.bool_)
            active_goals, budgets = advance_subroutine_state(
                assigned_goal_indices,
                context.active_goal_indices,
                context.remaining_subroutine_steps,
                action,
                done[None],
                num_primitive_actions=num_actions,
                max_subroutine_steps=config.subroutine.max_subroutine_steps,
            )
            context = context.replace(
                memories=jnp.roll(context.memories, -1, axis=1).at[:, -1].set(memories_out),
                memories_mask=mask,
                memories_mask_idx=mask_idx,
                prev_reset_done=done[None],
                active_goal_indices=active_goals,
                remaining_subroutine_steps=budgets,
            )
            return state.replace(
                policy_state=context,
                env_state=env_state,
                obs=obs,
                rollout_key=next_key,
                done=done,
                success=success,
                length=state.length + jnp.int32(1),
            )

        state = jax.lax.while_loop(lambda state: ~state.done, step_episode, state)
        return state.policy_state, (state.success, state.length)

    _, (successes, lengths) = jax.lax.scan(
        run_episode, policy_state, jnp.arange(config.evaluation.num_episodes_per_env, dtype=jnp.int32)
    )
    return EvaluationRolloutResults(successes.astype(jnp.bool_), lengths.astype(jnp.int32))


def run_periodic_evaluation(
    policy_train_state: Any,
    performance_estimates: jax.Array,
    goal_space_arrays: GoalSpaceArrays,
    evaluation_root: jax.Array,
    *,
    env: Any,
    env_params: Any,
    config: TrainConfig,
) -> EvaluationRolloutResults:
    goal_indices = jnp.arange(goal_space_arrays.goal_embeddings.shape[0], dtype=jnp.int32)
    worker_indices = jnp.arange(config.evaluation.num_envs_per_goal, dtype=jnp.int32)
    worker = _evaluate_goal_worker
    subroutine_kwargs = {}
    if config.learner_kind == "subroutine":
        worker = _evaluate_subroutine_goal_worker
        eligibility = prepare_call_eligibility(
            performance_estimates,
            competence_threshold=config.subroutine.competence_threshold,
            max_subroutine_steps=config.subroutine.max_subroutine_steps,
            num_primitive_actions=env.action_space(env_params).n,
        )
        subroutine_kwargs = dict(all_goal_embeddings=goal_space_arrays.goal_embeddings, eligibility=eligibility)

    def evaluate_goal(
        goal_index: jax.Array, goal_embedding: jax.Array, success_evaluator: jax.Array, condition_params: jax.Array
    ) -> EvaluationRolloutResults:
        evaluate_worker = Partial(
            worker,
            policy_train_state=policy_train_state,
            evaluation_root=evaluation_root,
            goal_index=goal_index,
            goal_embedding=goal_embedding,
            success_evaluator=success_evaluator,
            condition_params=condition_params,
            env=env,
            env_params=env_params,
            config=config,
            **subroutine_kwargs,
        )
        return jax.vmap(evaluate_worker)(worker_indices)

    return jax.vmap(evaluate_goal)(
        goal_indices,
        goal_space_arrays.goal_embeddings,
        goal_space_arrays.success_evaluator,
        goal_space_arrays.condition_params,
    )
