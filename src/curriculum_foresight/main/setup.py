from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import optax
from craftax.craftax_env import make_craftax_env_from_name
from flax.training.train_state import TrainState

from curriculum_foresight.curriculum.state import CurriculumPersistentState
from curriculum_foresight.graphs.feasibility_graph.runtime_graph import FeasibilityGraph, load_feasibility_graph
from curriculum_foresight.goals.encoding import load_goal_embeddings
from curriculum_foresight.goals.goal_space import apply_goal_weights, build_goal_space, load_goal_space_recipe
from curriculum_foresight.goals.types import GoalSpace
from curriculum_foresight.goals.weighting import single_target_goal_weights, uniform_goal_weights
from curriculum_foresight.graphs.lc_graph.base_graph import create_base_lc_adjacency
from curriculum_foresight.main.config import TrainConfig
from curriculum_foresight.networks.subroutine_actor_critic import SubroutineActorCritic
from curriculum_foresight.networks.primitive_actor_critic import PrimitiveActorCritic
from curriculum_foresight.performance_and_lp.learning_progress import init_learning_progress_estimator_state
from curriculum_foresight.performance_and_lp.observations import init_goal_performance_observations
from curriculum_foresight.performance_and_lp.performance_estimation import init_performance_estimator_state
from curriculum_foresight.policy.subroutines import prepare_call_eligibility
from curriculum_foresight.policy.transformer_memory import init_transformer_memories, init_transformer_memory_mask


@dataclass(frozen=True)
class TrainingSetup:
    rng: jax.Array
    goal_space: GoalSpace
    base_lc_adjacency: jax.Array
    feasibility_graph: FeasibilityGraph | None
    env: Any
    env_params: Any
    policy_train_state: TrainState
    curriculum_state: CurriculumPersistentState


def setup_goal_space(config: TrainConfig) -> GoalSpace:
    goal_space = build_goal_space(load_goal_space_recipe(config.goal_space_id))
    if config.goal_weighting_mode == "uniform":
        weights = uniform_goal_weights(goal_space)
    elif config.goal_weighting_mode == "single_target":
        weights = single_target_goal_weights(goal_space, config.target_task_id, config.target_weight_fraction)
    else:
        raise NotImplementedError(f"Unsupported goal weighting mode: {config.goal_weighting_mode!r}.")
    goal_space = apply_goal_weights(goal_space, weights)

    return load_goal_embeddings(goal_space, config.goal_embedding.encoder_id)


def setup_environment(config: TrainConfig) -> tuple[Any, Any]:
    env = make_craftax_env_from_name(config.env_id, auto_reset=False)
    env_params = env.default_params
    env_params = env_params.replace(max_timesteps=config.episode_max_steps)

    return env, env_params


def setup_actor_critic_train_state(
    rng: jax.Array, env: Any, env_params: Any, config: TrainConfig, *, num_goals: int
) -> tuple[jax.Array, TrainState]:
    num_actions = env.action_space(env_params).n
    network_kwargs = dict(
        env_id=config.env_id,
        obs_emb_dim=config.policy.obs_emb_dim,
        hidden_dim=config.policy.transformer_hidden_states_dim,
        num_attn_heads=config.policy.num_attn_heads,
        qkv_features=config.policy.qkv_features,
        num_layers_in_transformer=config.policy.num_transformer_blocks,
        gating_bias=config.policy.gating_bias,
        head_hidden_dim=config.policy.head_hidden_dim,
    )
    if config.learner_kind == "primitive":
        network = PrimitiveActorCritic(
            **network_kwargs,
            num_actions=num_actions,
            transformer_input_fusion_hidden_dim=config.policy.transformer_input_fusion_hidden_dim,
        )
    else:
        network = SubroutineActorCritic(
            **network_kwargs,
            num_primitive_actions=num_actions,
            num_goals=num_goals,
            max_subroutine_steps=config.subroutine.max_subroutine_steps,
        )

    init_batch_size = 2
    init_obs = jnp.zeros((init_batch_size, 1, *env.observation_space(env_params).shape), dtype=jnp.float32)
    init_inputs = {"observation": init_obs}

    init_memories = init_transformer_memories(init_batch_size, config.policy)
    init_mask = init_transformer_memory_mask(init_batch_size, config.policy)
    init_goal_embeddings = jnp.zeros((init_batch_size, config.goal_embedding.embedding_dim), dtype=jnp.float32)

    rng, init_rng = jax.random.split(rng)
    if config.learner_kind == "primitive":
        variables = network.init(init_rng, init_memories, init_inputs, init_mask, init_goal_embeddings)
    else:
        init_eligibility = prepare_call_eligibility(
            jnp.full((num_goals,), config.performance.ema_prior, dtype=jnp.float32),
            competence_threshold=config.subroutine.competence_threshold,
            max_subroutine_steps=config.subroutine.max_subroutine_steps,
            num_primitive_actions=num_actions,
        )
        variables = network.init(
            init_rng,
            init_memories,
            init_inputs,
            init_mask,
            init_goal_embeddings,
            init_goal_embeddings,
            jnp.zeros((init_batch_size,), dtype=jnp.int32),
            init_eligibility,
        )
    tx = optax.chain(
        optax.clip_by_global_norm(config.policy.max_grad_norm),
        optax.inject_hyperparams(optax.adam)(learning_rate=config.policy.learning_rate, eps=config.policy.adam_eps),
    )
    train_state = TrainState.create(apply_fn=network.apply, params=variables["params"], tx=tx)
    return rng, train_state


def set_up_for_training(config: TrainConfig) -> TrainingSetup:
    goal_space = setup_goal_space(config)
    base_lc_adjacency = create_base_lc_adjacency(goal_space, config.curriculum, learner_kind=config.learner_kind)
    feasibility_graph = None
    if config.curriculum.feasibility_mix > 0:
        feasibility_graph = load_feasibility_graph(
            config.curriculum.feasibility_graph_dir, goal_space, learner_kind=config.learner_kind
        )
    rng = jax.random.key(config.train_seed)
    env, env_params = setup_environment(config)
    num_goals = len(goal_space.task_ids)
    rng, policy_train_state = setup_actor_critic_train_state(rng, env, env_params, config, num_goals=num_goals)
    return TrainingSetup(
        rng=rng,
        goal_space=goal_space,
        base_lc_adjacency=base_lc_adjacency,
        feasibility_graph=feasibility_graph,
        env=env,
        env_params=env_params,
        policy_train_state=policy_train_state,
        curriculum_state=CurriculumPersistentState(
            performance_estimator_state=init_performance_estimator_state(num_goals, prior=config.performance.ema_prior),
            learning_progress_estimator_state=init_learning_progress_estimator_state(
                num_goals, max_history_observations=config.learning_progress.max_history_observations
            ),
            latest_performance_observations=init_goal_performance_observations(num_goals),
        ),
    )
