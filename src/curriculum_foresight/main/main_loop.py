from __future__ import annotations

import json
import time
from typing import Any

import jax
import jax.numpy as jnp
from jax.tree_util import Partial

from curriculum_foresight.curriculum.selection import compute_sampling_distribution, sample_environment_goal_indices
from curriculum_foresight.curriculum.signals import update_curriculum_iteration_signals
from curriculum_foresight.curriculum.state import update_curriculum_persistent_state
from curriculum_foresight.goals.goal_space import gather_selected_goal_batch
from curriculum_foresight.main.config import TrainConfig
from curriculum_foresight.main.subroutine_training import subroutine_train_on_goal_batch
from curriculum_foresight.main.primitive_training import primitive_train_on_goal_batch
from curriculum_foresight.main.setup import set_up_for_training
from curriculum_foresight.performance_and_lp.performance_estimation import update_performance_estimates
from curriculum_foresight.support.checkpointing import (
    TrainingCheckpointState,
    checkpoint_path,
    load_checkpoint,
    save_checkpoint_atomic,
    validate_persistence_files,
)
from curriculum_foresight.support.evaluation import (
    derive_evaluation_iterations,
    derive_evaluation_root,
    run_periodic_evaluation,
)
from curriculum_foresight.support.metrics import (
    MetricsRecorder,
    build_evaluation_record,
    build_run_metadata,
    build_training_record,
)
from curriculum_foresight.support.wandb_logging import WandbLogger, build_wandb_payload


def full_training(config: TrainConfig, *, resume: bool = False) -> dict[str, Any]:
    validate_persistence_files(config.run_dir, resume=resume)
    setup = set_up_for_training(config)
    rng = setup.rng
    policy_train_state = setup.policy_train_state
    curriculum_state = setup.curriculum_state
    target_goal_index = None
    num_target_anchor_environments = 0
    if config.goal_weighting_mode == "single_target":
        target_goal_index = setup.goal_space.task_ids.index(config.target_task_id)
    if config.num_target_anchor_draws_per_batch > 0:
        environments_per_goal_draw = config.num_envs_per_batch // config.num_goal_draws_per_batch
        num_target_anchor_environments = config.num_target_anchor_draws_per_batch * environments_per_goal_draw
    update_persistent_state = jax.jit(
        Partial(
            update_curriculum_persistent_state,
            ema_coefficient=config.performance.ema_coefficient,
            max_lookback_curriculum_iterations=config.learning_progress.max_lookback_curriculum_iterations,
        )
    )
    update_iteration_signals = jax.jit(
        Partial(update_curriculum_iteration_signals, config=config, feasibility_graph=setup.feasibility_graph)
    )
    compute_distribution = jax.jit(Partial(compute_sampling_distribution, config=config))
    training_kwargs = dict(env=setup.env, env_params=setup.env_params, config=config)
    if config.learner_kind == "subroutine":
        train_selected_goal_batch = jax.jit(
            Partial(
                subroutine_train_on_goal_batch,
                all_goal_embeddings=setup.goal_space.arrays.goal_embeddings,
                **training_kwargs,
            )
        )
    else:
        train_selected_goal_batch = jax.jit(
            Partial(primitive_train_on_goal_batch, num_goals=len(setup.goal_space.task_ids), **training_kwargs)
        )
    evaluate_periodically = jax.jit(
        Partial(run_periodic_evaluation, env=setup.env, env_params=setup.env_params, config=config)
    )
    evaluation_root = derive_evaluation_root(config.train_seed)
    evaluation_iterations = derive_evaluation_iterations(
        config.num_curriculum_iterations, config.evaluation.every_n_curriculum_iterations
    )
    evaluation_iteration_set = set(evaluation_iterations)

    completed_iterations = 0
    if resume:
        template_state = TrainingCheckpointState(
            completed_iteration=0,
            rng_key_data=jax.random.key_data(rng),
            policy_train_state=policy_train_state,
            curriculum_state=curriculum_state,
        )
        checkpoint_state = load_checkpoint(checkpoint_path(config.run_dir), template_state, config, setup.goal_space)
        completed_iterations = int(checkpoint_state.completed_iteration)
        rng = jax.random.wrap_key_data(jnp.asarray(checkpoint_state.rng_key_data))
        policy_train_state = checkpoint_state.policy_train_state
        curriculum_state = checkpoint_state.curriculum_state

    recorder = MetricsRecorder(config.run_dir)
    metadata = build_run_metadata(config, setup.goal_space)
    if resume:
        recorder.restore(completed_iterations, evaluation_iterations)
    elif config.run_dir is not None:
        (config.run_dir / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")

    logger = WandbLogger(config.logging, metadata["config"], config.run_dir)
    cumulative_wall_clock_offset = recorder.cumulative_wall_clock_offset
    cumulative_start = time.perf_counter()
    try:
        for curriculum_iteration in range(completed_iterations, config.num_curriculum_iterations):
            iteration_start = time.perf_counter()
            curriculum_state = update_persistent_state(curriculum_state, curriculum_iteration - 1)
            iteration_signals = update_iteration_signals(
                curriculum_state.performance_estimator_state.performance_estimates,
                curriculum_state.learning_progress_estimator_state.signed_lp,
                setup.base_lc_adjacency,
                setup.goal_space.arrays.weights,
            )
            goal_selection_distribution = compute_distribution(iteration_signals, curriculum_iteration)
            rng, environment_goal_indices = sample_environment_goal_indices(
                rng,
                goal_selection_distribution,
                num_envs=config.num_envs_per_batch,
                num_goal_draws=config.num_goal_draws_per_batch,
            )
            if config.num_target_anchor_draws_per_batch > 0:
                environment_goal_indices = environment_goal_indices.at[:num_target_anchor_environments].set(
                    target_goal_index
                )
            selected_goal_batch = gather_selected_goal_batch(setup.goal_space.arrays, environment_goal_indices)
            if config.learner_kind == "subroutine":
                training_outputs = train_selected_goal_batch(
                    rng,
                    policy_train_state,
                    selected_goal_batch,
                    performance_estimates=curriculum_state.performance_estimator_state.performance_estimates,
                )
            else:
                training_outputs = train_selected_goal_batch(rng, policy_train_state, selected_goal_batch)
            rng, policy_train_state, performance_observations, train_metrics = training_outputs
            curriculum_state = curriculum_state.replace(latest_performance_observations=performance_observations)
            jax.block_until_ready((curriculum_state, iteration_signals, training_outputs))
            curriculum_iteration_sec = time.perf_counter() - iteration_start

            completed_iteration = curriculum_iteration + 1
            evaluation_results = None
            evaluation_wall_clock_sec = None
            if completed_iteration in evaluation_iteration_set:
                evaluation_start = time.perf_counter()
                evaluation_performance_estimator_state = update_performance_estimates(
                    curriculum_state.performance_estimator_state,
                    performance_observations,
                    ema_coefficient=config.performance.ema_coefficient,
                )
                evaluation_results = evaluate_periodically(
                    policy_train_state,
                    evaluation_performance_estimator_state.performance_estimates,
                    setup.goal_space.arrays,
                    evaluation_root,
                )
                jax.block_until_ready(evaluation_results)
                evaluation_wall_clock_sec = time.perf_counter() - evaluation_start

            cumulative_wall_clock_sec = cumulative_wall_clock_offset + time.perf_counter() - cumulative_start
            training_record = build_training_record(
                completed_iteration,
                iteration_signals,
                goal_selection_distribution,
                train_metrics,
                config=config,
                target_goal_index=target_goal_index,
                cumulative_wall_clock_sec=cumulative_wall_clock_sec,
                curriculum_iteration_sec=curriculum_iteration_sec,
            )
            evaluation_record = None
            if evaluation_results is not None:
                evaluation_record = build_evaluation_record(
                    completed_iteration,
                    training_record["run/total_env_steps"],
                    evaluation_results,
                    evaluation_wall_clock_sec,
                )
            recorder.record(training_record, evaluation_record)
            if evaluation_record is not None:
                recorder.save()
                if config.run_dir is not None:
                    save_checkpoint_atomic(
                        checkpoint_path(config.run_dir),
                        TrainingCheckpointState(
                            completed_iteration=completed_iteration,
                            rng_key_data=jax.random.key_data(rng),
                            policy_train_state=policy_train_state,
                            curriculum_state=curriculum_state,
                        ),
                        config,
                        setup.goal_space,
                    )
            if logger.enabled:
                logger.log(
                    build_wandb_payload(
                        training_record,
                        evaluation_record,
                        num_env_steps_per_curriculum_iteration=config.num_env_steps_per_curriculum_iteration,
                        task_ids=metadata["task_ids"],
                        goal_weights=metadata["goal_weights"],
                        target_goal_index=target_goal_index,
                    )
                )
    finally:
        logger.finish()

    return {
        "goal_space": setup.goal_space,
        "base_lc_adjacency": setup.base_lc_adjacency,
        "rng": rng,
        "policy_train_state": policy_train_state,
        "curriculum_state": curriculum_state,
        "metrics": recorder.as_arrays(),
    }
