from __future__ import annotations

from typing import Any

import jax
from flax import struct
from flax.training.train_state import TrainState


class RunnerState(struct.PyTreeNode):
    rng: jax.Array
    policy_train_state: TrainState
    env_state: Any
    prev_obs: jax.Array
    prev_reset_done: jax.Array
    memories: jax.Array
    memories_mask: jax.Array
    memories_mask_idx: jax.Array
    success_tracking_state: Any


class SubroutineRunnerState(RunnerState):
    active_goal_indices: jax.Array
    remaining_subroutine_steps: jax.Array


class Transition(struct.PyTreeNode):
    obs: jax.Array
    action: jax.Array
    episode_done: jax.Array
    reward: jax.Array
    value: jax.Array
    log_prob: jax.Array
    memories_mask: jax.Array
    memories_indices: jax.Array


class SubroutineTransition(Transition):
    active_goal_indices: jax.Array
    remaining_subroutine_steps: jax.Array
