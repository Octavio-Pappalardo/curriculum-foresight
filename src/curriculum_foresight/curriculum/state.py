from __future__ import annotations

import jax
from flax import struct

from curriculum_foresight.performance_and_lp.learning_progress import (
    LearningProgressEstimatorState,
    update_learning_progress_estimates,
)
from curriculum_foresight.performance_and_lp.observations import GoalPerformanceObservations
from curriculum_foresight.performance_and_lp.performance_estimation import (
    PerformanceEstimatorState,
    update_performance_estimates,
)


class CurriculumPersistentState(struct.PyTreeNode):
    performance_estimator_state: PerformanceEstimatorState
    learning_progress_estimator_state: LearningProgressEstimatorState
    latest_performance_observations: GoalPerformanceObservations


def update_curriculum_persistent_state(
    state: CurriculumPersistentState,
    observation_iteration: jax.Array | int,
    *,
    ema_coefficient: float,
    max_lookback_curriculum_iterations: int,
) -> CurriculumPersistentState:
    performance_estimator_state = update_performance_estimates(
        state.performance_estimator_state, state.latest_performance_observations, ema_coefficient=ema_coefficient
    )
    learning_progress_estimator_state = update_learning_progress_estimates(
        state.learning_progress_estimator_state,
        state.latest_performance_observations,
        observation_iteration,
        max_lookback_curriculum_iterations=max_lookback_curriculum_iterations,
    )
    return state.replace(
        performance_estimator_state=performance_estimator_state,
        learning_progress_estimator_state=learning_progress_estimator_state,
    )
