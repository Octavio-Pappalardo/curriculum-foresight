from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from curriculum_foresight.graphs.feasibility_graph.config import FeasibilityGraphGenerationConfig
from curriculum_foresight.graphs.feasibility_graph.prompt import (
    FeasibilityGraphBlockResponse,
    GoalFeasibilityJudgment,
    render_feasibility_graph_prompt,
)
from curriculum_foresight.goals.types import GoalSpace


@dataclass(frozen=True)
class FeasibilityGraphRequest:
    request_id: str
    pass_index: int
    target_block_index: int
    source_ids: tuple[str, ...]
    target_ids: tuple[str, ...]
    max_feasibility_sources_per_target: int
    prompt: str


@dataclass(frozen=True)
class FeasibilityGraphGeneration:
    request_id: str
    raw_response_text: str
    response: FeasibilityGraphBlockResponse
    metadata: dict[str, object]


class FeasibilityGraphGenerationProvider(Protocol):
    @property
    def metadata(self) -> Mapping[str, object]: ...

    def generate_incrementally(
        self,
        requests: Sequence[FeasibilityGraphRequest],
        *,
        request_ordinal_offset: int = 0,
        first_request_attempt_offset: int = 0,
    ) -> Iterable[FeasibilityGraphGeneration]: ...


class FeasibilityGraphGenerationError(RuntimeError):
    def __init__(self, message: str, *, attempts_made: int = 0):
        super().__init__(message)
        self.attempts_made = attempts_made


@dataclass(frozen=True)
class FeasibilityGraphStatistics:
    goal_mass_sums: np.ndarray
    from_scratch_counts: np.ndarray


def build_feasibility_graph_requests(
    goal_space: GoalSpace, config: FeasibilityGraphGenerationConfig
) -> tuple[FeasibilityGraphRequest, ...]:
    num_goals = len(goal_space.task_ids)
    rng = np.random.default_rng(config.shuffle_seed)
    requests: list[FeasibilityGraphRequest] = []

    for pass_index in range(config.num_passes):
        source_indices = rng.permutation(num_goals)
        target_permutation = rng.permutation(num_goals)
        target_blocks = tuple(
            target_permutation[start : start + config.max_targets_per_prompt]
            for start in range(0, num_goals, config.max_targets_per_prompt)
        )
        source_ids = tuple(goal_space.task_ids[int(index)] for index in source_indices)

        for target_block_index, target_indices in enumerate(target_blocks):
            target_ids = tuple(goal_space.task_ids[int(index)] for index in target_indices)
            requests.append(
                FeasibilityGraphRequest(
                    request_id=f"pass-{pass_index:03d}__targets-{target_block_index:03d}",
                    pass_index=pass_index,
                    target_block_index=target_block_index,
                    source_ids=source_ids,
                    target_ids=target_ids,
                    max_feasibility_sources_per_target=config.max_feasibility_sources_per_target,
                    prompt=render_feasibility_graph_prompt(
                        goal_space,
                        source_indices,
                        target_indices,
                        max_feasibility_sources_per_target=config.max_feasibility_sources_per_target,
                        learner_kind=config.learner_kind,
                    ),
                )
            )

    return tuple(requests)


def parse_and_validate_feasibility_graph_response(
    raw_response_text: str, request: FeasibilityGraphRequest
) -> FeasibilityGraphBlockResponse:
    response = FeasibilityGraphBlockResponse.model_validate_json(raw_response_text)
    return _validate_and_canonicalize_feasibility_graph_response(response, request)


def accumulate_feasibility_graph_statistics(
    goal_space: GoalSpace,
    requests: Sequence[FeasibilityGraphRequest],
    responses: Sequence[FeasibilityGraphBlockResponse],
    *,
    num_passes: int,
) -> FeasibilityGraphStatistics:
    if num_passes <= 0:
        raise ValueError("num_passes must be positive.")
    if len(requests) != len(responses):
        msg = f"feasibility request/response length mismatch: {len(requests)} requests, {len(responses)} responses."
        raise ValueError(msg)

    actual_pass_indices = {request.pass_index for request in requests}
    expected_pass_indices = set(range(num_passes))
    if actual_pass_indices != expected_pass_indices:
        msg = (
            "feasibility requests have pass-index mismatch: "
            f"expected={sorted(expected_pass_indices)}, actual={sorted(actual_pass_indices)}."
        )
        raise ValueError(msg)

    goal_ids = goal_space.task_ids
    goal_id_set = set(goal_ids)
    goal_index_by_id = {task_id: index for index, task_id in enumerate(goal_ids)}
    num_goals = len(goal_ids)
    target_counts_by_pass = np.zeros((num_passes, num_goals), dtype=np.int32)

    for request in requests:
        if len(request.source_ids) != num_goals or set(request.source_ids) != goal_id_set:
            raise ValueError(f"feasibility request {request.request_id!r} does not contain every source exactly once.")

        for target_id in request.target_ids:
            if target_id not in goal_index_by_id:
                msg = f"feasibility request {request.request_id!r} contains unknown target id {target_id!r}."
                raise ValueError(msg)
            target_counts_by_pass[request.pass_index, goal_index_by_id[target_id]] += 1

    if not np.all(target_counts_by_pass == 1):
        raise ValueError("feasibility requests must cover every target exactly once in every pass.")

    goal_mass_sums = np.zeros((num_goals, num_goals), dtype=np.float64)
    from_scratch_counts = np.zeros(num_goals, dtype=np.int32)
    for response in responses:
        for judgment in response.judgments:
            target_index = goal_index_by_id[judgment.evaluated_goal_id]
            if judgment.feasible_from_scratch:
                from_scratch_counts[target_index] += 1
                continue
            source_mass = 1.0 / len(judgment.readiness_evidence_goal_ids)
            for source_id in judgment.readiness_evidence_goal_ids:
                goal_mass_sums[goal_index_by_id[source_id], target_index] += source_mass

    total_mass_by_target = goal_mass_sums.sum(axis=0) + from_scratch_counts
    if not np.allclose(total_mass_by_target, num_passes, rtol=0.0, atol=1e-12):
        raise ValueError("Total accumulated feasibility weight must equal num_passes for every target.")
    return FeasibilityGraphStatistics(goal_mass_sums=goal_mass_sums, from_scratch_counts=from_scratch_counts)


def _validate_and_canonicalize_feasibility_graph_response(
    response: FeasibilityGraphBlockResponse, request: FeasibilityGraphRequest
) -> FeasibilityGraphBlockResponse:
    evaluated_goal_ids = [judgment.evaluated_goal_id for judgment in response.judgments]
    if len(evaluated_goal_ids) != len(set(evaluated_goal_ids)):
        raise ValueError(f"Feasibility response {request.request_id!r}: duplicate evaluated goal IDs.")

    requested_target_ids = set(request.target_ids)
    evaluated_goal_id_set = set(evaluated_goal_ids)
    if evaluated_goal_id_set != requested_target_ids:
        missing = sorted(requested_target_ids - evaluated_goal_id_set)
        extra = sorted(evaluated_goal_id_set - requested_target_ids)
        msg = (
            f"Feasibility response {request.request_id!r}: "
            f"evaluated goal IDs do not match; missing={missing}, extra={extra}."
        )
        raise ValueError(msg)

    requested_source_ids = set(request.source_ids)
    judgments_by_goal: dict[str, tuple[bool, set[str]]] = {}
    for judgment in response.judgments:
        evidence_goal_ids = judgment.readiness_evidence_goal_ids
        if len(evidence_goal_ids) != len(set(evidence_goal_ids)):
            msg = (
                f"Feasibility response {request.request_id!r}, goal {judgment.evaluated_goal_id!r}: "
                f"duplicate readiness-evidence IDs."
            )
            raise ValueError(msg)
        unknown_evidence_goal_ids = set(evidence_goal_ids) - requested_source_ids
        if unknown_evidence_goal_ids:
            msg = (
                f"Feasibility response {request.request_id!r}, goal {judgment.evaluated_goal_id!r}: "
                f"unknown readiness-evidence IDs: {sorted(unknown_evidence_goal_ids)}."
            )
            raise ValueError(msg)
        if judgment.evaluated_goal_id in evidence_goal_ids:
            msg = (
                f"Feasibility response {request.request_id!r}, goal {judgment.evaluated_goal_id!r}: "
                f"a goal cannot be its own readiness evidence."
            )
            raise ValueError(msg)
        if len(evidence_goal_ids) > request.max_feasibility_sources_per_target:
            msg = (
                f"Feasibility response {request.request_id!r}, goal {judgment.evaluated_goal_id!r}: "
                f"readiness-evidence limit exceeded ({request.max_feasibility_sources_per_target})."
            )
            raise ValueError(msg)
        if judgment.feasible_from_scratch and evidence_goal_ids:
            msg = (
                f"Feasibility response {request.request_id!r}, goal {judgment.evaluated_goal_id!r}: "
                f"feasible_from_scratch is true, but readiness evidence is not empty."
            )
            raise ValueError(msg)
        if not judgment.feasible_from_scratch and not evidence_goal_ids:
            msg = (
                f"Feasibility response {request.request_id!r}, goal {judgment.evaluated_goal_id!r}: "
                f"feasible_from_scratch is false, but readiness evidence is empty."
            )
            raise ValueError(msg)
        judgments_by_goal[judgment.evaluated_goal_id] = (judgment.feasible_from_scratch, set(evidence_goal_ids))

    canonical_judgments = []
    for target_id in request.target_ids:
        feasible_from_scratch, selected_evidence_goal_ids = judgments_by_goal[target_id]
        canonical_judgments.append(
            GoalFeasibilityJudgment(
                evaluated_goal_id=target_id,
                feasible_from_scratch=feasible_from_scratch,
                readiness_evidence_goal_ids=[
                    source_id for source_id in request.source_ids if source_id in selected_evidence_goal_ids
                ],
            )
        )
    return FeasibilityGraphBlockResponse(judgments=canonical_judgments)
