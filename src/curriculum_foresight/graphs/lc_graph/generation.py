from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from curriculum_foresight.goals.types import GoalSpace
from curriculum_foresight.graphs.lc_graph.config import LCGraphGenerationConfig
from curriculum_foresight.graphs.lc_graph.prompt import (
    LCGraphBlockResponse,
    TargetEdgeSelection,
    render_lc_graph_prompt,
)


@dataclass(frozen=True)
class LCGraphRequest:
    request_id: str
    pass_index: int
    source_block_index: int
    target_block_index: int
    source_ids: tuple[str, ...]
    target_ids: tuple[str, ...]
    prompt: str


@dataclass(frozen=True)
class LCGraphGeneration:
    request_id: str
    raw_response_text: str
    response: LCGraphBlockResponse
    metadata: dict[str, object]


class LCGraphGenerationProvider(Protocol):
    @property
    def metadata(self) -> Mapping[str, object]: ...

    def generate_incrementally(
        self,
        requests: Sequence[LCGraphRequest],
        *,
        request_ordinal_offset: int = 0,
        first_request_attempt_offset: int = 0,
    ) -> Iterable[LCGraphGeneration]: ...


class LCGraphGenerationError(RuntimeError):
    def __init__(self, message: str, *, attempts_made: int = 0):
        super().__init__(message)
        self.attempts_made = attempts_made


def build_lc_graph_requests(goal_space: GoalSpace, config: LCGraphGenerationConfig) -> tuple[LCGraphRequest, ...]:
    num_goals = len(goal_space.task_ids)
    rng = np.random.default_rng(config.shuffle_seed)
    requests: list[LCGraphRequest] = []

    for pass_index in range(config.num_passes):
        source_permutation = rng.permutation(num_goals)
        target_permutation = rng.permutation(num_goals)
        source_blocks = tuple(
            source_permutation[start : start + config.max_sources_per_prompt]
            for start in range(0, num_goals, config.max_sources_per_prompt)
        )
        target_blocks = tuple(
            target_permutation[start : start + config.max_targets_per_prompt]
            for start in range(0, num_goals, config.max_targets_per_prompt)
        )

        for source_block_index, source_indices in enumerate(source_blocks):
            source_ids = tuple(goal_space.task_ids[int(index)] for index in source_indices)
            for target_block_index, target_indices in enumerate(target_blocks):
                target_ids = tuple(goal_space.task_ids[int(index)] for index in target_indices)
                request_id = (
                    f"pass-{pass_index:03d}__sources-{source_block_index:03d}__targets-{target_block_index:03d}"
                )
                requests.append(
                    LCGraphRequest(
                        request_id=request_id,
                        pass_index=pass_index,
                        source_block_index=source_block_index,
                        target_block_index=target_block_index,
                        source_ids=source_ids,
                        target_ids=target_ids,
                        prompt=render_lc_graph_prompt(
                            goal_space, source_indices, target_indices, learner_kind=config.learner_kind
                        ),
                    )
                )

    return tuple(requests)


def parse_and_validate_lc_graph_response(raw_response_text: str, request: LCGraphRequest) -> LCGraphBlockResponse:
    response = LCGraphBlockResponse.model_validate_json(raw_response_text)
    response_target_ids = [target.target_id for target in response.targets]
    if len(response_target_ids) != len(set(response_target_ids)):
        msg = f"LC graph response {request.request_id!r}: duplicate target IDs."
        raise ValueError(msg)

    requested_target_ids = set(request.target_ids)
    response_target_id_set = set(response_target_ids)
    if response_target_id_set != requested_target_ids:
        missing = sorted(requested_target_ids - response_target_id_set)
        extra = sorted(response_target_id_set - requested_target_ids)
        msg = f"LC graph response {request.request_id!r}: target IDs do not match; missing={missing}, extra={extra}."
        raise ValueError(msg)

    requested_source_ids = set(request.source_ids)
    selections_by_target: dict[str, set[str]] = {}
    for target in response.targets:
        if len(target.transfer_source_ids) != len(set(target.transfer_source_ids)):
            msg = f"LC graph response {request.request_id!r}, target {target.target_id!r}: duplicate source IDs."
            raise ValueError(msg)
        unknown_source_ids = set(target.transfer_source_ids) - requested_source_ids
        if unknown_source_ids:
            msg = (
                f"LC graph response {request.request_id!r}, target {target.target_id!r}: "
                f"unknown source IDs: {sorted(unknown_source_ids)}."
            )
            raise ValueError(msg)
        selections_by_target[target.target_id] = set(target.transfer_source_ids)

    canonical_targets: list[TargetEdgeSelection] = []
    for target_id in request.target_ids:
        selected_source_ids = selections_by_target[target_id]
        canonical_targets.append(
            TargetEdgeSelection(
                target_id=target_id,
                transfer_source_ids=[source_id for source_id in request.source_ids if source_id in selected_source_ids],
            )
        )

    return LCGraphBlockResponse(targets=canonical_targets)


def accumulate_lc_graph_edge_votes(goal_space: GoalSpace, generations: Sequence[LCGraphGeneration]) -> np.ndarray:
    goal_index_by_id = {task_id: index for index, task_id in enumerate(goal_space.task_ids)}
    num_goals = len(goal_space.task_ids)
    edge_votes = np.zeros((num_goals, num_goals), dtype=np.int32)
    for generation in generations:
        for target in generation.response.targets:
            target_index = goal_index_by_id[target.target_id]
            for source_id in target.transfer_source_ids:
                edge_votes[goal_index_by_id[source_id], target_index] += 1
    np.fill_diagonal(edge_votes, 0)
    return edge_votes
