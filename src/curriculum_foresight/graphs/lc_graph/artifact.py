from __future__ import annotations

import hashlib
import json
import shlex
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import numpy as np

from curriculum_foresight.goals.goal_space import build_goal_space, load_goal_space_recipe
from curriculum_foresight.graphs.common import (
    GRAPH_FILENAME,
    atomic_write_json,
    atomic_write_npz,
    gemini_config_from_metadata,
    initialize_graph_directory,
    load_graph_manifest,
    validate_graph_identity,
)
from curriculum_foresight.graphs.lc_graph.base_graph import load_lc_graph
from curriculum_foresight.graphs.lc_graph.config import LCGraphGenerationConfig
from curriculum_foresight.graphs.lc_graph.generation import (
    LCGraphGenerationProvider,
    LCGraphGeneration,
    LCGraphGenerationError,
    LCGraphRequest,
    accumulate_lc_graph_edge_votes,
    build_lc_graph_requests,
    parse_and_validate_lc_graph_response,
)


def generate_lc_graph_artifact(
    config: LCGraphGenerationConfig, provider: LCGraphGenerationProvider, *, output_dir: Path
) -> Path:
    output_dir = Path(output_dir).expanduser()
    goal_space = build_goal_space(load_goal_space_recipe(config.goal_space_id))
    requests = build_lc_graph_requests(goal_space, config)
    generation_settings = asdict(config)
    manifest = {
        "graph_kind": "lc",
        "goal_space_id": generation_settings.pop("goal_space_id"),
        "learner_kind": generation_settings.pop("learner_kind"),
        "ordered_task_ids": list(goal_space.task_ids),
        "unweighted_goal_space_fingerprint": goal_space.unweighted_goal_space_fingerprint,
        "request_plan_fingerprint": _request_plan_fingerprint(requests),
        "generation_config": generation_settings,
        "provider_metadata": dict(provider.metadata),
    }
    initialize_graph_directory(output_dir, manifest)
    return _continue_lc_graph_artifact(output_dir, provider)


def resume_lc_graph_artifact(output_dir: Path, provider: LCGraphGenerationProvider) -> Path:
    return _continue_lc_graph_artifact(Path(output_dir).expanduser(), provider)


def _continue_lc_graph_artifact(output_dir: Path, provider: LCGraphGenerationProvider) -> Path:
    manifest = load_graph_manifest(output_dir, "lc")
    config = LCGraphGenerationConfig(
        goal_space_id=manifest["goal_space_id"], learner_kind=manifest["learner_kind"], **manifest["generation_config"]
    )
    goal_space = build_goal_space(load_goal_space_recipe(config.goal_space_id))
    requests = build_lc_graph_requests(goal_space, config)
    validate_graph_identity(manifest, goal_space, config.learner_kind)
    if manifest["request_plan_fingerprint"] != _request_plan_fingerprint(requests):
        raise ValueError("LC graph request plan differs from the saved run.")
    generations = _load_completed_generations(output_dir, requests)
    completed_count = len(generations)
    with (output_dir / "state.json").open(encoding="utf-8") as f:
        state = json.load(f)
    failed_request_id = state["failed_request_id"]
    next_attempt_index = state["next_attempt_index"]
    if isinstance(next_attempt_index, bool) or not isinstance(next_attempt_index, int) or next_attempt_index < 0:
        raise ValueError("Saved LC graph retry count must be a nonnegative integer.")
    if completed_count and failed_request_id == requests[completed_count - 1].request_id:
        failed_request_id = None
        next_attempt_index = 0
        state = {"failed_request_id": None, "next_attempt_index": 0}
        atomic_write_json(output_dir / "state.json", state)
    expected_request_id = requests[completed_count].request_id if completed_count < len(requests) else None
    if failed_request_id not in (None, expected_request_id) or (failed_request_id is None and next_attempt_index != 0):
        raise ValueError("LC graph retry state does not match the next unfinished request.")

    if (output_dir / GRAPH_FILENAME).exists():
        if completed_count != len(requests):
            raise ValueError("completed LC graph has incomplete cached responses.")
        load_lc_graph(output_dir, goal_space, learner_kind=config.learner_kind)
        return output_dir

    initial_completed_count = completed_count
    if completed_count < len(requests):
        if gemini_config_from_metadata(manifest["provider_metadata"]) != gemini_config_from_metadata(provider.metadata):
            raise ValueError("LC graph provider settings differ from the saved run.")
        try:
            for generation in provider.generate_incrementally(
                requests[completed_count:],
                request_ordinal_offset=completed_count,
                first_request_attempt_offset=next_attempt_index,
            ):
                if completed_count >= len(requests) or generation.request_id != requests[completed_count].request_id:
                    raise LCGraphGenerationError("LC graph response has an unexpected request ID.")
                atomic_write_json(
                    output_dir / "responses" / f"{completed_count:06d}.json",
                    {
                        "request_id": generation.request_id,
                        "raw_response_text": generation.raw_response_text,
                        "metadata": generation.metadata,
                    },
                )
                generations.append(generation)
                completed_count += 1
                if state["failed_request_id"] is not None:
                    state = {"failed_request_id": None, "next_attempt_index": 0}
                    atomic_write_json(output_dir / "state.json", state)
        except LCGraphGenerationError as exc:
            if completed_count < len(requests) and exc.attempts_made > 0:
                attempt_offset = next_attempt_index if completed_count == initial_completed_count else 0
                state = {
                    "failed_request_id": requests[completed_count].request_id,
                    "next_attempt_index": attempt_offset + exc.attempts_made,
                }
                atomic_write_json(output_dir / "state.json", state)
            raise _in_progress_error(output_dir, attempts_made=exc.attempts_made) from exc
    if completed_count != len(requests):
        raise _in_progress_error(output_dir)

    edge_votes = accumulate_lc_graph_edge_votes(goal_space, generations)
    adjacency = (edge_votes.astype(np.float32) / config.num_passes).astype(np.float32, copy=False)
    atomic_write_npz(output_dir / GRAPH_FILENAME, adjacency=adjacency)
    return output_dir


def _in_progress_error(output_dir: Path, *, attempts_made: int = 0) -> LCGraphGenerationError:
    return LCGraphGenerationError(
        (
            f"LC graph generation is incomplete at {output_dir}. "
            f"Resume with: python scripts/lc_graph.py resume --output-dir {shlex.quote(str(output_dir))}"
        ),
        attempts_made=attempts_made,
    )


def _load_completed_generations(output_dir: Path, requests: Sequence[LCGraphRequest]) -> list[LCGraphGeneration]:
    responses_path = output_dir / "responses"
    if not responses_path.is_dir():
        raise ValueError("LC graph responses directory is missing.")
    record_paths = sorted(responses_path.glob("*.json"))
    expected_names = [f"{index:06d}.json" for index in range(len(record_paths))]
    if [path.name for path in record_paths] != expected_names or len(record_paths) > len(requests):
        raise ValueError("Missing or unexpected LC graph response files.")
    generations = []
    for request, record_path in zip(requests, record_paths, strict=False):
        with record_path.open(encoding="utf-8") as f:
            record = json.load(f)
        if record["request_id"] != request.request_id:
            raise ValueError(f"LC graph record {record_path.name!r} does not match its request.")
        response = parse_and_validate_lc_graph_response(record["raw_response_text"], request)
        generations.append(
            LCGraphGeneration(request.request_id, record["raw_response_text"], response, record["metadata"])
        )
    return generations


def _request_plan_fingerprint(requests: Sequence[LCGraphRequest]) -> str:
    payload = [
        {
            "request_id": request.request_id,
            "pass_index": request.pass_index,
            "source_block_index": request.source_block_index,
            "target_block_index": request.target_block_index,
            "source_ids": request.source_ids,
            "target_ids": request.target_ids,
            "prompt": request.prompt,
        }
        for request in requests
    ]
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
