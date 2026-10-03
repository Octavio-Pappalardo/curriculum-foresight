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
    gemini_config_from_metadata,
    initialize_graph_directory,
    load_graph_manifest,
    validate_graph_identity,
)
from curriculum_foresight.graphs.feasibility_graph.graph_storage import (
    load_feasibility_graph_artifact,
    save_feasibility_graph,
)
from curriculum_foresight.graphs.feasibility_graph.config import FeasibilityGraphGenerationConfig
from curriculum_foresight.graphs.feasibility_graph.generation import (
    FeasibilityGraphGenerationProvider,
    FeasibilityGraphGeneration,
    FeasibilityGraphGenerationError,
    FeasibilityGraphRequest,
    accumulate_feasibility_graph_statistics,
    build_feasibility_graph_requests,
    parse_and_validate_feasibility_graph_response,
)


def generate_feasibility_graph_artifact(
    config: FeasibilityGraphGenerationConfig, provider: FeasibilityGraphGenerationProvider, *, output_dir: Path
) -> Path:
    output_dir = Path(output_dir).expanduser()
    goal_space = build_goal_space(load_goal_space_recipe(config.goal_space_id))
    requests = build_feasibility_graph_requests(goal_space, config)
    generation_settings = asdict(config)
    manifest = {
        "graph_kind": "feasibility",
        "goal_space_id": generation_settings.pop("goal_space_id"),
        "learner_kind": generation_settings.pop("learner_kind"),
        "ordered_task_ids": list(goal_space.task_ids),
        "unweighted_goal_space_fingerprint": goal_space.unweighted_goal_space_fingerprint,
        "request_plan_fingerprint": _request_plan_fingerprint(requests),
        "generation_config": generation_settings,
        "provider_metadata": dict(provider.metadata),
    }
    initialize_graph_directory(output_dir, manifest)
    return _continue_feasibility_graph_artifact(output_dir, provider)


def resume_feasibility_graph_artifact(output_dir: Path, provider: FeasibilityGraphGenerationProvider) -> Path:
    return _continue_feasibility_graph_artifact(Path(output_dir).expanduser(), provider)


def _continue_feasibility_graph_artifact(output_dir: Path, provider: FeasibilityGraphGenerationProvider) -> Path:
    manifest = load_graph_manifest(output_dir, "feasibility")
    config = FeasibilityGraphGenerationConfig(
        goal_space_id=manifest["goal_space_id"], learner_kind=manifest["learner_kind"], **manifest["generation_config"]
    )
    goal_space = build_goal_space(load_goal_space_recipe(config.goal_space_id))
    requests = build_feasibility_graph_requests(goal_space, config)
    validate_graph_identity(manifest, goal_space, config.learner_kind)
    if manifest["request_plan_fingerprint"] != _request_plan_fingerprint(requests):
        raise ValueError("Feasibility graph request plan differs from the saved run.")
    generations = _load_completed_generations(output_dir, requests)
    completed_count = len(generations)
    with (output_dir / "state.json").open(encoding="utf-8") as f:
        state = json.load(f)
    failed_request_id = state["failed_request_id"]
    next_attempt_index = state["next_attempt_index"]
    if isinstance(next_attempt_index, bool) or not isinstance(next_attempt_index, int) or next_attempt_index < 0:
        raise ValueError("Saved feasibility graph retry count must be a nonnegative integer.")
    if completed_count and failed_request_id == requests[completed_count - 1].request_id:
        failed_request_id = None
        next_attempt_index = 0
        state = {"failed_request_id": None, "next_attempt_index": 0}
        atomic_write_json(output_dir / "state.json", state)
    expected_request_id = requests[completed_count].request_id if completed_count < len(requests) else None
    if failed_request_id not in (None, expected_request_id) or (failed_request_id is None and next_attempt_index != 0):
        raise ValueError("feasibility graph retry state does not match the next unfinished request.")

    if (output_dir / GRAPH_FILENAME).exists():
        if completed_count != len(requests):
            raise ValueError("completed feasibility graph has incomplete cached responses.")
        load_feasibility_graph_artifact(output_dir, goal_space, learner_kind=config.learner_kind)
        return output_dir

    initial_completed_count = completed_count
    if completed_count < len(requests):
        if gemini_config_from_metadata(manifest["provider_metadata"]) != gemini_config_from_metadata(provider.metadata):
            raise ValueError("Feasibility graph provider settings differ from the saved run.")
        try:
            for generation in provider.generate_incrementally(
                requests[completed_count:],
                request_ordinal_offset=completed_count,
                first_request_attempt_offset=next_attempt_index,
            ):
                if completed_count >= len(requests) or generation.request_id != requests[completed_count].request_id:
                    raise FeasibilityGraphGenerationError("Feasibility graph response has an unexpected request ID.")
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
        except FeasibilityGraphGenerationError as exc:
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

    statistics = accumulate_feasibility_graph_statistics(
        goal_space, requests, tuple(generation.response for generation in generations), num_passes=config.num_passes
    )
    goal_adjacency = (statistics.goal_mass_sums / config.num_passes).astype(np.float32)
    from_scratch_feasibility = (statistics.from_scratch_counts / config.num_passes).astype(np.float32)
    save_feasibility_graph(output_dir, goal_adjacency, from_scratch_feasibility)
    return output_dir


def _in_progress_error(output_dir: Path, *, attempts_made: int = 0) -> FeasibilityGraphGenerationError:
    return FeasibilityGraphGenerationError(
        (
            f"Feasibility graph generation is incomplete at {output_dir}. "
            f"Resume with: python scripts/feasibility_graph.py resume --output-dir {shlex.quote(str(output_dir))}"
        ),
        attempts_made=attempts_made,
    )


def _load_completed_generations(
    output_dir: Path, requests: Sequence[FeasibilityGraphRequest]
) -> list[FeasibilityGraphGeneration]:
    responses_path = output_dir / "responses"
    if not responses_path.is_dir():
        raise ValueError("feasibility graph responses directory is missing.")
    record_paths = sorted(responses_path.glob("*.json"))
    expected_names = [f"{index:06d}.json" for index in range(len(record_paths))]
    if [path.name for path in record_paths] != expected_names or len(record_paths) > len(requests):
        raise ValueError("Missing or unexpected feasibility graph response files.")
    generations = []
    for request, record_path in zip(requests, record_paths, strict=False):
        with record_path.open(encoding="utf-8") as f:
            record = json.load(f)
        if record["request_id"] != request.request_id:
            raise ValueError(f"feasibility graph record {record_path.name!r} does not match its request.")
        response = parse_and_validate_feasibility_graph_response(record["raw_response_text"], request)
        generations.append(
            FeasibilityGraphGeneration(request.request_id, record["raw_response_text"], response, record["metadata"])
        )
    return generations


def _request_plan_fingerprint(requests: Sequence[FeasibilityGraphRequest]) -> str:
    payload = [
        {
            "request_id": request.request_id,
            "pass_index": request.pass_index,
            "target_block_index": request.target_block_index,
            "source_ids": request.source_ids,
            "target_ids": request.target_ids,
            "max_feasibility_sources_per_target": request.max_feasibility_sources_per_target,
            "prompt": request.prompt,
        }
        for request in requests
    ]
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
