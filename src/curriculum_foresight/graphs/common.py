from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from curriculum_foresight.goals.types import GoalSpace


GRAPH_FILENAME = "graph.npz"


@dataclass
class GeminiProviderConfig:
    model: str = "gemini-3.7-flash"
    thinking_level: Literal["minimal", "low", "medium", "high"] = "high"
    generation_seed: int | None = 0
    max_output_tokens: int = 65536
    max_retries: int = 3

    def __post_init__(self) -> None:
        if not self.model:
            raise ValueError("Gemini model must be nonempty.")
        if self.thinking_level not in ("minimal", "low", "medium", "high"):
            raise ValueError(f"unsupported Gemini thinking level: {self.thinking_level!r}.")
        for field_name in ("generation_seed", "max_output_tokens", "max_retries"):
            value = getattr(self, field_name)
            if field_name == "generation_seed" and value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int):
                expected_type = "an integer or None" if field_name == "generation_seed" else "an integer"
                raise ValueError(f"{field_name} must be {expected_type}.")
        if self.generation_seed is not None and self.generation_seed < 0:
            raise ValueError("Gemini generation_seed must be nonnegative when provided.")
        if self.max_output_tokens <= 0:
            raise ValueError("Gemini max_output_tokens must be positive.")
        if self.max_retries < 0:
            raise ValueError("Gemini max_retries must be nonnegative.")


def gemini_provider_metadata(config: GeminiProviderConfig) -> dict[str, object]:
    from google import genai

    return {"provider": "gemini", **asdict(config), "sdk_version": genai.__version__}


def gemini_config_from_metadata(metadata: Mapping[str, object]) -> GeminiProviderConfig:
    if metadata.get("provider") != "gemini":
        raise ValueError(f"unsupported graph provider: {metadata.get('provider')!r}.")
    return GeminiProviderConfig(**{name: metadata[name] for name in GeminiProviderConfig.__dataclass_fields__})


def usage_metadata_to_dict(usage_metadata: object | None) -> dict[str, object] | None:
    if usage_metadata is None:
        return None
    return usage_metadata.model_dump(mode="json", exclude_none=True)


def load_graph_manifest(graph_dir: Path, graph_kind: str) -> dict[str, object]:
    with (graph_dir / "manifest.json").open(encoding="utf-8") as f:
        manifest = json.load(f)
    if manifest.get("graph_kind") != graph_kind:
        raise ValueError(f"expected {graph_kind!r} graph, got {manifest.get('graph_kind')!r}.")
    return manifest


def validate_graph_identity(manifest: Mapping[str, object], goal_space: GoalSpace, learner_kind: str) -> None:
    expected = {
        "goal_space_id": goal_space.goal_space_id,
        "ordered_task_ids": list(goal_space.task_ids),
        "unweighted_goal_space_fingerprint": goal_space.unweighted_goal_space_fingerprint,
        "learner_kind": learner_kind,
    }
    for name, value in expected.items():
        if manifest.get(name) != value:
            raise ValueError(f"graph {name} does not match the training goal space or learner.")


def initialize_graph_directory(output_dir: Path, manifest: Mapping[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError(f"graph output directory must be empty: {output_dir}.")
    (output_dir / "responses").mkdir()
    atomic_write_json(output_dir / "state.json", {"failed_request_id": None, "next_attempt_index": 0})
    atomic_write_json(output_dir / "manifest.json", manifest)


@contextmanager
def _atomic_output(path: Path, *, binary: bool = False):
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb" if binary else "w", encoding=None if binary else "utf-8") as f:
            yield f
            f.flush()
            os.fsync(f.fileno())
        temporary_path.replace(path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def atomic_write_json(path: Path, payload: object) -> None:
    with _atomic_output(path) as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def atomic_write_npz(path: Path, **arrays: np.ndarray) -> None:
    with _atomic_output(path, binary=True) as f:
        np.savez(f, **arrays)
