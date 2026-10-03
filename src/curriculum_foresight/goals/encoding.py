from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Protocol

import jax.numpy as jnp
import numpy as np

from curriculum_foresight.goals.encoding_specs import GoalEncoderId, GoalEncoderSpec, get_goal_encoder_spec
from curriculum_foresight.goals.goal_space import build_goal_space, load_goal_space_recipe
from curriculum_foresight.goals.types import GoalSpace

GOAL_EMBEDDINGS_ROOT = Path("artifacts/goal_embeddings")


class GoalEncoderRuntime(Protocol):
    @property
    def embedding_dim(self) -> int: ...

    def encode_goals(self, texts: Sequence[str]) -> np.ndarray: ...


class SentenceTransformerGoalEncoder:
    def __init__(self, spec: GoalEncoderSpec, batch_size: int):
        if spec.model_name is None or spec.revision is None:
            raise ValueError("a goal encoder specification must include a model name and revision.")
        if batch_size <= 0:
            raise ValueError("goal encoder batch size must be positive.")
        self._model_name = spec.model_name
        self._model_revision = spec.revision
        self._embedding_dim = spec.embedding_dim
        self._truncate_dim = spec.truncate_dim
        self._batch_size = batch_size
        self._model = None

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def encode_goals(self, texts: Sequence[str]) -> np.ndarray:
        import torch

        model = self._get_model()
        with torch.inference_mode():
            encode_kwargs = {
                "batch_size": self._batch_size,
                "show_progress_bar": True,
                "convert_to_numpy": True,
                "convert_to_tensor": False,
                "normalize_embeddings": True,
            }
            if self._truncate_dim is not None:
                encode_kwargs["truncate_dim"] = self._truncate_dim
            embeddings = model.encode(list(texts), **encode_kwargs)
        return np.asarray(embeddings, dtype=np.float32)

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name, revision=self._model_revision)
        return self._model


def make_goal_encoder(encoder_id: GoalEncoderId | str, *, batch_size: int = 8) -> GoalEncoderRuntime:
    spec = get_goal_encoder_spec(encoder_id)
    return SentenceTransformerGoalEncoder(spec, batch_size)


def encode_goal_space(goal_space: GoalSpace, encoder: GoalEncoderRuntime) -> GoalSpace:
    goal_embeddings = np.asarray(encoder.encode_goals(goal_space.goal_texts), dtype=np.float32)
    expected_shape = (len(goal_space.goal_texts), encoder.embedding_dim)
    _validate_goal_embeddings(goal_embeddings, expected_shape, "goal encoder output")

    arrays = goal_space.arrays.replace(goal_embeddings=jnp.asarray(goal_embeddings, dtype=jnp.float32))
    return replace(goal_space, arrays=arrays)


def goal_embeddings_path(goal_space_id: str, encoder_id: GoalEncoderId | str) -> Path:
    get_goal_encoder_spec(encoder_id)
    return GOAL_EMBEDDINGS_ROOT / goal_space_id / f"{encoder_id}.npz"


def generate_goal_embeddings(goal_space_id: str, encoder_id: GoalEncoderId | str, *, batch_size: int) -> Path:
    output_path = goal_embeddings_path(goal_space_id, encoder_id)
    if output_path.exists():
        msg = f"Goal embeddings already exist: {output_path}. Remove the file before regenerating."
        raise FileExistsError(msg)

    goal_space = build_goal_space(load_goal_space_recipe(goal_space_id))
    encoded_goal_space = encode_goal_space(goal_space, make_goal_encoder(encoder_id, batch_size=batch_size))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        goal_embeddings=np.asarray(encoded_goal_space.arrays.goal_embeddings, dtype=np.float32),
        unweighted_goal_space_fingerprint=np.asarray(goal_space.unweighted_goal_space_fingerprint),
    )
    return output_path


def load_goal_embeddings(goal_space: GoalSpace, encoder_id: GoalEncoderId | str) -> GoalSpace:
    spec = get_goal_encoder_spec(encoder_id)
    artifact_path = goal_embeddings_path(goal_space.goal_space_id, encoder_id)
    if not artifact_path.is_file():
        command = "python scripts/goal_embeddings.py"
        raise FileNotFoundError(f"goal embeddings not found at {artifact_path}. Generate them with:\n{command}")

    with np.load(artifact_path, allow_pickle=False) as artifact:
        required_members = {"goal_embeddings", "unweighted_goal_space_fingerprint"}
        missing_members = required_members.difference(artifact.files)
        if missing_members:
            raise ValueError(f"goal embeddings at {artifact_path} are missing: {', '.join(sorted(missing_members))}.")
        fingerprint = artifact["unweighted_goal_space_fingerprint"]
        if fingerprint.shape != () or fingerprint.dtype.kind != "U":
            raise ValueError(f"goal embeddings at {artifact_path} require a scalar string fingerprint.")
        if fingerprint.item() != goal_space.unweighted_goal_space_fingerprint:
            raise ValueError(
                f"goal embeddings at {artifact_path} do not match goal space {goal_space.goal_space_id!r}."
            )
        goal_embeddings = np.asarray(artifact["goal_embeddings"], dtype=np.float32)

    expected_shape = (len(goal_space.goal_texts), spec.embedding_dim)
    _validate_goal_embeddings(goal_embeddings, expected_shape, f"goal embeddings at {artifact_path}")
    arrays = goal_space.arrays.replace(goal_embeddings=jnp.asarray(goal_embeddings, dtype=jnp.float32))
    return replace(goal_space, arrays=arrays)


def _validate_goal_embeddings(embeddings: np.ndarray, expected_shape: tuple[int, int], source: str) -> None:
    if embeddings.shape != expected_shape:
        raise ValueError(f"{source} must have shape {expected_shape}, got {embeddings.shape}.")
    if not np.all(np.isfinite(embeddings)):
        raise ValueError(f"{source} must contain only finite values.")
