from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


GoalEncoderId = Literal["qwen3_embedding_4b_512"]


@dataclass(frozen=True)
class GoalEncoderSpec:
    model_name: str | None
    embedding_dim: int
    truncate_dim: int | None = None
    revision: str | None = None


GOAL_ENCODER_SPECS: dict[str, GoalEncoderSpec] = {
    "qwen3_embedding_4b_512": GoalEncoderSpec(
        model_name="Qwen/Qwen3-Embedding-4B",
        embedding_dim=512,
        truncate_dim=512,
        revision="5cf2132abc99cad020ac570b19d031efec650f2b",
    )
}

SUPPORTED_GOAL_ENCODER_IDS = tuple(GOAL_ENCODER_SPECS)


def get_goal_encoder_spec(encoder_id: str) -> GoalEncoderSpec:
    try:
        return GOAL_ENCODER_SPECS[encoder_id]
    except KeyError as exc:
        supported = ", ".join(SUPPORTED_GOAL_ENCODER_IDS)
        msg = f"goal encoder id must be one of: {supported}. Got {encoder_id!r}."
        raise ValueError(msg) from exc
