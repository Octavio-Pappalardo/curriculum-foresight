from __future__ import annotations

from dataclasses import dataclass

from curriculum_foresight.graphs.prompt_context import LEARNER_CONTEXTS


@dataclass
class LCGraphGenerationConfig:
    goal_space_id: str = "craftax_256"
    max_sources_per_prompt: int = 64
    max_targets_per_prompt: int = 16
    num_passes: int = 5
    shuffle_seed: int = 0
    learner_kind: str = "primitive"

    def __post_init__(self) -> None:
        if self.learner_kind not in LEARNER_CONTEXTS:
            raise ValueError(f"unknown learner_kind: {self.learner_kind!r}.")
        if not self.goal_space_id:
            msg = "goal_space_id must be nonempty."
            raise ValueError(msg)
        if self.max_sources_per_prompt <= 0:
            msg = "max_sources_per_prompt must be positive."
            raise ValueError(msg)
        if self.max_targets_per_prompt <= 0:
            msg = "max_targets_per_prompt must be positive."
            raise ValueError(msg)
        if self.num_passes <= 0:
            msg = "num_passes must be positive."
            raise ValueError(msg)
        if self.shuffle_seed < 0:
            msg = "shuffle_seed must be nonnegative."
            raise ValueError(msg)
