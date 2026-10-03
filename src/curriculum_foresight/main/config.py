from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, get_args

import numpy as np

from curriculum_foresight.goals.encoding_specs import GoalEncoderId, get_goal_encoder_spec

EnvId = Literal["Craftax-Symbolic-v1"]
GoalWeightingMode = Literal["uniform", "single_target"]
LearnerKind = Literal["primitive", "subroutine"]
BaseLCGraphSource = Literal["none", "artifact", "permuted"]
AlgorithmId = Literal["alp-downstream", "absolute-lp", "intermediate-difficulty", "uniform", "target-only"]


@dataclass
class GoalEmbeddingConfig:
    encoder_id: GoalEncoderId = "qwen3_embedding_4b_512"
    embedding_dim: int = field(init=False)

    def __post_init__(self) -> None:
        if self.encoder_id not in get_args(GoalEncoderId):
            raise ValueError(f"Unsupported goal_embedding.encoder_id: {self.encoder_id!r}.")
        self.embedding_dim = get_goal_encoder_spec(self.encoder_id).embedding_dim


@dataclass
class PolicyConfig:
    transformer_input_fusion_hidden_dim: int = 512

    obs_emb_dim: int = 512

    past_context_length: int = 64
    subsequence_length_in_loss_calculation: int = 64
    num_attn_heads: int = 4
    num_transformer_blocks: int = 2
    transformer_hidden_states_dim: int = 256
    qkv_features: int = 256
    gating_bias: float = 2.0

    head_hidden_dim: int = 512

    learning_rate: float = 2e-4
    adam_eps: float = 1e-5
    max_grad_norm: float = 1.0

    update_epochs: int = 1
    num_minibatches: int = 8
    clip_eps: float = 0.2
    gamma: float = 0.999
    gae_lambda: float = 0.97
    ent_coef: float = 0.005
    vf_coef: float = 0.5


@dataclass
class SubroutineConfig:
    max_subroutine_steps: int = 32
    competence_threshold: float = 1 / 3
    reference_kl_coef: float = 0.005


@dataclass
class PerformanceConfig:
    last_k_success_k: int = 5
    ema_prior: float = 0.0
    ema_coefficient: float = 0.3


@dataclass
class LearningProgressConfig:
    max_history_observations: int = 7
    max_lookback_curriculum_iterations: int = 12
    num_warmup_curriculum_iterations: int = 16


@dataclass
class EvaluationConfig:
    every_n_curriculum_iterations: int = 30
    num_envs_per_goal: int = 16
    num_episodes_per_env: int = 8


@dataclass
class CurriculumConfig:
    base_graph_source: BaseLCGraphSource = "artifact"
    lc_graph_dir: Path | None = None
    graph_permutation_seed: int = 0
    solved_success_threshold: float = 0.9
    intermediate_difficulty_min: float = 0.1
    intermediate_difficulty_max: float = 0.9
    max_path_length: int = 2
    path_discount: float = 0.3
    downstream_coefficient: float = 1.0
    learning_progress_exponent: float = 0.5
    uniform_mix: float = 0.1
    feasibility_mix: float = 0.1
    feasibility_performance_threshold: float = 0.05
    feasibility_graph_dir: Path | None = None

    def __post_init__(self) -> None:
        if not (
            0.0 <= self.uniform_mix <= 1.0
            and 0.0 <= self.feasibility_mix <= 1.0
            and self.uniform_mix + self.feasibility_mix <= 1.0
        ):
            raise ValueError("uniform_mix and feasibility_mix must each be in [0, 1] and sum to at most 1.")


@dataclass
class LoggingConfig:
    enable_wandb: bool = False
    wandb_project: str = "curriculum-foresight"
    wandb_entity: str | None = None
    wandb_group: str | None = None
    wandb_run_name: str | None = None
    wandb_tags: tuple[str, ...] = ()
    wandb_notes: str | None = None


@dataclass
class TrainConfig:
    algorithm_id: AlgorithmId = "alp-downstream"
    run_dir: Path | None = None
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    train_seed: int = 42
    env_id: EnvId = "Craftax-Symbolic-v1"
    goal_space_id: str = "craftax_256"
    goal_weighting_mode: GoalWeightingMode = "uniform"
    target_task_id: str | None = None
    target_weight_fraction: float | None = None
    num_envs_per_batch: int = 512
    num_goal_draws_per_batch: int = 64
    num_target_anchor_draws_per_batch: int = 0
    goal_embedding: GoalEmbeddingConfig = field(default_factory=GoalEmbeddingConfig)

    episode_max_steps: int = 4096

    num_steps_per_env_per_curriculum_iteration: int = 4096
    num_steps_per_update: int = 256
    total_timesteps: int = 3_019_898_880
    learner_kind: LearnerKind = "primitive"
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    subroutine: SubroutineConfig | None = None
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    learning_progress: LearningProgressConfig = field(default_factory=LearningProgressConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    num_policy_updates_per_curriculum_iteration: int = field(init=False)
    num_env_steps_per_curriculum_iteration: int = field(init=False)
    num_curriculum_iterations: int = field(init=False)

    def __post_init__(self) -> None:
        for name, value, choices in (
            ("algorithm_id", self.algorithm_id, AlgorithmId),
            ("learner_kind", self.learner_kind, LearnerKind),
            ("env_id", self.env_id, EnvId),
            ("goal_weighting_mode", self.goal_weighting_mode, GoalWeightingMode),
            ("curriculum.base_graph_source", self.curriculum.base_graph_source, BaseLCGraphSource),
        ):
            if value not in get_args(choices):
                raise ValueError(f"Unsupported {name}: {value!r}.")

        if self.run_dir is not None:
            self.run_dir = Path(self.run_dir).expanduser()
        if self.curriculum.lc_graph_dir is not None:
            self.curriculum.lc_graph_dir = Path(self.curriculum.lc_graph_dir).expanduser()
        if self.curriculum.feasibility_graph_dir is not None:
            self.curriculum.feasibility_graph_dir = Path(self.curriculum.feasibility_graph_dir).expanduser()

        if self.num_steps_per_env_per_curriculum_iteration <= 0 or self.num_steps_per_update <= 0:
            raise ValueError("num_steps_per_env_per_curriculum_iteration and num_steps_per_update must be positive.")
        if self.num_steps_per_env_per_curriculum_iteration % self.num_steps_per_update != 0:
            raise ValueError("num_steps_per_env_per_curriculum_iteration must be divisible by num_steps_per_update.")

        self.num_policy_updates_per_curriculum_iteration = (
            self.num_steps_per_env_per_curriculum_iteration // self.num_steps_per_update
        )
        self.num_env_steps_per_curriculum_iteration = (
            self.num_envs_per_batch * self.num_steps_per_env_per_curriculum_iteration
        )
        self.num_curriculum_iterations = (
            self.total_timesteps + self.num_env_steps_per_curriculum_iteration - 1
        ) // self.num_env_steps_per_curriculum_iteration


def config_to_dict(config: TrainConfig) -> dict[str, Any]:
    def json_default(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"Cannot serialize {type(value).__name__} to JSON.")

    return json.loads(json.dumps(asdict(config), default=json_default))
