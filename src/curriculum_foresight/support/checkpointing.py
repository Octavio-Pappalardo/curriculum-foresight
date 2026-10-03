from __future__ import annotations

import os
import tempfile
from pathlib import Path

import jax
from flax import serialization, struct
from flax.training.train_state import TrainState

from curriculum_foresight.curriculum.state import CurriculumPersistentState
from curriculum_foresight.goals.types import GoalSpace
from curriculum_foresight.main.config import TrainConfig, config_to_dict


CHECKPOINT_FILENAME = "latest_checkpoint.msgpack"
METRICS_FILENAME = "metrics.npz"


class TrainingCheckpointState(struct.PyTreeNode):
    completed_iteration: int
    rng_key_data: jax.Array
    policy_train_state: TrainState
    curriculum_state: CurriculumPersistentState


def checkpoint_path(run_dir: Path) -> Path:
    return Path(run_dir) / CHECKPOINT_FILENAME


def validate_persistence_files(run_dir: Path | None, *, resume: bool) -> None:
    if run_dir is None:
        if resume:
            raise ValueError("Resuming requires config.run_dir.")
        return

    run_dir = Path(run_dir)
    paths = (run_dir / METRICS_FILENAME, checkpoint_path(run_dir))
    if resume:
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing files required to resume: {', '.join(missing)}")
        return

    existing = [str(path) for path in (*paths, run_dir / "run.json") if path.exists()]
    if existing:
        raise FileExistsError(f"Cannot overwrite existing run files: {', '.join(existing)}")


def checkpoint_metadata(config: TrainConfig, goal_space: GoalSpace) -> dict[str, object]:
    resolved_config = config_to_dict(config)
    del resolved_config["logging"]
    del resolved_config["run_dir"]
    return {
        "resolved_config": resolved_config,
        "unweighted_goal_space_fingerprint": goal_space.unweighted_goal_space_fingerprint,
        "weighted_goal_space_fingerprint": goal_space.weighted_goal_space_fingerprint,
    }


def save_checkpoint_atomic(
    path: Path, state: TrainingCheckpointState, config: TrainConfig, goal_space: GoalSpace
) -> None:
    path = Path(path)
    payload = {"metadata": checkpoint_metadata(config, goal_space), "state": serialization.to_state_dict(state)}
    temporary_path: Path | None = None
    try:
        encoded = serialization.msgpack_serialize(payload)
        with tempfile.NamedTemporaryFile(
            mode="w+b", prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(encoded)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def load_checkpoint(
    path: Path, template_state: TrainingCheckpointState, config: TrainConfig, goal_space: GoalSpace
) -> TrainingCheckpointState:
    payload = serialization.msgpack_restore(Path(path).read_bytes())
    if payload["metadata"] != checkpoint_metadata(config, goal_space):
        raise ValueError("checkpoint metadata does not match the supplied configuration and goal space.")
    return serialization.from_state_dict(template_state, payload["state"])
