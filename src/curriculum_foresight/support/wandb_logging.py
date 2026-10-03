from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from curriculum_foresight.main.config import LoggingConfig


def build_wandb_payload(
    training_row: Mapping[str, Any],
    evaluation_row: Mapping[str, Any] | None,
    *,
    num_env_steps_per_curriculum_iteration: int,
    task_ids: Sequence[str],
    goal_weights: Sequence[float],
    target_goal_index: int | None,
) -> dict[str, int | float]:
    payload = {key: np.asarray(value).item() for key, value in training_row.items() if np.ndim(value) == 0}
    payload["time/env_steps_per_sec"] = num_env_steps_per_curriculum_iteration / float(
        training_row["time/curriculum_iteration_sec"]
    )
    if evaluation_row is not None:
        success_rates = np.asarray(evaluation_row["evaluation/episode_successes"], dtype=np.float64).mean(axis=(1, 2))
        weights = np.asarray(goal_weights, dtype=np.float64)
        payload.update(
            {
                "evaluation/success_rates/weighted_mean": float(np.sum(weights * success_rates) / np.sum(weights)),
                "evaluation/success_rates/unweighted_mean": float(success_rates.mean()),
                "evaluation/episode_lengths/mean": float(np.mean(evaluation_row["evaluation/episode_lengths"])),
                "time/evaluation_sec": float(evaluation_row["evaluation/wall_clock_sec"]),
            }
        )
        if target_goal_index is not None:
            payload["evaluation/success_rates/target"] = float(success_rates[target_goal_index])
        payload.update(
            {
                f"per_goal_success_rate/{task_id}": float(rate)
                for task_id, rate in zip(task_ids, success_rates, strict=True)
            }
        )
    return payload


class WandbLogger:
    def __init__(self, config: LoggingConfig, resolved_config: dict[str, Any], run_dir: Path | None) -> None:
        self._run = None
        self.enabled = False
        if not config.enable_wandb:
            return
        try:
            import wandb

            self._run = wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity,
                group=config.wandb_group,
                name=config.wandb_run_name,
                tags=config.wandb_tags,
                notes=config.wandb_notes,
                job_type="train",
                config=resolved_config,
                dir=str(run_dir) if run_dir is not None else None,
            )
            self._run.define_metric("run/total_env_steps")
            self._run.define_metric("*", step_metric="run/total_env_steps")
            self.enabled = True
        except Exception as error:
            warnings.warn(f"W&B initialization failed; logging disabled: {error}", RuntimeWarning, stacklevel=2)

    def log(self, payload: dict[str, int | float]) -> None:
        if not self.enabled:
            return
        try:
            self._run.log(payload)
        except Exception as error:
            self.enabled = False
            warnings.warn(f"W&B logging failed; further logging disabled: {error}", RuntimeWarning, stacklevel=2)

    def finish(self) -> None:
        run, self._run = self._run, None
        self.enabled = False
        if run is not None:
            try:
                run.finish()
            except Exception as error:
                warnings.warn(f"W&B finish failed: {error}", RuntimeWarning, stacklevel=2)
