from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import fields
from pathlib import Path

from curriculum_foresight.main.config import (
    CurriculumConfig,
    EvaluationConfig,
    GoalEmbeddingConfig,
    LearningProgressConfig,
    LoggingConfig,
    PerformanceConfig,
    PolicyConfig,
    SubroutineConfig,
    TrainConfig,
    config_to_dict,
)


METHODS = {
    "alp-d-f": ("alp-downstream", "artifact", 0.1, 0.1),
    "alp-d": ("alp-downstream", "artifact", 0.2, 0.0),
    "alp-f": ("absolute-lp", "none", 0.1, 0.1),
    "alp": ("absolute-lp", "none", 0.2, 0.0),
    "uniform": ("uniform", "none", 1.0, 0.0),
    "intermediate-difficulty": ("intermediate-difficulty", "none", 0.2, 0.0),
    "target": ("target-only", "none", 1.0, 0.0),
    "permuted": ("alp-downstream", "permuted", 0.1, 0.1),
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a curriculum method. Run from the repository root with prepared embeddings and graphs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/train.py start --method alp --seed 1 --run-dir runs/alp-1\n"
            "  python scripts/train.py start --method alp-d-f --seed 1 --run-dir runs/alp-d-f-1 \\\n"
            "      --lc-graph-dir artifacts/lc --feasibility-graph-dir artifacts/feasibility --dry-run\n"
            "  python scripts/train.py start --method target --target inventory.wood.ge_64 \\\n"
            "      --seed 1 --run-dir runs/wood-1\n"
            "  python scripts/train.py resume --run-dir runs/alp-1\n\n"
            "Graph preparation:\n"
            "  python scripts/lc_graph.py --help\n"
            "  python scripts/feasibility_graph.py --help"
        ),
    )
    subparsers = parser.add_subparsers(dest="operation", required=True)
    start = subparsers.add_parser("start", help="Start a new run with paper defaults.")
    start.add_argument("--method", choices=METHODS, required=True)
    start.add_argument("--seed", type=int, required=True)
    start.add_argument("--learner", choices=("primitive", "subroutine"), default="primitive")
    start.add_argument("--target", help="Exact benchmark task ID; omitted means equally weighted goals.")
    start.add_argument("--lc-graph-dir", type=Path, help="Required for ALP+D, ALP+D+F, and PERM.")
    start.add_argument("--feasibility-graph-dir", type=Path, help="Required for methods with feasibility guidance.")
    start.add_argument(
        "--total-timesteps",
        type=int,
        help="Override the setting's paper budget; rounds up to full curriculum iterations.",
    )
    start.add_argument(
        "--target-weight-fraction", type=float, help="Target's fraction of objective weight, in (0, 1]; default: 1."
    )
    start.add_argument(
        "--graph-permutation-seed", type=int, help="Required only for PERM; independent of the training seed."
    )
    resume = subparsers.add_parser("resume", help="Restore settings and state from an existing run.")
    for command in (start, resume):
        command.add_argument("--run-dir", type=Path, required=True)
        command.add_argument(
            "--dry-run", action="store_true", help="Print settings and check inputs without training or writing files."
        )
        if command is resume:
            command.add_argument(
                "--wandb",
                action=argparse.BooleanOptionalAction,
                default=None,
                help="Override the saved W&B logging setting.",
            )
        else:
            command.add_argument("--wandb", action="store_true", default=None, help="Enable W&B logging.")
        for option in ("project", "entity", "group", "run-name"):
            command.add_argument(f"--wandb-{option}", help=f"Set the W&B {option.replace('-', ' ')}.")
    return parser


def build_config(args: argparse.Namespace) -> TrainConfig:
    algorithm, graph_source, uniform_mix, feasibility_mix = METHODS[args.method]
    if args.method == "target" and args.target is None:
        raise ValueError("The target method requires --target.")
    if args.target_weight_fraction is not None and args.target is None:
        raise ValueError("--target-weight-fraction requires --target.")
    target_fraction = None
    if args.target is not None:
        from curriculum_foresight.goals.goal_space import load_goal_space_recipe

        recipe = load_goal_space_recipe("craftax_256")
        if args.target not in {task.task_id for task in recipe.task_specs}:
            raise ValueError(f"Unknown benchmark task ID: {args.target!r}.")
        target_fraction = 1.0 if args.target_weight_fraction is None else args.target_weight_fraction
        if not 0.0 < target_fraction <= 1.0:
            raise ValueError("--target-weight-fraction must be in (0, 1].")
        if args.method == "target" and target_fraction != 1.0:
            raise ValueError("The target method requires target-weight fraction 1.")

    for option, value, required in (
        ("--lc-graph-dir", args.lc_graph_dir, graph_source != "none"),
        ("--feasibility-graph-dir", args.feasibility_graph_dir, feasibility_mix > 0),
        ("--graph-permutation-seed", args.graph_permutation_seed, graph_source == "permuted"),
    ):
        if required and value is None:
            raise ValueError(f"{args.method} requires {option}.")
        if not required and value is not None:
            raise ValueError(f"{option} does not apply to {args.method}.")

    budget = 3_019_898_880
    if args.target is not None:
        budget = 1_509_949_440
    elif args.learner == "subroutine":
        budget = 5_033_164_800
    if args.total_timesteps is not None:
        budget = args.total_timesteps
    if budget <= 0:
        raise ValueError("--total-timesteps must be positive.")

    return TrainConfig(
        algorithm_id=algorithm,
        train_seed=args.seed,
        run_dir=args.run_dir.expanduser().resolve(),
        learner_kind=args.learner,
        subroutine=SubroutineConfig() if args.learner == "subroutine" else None,
        goal_weighting_mode="single_target" if args.target is not None else "uniform",
        target_task_id=args.target,
        target_weight_fraction=target_fraction,
        num_target_anchor_draws_per_batch=64 if args.method == "target" else (4 if args.target is not None else 0),
        total_timesteps=budget,
        evaluation=EvaluationConfig(every_n_curriculum_iterations=15 if args.target is not None else 30),
        curriculum=CurriculumConfig(
            base_graph_source=graph_source,
            lc_graph_dir=args.lc_graph_dir.expanduser().resolve() if args.lc_graph_dir is not None else None,
            feasibility_graph_dir=args.feasibility_graph_dir.expanduser().resolve()
            if args.feasibility_graph_dir is not None
            else None,
            graph_permutation_seed=args.graph_permutation_seed if args.graph_permutation_seed is not None else 0,
            uniform_mix=uniform_mix,
            feasibility_mix=feasibility_mix,
        ),
    )


def load_run_config(run_dir: Path) -> TrainConfig:
    run_dir = run_dir.expanduser().resolve()
    data = json.loads((run_dir / "run.json").read_text())["config"]
    data["run_dir"] = run_dir
    data["logging"]["wandb_tags"] = tuple(data["logging"]["wandb_tags"])
    for name, config_class in (
        ("goal_embedding", GoalEmbeddingConfig),
        ("policy", PolicyConfig),
        ("subroutine", SubroutineConfig),
        ("performance", PerformanceConfig),
        ("learning_progress", LearningProgressConfig),
        ("evaluation", EvaluationConfig),
        ("curriculum", CurriculumConfig),
        ("logging", LoggingConfig),
    ):
        values = data[name]
        if values is not None:
            for item in fields(config_class):
                if not item.init:
                    values.pop(item.name, None)
            data[name] = config_class(**values)
    for item in fields(TrainConfig):
        if not item.init:
            data.pop(item.name, None)
    return TrainConfig(**data)


def _apply_logging_options(config: TrainConfig, args: argparse.Namespace) -> None:
    if args.wandb is not None:
        config.logging.enable_wandb = args.wandb
    for name in ("wandb_project", "wandb_entity", "wandb_group", "wandb_run_name"):
        value = getattr(args, name)
        if value is not None:
            setattr(config.logging, name, value)


def _print_summary(config: TrainConfig) -> None:
    curriculum = config.curriculum
    objective = "all goals, equal weights"
    if config.target_task_id is not None:
        objective = f"{config.target_task_id}, target weight {config.target_weight_fraction:g}"
    print(f"Algorithm: {config.algorithm_id} | Learner: {config.learner_kind} | Seed: {config.train_seed}")
    print(
        f"Objective: {objective} | Target selections: {config.num_target_anchor_draws_per_batch}/{config.num_goal_draws_per_batch}"
    )
    actual_steps = config.num_curriculum_iterations * config.num_env_steps_per_curriculum_iteration
    print(
        f"Budget: {config.total_timesteps:,} requested steps; {actual_steps:,} steps across {config.num_curriculum_iterations:,} iterations"
    )
    print(
        f"Sampling: uniform {curriculum.uniform_mix:g}, feasibility {curriculum.feasibility_mix:g}, primary {1 - curriculum.uniform_mix - curriculum.feasibility_mix:g}"
    )
    print(f"LC graph ({curriculum.base_graph_source}): {curriculum.lc_graph_dir}")
    if curriculum.base_graph_source == "permuted":
        print(f"Graph permutation seed: {curriculum.graph_permutation_seed}")
    print(f"Feasibility graph: {curriculum.feasibility_graph_dir}")
    print(f"Run directory: {config.run_dir}", flush=True)


def _check_artifacts(config: TrainConfig) -> None:
    from curriculum_foresight.main.setup import setup_goal_space
    from curriculum_foresight.graphs.lc_graph.base_graph import create_base_lc_adjacency
    from curriculum_foresight.graphs.feasibility_graph.runtime_graph import load_feasibility_graph

    goal_space = setup_goal_space(config)
    create_base_lc_adjacency(goal_space, config.curriculum, learner_kind=config.learner_kind)
    if config.curriculum.feasibility_mix > 0:
        load_feasibility_graph(config.curriculum.feasibility_graph_dir, goal_space, learner_kind=config.learner_kind)


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    resume = args.operation == "resume"
    try:
        config = load_run_config(args.run_dir) if resume else build_config(args)
        _apply_logging_options(config, args)
        _print_summary(config)
        if args.dry_run:
            from curriculum_foresight.support.checkpointing import validate_persistence_files

            validate_persistence_files(config.run_dir, resume=resume)
            print(json.dumps(config_to_dict(config), indent=2))
            _check_artifacts(config)
            print("Dry run passed.")
            return
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))

    from curriculum_foresight.main.main_loop import full_training

    full_training(config, resume=resume)


if __name__ == "__main__":
    main()
