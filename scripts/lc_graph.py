from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from curriculum_foresight.graphs.common import GeminiProviderConfig, gemini_config_from_metadata, load_graph_manifest
from curriculum_foresight.graphs.prompt_context import LEARNER_CONTEXTS
from curriculum_foresight.graphs.lc_graph.config import LCGraphGenerationConfig


def _build_parser() -> argparse.ArgumentParser:
    generation_defaults = LCGraphGenerationConfig()
    provider_defaults = GeminiProviderConfig()
    parser = argparse.ArgumentParser(
        description="Generate learning-connectivity graphs for the 256-goal Craftax benchmark."
    )
    subparsers = parser.add_subparsers(dest="operation", required=True)

    generate_parser = subparsers.add_parser("generate", help="Generate a weighted LC graph.")
    generate_parser.add_argument("--learner", choices=LEARNER_CONTEXTS, default=generation_defaults.learner_kind)
    generate_parser.add_argument("--output-dir", type=Path, required=True)
    generate_parser.add_argument(
        "--max-sources-per-prompt", type=int, default=generation_defaults.max_sources_per_prompt
    )
    generate_parser.add_argument(
        "--max-targets-per-prompt", type=int, default=generation_defaults.max_targets_per_prompt
    )
    generate_parser.add_argument("--num-passes", type=int, default=generation_defaults.num_passes)
    generate_parser.add_argument("--shuffle-seed", type=int, default=generation_defaults.shuffle_seed)
    generate_parser.add_argument("--model", default=provider_defaults.model)
    generate_parser.add_argument(
        "--thinking-level", choices=("minimal", "low", "medium", "high"), default=provider_defaults.thinking_level
    )
    generation_seed_group = generate_parser.add_mutually_exclusive_group()
    generation_seed_group.add_argument("--generation-seed", type=int, default=provider_defaults.generation_seed)
    generation_seed_group.add_argument(
        "--no-generation-seed",
        action="store_const",
        const=None,
        dest="generation_seed",
        help="Do not specify a generation seed.",
    )
    generate_parser.add_argument("--max-output-tokens", type=int, default=provider_defaults.max_output_tokens)
    generate_parser.add_argument("--max-retries", type=int, default=provider_defaults.max_retries)

    resume_parser = subparsers.add_parser("resume", help="Resume LC graph generation or reconstruct a completed graph.")
    resume_parser.add_argument("--output-dir", type=Path, required=True)

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    from dotenv import load_dotenv

    from curriculum_foresight.graphs.lc_graph.artifact import generate_lc_graph_artifact, resume_lc_graph_artifact
    from curriculum_foresight.graphs.lc_graph.gemini_provider import GeminiLCGraphProvider

    load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env", override=False)
    if args.operation == "generate":
        generation_config = LCGraphGenerationConfig(
            learner_kind=args.learner,
            max_sources_per_prompt=args.max_sources_per_prompt,
            max_targets_per_prompt=args.max_targets_per_prompt,
            num_passes=args.num_passes,
            shuffle_seed=args.shuffle_seed,
        )
        provider_config = GeminiProviderConfig(
            model=args.model,
            thinking_level=args.thinking_level,
            generation_seed=args.generation_seed,
            max_output_tokens=args.max_output_tokens,
            max_retries=args.max_retries,
        )
        artifact_path = generate_lc_graph_artifact(
            generation_config, GeminiLCGraphProvider(provider_config, show_progress=True), output_dir=args.output_dir
        )
        print(f"LC graph ready: {artifact_path}")
        return

    manifest = load_graph_manifest(args.output_dir.expanduser(), "lc")
    provider_config = gemini_config_from_metadata(manifest["provider_metadata"])
    artifact_path = resume_lc_graph_artifact(
        args.output_dir, GeminiLCGraphProvider(provider_config, show_progress=True)
    )
    print(f"LC graph ready: {artifact_path}")


if __name__ == "__main__":
    main()
