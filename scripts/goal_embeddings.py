from __future__ import annotations

import argparse
from collections.abc import Sequence

from curriculum_foresight.goals.encoding import generate_goal_embeddings


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate frozen Qwen3-Embedding-4B embeddings for the 256-goal Craftax benchmark."
    )
    parser.add_argument("--batch-size", type=int, default=8)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    artifact_path = generate_goal_embeddings("craftax_256", "qwen3_embedding_4b_512", batch_size=args.batch_size)
    print(f"Created goal embeddings: {artifact_path}")


if __name__ == "__main__":
    main()
