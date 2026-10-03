# Anticipating the Consequences of Curriculum Decisions with Large Language Models

This repository contains the code for the paper *Anticipating the Consequences of Curriculum Decisions with Large Language Models*. It implements an LLM-informed automatic curriculum learning method that augments learning-progress-based task selection with estimates of the potential downstream benefits of training on a task and of whether direct training on a low-performance task is currently likely to produce progress.

## Installation

The code uses Python 3.12 and JAX with CUDA 12 support. Install the dependencies using [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
uv sync --locked --all-extras
source .venv/bin/activate
```

Run the commands below from the repository root.

## Prepare embeddings and graphs

### Goal embeddings

All methods evaluated in the paper use fixed Qwen3-Embedding-4B embeddings for the 256 goals in the benchmark. Generate these embeddings with:

```bash
python scripts/goal_embeddings.py
```

The command downloads the embedding model and saves the embeddings to `artifacts/goal_embeddings/craftax_256/qwen3_embedding_4b_512.npz`.

### Task graphs

The method table below lists the learning-connectivity (LC) and feasibility-evidence (FE) graphs required by each method.

Graph generation uses the Gemini API. Add your [API key](https://ai.google.dev/gemini-api/docs/api-key) to a `.env` file in the repository root:

```dotenv
GEMINI_API_KEY=your_api_key
```

Generate graphs for the primitive-action learner:

```bash
python scripts/lc_graph.py generate --learner primitive \
    --output-dir LC_GRAPH_DIR
python scripts/feasibility_graph.py generate --learner primitive \
    --output-dir FEASIBILITY_GRAPH_DIR
```

For the subroutine learner, use `--learner subroutine`.

To resume interrupted generation, use the corresponding script with `resume` and the same output directory:

```bash
python scripts/lc_graph.py resume --output-dir LC_GRAPH_DIR
python scripts/feasibility_graph.py resume --output-dir FEASIBILITY_GRAPH_DIR
```

## Run experiments

Use `scripts/train.py` to train a selected curriculum method and learner with either equally weighted goals or a single-target objective. The examples below use ALP+D+F, the proposed method; the other methods are baselines and ablations.

| Method | `--method` | Required graphs |
| --- | --- | --- |
| ALP+D+F | `alp-d-f` | LC and FE |
| ALP+D | `alp-d` | LC |
| ALP+F | `alp-f` | FE |
| ALP | `alp` | None |
| UNIF | `uniform` | None |
| I-DIFF | `intermediate-difficulty` | None |
| TARGET | `target` | None |
| PERM | `permuted` | LC and FE |

Provide `--lc-graph-dir` and `--feasibility-graph-dir` when the method requires them, using graphs generated for the selected learner. PERM additionally requires `--graph-permutation-seed`, which is independent of the training seed. Set `--learner` to `primitive` or `subroutine`. See `python scripts/train.py start --help` for all options.

### Equally weighted goals

Train across all 256 goals with equal objective weights:

```bash
python scripts/train.py start --method alp-d-f --learner LEARNER --seed 1 \
    --run-dir RUN_DIR \
    --lc-graph-dir LC_GRAPH_DIR \
    --feasibility-graph-dir FEASIBILITY_GRAPH_DIR
```

### Single target

Assign all objective weight to a single target, with other goals available for auxiliary training:

```bash
python scripts/train.py start --method alp-d-f --learner LEARNER --seed 1 \
    --target TARGET_TASK_ID --run-dir RUN_DIR \
    --lc-graph-dir LC_GRAPH_DIR \
    --feasibility-graph-dir FEASIBILITY_GRAPH_DIR
```

The paper's four target tasks are:

| Target | Task ID |
| --- | --- |
| Dungeon | `player_level.ge_1` |
| Skeleton | `mob.ranged.skeleton.defeated` |
| Wood | `inventory.wood.ge_64` |
| Furnace | `relation.block.furnace.adjacent_4` |

## Results and resuming

Each run saves its outputs under `--run-dir`:

- `run.json` records the run's settings, task IDs, and objective weights.
- `metrics.npz` contains the recorded training statistics, curriculum information, and evaluation results.
- `latest_checkpoint.msgpack` contains the saved policy and training state needed to resume the run.

Resume from the latest saved checkpoint with:

```bash
python scripts/train.py resume --run-dir RUN_DIR
```

This restores the run's saved configuration and training state.
