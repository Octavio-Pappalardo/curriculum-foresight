from __future__ import annotations

import copy
import json
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from curriculum_foresight.goals.types import GoalSpace
from curriculum_foresight.graphs.prompt_context import (
    CRAFTAX_DETAILED_ENVIRONMENT_CONTEXT,
    CUSTOM_GOAL_SUITE_CONTEXT,
    LEARNER_CONTEXTS,
)


class TargetEdgeSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_id: str = Field(description="Exact ID of one supplied target goal.")
    transfer_source_ids: list[str] = Field(
        description="Exact IDs of all supplied source goals satisfying the edge criterion "
        "for this target. Empty if none qualify."
    )


class LCGraphBlockResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    targets: list[TargetEdgeSelection] = Field(description="One selection for each supplied target goal.")


def build_lc_graph_response_json_schema(source_ids: Sequence[str], target_ids: Sequence[str]) -> dict[str, object]:
    schema = copy.deepcopy(LCGraphBlockResponse.model_json_schema())
    selection_schema = schema["$defs"]["TargetEdgeSelection"]
    selection_schema["properties"]["target_id"]["enum"] = list(target_ids)
    selection_schema["properties"]["transfer_source_ids"]["items"]["enum"] = list(source_ids)
    return schema


def render_lc_graph_prompt(
    goal_space: GoalSpace,
    source_indices: Sequence[int],
    target_indices: Sequence[int],
    *,
    learner_kind: str = "primitive",
) -> str:
    source_records = [
        {"id": goal_space.task_ids[int(index)], "goal": goal_space.goal_texts[int(index)]} for index in source_indices
    ]
    target_records = [
        {"id": goal_space.task_ids[int(index)], "goal": goal_space.goal_texts[int(index)]} for index in target_indices
    ]
    sources_json = json.dumps(source_records, ensure_ascii=False, indent=2)
    targets_json = json.dumps(target_records, ensure_ascii=False, indent=2)

    return f"""\
Task: infer directed learning-connectivity edges between the supplied Craftax goals. Your predictions will be used to construct a learning-connectivity graph for an automatic curriculum-learning algorithm.

## Environment context

{CRAFTAX_DETAILED_ENVIRONMENT_CONTEXT.strip()}

{CUSTOM_GOAL_SUITE_CONTEXT.strip()}

## Learner context

{LEARNER_CONTEXTS[learner_kind].strip()}

## Meaning of an edge

Assume that training on the source goal meaningfully improves the learner’s ability to achieve it. Include an edge `source -> target` only when the resulting learning is reasonably expected to improve the policy’s ability to achieve the target or make subsequent training on the target more productive.

That is, include an edge when you assess that either of the following holds:

1. **Immediate transfer:** source learning directly improves the policy’s ability to achieve the target in a new episode where the policy is conditioned on the target, before any target-specific training.
2. **Enabling transfer:** source learning makes later training on the target more likely to produce meaningful policy improvement. It may make future progress on the target possible or allow it to occur faster or more reliably.

Include only transfer that is expected to be meaningful and attributable to what is learned while improving on the source; benefits expected from getting any generic additional experience on the environment are not sufficient.

During training, the curriculum algorithm decides which goals to train. It uses these edges to value progress on a source partly according to whether the resulting learning could also benefit other goals later.

Textual similarity, thematic relatedness, or a game-mechanical prerequisite relationship may provide evidence of transfer when judging an edge, but they do not determine the judgment by themselves. Learning transfer need not follow relative goal difficulty or prerequisite order; a harder source may also benefit an easier target. Within-episode state changes associated with completing a source are temporary and should be distinguished from learning retained in the shared policy. Include an edge only when progress on the source is expected to produce retained learning that benefits the target under the definition above. Include every qualifying edge for which progress on the source is reasonably expected to improve the policy’s ability to achieve the target or make subsequent training on the target more productive.

## Candidate goals

Potential sources:
{sources_json}

Targets:
{targets_json}

## Response requirements

Populate the supplied structured response with exactly one record for every prompted target.

* Reproduce every `target_id` exactly as it appears in the prompted target list.
* Use only exact IDs from the prompted source list in `transfer_source_ids`.
* Use an empty `transfer_source_ids` list when no source qualifies.
* Do not duplicate target IDs or source IDs.
* Do not include a goal as a transfer source for itself.
* Return only the supplied structured response.

"""
