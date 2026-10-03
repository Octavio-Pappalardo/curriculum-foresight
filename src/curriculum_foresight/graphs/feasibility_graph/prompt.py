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


class GoalFeasibilityJudgment(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    evaluated_goal_id: str = Field(description="Exact ID of one goal from `Goals to evaluate`.")
    feasible_from_scratch: bool = Field(
        description="Whether a learner whose trainable policy parameters have just been initialized is likely to "
        "make progress by training directly on the evaluated goal."
    )
    readiness_evidence_goal_ids: list[str] = Field(
        description="Exact IDs of the goals selected as readiness evidence for the evaluated goal from `Goals "
        "available as readiness evidence`. Empty exactly when `feasible_from_scratch` is true."
    )


class FeasibilityGraphBlockResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    judgments: list[GoalFeasibilityJudgment] = Field(description="One judgment for every goal in `Goals to evaluate`.")


def build_feasibility_graph_response_json_schema(
    target_ids: Sequence[str], *, max_feasibility_sources_per_target: int
) -> dict[str, object]:
    schema = copy.deepcopy(FeasibilityGraphBlockResponse.model_json_schema())
    judgments_schema = schema["properties"]["judgments"]
    judgments_schema["minItems"] = len(target_ids)
    judgments_schema["maxItems"] = len(target_ids)
    judgment_schema = schema["$defs"]["GoalFeasibilityJudgment"]
    judgment_schema["properties"]["evaluated_goal_id"]["enum"] = list(target_ids)
    readiness_evidence_ids_schema = judgment_schema["properties"]["readiness_evidence_goal_ids"]
    readiness_evidence_ids_schema["maxItems"] = max_feasibility_sources_per_target
    return schema


def render_feasibility_graph_prompt(
    goal_space: GoalSpace,
    source_indices: Sequence[int],
    target_indices: Sequence[int],
    *,
    max_feasibility_sources_per_target: int,
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
Task: assess when direct training on each supplied Craftax goal is likely to produce measurable learning progress. Your judgments will be used to construct a feasibility graph for an automatic curriculum-learning algorithm.

## Environment context

{CRAFTAX_DETAILED_ENVIRONMENT_CONTEXT.strip()}

## Goal-suite context

{CUSTOM_GOAL_SUITE_CONTEXT.strip()}

## Learner context

{LEARNER_CONTEXTS[learner_kind].strip()}

## What to decide for each goal

For each goal in `Goals to evaluate`, imagine that the learner does not yet know how to achieve it. You must help decide when training directly on that goal (using it to condition the policy and rewarding only success on it over repeated fresh episodes) is likely to begin improving the policy’s success on it.

To make this judgment, identify the goals from `Goals available as readiness evidence` on which you would ordinarily expect the learner to exhibit competence before direct training on the evaluated goal is likely to begin producing improvement. The learner is competent on a goal when it can achieve that goal with nontrivial, reasonably reliable success while conditioned on it.

This judgment requires careful reasoning about the characteristics of the learner and learning setting, the goals’ success conditions and relationships, and the relevant Craftax mechanics and environment structure.

During training, the algorithm will use these judgments only while the evaluated goal’s own success rate remains zero or low. When `feasible_from_scratch` is false, it will average the learner’s current success rates on the selected readiness-evidence goals. A higher average will be interpreted as stronger evidence of readiness and make the evaluated goal more likely to be selected for direct training. A `feasible_from_scratch` judgment instead provides readiness evidence that does not depend on the learner’s success on another goal.

Set `feasible_from_scratch` to true only when a learner whose trainable policy parameters have just been initialized is likely to encounter rewarded success often enough for direct training on the evaluated goal to begin improving the policy. The fact that the goal can be achieved in principle is not sufficient.

Otherwise, set `feasible_from_scratch` to false and select between 1 and {max_feasibility_sources_per_target} goals from `Goals available as readiness evidence`. Select the most informative goals that satisfy the criterion above. Do not select a goal that the learner would ordinarily be expected to master only after the evaluated goal. Among the remaining qualifying goals, omit earlier or broader goals when competence on a more advanced or specific goal already provides the same evidence. The upper bound is only a cap, not a target. Include only goals whose competence would provide meaningful readiness evidence on its own, and do not include goals whose evidence is subsumed by that of other selected goals.

Before returning the response, review every judgment for consistency with the relevant Craftax mechanics, the supplied goal definitions, and the learner and readiness-evidence definitions above.

## Goals

Goals available as readiness evidence:
{sources_json}

Goals to evaluate:
{targets_json}

## Response requirements

Return exactly one entry in `judgments` for every goal in `Goals to evaluate`.

* In each entry, copy the corresponding goal’s `id` exactly into `evaluated_goal_id`.
* If `feasible_from_scratch` is true, return an empty `readiness_evidence_goal_ids` list.
* If `feasible_from_scratch` is false, return between 1 and {max_feasibility_sources_per_target} exact IDs from `Goals available as readiness evidence` in `readiness_evidence_goal_ids`.
* Do not repeat an `evaluated_goal_id` or any ID within a `readiness_evidence_goal_ids` list.
* Do not include the evaluated goal’s own ID in its `readiness_evidence_goal_ids` list.
* Return only the supplied structured response.

"""
