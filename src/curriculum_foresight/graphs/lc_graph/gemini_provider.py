from __future__ import annotations

import time
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

from google import genai
from google.genai import types as genai_types
from tqdm import tqdm

from curriculum_foresight.graphs.common import GeminiProviderConfig, gemini_provider_metadata, usage_metadata_to_dict
from curriculum_foresight.graphs.lc_graph.generation import (
    LCGraphGeneration,
    LCGraphGenerationError,
    LCGraphRequest,
    parse_and_validate_lc_graph_response,
)
from curriculum_foresight.graphs.lc_graph.prompt import build_lc_graph_response_json_schema


class GeminiLCGraphProvider:
    def __init__(self, config: GeminiProviderConfig, *, client: Any | None = None, show_progress: bool = False):
        self._config = config
        self._client = client
        self._show_progress = show_progress

    @property
    def metadata(self) -> Mapping[str, object]:
        return gemini_provider_metadata(self._config)

    def generate_incrementally(
        self,
        requests: Sequence[LCGraphRequest],
        *,
        request_ordinal_offset: int = 0,
        first_request_attempt_offset: int = 0,
    ) -> Iterator[LCGraphGeneration]:
        request_iter = (
            tqdm(requests, total=len(requests), desc="Generating LC graph", unit="request")
            if self._show_progress
            else requests
        )
        for relative_ordinal, request in enumerate(request_iter):
            yield self._generate_one(
                request,
                request_ordinal_offset + relative_ordinal,
                attempt_offset=first_request_attempt_offset if relative_ordinal == 0 else 0,
            )

    def _generate_one(
        self, request: LCGraphRequest, request_ordinal: int, *, attempt_offset: int = 0
    ) -> LCGraphGeneration:
        if self._client is None:
            self._client = genai.Client()
        max_attempts = self._config.max_retries + 1
        last_error: Exception | None = None
        for local_attempt_index in range(max_attempts):
            attempt_index = attempt_offset + local_attempt_index
            successful_seed = self._attempt_seed(request_ordinal, attempt_index)
            generation_config_kwargs: dict[str, object] = {
                "response_mime_type": "application/json",
                "response_json_schema": build_lc_graph_response_json_schema(request.source_ids, request.target_ids),
                "thinking_config": genai_types.ThinkingConfig(thinking_level=self._config.thinking_level),
                "max_output_tokens": self._config.max_output_tokens,
            }
            if successful_seed is not None:
                generation_config_kwargs["seed"] = successful_seed

            try:
                response = self._client.models.generate_content(
                    model=self._config.model,
                    contents=request.prompt,
                    config=genai_types.GenerateContentConfig(**generation_config_kwargs),
                )
                raw_response_text = getattr(response, "text", None)
                if not isinstance(raw_response_text, str):
                    msg = f"Gemini response for LC graph request {request.request_id!r} did not contain text."
                    raise ValueError(msg)
                canonical_response = parse_and_validate_lc_graph_response(raw_response_text, request)
            except Exception as exc:
                last_error = exc
                if local_attempt_index < self._config.max_retries:
                    time.sleep(2**local_attempt_index)
                continue

            metadata: dict[str, object] = {"num_attempts": attempt_index + 1}
            if successful_seed is not None:
                metadata["successful_seed"] = successful_seed
            usage_metadata = usage_metadata_to_dict(getattr(response, "usage_metadata", None))
            if usage_metadata is not None:
                metadata["usage_metadata"] = usage_metadata
            return LCGraphGeneration(
                request_id=request.request_id,
                raw_response_text=raw_response_text,
                response=canonical_response,
                metadata=metadata,
            )

        total_known_attempts = attempt_offset + max_attempts
        msg = (
            f"Gemini LC graph generation failed for request {request.request_id!r} "
            f"after {max_attempts} attempts ({total_known_attempts} including previous attempts)."
        )
        raise LCGraphGenerationError(msg, attempts_made=max_attempts) from last_error

    def _attempt_seed(self, request_ordinal: int, attempt_index: int) -> int | None:
        if self._config.generation_seed is None:
            return None
        return self._config.generation_seed + request_ordinal * (self._config.max_retries + 1) + attempt_index
