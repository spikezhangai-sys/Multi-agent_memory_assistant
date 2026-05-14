from __future__ import annotations

import json
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from driftscope.llm.client import StructuredLLM


class TopicPredictor(Protocol):
    """Predict candidate topic paths for a query. Used by HybridRetrieverAgent
    to fan out retrieval over multiple topic interpretations."""

    def predict(self, query: str, available_topics: list[str]) -> list[str]:
        ...


class TopicPredictionList(BaseModel):
    model_config = ConfigDict(extra="forbid")

    topics: list[str] = Field(
        default_factory=list,
        description=(
            "Ordered list of topic paths the query is most likely about. "
            "Each entry must be one of the paths in the available list."
        ),
    )


class LLMTopicPredictor:
    """LLM-backed topic predictor.

    Given a query and the set of currently known topic leaves, asks the LLM
    to return up to ``max_topics`` paths that the query plausibly relates to.
    Output is constrained to existing paths so the retriever can use them
    directly as predicted_topic in scoring passes.
    """

    def __init__(
        self,
        llm: StructuredLLM,
        *,
        max_topics: int = 4,
        max_available_topics_in_prompt: int = 200,
    ) -> None:
        self.llm = llm
        self.max_topics = max(1, max_topics)
        self.max_available_topics_in_prompt = max_available_topics_in_prompt

    def predict(self, query: str, available_topics: list[str]) -> list[str]:
        if not query.strip() or not available_topics:
            return []

        truncated = available_topics[: self.max_available_topics_in_prompt]
        user_prompt = json.dumps(
            {
                "query": query,
                "available_topics": truncated,
                "max_topics": self.max_topics,
            },
            ensure_ascii=False,
        )
        try:
            raw = self.llm.generate_structured(
                system_prompt=_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                response_model=TopicPredictionList,
            )
            if not isinstance(raw, TopicPredictionList):
                raw = TopicPredictionList.model_validate(raw)
        except Exception:
            return []

        allowed = set(truncated)
        seen: set[str] = set()
        ordered: list[str] = []
        for path in raw.topics:
            if path in allowed and path not in seen:
                seen.add(path)
                ordered.append(path)
            if len(ordered) >= self.max_topics:
                break
        return ordered


_SYSTEM_PROMPT = """You are the Topic Predictor for a memory retrieval system.

Given a user's query and the list of currently known topic leaves in the memory store, return the topic paths that the query is most plausibly about.

# Rules
- Return AT MOST `max_topics` paths, ordered from most likely to least likely.
- Each path MUST be one of the paths in `available_topics`. Do NOT invent new paths.
- The query may have multiple semantic dimensions (e.g. "bike-related expenses" combines a category dimension `vehicle/helmet/bike_accessory` with a financial dimension `expense/purchase`). Pick paths that cover ALL dimensions you detect.
- If the query is broad (e.g. "what did I say about my health") return multiple sibling leaves under the relevant category.
- Return an empty list only if NO available topic is even tangentially relevant.

Output strict JSON: {"topics": ["user.path.one", "user.path.two", ...]}.
"""
