from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import numpy as np
from pydantic import BaseModel

from driftscope.agents.retriever_agent import HybridRetrieverAgent
from driftscope.agents.topic_predictor import LLMTopicPredictor, TopicPredictionList
from driftscope.agents.types import RetrievalInput
from driftscope.config.loader import load_default_config
from driftscope.core.memory_base import MemoryBase
from driftscope.core.schema import Confidence, MemoryEntry, Scope, TimeRange


class _DirectionalEmbedder:
    model = "directional-test"
    dim = 6

    _vectors = {
        "food": np.array([1, 0, 0, 0, 0, 0], dtype=np.float32),
        "cooking": np.array([0.7, 0.7, 0, 0, 0, 0], dtype=np.float32),
        "electronics": np.array([0, 0, 1, 0, 0, 0], dtype=np.float32),
    }

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            key = text.strip().lower()
            vec = self._vectors.get(key)
            if vec is None:
                out[row, 5] = 1.0
            else:
                out[row] = vec
        return out


def _make_memory(*, topic_id: str, content: str, timestamp: datetime) -> MemoryEntry:
    return MemoryEntry(
        content=content,
        type="preference",
        topic_id=topic_id,
        scope=Scope(kind="personal"),
        src="user_explicit",
        conf=Confidence(prior=0.9, llm_self=0.8, combined=0.87),
        valid_time=TimeRange(start=timestamp),
        ingest_time=timestamp,
    )


class _StaticPredictor:
    """Test double for TopicPredictor that returns a preset list."""

    def __init__(self, paths: list[str]) -> None:
        self._paths = paths
        self.calls: list[tuple[str, list[str]]] = []

    def predict(self, query: str, available_topics: list[str]) -> list[str]:
        self.calls.append((query, list(available_topics)))
        return [p for p in self._paths if p in set(available_topics)]


class _ScriptedLLM:
    """Mock StructuredLLM that returns a preset payload for any call."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.calls: list[dict[str, str]] = []

    def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[BaseModel],
    ) -> BaseModel:
        self.calls.append({"system": system_prompt, "user": user_prompt})
        return response_model.model_validate(self.payload)


def test_llm_topic_predictor_returns_only_known_paths() -> None:
    llm = _ScriptedLLM({
        "topics": [
            "user.preference.cooking",       # known
            "user.preference.unknown_path",  # filtered out
            "user.preference.electronics",   # known
        ]
    })
    predictor = LLMTopicPredictor(llm, max_topics=4)

    available = ["user.preference.cooking", "user.preference.electronics", "user.preference.food"]
    result = predictor.predict("anything", available)

    assert result == ["user.preference.cooking", "user.preference.electronics"]
    assert len(llm.calls) == 1


def test_llm_topic_predictor_caps_at_max_topics() -> None:
    llm = _ScriptedLLM({
        "topics": [
            "user.preference.cooking",
            "user.preference.electronics",
            "user.preference.food",
        ]
    })
    predictor = LLMTopicPredictor(llm, max_topics=2)
    result = predictor.predict(
        "anything",
        ["user.preference.cooking", "user.preference.electronics", "user.preference.food"],
    )
    assert result == ["user.preference.cooking", "user.preference.electronics"]


def test_llm_topic_predictor_returns_empty_on_llm_failure() -> None:
    class BoomLLM:
        def generate_structured(self, **_: Any) -> Any:
            raise RuntimeError("LLM down")

    predictor = LLMTopicPredictor(BoomLLM())
    result = predictor.predict("anything", ["user.preference.cooking"])
    assert result == []


def test_hybrid_retriever_uses_predictor_to_promote_sibling_leaf() -> None:
    """With multi-topic predictor returning two siblings, a memory whose only
    strong signal is an exact topic match (not keyword content overlap) should
    still surface in top results — which is impossible with single-topic.
    """
    embedder = _DirectionalEmbedder()
    mb = MemoryBase(":memory:", embedder=embedder)
    cooking_path = mb.canonicalize_topic("user.preference", "cooking")
    electronics_path = mb.canonicalize_topic("user.preference", "electronics")
    assert cooking_path == "user.preference.cooking"
    assert electronics_path == "user.preference.electronics"

    now = datetime(2026, 4, 1, tzinfo=UTC)
    cooking_mem = _make_memory(
        topic_id=cooking_path,
        content="loves new recipes and cuisine",
        timestamp=now,
    )
    electronics_mem = _make_memory(
        topic_id=electronics_path,
        content="collects vintage stereo gear",  # no keyword overlap with query
        timestamp=now,
    )
    mb.add(cooking_mem)
    mb.add(electronics_mem)

    config = load_default_config()
    predictor = _StaticPredictor([cooking_path, electronics_path])
    retriever = HybridRetrieverAgent(
        memory_base=mb,
        embedder=embedder,
        config=config,
        topic_predictor=predictor,
    )

    result = retriever.run(
        RetrievalInput(
            query="food cuisine recipes",
            scope=Scope(kind="personal"),
            timestamp=now,
            allow_sensitive_raw=True,
        )
    )

    # Predictor was consulted with the available leaf paths.
    assert predictor.calls, "topic predictor should have been called"

    ranked_ids = [m.memory.id for m in result.ranked_memories]
    assert cooking_mem.id in ranked_ids
    assert electronics_mem.id in ranked_ids, (
        "sibling-leaf memory must reach top-K when its leaf is one of the candidate topics"
    )

    # gating_stats should advertise multi-topic fan-out happened.
    assert result.gating_stats.get("candidate_topics") == 2


def test_hybrid_retriever_falls_back_to_legacy_when_predictor_returns_empty() -> None:
    embedder = _DirectionalEmbedder()
    mb = MemoryBase(":memory:", embedder=embedder)
    cooking_path = mb.canonicalize_topic("user.preference", "cooking")
    now = datetime(2026, 4, 1, tzinfo=UTC)
    cooking_mem = _make_memory(
        topic_id=cooking_path,
        content="loves new recipes and cuisine",
        timestamp=now,
    )
    mb.add(cooking_mem)

    empty_predictor = _StaticPredictor([])  # always returns []
    retriever = HybridRetrieverAgent(
        memory_base=mb,
        embedder=embedder,
        config=load_default_config(),
        topic_predictor=empty_predictor,
    )
    result = retriever.run(
        RetrievalInput(
            query="food cuisine recipes",
            scope=Scope(kind="personal"),
            timestamp=now,
            allow_sensitive_raw=True,
        )
    )

    # Empty predictor result → legacy single-topic path → candidate_topics = 1.
    assert result.gating_stats.get("candidate_topics") == 1
    ranked_ids = [m.memory.id for m in result.ranked_memories]
    assert cooking_mem.id in ranked_ids
