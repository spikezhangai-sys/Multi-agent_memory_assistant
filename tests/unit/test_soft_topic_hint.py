from __future__ import annotations

from datetime import UTC, datetime

import numpy as np

from driftscope.agents.retriever_agent import HybridRetrieverAgent
from driftscope.agents.types import RetrievalInput
from driftscope.config.loader import load_default_config
from driftscope.core.memory_base import MemoryBase
from driftscope.core.schema import Confidence, MemoryEntry, Scope, TimeRange


class DirectionalEmbedder:
    """Each known text maps to a fixed unit vector; unknown texts to axis 0.

    Vectors are hand-chosen so that:
      - cosine(food, cooking) ≈ 0.7 (above soft-hint threshold 0.55,
        below canonicalize threshold 0.85)
      - cosine(food, electronics) = 0
    Used to drive deterministic outcomes for topic-hint tests without
    depending on a real sentence transformer.
    """

    model = "directional-test"
    dim = 8

    _vector_by_text = {
        # Seed suffixes: each on its own axis (unit vectors).
        "food": np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32),
        "diet": np.array([0, 0, 1, 0, 0, 0, 0, 0], dtype=np.float32),
        "workstyle": np.array([0, 0, 0, 1, 0, 0, 0, 0], dtype=np.float32),
        "software": np.array([0, 0, 0, 0, 1, 0, 0, 0], dtype=np.float32),
        # Runtime: cooking leans into food's axis (0.7) + its own (0.7).
        "cooking": np.array([0, 0.70710677, 0, 0, 0, 0.70710677, 0, 0], dtype=np.float32),
        "recipes": np.array([0, 0.70710677, 0, 0, 0, 0.70710677, 0, 0], dtype=np.float32),
        # Runtime: electronics on its own axis, orthogonal to food.
        "electronics": np.array([0, 0, 0, 0, 0, 0, 1, 0], dtype=np.float32),
    }

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            key = text.strip().lower().replace(" ", "_")
            vec = self._vector_by_text.get(key)
            if vec is None:
                out[row, 7] = 1.0
            else:
                out[row] = vec
        return out


def _make_memory(
    *,
    topic_id: str,
    content: str,
    timestamp: datetime,
) -> MemoryEntry:
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


def test_soft_topic_hint_exact_match_scores_one() -> None:
    embedder = DirectionalEmbedder()
    mb = MemoryBase(":memory:", embedder=embedder)
    mb._ensure_seeds_embedded("user.preference")

    retriever = HybridRetrieverAgent(memory_base=mb, embedder=embedder)
    leaves = {path: vec for path, vec in mb.known_leaves_in_category("user.preference") if vec is not None}

    hint = retriever._topic_hint_score(
        memory_topic_id="user.preference.food",
        predicted_topic="user.preference.food",
        predicted_category="user.preference",
        leaf_vectors_by_path=leaves,
    )
    assert hint == 1.0


def test_soft_topic_hint_same_category_similar_leaf_scores_between_floor_and_one() -> None:
    embedder = DirectionalEmbedder()
    mb = MemoryBase(":memory:", embedder=embedder)
    mb.canonicalize_topic("user.preference", "cooking")

    retriever = HybridRetrieverAgent(memory_base=mb, embedder=embedder)
    leaves = {path: vec for path, vec in mb.known_leaves_in_category("user.preference") if vec is not None}

    hint = retriever._topic_hint_score(
        memory_topic_id="user.preference.cooking",
        predicted_topic="user.preference.food",
        predicted_category="user.preference",
        leaf_vectors_by_path=leaves,
    )
    floor = retriever.config.retrieval.topic_soft_hint_floor
    assert floor <= hint <= 1.0
    assert hint < 1.0


def test_soft_topic_hint_same_category_dissimilar_leaf_scores_sibling_floor() -> None:
    embedder = DirectionalEmbedder()
    mb = MemoryBase(":memory:", embedder=embedder)
    mb.canonicalize_topic("user.preference", "electronics")

    retriever = HybridRetrieverAgent(memory_base=mb, embedder=embedder)
    leaves = {path: vec for path, vec in mb.known_leaves_in_category("user.preference") if vec is not None}

    hint = retriever._topic_hint_score(
        memory_topic_id="user.preference.electronics",
        predicted_topic="user.preference.food",
        predicted_category="user.preference",
        leaf_vectors_by_path=leaves,
    )
    assert hint == retriever.config.retrieval.topic_sibling_floor


def test_soft_topic_hint_cross_category_is_zero_even_when_similar() -> None:
    embedder = DirectionalEmbedder()
    mb = MemoryBase(":memory:", embedder=embedder)

    retriever = HybridRetrieverAgent(memory_base=mb, embedder=embedder)
    # predicted in user.preference, memory in user.constraint → cross-category
    hint = retriever._topic_hint_score(
        memory_topic_id="user.constraint.diet",
        predicted_topic="user.preference.food",
        predicted_category="user.preference",
        leaf_vectors_by_path={},  # not consulted when categories differ
    )
    assert hint == 0.0


def test_query_embedding_predicts_runtime_leaf_when_keyword_match_misses() -> None:
    """Write-side registers `user.preference.music`; read-side query 'music'
    should resolve to it via embedding fallback (no seed has 'music' keyword)."""

    class QueryAwareEmbedder:
        """`music` suffix and `music` query both land on axis 5; orthogonal
        to all seed suffixes so the keyword matcher stays dormant too."""

        model = "qa-test"
        dim = 8

        _axis_by_text = {
            "food": 1,
            "diet": 2,
            "workstyle": 3,
            "software": 4,
            "music": 5,
        }

        def embed(self, texts: list[str]) -> np.ndarray:
            out = np.zeros((len(texts), self.dim), dtype=np.float32)
            for row, text in enumerate(texts):
                # Find which known key the text contains (query is a sentence).
                text_lower = text.lower()
                axis = None
                for key, ax in self._axis_by_text.items():
                    if key in text_lower:
                        axis = ax
                        break
                out[row, axis if axis is not None else 7] = 1.0
            return out

    embedder = QueryAwareEmbedder()
    mb = MemoryBase(":memory:", embedder=embedder)
    music_path = mb.canonicalize_topic("user.preference", "music")
    assert music_path == "user.preference.music"
    assert music_path not in {seed.path for seed in mb.topic_tree.seeds_in_category("user.preference")}

    now = datetime(2026, 4, 1, tzinfo=UTC)
    memory = _make_memory(topic_id=music_path, content="Loves jazz records", timestamp=now)
    mb.add(memory)

    retriever = HybridRetrieverAgent(memory_base=mb, embedder=embedder, config=load_default_config())
    result = retriever.run(
        RetrievalInput(
            query="What kind of music do I enjoy?",
            scope=Scope(kind="personal"),
            timestamp=now,
            allow_sensitive_raw=True,
        )
    )
    assert result.predicted_topic == "user.preference.music"
    assert result.gating_stats.get("predicted_via_embedding") == 1
    ranking = {m.memory.id: m for m in result.ranked_memories}
    assert memory.id in ranking
    assert ranking[memory.id].score_breakdown["topic_hint"] == 1.0


def test_query_embedding_fallback_skipped_when_below_threshold() -> None:
    """Query vector orthogonal to all known leaf vectors → predicted stays None."""

    class OrthogonalEmbedder:
        """Every known seed/runtime suffix maps to a distinct axis 0..N-2;
        every other text (the query here) maps to the last reserved axis
        so the query vector is orthogonal to all known leaves."""

        model = "orthogonal"
        dim = 64
        _query_axis = 63

        _known_suffixes = (
            # Seeds — keep in sync with topic_tree.yaml
            "food", "diet", "workstyle", "software",
            "location", "education", "errands", "cultural_visit",
            "fitness", "performance", "style", "schedule",
            "alpha", "shifts",
            "vehicle", "electronics", "subscriptions", "home",
            "family", "pets", "doctor_visit", "condition",
            # Runtime suffixes created by tests in this file
            "music",
        )
        _axis_by_text = {suffix: i for i, suffix in enumerate(_known_suffixes)}

        def embed(self, texts: list[str]) -> np.ndarray:
            out = np.zeros((len(texts), self.dim), dtype=np.float32)
            for row, text in enumerate(texts):
                key = text.strip().lower().replace(" ", "_")
                axis = self._axis_by_text.get(key, self._query_axis)
                out[row, axis] = 1.0
            return out

    embedder = OrthogonalEmbedder()
    mb = MemoryBase(":memory:", embedder=embedder)
    mb.canonicalize_topic("user.preference", "music")
    retriever = HybridRetrieverAgent(memory_base=mb, embedder=embedder, config=load_default_config())
    now = datetime(2026, 4, 1, tzinfo=UTC)
    result = retriever.run(
        RetrievalInput(
            query="zxqv lorem ipsum",
            scope=Scope(kind="personal"),
            timestamp=now,
            allow_sensitive_raw=True,
        )
    )
    # Query lands on axis 15, which no leaf occupies → cosine 0 everywhere.
    assert result.predicted_topic is None
    assert result.gating_stats.get("predicted_via_embedding") == 0


def test_soft_topic_hint_run_integration_ranks_similar_leaf_above_unrelated() -> None:
    """Full retrieval call: same-category similar leaf should rank above an
    unrelated memory when other signals are equal."""
    embedder = DirectionalEmbedder()
    mb = MemoryBase(":memory:", embedder=embedder)
    # Pre-register 'cooking' (similar to 'food') and 'electronics' (dissimilar).
    cooking_path = mb.canonicalize_topic("user.preference", "cooking")
    electronics_path = mb.canonicalize_topic("user.preference", "electronics")
    assert cooking_path == "user.preference.cooking"
    assert electronics_path == "user.preference.electronics"

    now = datetime(2026, 4, 1, tzinfo=UTC)
    similar = _make_memory(
        topic_id=cooking_path,
        content="Loves trying new cuisine styles",
        timestamp=now,
    )
    unrelated = _make_memory(
        topic_id=electronics_path,
        content="Collects vintage audio gear",
        timestamp=now,
    )
    mb.add(similar)
    mb.add(unrelated)

    retriever = HybridRetrieverAgent(memory_base=mb, embedder=embedder, config=load_default_config())
    # Craft a query that matches the 'food' seed so predicted_topic=user.preference.food
    result = retriever.run(
        RetrievalInput(
            query="我最近喜欢研究新的食物 cuisine 口味",
            scope=Scope(kind="personal"),
            timestamp=now,
            allow_sensitive_raw=True,
        )
    )
    ranking = {m.memory.id: m for m in result.ranked_memories}
    assert similar.id in ranking
    assert unrelated.id in ranking
    assert ranking[similar.id].score > ranking[unrelated.id].score
    sibling_floor = retriever.config.retrieval.topic_sibling_floor
    assert ranking[similar.id].score_breakdown["topic_hint"] > sibling_floor
    assert ranking[unrelated.id].score_breakdown["topic_hint"] == sibling_floor
