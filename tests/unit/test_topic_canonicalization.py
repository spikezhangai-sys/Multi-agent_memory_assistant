from __future__ import annotations

import numpy as np

from driftscope.core.memory_base import MemoryBase
from driftscope.core.topic_tree import (
    TopicCategory,
    TopicLeaf,
    TopicTree,
    category_of_path,
    normalize_leaf_suffix,
)
from driftscope.embeddings.client import MockEmbedder


def _small_tree() -> TopicTree:
    categories = [
        TopicCategory(id="user.preference", description="tastes", default_type="preference"),
        TopicCategory(id="user.constraint", description="limits", default_type="constraint"),
    ]
    seeds = [
        TopicLeaf(path="user.preference.food", description="food tastes", keywords=["food"]),
        TopicLeaf(path="user.constraint.diet", description="diet limits", keywords=["diet"]),
    ]
    return TopicTree(categories=categories, seed_leaves=seeds)


def test_normalize_leaf_suffix_canonicalizes_casing_and_spaces() -> None:
    assert normalize_leaf_suffix("Books") == "books"
    assert normalize_leaf_suffix("Books With Space") == "books_with_space"
    assert normalize_leaf_suffix("books-fiction") == "books_fiction"
    assert normalize_leaf_suffix("  trailing  ") == "trailing"


def test_normalize_leaf_suffix_rejects_invalid() -> None:
    assert normalize_leaf_suffix("") is None
    assert normalize_leaf_suffix("食物") is None
    assert normalize_leaf_suffix(".foo") is None
    assert normalize_leaf_suffix("a" * 200) is None


def test_category_of_path_strips_last_segment() -> None:
    assert category_of_path("user.preference.food") == "user.preference"
    assert category_of_path("single") == "single"


def test_topic_tree_has_category_enforces_closed_set() -> None:
    tree = _small_tree()
    assert tree.has_category("user.preference")
    assert not tree.has_category("user.whatever")


def test_topic_tree_compose_leaf_path_validates_category_and_suffix() -> None:
    tree = _small_tree()
    assert tree.compose_leaf_path("user.preference", "books") == "user.preference.books"
    assert tree.compose_leaf_path("user.whatever", "books") is None
    assert tree.compose_leaf_path("user.preference", "") is None


def test_topic_tree_rejects_seed_whose_category_is_missing() -> None:
    import pytest

    categories = [TopicCategory(id="user.preference", description="tastes", default_type="preference")]
    seeds = [TopicLeaf(path="missing.bucket.foo", description="x")]
    with pytest.raises(ValueError, match="no registered category"):
        TopicTree(categories=categories, seed_leaves=seeds)


def test_canonicalize_topic_returns_seed_path_for_seed_suffix() -> None:
    mb = MemoryBase(":memory:")
    resolved = mb.canonicalize_topic("user.preference", "food")
    assert resolved == "user.preference.food"


def test_canonicalize_topic_returns_none_for_unknown_category() -> None:
    mb = MemoryBase(":memory:")
    assert mb.canonicalize_topic("user.whatever", "books") is None


def test_canonicalize_topic_returns_none_for_invalid_suffix() -> None:
    mb = MemoryBase(":memory:")
    assert mb.canonicalize_topic("user.preference", "") is None
    assert mb.canonicalize_topic("user.preference", "食物") is None


def test_canonicalize_topic_registers_novel_leaf_without_embedder() -> None:
    mb = MemoryBase(":memory:")  # no embedder
    resolved = mb.canonicalize_topic("user.preference", "books")
    assert resolved == "user.preference.books"
    assert mb.is_known_topic("user.preference.books")
    # second call is idempotent
    assert mb.canonicalize_topic("user.preference", "books") == "user.preference.books"


def test_canonicalize_topic_merges_similar_leaves_via_embedding() -> None:
    class DirectionalEmbedder:
        """Deterministic embedder: each suffix gets a unit vector along a
        preassigned axis; 'books' and 'reading' share the same axis so they
        collapse under canonicalization."""

        model = "directional"
        dim = 4

        _axis_by_suffix = {
            "food": 0,
            "diet": 1,
            "books": 2,
            "reading": 2,
            "workstyle": 3,
        }

        def embed(self, texts: list[str]) -> np.ndarray:
            vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
            for row, text in enumerate(texts):
                key = text.strip().lower().replace(" ", "_")
                axis = self._axis_by_suffix.get(key, 0)
                vectors[row, axis] = 1.0
            return vectors

    mb = MemoryBase(":memory:", embedder=DirectionalEmbedder())
    first = mb.canonicalize_topic("user.preference", "books")
    assert first == "user.preference.books"
    second = mb.canonicalize_topic("user.preference", "reading")
    assert second == "user.preference.books"  # collapsed to canonical


def test_canonicalize_topic_keeps_distinct_leaves_below_threshold() -> None:
    mb = MemoryBase(":memory:", embedder=MockEmbedder(dim=128))
    first = mb.canonicalize_topic("user.preference", "books")
    second = mb.canonicalize_topic("user.preference", "unrelated_gardening_slug")
    assert first == "user.preference.books"
    assert second == "user.preference.unrelated_gardening_slug"


def test_is_known_topic_spans_seeds_and_registered() -> None:
    mb = MemoryBase(":memory:")
    assert mb.is_known_topic("user.preference.food")
    assert not mb.is_known_topic("user.preference.books")
    mb.canonicalize_topic("user.preference", "books")
    assert mb.is_known_topic("user.preference.books")


def test_memory_base_add_accepts_registered_leaf_topic() -> None:
    from datetime import UTC, datetime

    from driftscope.core.schema import Confidence, MemoryEntry, Scope, TimeRange

    mb = MemoryBase(":memory:")
    path = mb.canonicalize_topic("user.preference", "board_games")
    assert path == "user.preference.board_games"
    now = datetime(2026, 4, 1, tzinfo=UTC)
    entry = MemoryEntry(
        content="User enjoys complex eurogames",
        type="preference",
        topic_id=path,
        scope=Scope(kind="personal"),
        src="user_explicit",
        conf=Confidence(prior=0.9, llm_self=0.8, combined=0.87),
        valid_time=TimeRange(start=now),
        ingest_time=now,
    )
    mb.add(entry)  # must not raise
    assert mb.get(entry.id).topic_id == "user.preference.board_games"
