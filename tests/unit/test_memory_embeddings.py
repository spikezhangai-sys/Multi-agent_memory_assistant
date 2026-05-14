from datetime import UTC, datetime

import numpy as np

from driftscope.core.memory_base import MemoryBase
from driftscope.core.schema import Scope
from driftscope.embeddings import MockEmbedder
from tests.unit.helpers import make_memory


def test_memory_base_stores_and_retrieves_embeddings() -> None:
    embedder = MockEmbedder(dim=64)
    store = MemoryBase(embedder=embedder)
    memory = make_memory(
        content="I moved to the Caldermere district last month.",
        topic_id=None,
        scope=Scope(kind="personal"),
    )

    store.add(memory)

    result = store.get_embedding(memory.id)
    assert result is not None
    model, vector = result
    assert model == "mock"
    assert vector.shape == (64,)
    assert np.isclose(np.linalg.norm(vector), 1.0)


def test_query_visible_with_vectors_pairs_every_memory() -> None:
    embedder = MockEmbedder(dim=32)
    store = MemoryBase(embedder=embedder)
    scope = Scope(kind="personal")
    store.add(
        make_memory(
            content="I moved to the Caldermere district last month.",
            topic_id=None,
            scope=scope,
        )
    )
    store.add(
        make_memory(
            content="I love quiet cafes.",
            topic_id="user.preference.food",
            scope=scope,
            memory_type="preference",
        )
    )

    paired = store.query_visible_with_vectors(scope, datetime(2026, 4, 2, tzinfo=UTC))
    assert len(paired) == 2
    for memory, vector in paired:
        assert vector is not None
        assert vector.shape == (32,)
