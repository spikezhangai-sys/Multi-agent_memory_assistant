from datetime import UTC, datetime

import numpy as np

from driftscope.agents.retriever_agent import HybridRetrieverAgent
from driftscope.agents.types import RetrievalInput
from driftscope.config.loader import load_default_config
from driftscope.core.memory_base import MemoryBase
from driftscope.core.schema import Scope
from driftscope.embeddings import MockEmbedder
from driftscope.retrieval.query_time_parser import QueryTimeHint, QueryTimeParser
from tests.unit.helpers import make_memory


class _StubEmbedder:
    model = "stub"
    dim = 4

    def __init__(self, mapping: dict[str, list[float]]) -> None:
        self.mapping = mapping

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            vec = self.mapping.get(text, [0.0] * self.dim)
            vectors.append(vec)
        arr = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return arr / norms


def test_hybrid_retriever_uses_dense_match_when_sparse_misses() -> None:
    scope = Scope(kind="personal")
    blazer_memory = "I got her a teal blazer."
    distractor = "I love quiet weekends."
    embedder = _StubEmbedder(
        {
            # Query aligns with blazer_memory content, not with distractor.
            "What did I buy for my aunt's retirement?": [1.0, 0.0, 0.0, 0.0],
            blazer_memory: [0.95, 0.3, 0.0, 0.0],
            distractor: [0.0, 0.0, 1.0, 0.0],
        }
    )

    store = MemoryBase(embedder=embedder)
    blazer = make_memory(
        content=blazer_memory,
        topic_id=None,
        scope=scope,
    )
    other = make_memory(
        content=distractor,
        topic_id=None,
        scope=scope,
    )
    store.add(blazer)
    store.add(other)

    agent = HybridRetrieverAgent(memory_base=store, embedder=embedder)
    result = agent.run(
        RetrievalInput(
            query="What did I buy for my aunt's retirement?",
            scope=scope,
            timestamp=datetime(2026, 4, 2, tzinfo=UTC),
        )
    )

    assert result.ranked_memories[0].memory.id == blazer.id
    assert "dense" in result.ranked_memories[0].matched_by


def test_hybrid_retriever_falls_back_to_sparse_without_embedder() -> None:
    scope = Scope(kind="personal")
    store = MemoryBase()
    location = make_memory(
        content="我现在住在上海",
        topic_id="user.profile.location",
        scope=scope,
    )
    food = make_memory(
        content="我最近喜欢吃日料",
        topic_id="user.preference.food",
        scope=scope,
        memory_type="preference",
    )
    store.add(location)
    store.add(food)

    agent = HybridRetrieverAgent(memory_base=store)
    result = agent.run(
        RetrievalInput(
            query="我住在哪？",
            scope=scope,
            timestamp=datetime(2026, 4, 2, tzinfo=UTC),
        )
    )

    assert result.ranked_memories[0].memory.id == location.id


def test_hybrid_retriever_applies_time_proximity_boost() -> None:
    scope = Scope(kind="personal")
    now = datetime(2026, 4, 10, tzinfo=UTC)

    class _FixedParser(QueryTimeParser):
        def __init__(self) -> None:
            pass

        def parse(self, *, query, now):  # type: ignore[override]
            target = datetime(2026, 4, 9, tzinfo=UTC)
            return QueryTimeHint(center=target, start=target, end=target)

    config = load_default_config()
    config.retrieval.query_time_parse_enabled = True
    config.retrieval.lambda_time = 1.0

    store = MemoryBase()
    recent = make_memory(
        content="Ate sushi at Nobu.",
        topic_id=None,
        scope=scope,
        memory_type="episodic",
        event_time=datetime(2026, 4, 9, tzinfo=UTC),
        ingest_time=datetime(2026, 4, 9, tzinfo=UTC),
    )
    distant = make_memory(
        content="Ate sushi at Nobu back in January.",
        topic_id=None,
        scope=scope,
        memory_type="episodic",
        event_time=datetime(2026, 1, 15, tzinfo=UTC),
        ingest_time=datetime(2026, 1, 15, tzinfo=UTC),
    )
    store.add(recent)
    store.add(distant)

    agent = HybridRetrieverAgent(
        memory_base=store,
        config=config,
        query_time_parser=_FixedParser(),
    )
    result = agent.run(
        RetrievalInput(
            query="Where did I eat yesterday?",
            scope=scope,
            timestamp=now,
        )
    )

    assert result.ranked_memories[0].memory.id == recent.id
    assert result.ranked_memories[0].score_breakdown["time_prox"] > result.ranked_memories[-1].score_breakdown["time_prox"]


def test_hybrid_retriever_applies_quoted_phrase_boost() -> None:
    scope = Scope(kind="personal")
    config = load_default_config()
    config.retrieval.lambda_quoted = 1.0

    store = MemoryBase()
    matching = make_memory(
        content="The assistant suggested options like 'sexual compulsions' and other framings.",
        topic_id=None,
        scope=scope,
    )
    distractor = make_memory(
        content="Generic background notes with no exact phrase match.",
        topic_id=None,
        scope=scope,
    )
    store.add(matching)
    store.add(distractor)

    agent = HybridRetrieverAgent(memory_base=store, config=config)
    result = agent.run(
        RetrievalInput(
            query="Remind me what you suggested — was it 'sexual compulsions' or something else?",
            scope=scope,
            timestamp=datetime(2026, 4, 2, tzinfo=UTC),
        )
    )

    assert result.ranked_memories[0].memory.id == matching.id
    top = result.ranked_memories[0]
    assert top.score_breakdown["quoted"] > 0.0
    assert "quoted_phrase" in top.matched_by


def test_hybrid_retriever_applies_person_name_boost() -> None:
    scope = Scope(kind="personal")
    config = load_default_config()
    config.retrieval.lambda_person = 1.0

    store = MemoryBase()
    with_name = make_memory(
        content="Started taking ukulele lessons with my friend Rachel today.",
        topic_id=None,
        scope=scope,
    )
    without_name = make_memory(
        content="Took a generic music class with a friend.",
        topic_id=None,
        scope=scope,
    )
    store.add(with_name)
    store.add(without_name)

    agent = HybridRetrieverAgent(memory_base=store, config=config)
    result = agent.run(
        RetrievalInput(
            query="What did I do with Rachel?",
            scope=scope,
            timestamp=datetime(2026, 4, 2, tzinfo=UTC),
        )
    )

    assert result.ranked_memories[0].memory.id == with_name.id
    top = result.ranked_memories[0]
    assert top.score_breakdown["person"] > 0.0
    assert "person_name" in top.matched_by


def test_hybrid_retriever_uses_rule_based_time_parser_for_yesterday() -> None:
    from driftscope.retrieval.rule_time_parser import RuleBasedQueryTimeParser

    scope = Scope(kind="personal")
    now = datetime(2026, 4, 10, 12, 0, tzinfo=UTC)

    config = load_default_config()
    config.retrieval.query_time_parse_enabled = True
    config.retrieval.lambda_time = 1.0

    store = MemoryBase()
    recent = make_memory(
        content="Visited the lakeside cafe.",
        topic_id=None,
        scope=scope,
        memory_type="episodic",
        event_time=datetime(2026, 4, 9, 18, 0, tzinfo=UTC),
        ingest_time=datetime(2026, 4, 9, 18, 0, tzinfo=UTC),
    )
    distant = make_memory(
        content="Visited the lakeside cafe months back.",
        topic_id=None,
        scope=scope,
        memory_type="episodic",
        event_time=datetime(2026, 1, 5, tzinfo=UTC),
        ingest_time=datetime(2026, 1, 5, tzinfo=UTC),
    )
    store.add(recent)
    store.add(distant)

    agent = HybridRetrieverAgent(
        memory_base=store,
        config=config,
        query_time_parser=RuleBasedQueryTimeParser(),
    )
    result = agent.run(
        RetrievalInput(
            query="Where did I go yesterday?",
            scope=scope,
            timestamp=now,
        )
    )

    assert result.ranked_memories[0].memory.id == recent.id
    assert result.ranked_memories[0].score_breakdown["time_prox"] > 0.0
