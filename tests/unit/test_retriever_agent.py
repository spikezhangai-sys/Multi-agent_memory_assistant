from datetime import UTC, datetime

from driftscope.agents.retriever_agent import HeuristicRetrieverAgent
from driftscope.agents.types import RetrievalInput
from driftscope.core.memory_base import MemoryBase
from driftscope.core.schema import Scope
from tests.unit.helpers import make_memory


def test_retriever_returns_top_ranked_fact_for_topic_query() -> None:
    store = MemoryBase()
    scope = Scope(kind="personal")
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

    agent = HeuristicRetrieverAgent(memory_base=store)
    result = agent.run(
        RetrievalInput(
            query="我住在哪？",
            scope=scope,
            timestamp=datetime(2026, 4, 2, tzinfo=UTC),
        )
    )

    assert result.predicted_topic == "user.profile.location"
    assert result.ranked_memories[0].memory.id == location.id


def test_retriever_injects_constraints_without_top_k_truncation() -> None:
    store = MemoryBase()
    scope = Scope(kind="personal")
    preference = make_memory(
        content="我想吃点清淡的晚餐",
        topic_id="user.preference.food",
        scope=scope,
        memory_type="preference",
    )
    constraint = make_memory(
        content="我对花生过敏",
        topic_id="user.constraint.diet",
        scope=scope,
        memory_type="constraint",
    )
    store.add(preference)
    store.add(constraint)

    agent = HeuristicRetrieverAgent(memory_base=store)
    result = agent.run(
        RetrievalInput(
            query="给我推荐晚餐，注意我不能吃什么？",
            scope=scope,
            timestamp=datetime(2026, 4, 2, tzinfo=UTC),
        )
    )

    assert any(item.id == constraint.id for item in result.injected_constraints)


def test_retriever_downweights_assistant_summary_memories() -> None:
    store = MemoryBase()
    scope = Scope(kind="personal")
    user_fact = make_memory(
        content="我现在住在上海",
        topic_id="user.profile.location",
        scope=scope,
    )
    assistant_summary = make_memory(
        content="你现在住在上海",
        topic_id="user.profile.location",
        scope=scope,
        origin_role="assistant",
        source_kind="summary",
    )
    store.add(user_fact)
    store.add(assistant_summary)

    agent = HeuristicRetrieverAgent(memory_base=store)
    result = agent.run(
        RetrievalInput(
            query="我住在哪？",
            scope=scope,
            timestamp=datetime(2026, 4, 2, tzinfo=UTC),
        )
    )

    assert result.ranked_memories[0].memory.id == user_fact.id
    ranked_ids = [item.memory.id for item in result.ranked_memories]
    assert assistant_summary.id in ranked_ids


def test_retriever_can_return_memory_without_topic_id() -> None:
    store = MemoryBase()
    scope = Scope(kind="personal")
    degree = make_memory(
        content="I graduated with a degree in Business Administration.",
        topic_id=None,
        scope=scope,
    )
    distractor = make_memory(
        content="我最近喜欢吃日料",
        topic_id="user.preference.food",
        scope=scope,
        memory_type="preference",
    )
    store.add(degree)
    store.add(distractor)

    agent = HeuristicRetrieverAgent(memory_base=store)
    result = agent.run(
        RetrievalInput(
            query="What degree did I graduate with?",
            scope=scope,
            timestamp=datetime(2026, 4, 2, tzinfo=UTC),
        )
    )

    assert result.ranked_memories[0].memory.id == degree.id


def test_retriever_uses_raw_sensitive_content_when_allowed() -> None:
    store = MemoryBase()
    scope = Scope(kind="personal")
    memory = make_memory(
        content="My passport number is ZX123456 and it expires in 2030.",
        topic_id=None,
        scope=scope,
        sensitive=True,
        summary_for_retrieval="Sensitive identity document details.",
    )
    store.add(memory)

    agent = HeuristicRetrieverAgent(memory_base=store)
    hidden_result = agent.run(
        RetrievalInput(
            query="What is my passport number?",
            scope=scope,
            timestamp=datetime(2026, 4, 2, tzinfo=UTC),
            allow_sensitive_raw=False,
        )
    )
    visible_result = agent.run(
        RetrievalInput(
            query="What is my passport number?",
            scope=scope,
            timestamp=datetime(2026, 4, 2, tzinfo=UTC),
            allow_sensitive_raw=True,
        )
    )

    assert hidden_result.ranked_memories[0].matched_by == ["fallback_visible"]
    assert visible_result.ranked_memories[0].memory.id == memory.id
    assert "lexical_overlap" in visible_result.ranked_memories[0].matched_by
    assert visible_result.ranked_memories[0].score > hidden_result.ranked_memories[0].score
