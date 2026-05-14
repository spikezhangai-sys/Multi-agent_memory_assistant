from datetime import UTC, datetime

from driftscope.agents.types import UpdateInput
from driftscope.agents.update_agent import HeuristicUpdateAgent
from driftscope.core.schema import Scope
from tests.unit.helpers import make_memory


def test_update_agent_builds_add_proposal_for_new_fact() -> None:
    agent = HeuristicUpdateAgent()
    result = agent.run(
        UpdateInput(
            user_input="我现在住在上海",
            scope=Scope(kind="personal"),
            timestamp=datetime(2026, 4, 1, tzinfo=UTC),
            nearby_memories=[],
        )
    )

    assert result.intent == "add"
    assert result.candidate is not None
    assert result.candidate.topic_id == "user.profile.location"


def test_update_agent_keeps_simple_topic_match_as_add_even_with_nearby_memory() -> None:
    agent = HeuristicUpdateAgent()
    existing = make_memory(
        content="我现在住在上海",
        topic_id="user.profile.location",
        scope=Scope(kind="personal"),
    )
    result = agent.run(
        UpdateInput(
            user_input="我现在搬到北京了",
            scope=Scope(kind="personal"),
            timestamp=datetime(2026, 4, 2, tzinfo=UTC),
            nearby_memories=[existing],
        )
    )

    assert result.intent == "add"
    assert result.candidate is not None
    assert result.candidate.topic_id == "user.profile.location"


def test_heuristic_update_agent_run_batch_returns_indexed_proposals() -> None:
    agent = HeuristicUpdateAgent()
    results = agent.run_batch(
        [
            UpdateInput(
                user_input="我现在住在上海",
                scope=Scope(kind="personal"),
                timestamp=datetime(2026, 4, 1, tzinfo=UTC),
                nearby_memories=[],
            ),
            UpdateInput(
                user_input="asdfqwerzxcv",
                scope=Scope(kind="personal"),
                timestamp=datetime(2026, 4, 1, tzinfo=UTC),
                nearby_memories=[],
            ),
        ]
    )

    assert len(results) == 1
    assert results[0].source_turn_index == 0
    assert results[0].proposal.intent == "add"


def test_heuristic_update_agent_ignores_text_without_topic_match() -> None:
    agent = HeuristicUpdateAgent()
    result = agent.run(
        UpdateInput(
            user_input="I once read a very interesting book on a rainy afternoon.",
            scope=Scope(kind="personal"),
            timestamp=datetime(2026, 4, 1, tzinfo=UTC),
            nearby_memories=[],
        )
    )

    assert result.intent == "ignore"
    assert result.candidate is None
