from driftscope.agents.response_agent import HeuristicResponseAgent
from driftscope.agents.types import CandidateMatch, ResponseInput, RetrievalResult
from driftscope.core.schema import Scope
from tests.unit.helpers import make_memory


def test_response_agent_abstains_without_evidence() -> None:
    agent = HeuristicResponseAgent()
    result = agent.run(
        ResponseInput(
            query="我住在哪？",
            retrieval=RetrievalResult(),
        )
    )

    assert result.abstained is True
    assert result.cited_memory_ids == []


def test_response_agent_combines_primary_memory_and_constraints() -> None:
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
    agent = HeuristicResponseAgent()
    result = agent.run(
        ResponseInput(
            query="推荐晚餐",
            retrieval=RetrievalResult(
                ranked_memories=[
                    CandidateMatch(
                        memory=preference,
                        score=0.9,
                        score_breakdown={},
                        matched_by=[],
                    )
                ],
                injected_constraints=[constraint],
            ),
        )
    )

    assert "清淡" in result.answer
    assert "花生过敏" in result.answer
    assert preference.id in result.cited_memory_ids
    assert constraint.id in result.cited_memory_ids
