from pydantic import BaseModel

from driftscope.agents.response_agent import LLMResponseAgent
from driftscope.agents.types import CandidateMatch, ResponseInput, RetrievalResult
from driftscope.core.schema import Scope
from tests.unit.helpers import make_memory


class FakeLLM:
    def __init__(self, payload):
        self.payload = payload

    def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[BaseModel],
    ):
        return self.payload


class RaisingLLM:
    def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[BaseModel],
    ):
        raise ValueError("invalid json")


def test_llm_response_agent_filters_invalid_citations() -> None:
    memory = make_memory(
        content="我现在住在上海",
        topic_id="user.profile.location",
        scope=Scope(kind="personal"),
    )
    agent = LLMResponseAgent(
        FakeLLM(
            {
                "answer": "你住在上海",
                "cited_memory_ids": [memory.id, "fake-id"],
                "context_only_ids": ["fake-id"],
                "abstained": False,
            }
        )
    )
    result = agent.run(
        ResponseInput(
            query="我住在哪？",
            retrieval=RetrievalResult(
                ranked_memories=[
                    CandidateMatch(
                        memory=memory,
                        score=0.9,
                        score_breakdown={},
                        matched_by=[],
                    )
                ]
            ),
        )
    )

    assert result.cited_memory_ids == [memory.id]
    assert result.context_only_ids == []


def test_llm_response_agent_falls_back_to_heuristic_when_llm_output_is_invalid() -> None:
    memory = make_memory(
        content="我现在住在上海",
        topic_id="user.profile.location",
        scope=Scope(kind="personal"),
    )
    agent = LLMResponseAgent(RaisingLLM())
    result = agent.run(
        ResponseInput(
            query="我住在哪？",
            retrieval=RetrievalResult(
                ranked_memories=[
                    CandidateMatch(
                        memory=memory,
                        score=0.9,
                        score_breakdown={},
                        matched_by=[],
                    )
                ]
            ),
        )
    )

    assert result.abstained is False
    assert result.cited_memory_ids == [memory.id]
    assert result.context_only_ids == []
    assert result.abstain_reason is not None
    assert result.abstain_reason.startswith("llm_parse_failure_fallback_heuristic")


def test_llm_response_agent_abstains_when_llm_fails_with_no_evidence() -> None:
    agent = LLMResponseAgent(RaisingLLM())
    result = agent.run(
        ResponseInput(
            query="我住在哪？",
            retrieval=RetrievalResult(ranked_memories=[]),
        )
    )

    assert result.abstained is True
    assert result.cited_memory_ids == []
