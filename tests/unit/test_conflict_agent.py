from datetime import UTC, datetime

from pydantic import BaseModel

from driftscope.agents.conflict_agent import ConflictAgent
from driftscope.agents.types import CandidateMatch, ConflictInput, ConflictResolution, UpdateProposal
from driftscope.core.schema import Scope, TopicQuery
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


def make_conflict_input() -> ConflictInput:
    scope = Scope(kind="personal")
    target = make_memory(
        content="我现在住在上海",
        topic_id="user.profile.location",
        scope=scope,
    )
    proposal = UpdateProposal(
        intent="supersede_full",
        candidate=make_memory(
            content="我搬到北京了",
            topic_id="user.profile.location",
            scope=scope,
        ),
        target_hint=TopicQuery(topic_id="user.profile.location", keywords=["搬到", "北京"]),
        transition_type="corrected",
    )
    return ConflictInput(
        proposal=proposal,
        scope=scope,
        timestamp=datetime(2026, 4, 2, tzinfo=UTC),
        candidates=[CandidateMatch(memory=target, score=0.9)],
    )


def test_conflict_agent_accepts_valid_llm_output() -> None:
    input_obj = make_conflict_input()
    llm = FakeLLM(
        {
            "action": "confirm_supersede",
            "target_id": input_obj.candidates[0].memory.id,
            "transition_type": "corrected",
            "confidence": 0.9,
            "reason": "same memory slot",
        }
    )
    result = ConflictAgent(llm).run(input_obj)

    assert result.used_fallback is False
    assert result.resolution.action == "confirm_supersede"


def test_conflict_agent_falls_back_when_llm_selects_unknown_target() -> None:
    input_obj = make_conflict_input()
    llm = FakeLLM(
        {
            "action": "confirm_supersede",
            "target_id": "made_up",
            "transition_type": "corrected",
            "confidence": 0.9,
            "reason": "hallucinated id",
        }
    )
    result = ConflictAgent(llm).run(input_obj)

    assert result.used_fallback is True
    assert result.resolution.action == "request_clarification"
    assert result.validation_errors

