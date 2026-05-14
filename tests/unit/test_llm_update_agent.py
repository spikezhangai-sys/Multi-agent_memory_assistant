from datetime import UTC, datetime

from pydantic import BaseModel

from driftscope.agents.types import UpdateInput
from driftscope.agents.update_agent import LLMUpdateAgent
from driftscope.config.loader import load_default_config
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


class SequenceLLM:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls = 0

    def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[BaseModel],
    ):
        if not self.payloads:
            raise AssertionError("unexpected extra LLM call")
        self.calls += 1
        return self.payloads.pop(0)


class RaisingLLM:
    def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[BaseModel],
    ):
        raise ValueError("invalid json")


def test_llm_update_agent_builds_supersede_proposal() -> None:
    llm = FakeLLM(
        {
            "intent": "supersede_full",
            "candidate_content": "我搬到北京了",
            "candidate_type": "fact",
            "topic_id": "user.profile.location",
            "keywords": ["搬到", "北京", "住"],
            "transition_type": "corrected",
        }
    )
    agent = LLMUpdateAgent(llm)
    existing = make_memory(
        content="我现在住在上海",
        topic_id="user.profile.location",
        scope=Scope(kind="personal"),
    )
    result = agent.run(
        UpdateInput(
            user_input="我搬到北京了",
            scope=Scope(kind="personal"),
            timestamp=datetime(2026, 4, 2, tzinfo=UTC),
            nearby_memories=[existing],
        )
    )

    assert result.intent == "supersede_full"
    assert result.candidate is not None
    assert result.candidate.content == "我搬到北京了"
    assert result.target_hint is not None


def test_llm_update_agent_builds_multiple_batch_proposals() -> None:
    llm = FakeLLM(
        {
            "proposals": [
                {
                    "source_turn_index": 0,
                    "intent": "add",
                    "candidate_content": "我住在上海",
                    "candidate_type": "fact",
                    "topic_id": "user.profile.location",
                    "keywords": ["住", "上海"],
                },
                {
                    "source_turn_index": 1,
                    "intent": "add",
                    "candidate_content": "我喜欢吃日料",
                    "candidate_type": "preference",
                    "topic_id": "user.preference.food",
                    "keywords": ["喜欢", "日料"],
                },
            ]
        }
    )
    agent = LLMUpdateAgent(llm)
    results = agent.run_batch(
        [
            UpdateInput(
                user_input="我住在上海",
                scope=Scope(kind="personal"),
                timestamp=datetime(2026, 4, 2, tzinfo=UTC),
                nearby_memories=[],
            ),
            UpdateInput(
                user_input="我喜欢吃日料",
                scope=Scope(kind="personal"),
                timestamp=datetime(2026, 4, 2, tzinfo=UTC),
                nearby_memories=[],
            ),
        ]
    )

    assert [item.source_turn_index for item in results] == [0, 1]
    assert results[0].proposal.candidate is not None
    assert results[0].proposal.candidate.content == "我住在上海"
    assert results[1].proposal.candidate is not None
    assert results[1].proposal.candidate.content == "我喜欢吃日料"


def test_llm_update_agent_omits_nearby_memories_when_disabled() -> None:
    existing = make_memory(
        content="User lives in Shanghai",
        topic_id="user.profile.location",
        scope=Scope(kind="personal"),
    )
    agent = LLMUpdateAgent(FakeLLM({"proposals": []}))

    prompt = agent._build_batch_prompt(
        [
            UpdateInput(
                user_input="User moved to Beijing",
                scope=Scope(kind="personal"),
                timestamp=datetime(2026, 4, 2, tzinfo=UTC),
                nearby_memories=[existing],
            )
        ]
    )

    assert "nearby_memories" not in prompt


def test_llm_update_agent_keeps_nearby_memories_when_enabled() -> None:
    config = load_default_config()
    config.update.nearby_k = 1
    existing = make_memory(
        content="User lives in Shanghai",
        topic_id="user.profile.location",
        scope=Scope(kind="personal"),
    )
    agent = LLMUpdateAgent(FakeLLM({"proposals": []}), config=config)

    prompt = agent._build_batch_prompt(
        [
            UpdateInput(
                user_input="User moved to Beijing",
                scope=Scope(kind="personal"),
                timestamp=datetime(2026, 4, 2, tzinfo=UTC),
                nearby_memories=[existing],
            )
        ]
    )

    assert "nearby_memories" in prompt
    assert "User lives in Shanghai" in prompt


def test_llm_update_agent_accepts_list_candidate_content_in_strict_batch_output() -> None:
    llm = FakeLLM(
        {
            "proposals": [
                {
                    "source_turn_index": 0,
                    "intent": "add",
                    "candidate_content": ["I've been listening to audiobooks during my daily commute, which takes 45 minutes each way."],
                    "candidate_type": "fact",
                    "topic_id": None,
                    "keywords": ["commute", "45 minutes"],
                }
            ]
        }
    )
    agent = LLMUpdateAgent(llm)
    results = agent.run_batch(
        [
            UpdateInput(
                user_input="I've been listening to audiobooks during my daily commute, which takes 45 minutes each way.",
                scope=Scope(kind="personal"),
                timestamp=datetime(2026, 4, 2, tzinfo=UTC),
                nearby_memories=[],
            )
            ,
            UpdateInput(
                user_input="Do you have any audiobook recommendations?",
                scope=Scope(kind="personal"),
                timestamp=datetime(2026, 4, 2, tzinfo=UTC),
                nearby_memories=[],
                origin_role="assistant",
            )
        ]
    )

    assert len(results) == 1
    assert results[0].proposal.candidate is not None
    assert results[0].proposal.candidate.content == "I've been listening to audiobooks during my daily commute, which takes 45 minutes each way."


def test_llm_update_agent_ignores_unknown_batch_item_without_candidate_payload() -> None:
    llm = FakeLLM(
        {
            "items": [
                {
                    "turn_index": 0,
                    "action": "timeline_event",
                }
            ]
        }
    )
    agent = LLMUpdateAgent(llm)
    results = agent.run_batch(
        [
            UpdateInput(
                user_input="今天心情不错",
                scope=Scope(kind="personal"),
                timestamp=datetime(2026, 4, 2, tzinfo=UTC),
                nearby_memories=[],
            ),
            UpdateInput(
                user_input="随便聊聊",
                scope=Scope(kind="personal"),
                timestamp=datetime(2026, 4, 3, tzinfo=UTC),
                nearby_memories=[],
            ),
        ]
    )

    assert results == []


def test_llm_update_agent_skips_batch_when_llm_output_is_invalid() -> None:
    agent = LLMUpdateAgent(RaisingLLM())
    results = agent.run_batch(
        [
            UpdateInput(
                user_input="Just chatting about nothing stable.",
                scope=Scope(kind="personal"),
                timestamp=datetime(2026, 4, 2, tzinfo=UTC),
                nearby_memories=[],
            ),
            UpdateInput(
                user_input="Still nothing to remember here.",
                scope=Scope(kind="personal"),
                timestamp=datetime(2026, 4, 2, tzinfo=UTC),
                nearby_memories=[],
            ),
        ]
    )

    assert results == []


def test_llm_update_agent_respects_null_topic_when_llm_omits_it() -> None:
    llm = FakeLLM(
        {
            "intent": "add",
            "candidate_content": "I graduated with a degree in Business Administration.",
            "candidate_type": "fact",
            "topic_id": None,
            "keywords": ["graduated", "degree", "business"],
        }
    )
    agent = LLMUpdateAgent(llm)
    result = agent.run(
        UpdateInput(
            user_input="I graduated with a degree in Business Administration.",
            scope=Scope(kind="personal"),
            timestamp=datetime(2026, 4, 2, tzinfo=UTC),
            nearby_memories=[],
        )
    )

    assert result.intent == "add"
    assert result.candidate is not None
    assert result.candidate.topic_id is None
    assert result.candidate.content == "I graduated with a degree in Business Administration."


def test_llm_update_agent_coerces_legacy_ignore_intent_to_add() -> None:
    """Legacy behavior: when an LLM somehow emits intent='ignore' (e.g. via
    a non-strict provider or relaxed-payload fallback), the system should
    coerce to 'add' rather than silently drop the populated candidate. The
    closed-enum schema in strict mode prevents fresh emissions of 'ignore',
    but this defensive coercion preserves any candidate that slips through.
    """
    llm = FakeLLM({"intent": "ignore"})
    agent = LLMUpdateAgent(llm)
    result = agent.run(
        UpdateInput(
            user_input="I graduated with a degree in Business Administration.",
            scope=Scope(kind="personal"),
            timestamp=datetime(2026, 4, 2, tzinfo=UTC),
            nearby_memories=[],
        )
    )

    assert result.intent == "add"
    assert result.candidate is not None
    assert "Business Administration" in result.candidate.content


def test_llm_update_agent_retries_user_turn_after_batch_miss() -> None:
    llm = SequenceLLM(
        [
            {"proposals": []},
            {
                "proposals": [
                    {
                        "source_turn_index": 0,
                        "intent": "add",
                        "candidate_content": "User's daily commute takes 45 minutes each way.",
                        "candidate_type": "fact",
                        "topic_id": None,
                        "keywords": ["commute", "45 minutes", "each way"],
                    }
                ]
            },
        ]
    )
    config = load_default_config()
    config.update.batch_supplemental_enabled = True
    agent = LLMUpdateAgent(llm, config=config)

    results = agent.run_batch(
        [
            UpdateInput(
                user_input="I've been listening to audiobooks during my daily commute, which takes 45 minutes each way.",
                scope=Scope(kind="personal"),
                timestamp=datetime(2026, 4, 2, tzinfo=UTC),
                nearby_memories=[],
                origin_role="user",
            ),
            UpdateInput(
                user_input="Do you have any fiction recommendations?",
                scope=Scope(kind="personal"),
                timestamp=datetime(2026, 4, 2, tzinfo=UTC),
                nearby_memories=[],
                origin_role="assistant",
            ),
        ]
    )

    assert llm.calls == 2
    assert len(results) == 1
    assert results[0].source_turn_index == 0
    assert results[0].proposal.candidate is not None
    assert "45 minutes each way" in results[0].proposal.candidate.content


def test_llm_update_agent_retries_plain_user_turn_without_first_person_marker() -> None:
    llm = SequenceLLM(
        [
            {"proposals": []},
            {
                "proposals": [
                    {
                        "source_turn_index": 0,
                        "intent": "add",
                        "candidate_content": "Daily commute takes 45 minutes each way.",
                        "candidate_type": "fact",
                        "topic_id": None,
                        "keywords": ["commute", "45 minutes", "each way"],
                    }
                ]
            },
        ]
    )
    config = load_default_config()
    config.update.batch_supplemental_enabled = True
    agent = LLMUpdateAgent(llm, config=config)

    results = agent.run_batch(
        [
            UpdateInput(
                user_input="Daily commute takes 45 minutes each way.",
                scope=Scope(kind="personal"),
                timestamp=datetime(2026, 4, 2, tzinfo=UTC),
                nearby_memories=[],
                origin_role="user",
            ),
            UpdateInput(
                user_input="Anything else?",
                scope=Scope(kind="personal"),
                timestamp=datetime(2026, 4, 2, tzinfo=UTC),
                nearby_memories=[],
                origin_role="assistant",
            ),
        ]
    )

    assert llm.calls == 2
    assert len(results) == 1
    assert results[0].proposal.candidate is not None
    assert "45 minutes each way" in results[0].proposal.candidate.content


def test_llm_update_agent_skips_supplemental_by_default_when_batch_returns_empty() -> None:
    """Default config has batch_supplemental_enabled=False; the batch prompt is
    treated as authoritative and no per-turn re-asks happen."""
    llm = SequenceLLM(
        [
            {"proposals": []},
            # second LLM payload would only be consumed if supplemental fired
            {
                "proposals": [
                    {
                        "source_turn_index": 0,
                        "intent": "add",
                        "candidate_content": "Should never be reached.",
                        "candidate_type": "fact",
                        "topic_id": None,
                        "keywords": ["unused"],
                    }
                ]
            },
        ]
    )
    agent = LLMUpdateAgent(llm)  # default config: supplemental off

    results = agent.run_batch(
        [
            UpdateInput(
                user_input="Daily commute takes 45 minutes each way.",
                scope=Scope(kind="personal"),
                timestamp=datetime(2026, 4, 2, tzinfo=UTC),
                nearby_memories=[],
                origin_role="user",
            ),
            UpdateInput(
                user_input="Anything else?",
                scope=Scope(kind="personal"),
                timestamp=datetime(2026, 4, 2, tzinfo=UTC),
                nearby_memories=[],
                origin_role="assistant",
            ),
        ]
    )

    assert llm.calls == 1, "supplemental should not fire when default flag is False"
    assert results == []


def test_llm_update_agent_does_not_add_extra_fallback_when_batch_already_returned_one_proposal() -> None:
    llm = SequenceLLM(
        [
            {
                "proposals": [
                    {
                        "source_turn_index": 0,
                        "intent": "add",
                        "candidate_content": "User enjoys audiobooks during the commute.",
                        "candidate_type": "fact",
                        "topic_id": None,
                        "keywords": ["audiobooks", "commute"],
                    }
                ]
            }
        ]
    )
    agent = LLMUpdateAgent(llm)

    results = agent.run_batch(
        [
            UpdateInput(
                user_input="I've been listening to audiobooks during my daily commute, which takes 45 minutes each way.",
                scope=Scope(kind="personal"),
                timestamp=datetime(2026, 4, 2, tzinfo=UTC),
                nearby_memories=[],
                origin_role="user",
            ),
            UpdateInput(
                user_input="Anything else I should try?",
                scope=Scope(kind="personal"),
                timestamp=datetime(2026, 4, 2, tzinfo=UTC),
                nearby_memories=[],
                origin_role="assistant",
            ),
        ]
    )

    assert llm.calls == 1
    contents = [item.proposal.candidate.content for item in results if item.proposal.candidate is not None]
    assert "User enjoys audiobooks during the commute." in contents
    assert not any("45 minutes each way" in content for content in contents)


def test_llm_update_agent_run_many_extracts_multiple_proposals_from_single_turn() -> None:
    llm = FakeLLM(
        {
            "proposals": [
                {
                    "source_turn_index": 0,
                    "intent": "add",
                    "candidate_content": "For aunt Lorena's retirement, got a teal blazer and a matching brooch.",
                    "candidate_type": "episodic",
                    "topic_id": None,
                    "keywords": ["aunt", "teal blazer", "brooch"],
                    "evidence": "a teal blazer and a matching brooch",
                    "importance": 0.6,
                    "sensitivity": "low",
                    "event_time": "2026-04-01T00:00:00+00:00",
                },
                {
                    "source_turn_index": 0,
                    "intent": "add",
                    "candidate_content": "User attends aunt Lorena's retirement celebrations.",
                    "candidate_type": "fact",
                    "topic_id": None,
                    "keywords": ["aunt", "retirement"],
                    "evidence": "aunt Lorena's retirement",
                    "importance": 0.4,
                    "sensitivity": "low",
                },
            ]
        }
    )
    agent = LLMUpdateAgent(llm)
    proposals = agent.run_many(
        UpdateInput(
            user_input="For my aunt Lorena's retirement, I got her a teal blazer and a matching brooch to go with it.",
            scope=Scope(kind="personal"),
            timestamp=datetime(2026, 4, 1, tzinfo=UTC),
            nearby_memories=[],
            origin_role="user",
        )
    )

    assert len(proposals) == 2
    contents = [p.candidate.content for p in proposals if p.candidate is not None]
    assert any("teal blazer" in c for c in contents)
    assert any("aunt" in c for c in contents)


def test_llm_update_agent_run_many_falls_back_to_single_prompt_when_batch_empty() -> None:
    llm = SequenceLLM(
        [
            {"proposals": []},
            {
                "intent": "add",
                "candidate_content": "User moved to the Caldermere district last month.",
                "candidate_type": "fact",
                "topic_id": None,
                "keywords": ["Caldermere", "moved"],
                "evidence": "moved into my own studio in the Caldermere district",
                "importance": 0.7,
                "sensitivity": "low",
            },
        ]
    )
    config = load_default_config()
    config.update.batch_supplemental_enabled = True
    agent = LLMUpdateAgent(llm, config=config)
    proposals = agent.run_many(
        UpdateInput(
            user_input="Can you recommend a houseplant? I moved into my own studio in the Caldermere district last month.",
            scope=Scope(kind="personal"),
            timestamp=datetime(2026, 4, 1, tzinfo=UTC),
            nearby_memories=[],
            origin_role="user",
        )
    )

    assert llm.calls == 2
    assert len(proposals) == 1
    assert proposals[0].candidate is not None
    assert "Caldermere" in proposals[0].candidate.content
