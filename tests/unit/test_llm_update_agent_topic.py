from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel

from driftscope.agents.types import UpdateInput
from driftscope.agents.update_agent import LLMUpdateAgent
from driftscope.core.memory_base import MemoryBase
from driftscope.core.schema import Scope


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


def _user_payload(**overrides):
    base = {
        "intent": "add",
        "candidate_content": "User enjoys reading classic novels",
        "candidate_type": "preference",
        "evidence": "reading classic novels",
        "importance": 0.6,
        "sensitivity": "low",
    }
    base.update(overrides)
    return base


def test_llm_agent_canonicalizes_category_plus_leaf_suffix_via_memory_base() -> None:
    mb = MemoryBase(":memory:")  # no embedder — novel suffix registers verbatim
    agent = LLMUpdateAgent(FakeLLM(_user_payload(category="user.preference", leaf_suffix="books")),
                           canonicalizer=mb.canonicalize_topic)
    result = agent.run(
        UpdateInput(
            user_input="reading classic novels",
            scope=Scope(kind="personal"),
            timestamp=datetime(2026, 4, 2, tzinfo=UTC),
        )
    )
    assert result.intent == "add"
    assert result.candidate.topic_id == "user.preference.books"
    assert mb.is_known_topic("user.preference.books")


def test_llm_agent_composes_seed_path_when_leaf_suffix_matches_seed() -> None:
    mb = MemoryBase(":memory:")
    agent = LLMUpdateAgent(
        FakeLLM(_user_payload(category="user.preference", leaf_suffix="food",
                              candidate_content="I love spicy Sichuan food",
                              evidence="spicy Sichuan food")),
        canonicalizer=mb.canonicalize_topic,
    )
    result = agent.run(
        UpdateInput(
            user_input="spicy Sichuan food",
            scope=Scope(kind="personal"),
            timestamp=datetime(2026, 4, 2, tzinfo=UTC),
        )
    )
    assert result.candidate.topic_id == "user.preference.food"


def test_llm_agent_rejects_unknown_category() -> None:
    mb = MemoryBase(":memory:")
    agent = LLMUpdateAgent(
        FakeLLM(_user_payload(category="user.unknown", leaf_suffix="anything")),
        canonicalizer=mb.canonicalize_topic,
    )
    result = agent.run(
        UpdateInput(
            user_input="reading classic novels",
            scope=Scope(kind="personal"),
            timestamp=datetime(2026, 4, 2, tzinfo=UTC),
        )
    )
    assert result.intent == "add"
    assert result.candidate.topic_id is None


def test_llm_agent_without_canonicalizer_only_accepts_seed_matches() -> None:
    agent = LLMUpdateAgent(
        FakeLLM(_user_payload(category="user.preference", leaf_suffix="books")),
        canonicalizer=None,
    )
    result = agent.run(
        UpdateInput(
            user_input="reading classic novels",
            scope=Scope(kind="personal"),
            timestamp=datetime(2026, 4, 2, tzinfo=UTC),
        )
    )
    assert result.candidate.topic_id is None  # novel suffix rejected without canonicalizer


def test_llm_agent_accepts_legacy_topic_id_path() -> None:
    """Back-compat: if LLM emits legacy topic_id of a seed, still honored."""
    agent = LLMUpdateAgent(FakeLLM(_user_payload(topic_id="user.preference.food",
                                                  candidate_content="I love salsify",
                                                  evidence="I love salsify")))
    result = agent.run(
        UpdateInput(
            user_input="I love salsify",
            scope=Scope(kind="personal"),
            timestamp=datetime(2026, 4, 2, tzinfo=UTC),
        )
    )
    assert result.candidate.topic_id == "user.preference.food"


def test_llm_agent_ignores_category_without_leaf_suffix() -> None:
    mb = MemoryBase(":memory:")
    agent = LLMUpdateAgent(
        FakeLLM(_user_payload(category="user.preference", leaf_suffix=None)),
        canonicalizer=mb.canonicalize_topic,
    )
    result = agent.run(
        UpdateInput(
            user_input="reading classic novels",
            scope=Scope(kind="personal"),
            timestamp=datetime(2026, 4, 2, tzinfo=UTC),
        )
    )
    assert result.candidate.topic_id is None


def test_llm_agent_salvages_full_path_in_category_with_duplicate_leaf() -> None:
    """LLM mistake: category='user.activity.cultural_visit' + leaf_suffix='cultural_visit'."""
    mb = MemoryBase(":memory:")
    agent = LLMUpdateAgent(
        FakeLLM(
            _user_payload(
                category="user.activity.cultural_visit",
                leaf_suffix="cultural_visit",
                candidate_type="episodic",
                candidate_content="User visited the design museum exhibit.",
                evidence="design museum exhibit",
                event_time=datetime(2026, 4, 1, tzinfo=UTC),
            )
        ),
        canonicalizer=mb.canonicalize_topic,
    )
    result = agent.run(
        UpdateInput(
            user_input="design museum exhibit",
            scope=Scope(kind="personal"),
            timestamp=datetime(2026, 4, 2, tzinfo=UTC),
        )
    )
    assert result.candidate.topic_id == "user.activity.cultural_visit"


def test_llm_agent_salvages_full_path_in_category_with_null_leaf() -> None:
    """LLM mistake: category='user.possessions.home' + leaf_suffix=None."""
    mb = MemoryBase(":memory:")
    agent = LLMUpdateAgent(
        FakeLLM(
            _user_payload(
                category="user.possessions.home",
                leaf_suffix=None,
                candidate_content="User redecorated the living room.",
                evidence="redecorated the living room",
            )
        ),
        canonicalizer=mb.canonicalize_topic,
    )
    result = agent.run(
        UpdateInput(
            user_input="redecorated the living room",
            scope=Scope(kind="personal"),
            timestamp=datetime(2026, 4, 2, tzinfo=UTC),
        )
    )
    assert result.candidate.topic_id == "user.possessions.home"


def test_llm_agent_combines_embedded_and_explicit_suffix() -> None:
    """LLM mistake: category='user.possessions.vehicle' + leaf_suffix='oil_change'.

    Interpretation: LLM put a seed leaf path in category AND specified a
    finer-grained suffix. Preserve both by combining them into one leaf."""
    mb = MemoryBase(":memory:")
    agent = LLMUpdateAgent(
        FakeLLM(
            _user_payload(
                category="user.possessions.vehicle",
                leaf_suffix="oil_change",
                candidate_content="User scheduled an oil change.",
                evidence="scheduled an oil change",
            )
        ),
        canonicalizer=mb.canonicalize_topic,
    )
    result = agent.run(
        UpdateInput(
            user_input="scheduled an oil change",
            scope=Scope(kind="personal"),
            timestamp=datetime(2026, 4, 2, tzinfo=UTC),
        )
    )
    assert result.candidate.topic_id == "user.possessions.vehicle_oil_change"


def test_llm_agent_rejects_full_path_in_category_with_unknown_prefix() -> None:
    """Unknown category prefix even after stripping → None."""
    mb = MemoryBase(":memory:")
    agent = LLMUpdateAgent(
        FakeLLM(
            _user_payload(
                category="nonsense.path.here",
                leaf_suffix="foo",
            )
        ),
        canonicalizer=mb.canonicalize_topic,
    )
    result = agent.run(
        UpdateInput(
            user_input="reading classic novels",
            scope=Scope(kind="personal"),
            timestamp=datetime(2026, 4, 2, tzinfo=UTC),
        )
    )
    assert result.candidate.topic_id is None
