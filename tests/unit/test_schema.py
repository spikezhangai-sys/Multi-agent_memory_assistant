from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from driftscope.core.schema import Confidence, MemoryEntry, Scope, TimeRange, TurnInput


def test_scope_requires_ref_for_project() -> None:
    with pytest.raises(ValidationError):
        Scope(kind="project")


def test_sensitive_memory_requires_summary() -> None:
    with pytest.raises(ValidationError):
        MemoryEntry(
            content="我对花生过敏",
            type="constraint",
            topic_id="user.constraint.diet",
            scope=Scope(kind="personal"),
            src="user_explicit",
            conf=Confidence(prior=0.9, llm_self=0.8, combined=0.87),
            valid_time=TimeRange(start=datetime(2026, 1, 1, tzinfo=UTC)),
            ingest_time=datetime(2026, 1, 1, tzinfo=UTC),
            sensitive=True,
        )


def test_revoked_state_requires_revoked_at() -> None:
    with pytest.raises(ValidationError):
        MemoryEntry(
            content="我现在住在上海",
            type="fact",
            topic_id="user.profile.location",
            scope=Scope(kind="personal"),
            src="user_explicit",
            conf=Confidence(prior=0.9, llm_self=0.8, combined=0.87),
            valid_time=TimeRange(start=datetime(2026, 1, 1, tzinfo=UTC)),
            ingest_time=datetime(2026, 1, 1, tzinfo=UTC),
            state="revoked",
        )


def test_turn_input_requires_payload() -> None:
    with pytest.raises(ValidationError):
        TurnInput(
            scope=Scope(kind="personal"),
            timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        )


def test_memory_entry_allows_missing_topic_id() -> None:
    memory = MemoryEntry(
        content="I graduated with a degree in Business Administration.",
        type="fact",
        topic_id=None,
        scope=Scope(kind="personal"),
        src="user_explicit",
        conf=Confidence(prior=0.9, llm_self=0.8, combined=0.87),
        valid_time=TimeRange(start=datetime(2026, 1, 1, tzinfo=UTC)),
        ingest_time=datetime(2026, 1, 1, tzinfo=UTC),
    )

    assert memory.topic_id is None
