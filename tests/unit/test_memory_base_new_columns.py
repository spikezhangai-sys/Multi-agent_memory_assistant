from datetime import UTC, datetime

from driftscope.core.memory_base import MemoryBase
from driftscope.core.schema import Scope
from tests.unit.helpers import make_memory


def test_memory_base_persists_event_time_and_evidence_round_trip() -> None:
    store = MemoryBase()
    event_time = datetime(2026, 3, 20, 10, 0, tzinfo=UTC)
    memory = make_memory(
        content="Attended cousin's wedding in Brooklyn.",
        topic_id=None,
        scope=Scope(kind="personal"),
        memory_type="episodic",
        event_time=event_time,
        evidence="cousin's wedding",
        importance=0.7,
        sensitivity="low",
        ttl_days=None,
    )

    store.add(memory)
    fetched = store.get(memory.id)

    assert fetched.event_time == event_time
    assert fetched.evidence == "cousin's wedding"
    assert fetched.importance == 0.7
    assert fetched.sensitivity == "low"
    assert fetched.ttl_days is None


def test_memory_base_round_trips_ttl_days_and_missing_fields() -> None:
    store = MemoryBase()
    memory = make_memory(
        content="Temporary status: feeling tired today.",
        topic_id=None,
        scope=Scope(kind="personal"),
        ttl_days=3,
    )

    store.add(memory)
    fetched = store.get(memory.id)

    assert fetched.ttl_days == 3
    assert fetched.event_time is None
    assert fetched.importance is None
