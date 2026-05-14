from __future__ import annotations

from datetime import UTC, datetime, timedelta

from driftscope.core.schema import Confidence, MemoryEntry, Scope, SupersedeLink, TimeRange


def make_memory(
    *,
    content: str,
    topic_id: str | None,
    scope: Scope,
    memory_type: str = "fact",
    origin_role: str = "user",
    source_kind: str = "explicit",
    state: str = "active",
    revoked_at: datetime | None = None,
    supersedes: list[SupersedeLink] | None = None,
    ingest_time: datetime | None = None,
    sensitive: bool = False,
    summary_for_retrieval: str | None = None,
    event_time: datetime | None = None,
    evidence: str | None = None,
    importance: float | None = None,
    sensitivity: str | None = None,
    ttl_days: int | None = None,
) -> MemoryEntry:
    now = ingest_time or datetime(2026, 4, 1, tzinfo=UTC)
    src = "user_explicit" if origin_role == "user" and source_kind == "explicit" else "inferred"
    return MemoryEntry(
        content=content,
        type=memory_type,
        topic_id=topic_id,
        scope=scope,
        src=src,
        origin_role=origin_role,
        source_kind=source_kind,
        conf=Confidence(prior=0.9, llm_self=0.8, combined=0.87),
        valid_time=TimeRange(start=now - timedelta(days=1)),
        ingest_time=now,
        state=state,
        revoked_at=revoked_at,
        supersedes=supersedes or [],
        sensitive=sensitive,
        summary_for_retrieval=summary_for_retrieval,
        event_time=event_time,
        evidence=evidence,
        importance=importance,
        sensitivity=sensitivity,
        ttl_days=ttl_days,
    )
