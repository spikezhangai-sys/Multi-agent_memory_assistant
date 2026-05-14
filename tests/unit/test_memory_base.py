from datetime import UTC, datetime, timedelta

import pytest

from driftscope.core.memory_base import MemoryBase
from driftscope.core.schema import Confidence, MemoryEntry, Scope, SupersedeLink, TimeRange


def make_memory(
    *,
    content: str,
    topic_id: str,
    scope: Scope,
    memory_type: str = "fact",
    origin_role: str = "user",
    source_kind: str = "explicit",
    state: str = "active",
    revoked_at: datetime | None = None,
    supersedes: list[SupersedeLink] | None = None,
) -> MemoryEntry:
    now = datetime(2026, 4, 1, tzinfo=UTC)
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
    )


def test_add_and_get_memory() -> None:
    store = MemoryBase()
    memory = make_memory(
        content="我现在住在上海",
        topic_id="user.profile.location",
        scope=Scope(kind="personal"),
    )
    store.add(memory)

    fetched = store.get(memory.id)
    assert fetched.content == memory.content
    assert fetched.scope == memory.scope


def test_query_visible_respects_scope_rules() -> None:
    store = MemoryBase()
    now = datetime(2026, 4, 1, tzinfo=UTC)
    global_memory = make_memory(
        content="系统默认语言是中文",
        topic_id="user.communication.style",
        scope=Scope(kind="global"),
        memory_type="preference",
    )
    personal_memory = make_memory(
        content="我喜欢简洁回答",
        topic_id="user.communication.style",
        scope=Scope(kind="personal"),
        memory_type="preference",
    )
    project_alpha = make_memory(
        content="alpha 项目使用 Python",
        topic_id="project.context.alpha",
        scope=Scope(kind="project", ref="alpha"),
    )
    project_beta = make_memory(
        content="beta 项目使用 Go",
        topic_id="project.context.alpha",
        scope=Scope(kind="project", ref="beta"),
    )
    for memory in [global_memory, personal_memory, project_alpha, project_beta]:
        store.add(memory)

    visible = store.query_visible(Scope(kind="project", ref="alpha"), now)
    visible_ids = {memory.id for memory in visible}
    assert global_memory.id in visible_ids
    assert personal_memory.id in visible_ids
    assert project_alpha.id in visible_ids
    assert project_beta.id not in visible_ids


def test_supersede_chain_and_rollback() -> None:
    store = MemoryBase()
    old_memory = make_memory(
        content="我现在住在上海",
        topic_id="user.profile.location",
        scope=Scope(kind="personal"),
    )
    store.add(old_memory)
    store.update_state(old_memory.id, "superseded")

    new_memory = make_memory(
        content="我搬到北京了",
        topic_id="user.profile.location",
        scope=Scope(kind="personal"),
        supersedes=[
            SupersedeLink(target=old_memory.id, transition_type="corrected"),
        ],
    )
    store.add(new_memory)

    backward = store.get_supersede_chain(new_memory.id, "backward")
    forward = store.get_supersede_chain(old_memory.id, "forward")
    assert [item.id for item in backward] == [old_memory.id]
    assert [item.id for item in forward] == [new_memory.id]

    revoked = make_memory(
        content="我不能吃花生",
        topic_id="user.constraint.diet",
        scope=Scope(kind="personal"),
        memory_type="constraint",
        state="revoked",
        revoked_at=datetime(2026, 4, 2, tzinfo=UTC),
    )
    store.add(revoked)
    assert store.rollback(revoked.id) is True
    assert store.get(revoked.id).state == "active"


def test_reject_unknown_topic() -> None:
    store = MemoryBase()
    memory = make_memory(
        content="未知主题",
        topic_id="not.a.real.topic",
        scope=Scope(kind="personal"),
    )
    with pytest.raises(ValueError):
        store.add(memory)


def test_accepts_memory_without_topic() -> None:
    store = MemoryBase()
    memory = make_memory(
        content="I graduated with a degree in Business Administration.",
        topic_id=None,
        scope=Scope(kind="personal"),
    )

    store.add(memory)

    fetched = store.get(memory.id)
    assert fetched.topic_id is None
