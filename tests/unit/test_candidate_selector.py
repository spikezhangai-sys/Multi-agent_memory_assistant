from datetime import UTC, datetime

from driftscope.agents.candidate_selector import CandidateSelector, CandidateSelectorConfig
from driftscope.agents.types import UpdateProposal
from driftscope.core.memory_base import MemoryBase
from driftscope.core.schema import Scope, TopicQuery
from tests.unit.helpers import make_memory


def test_selector_prefers_same_scope_same_topic_candidate() -> None:
    store = MemoryBase()
    scope = Scope(kind="personal")
    target = make_memory(
        content="我现在住在上海",
        topic_id="user.profile.location",
        scope=scope,
    )
    other_scope = make_memory(
        content="alpha 项目在上海",
        topic_id="user.profile.location",
        scope=Scope(kind="project", ref="alpha"),
    )
    store.add(target)
    store.add(other_scope)

    proposal = UpdateProposal(
        intent="supersede_full",
        candidate=make_memory(
            content="我搬到北京了",
            topic_id="user.profile.location",
            scope=scope,
        ),
        target_hint=TopicQuery(topic_id="user.profile.location", keywords=["搬到", "北京", "住在"]),
        transition_type="corrected",
    )
    selector = CandidateSelector(
        CandidateSelectorConfig(min_score=0.0, ambiguity_margin=0.08),
    )
    selection = selector.select(
        proposal=proposal,
        memory_base=store,
        scope=scope,
        timestamp=datetime(2026, 4, 2, tzinfo=UTC),
    )

    assert [match.memory.id for match in selection.candidates] == [target.id]
    assert selection.ambiguous_candidates is False


def test_selector_marks_ambiguous_when_top_scores_are_tied() -> None:
    store = MemoryBase()
    scope = Scope(kind="personal")
    first = make_memory(
        content="我现在住在上海",
        topic_id="user.profile.location",
        scope=scope,
    )
    second = make_memory(
        content="我现在住在上海",
        topic_id="user.profile.location",
        scope=scope,
    )
    store.add(first)
    store.add(second)

    proposal = UpdateProposal(
        intent="supersede_full",
        candidate=make_memory(
            content="我搬到北京了",
            topic_id="user.profile.location",
            scope=scope,
        ),
        target_hint=TopicQuery(topic_id="user.profile.location", keywords=["住在", "搬到"]),
        transition_type="corrected",
    )
    selector = CandidateSelector(CandidateSelectorConfig(min_score=0.0, ambiguity_margin=0.5))
    selection = selector.select(
        proposal=proposal,
        memory_base=store,
        scope=scope,
        timestamp=datetime(2026, 4, 2, tzinfo=UTC),
    )

    assert len(selection.candidates) == 2
    assert selection.ambiguous_candidates is True


def test_selector_excludes_assistant_summary_memories_from_conflict_targets() -> None:
    store = MemoryBase()
    scope = Scope(kind="personal")
    assistant_summary = make_memory(
        content="你现在住在上海",
        topic_id="user.profile.location",
        scope=scope,
        origin_role="assistant",
        source_kind="summary",
    )
    store.add(assistant_summary)

    proposal = UpdateProposal(
        intent="supersede_full",
        candidate=make_memory(
            content="我搬到北京了",
            topic_id="user.profile.location",
            scope=scope,
        ),
        target_hint=TopicQuery(topic_id="user.profile.location", keywords=["搬到", "北京", "住在"]),
        transition_type="corrected",
    )
    selector = CandidateSelector(CandidateSelectorConfig(min_score=0.0))
    selection = selector.select(
        proposal=proposal,
        memory_base=store,
        scope=scope,
        timestamp=datetime(2026, 4, 2, tzinfo=UTC),
    )

    assert selection.candidates == []


def test_selector_treats_topic_as_hint_not_hard_filter() -> None:
    store = MemoryBase()
    scope = Scope(kind="personal")
    target = make_memory(
        content="I graduated with a degree in Business Administration.",
        topic_id=None,
        scope=scope,
    )
    store.add(target)

    proposal = UpdateProposal(
        intent="supersede_full",
        candidate=make_memory(
            content="I graduated with a degree in Finance.",
            topic_id="project.context.alpha",
            scope=scope,
        ),
        target_hint=TopicQuery(topic_id="project.context.alpha", keywords=["graduated", "degree", "finance"]),
        transition_type="corrected",
    )
    selector = CandidateSelector(CandidateSelectorConfig(min_score=0.0))
    selection = selector.select(
        proposal=proposal,
        memory_base=store,
        scope=scope,
        timestamp=datetime(2026, 4, 2, tzinfo=UTC),
    )

    assert [match.memory.id for match in selection.candidates] == [target.id]


def test_selector_returns_candidates_for_add_intent_with_existing_same_topic_memory() -> None:
    """`add` proposals must now surface conflict candidates (used to short-circuit
    to empty). This is what lets a user statement like "remember when I got
    pre-approved for $400k from Wells Fargo" find the older $350k fact and
    route through the conflict agent instead of accumulating both as active.
    """
    store = MemoryBase()
    scope = Scope(kind="personal")
    existing = make_memory(
        content="User got pre-approved for $350,000 from Wells Fargo.",
        topic_id="user.possessions.home",
        scope=scope,
    )
    store.add(existing)

    proposal = UpdateProposal(
        intent="add",
        candidate=make_memory(
            content="User got pre-approved for $400,000 from Wells Fargo.",
            topic_id="user.possessions.home",
            scope=scope,
        ),
    )
    selector = CandidateSelector(CandidateSelectorConfig(min_score=0.0))
    selection = selector.select(
        proposal=proposal,
        memory_base=store,
        scope=scope,
        timestamp=datetime(2026, 4, 2, tzinfo=UTC),
    )

    assert [match.memory.id for match in selection.candidates] == [existing.id]


def test_selector_excludes_raw_session_from_conflict_pool() -> None:
    """raw_session memories are verbatim audit trail and must never appear as
    conflict candidates — without this, an `add` proposal would self-conflict
    against the raw_session sidecar written for the same turn.
    """
    store = MemoryBase()
    scope = Scope(kind="personal")
    raw = make_memory(
        content="I just got pre-approved for $400,000 from Wells Fargo.",
        topic_id=None,
        scope=scope,
        memory_type="raw_session",
    )
    store.add(raw)

    proposal = UpdateProposal(
        intent="add",
        candidate=make_memory(
            content="User got pre-approved for $400,000 from Wells Fargo.",
            topic_id="user.possessions.home",
            scope=scope,
        ),
    )
    selector = CandidateSelector(CandidateSelectorConfig(min_score=0.0))
    selection = selector.select(
        proposal=proposal,
        memory_base=store,
        scope=scope,
        timestamp=datetime(2026, 4, 2, tzinfo=UTC),
    )

    assert selection.candidates == []
