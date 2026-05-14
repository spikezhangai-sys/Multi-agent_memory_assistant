from datetime import UTC, datetime

from driftscope.agents.types import CandidateMatch, ConflictResolution, UpdateProposal
from driftscope.core.memory_base import MemoryBase
from driftscope.core.schema import Scope, TopicQuery
from driftscope.pipeline.transitions import (
    apply_conflict_resolution,
    is_rollback_legal,
    locate_by_hint,
    try_deterministic_correction,
)
from tests.unit.helpers import make_memory


def test_apply_conflict_resolution_supersedes_old_memory() -> None:
    store = MemoryBase()
    scope = Scope(kind="personal")
    old_memory = make_memory(
        content="我现在住在上海",
        topic_id="user.profile.location",
        scope=scope,
    )
    store.add(old_memory)

    new_memory = make_memory(
        content="我搬到北京了",
        topic_id="user.profile.location",
        scope=scope,
    )
    proposal = UpdateProposal(
        intent="supersede_full",
        candidate=new_memory,
        target_hint=TopicQuery(topic_id="user.profile.location", keywords=["搬到", "北京"]),
        transition_type="corrected",
    )
    resolution = ConflictResolution(
        action="confirm_supersede",
        target_id=old_memory.id,
        transition_type="corrected",
        confidence=0.9,
        reason="same slot",
    )

    applied = apply_conflict_resolution(
        memory_base=store,
        proposal=proposal,
        resolution=resolution,
        timestamp=datetime(2026, 4, 2, tzinfo=UTC),
    )

    assert applied is True
    assert store.get(old_memory.id).state == "superseded"
    assert store.get(old_memory.id).valid_time.end == datetime(2026, 4, 2, tzinfo=UTC)
    assert store.get(new_memory.id).supersedes[0].target == old_memory.id


def test_apply_conflict_resolution_revoke_sets_valid_end() -> None:
    store = MemoryBase()
    scope = Scope(kind="personal")
    target = make_memory(
        content="我不能吃花生",
        topic_id="user.constraint.diet",
        scope=scope,
        memory_type="constraint",
    )
    store.add(target)

    proposal = UpdateProposal(
        intent="revoke",
        target_hint=TopicQuery(topic_id="user.constraint.diet", keywords=["花生"]),
        transition_type="user_revoked",
    )
    resolution = ConflictResolution(
        action="confirm_revoke",
        target_id=target.id,
        transition_type="user_revoked",
        confidence=0.9,
        reason="user revoked memory",
    )

    applied = apply_conflict_resolution(
        memory_base=store,
        proposal=proposal,
        resolution=resolution,
        timestamp=datetime(2026, 4, 2, tzinfo=UTC),
    )

    assert applied is True
    revoked = store.get(target.id)
    assert revoked.state == "revoked"
    assert revoked.valid_time.end == datetime(2026, 4, 2, tzinfo=UTC)


def test_locate_by_hint_returns_none_on_tie() -> None:
    scope = Scope(kind="personal")
    first = make_memory(
        content="我不能吃花生",
        topic_id="user.constraint.diet",
        scope=scope,
        memory_type="constraint",
        state="revoked",
        revoked_at=datetime(2026, 4, 2, tzinfo=UTC),
    )
    second = make_memory(
        content="我不能吃花生",
        topic_id="user.constraint.diet",
        scope=scope,
        memory_type="constraint",
        state="revoked",
        revoked_at=datetime(2026, 4, 2, tzinfo=UTC),
    )

    hint = TopicQuery(topic_id="user.constraint.diet", keywords=["花生"])
    assert locate_by_hint([first, second], hint) is None


def _make_correction_scenario(
    *,
    user_text: str = "Actually, I moved to 北京, not 上海",
    proposal_topic: str | None = "user.profile.location",
    proposal_type: str = "fact",
    top_topic: str = "user.profile.location",
    top_type: str = "fact",
    top_content_sim: float = 0.4,
    top_matched_by: list[str] | None = None,
    second_content_sim: float | None = 0.2,
    intent: str = "supersede_full",
    transition_type: str | None = "corrected",
    ambiguous: bool = True,
) -> tuple[UpdateProposal, list[CandidateMatch], bool, str]:
    scope = Scope(kind="personal")
    candidate_memory = make_memory(
        content="我搬到北京了",
        topic_id=proposal_topic,
        scope=scope,
        memory_type=proposal_type,
    )
    if intent == "supersede_full":
        proposal = UpdateProposal(
            intent="supersede_full",
            candidate=candidate_memory,
            target_hint=TopicQuery(
                topic_id=proposal_topic or "user.profile.location",
                keywords=["北京"],
            ),
            transition_type=transition_type,
        )
    else:
        proposal = UpdateProposal(intent="add", candidate=candidate_memory)

    top_memory = make_memory(
        content="我现在住在上海",
        topic_id=top_topic,
        scope=scope,
        memory_type=top_type,
    )
    matched_by = top_matched_by if top_matched_by is not None else ["content_overlap", "type_exact"]
    candidates = [
        CandidateMatch(
            memory=top_memory,
            score=0.8,
            score_breakdown={"content_sim": top_content_sim, "keyword_overlap": 0.4},
            matched_by=matched_by,
        )
    ]
    if second_content_sim is not None:
        second_memory = make_memory(
            content="我在星巴克的会员等级是金",
            topic_id=top_topic,
            scope=scope,
            memory_type=top_type,
        )
        candidates.append(
            CandidateMatch(
                memory=second_memory,
                score=0.78,
                score_breakdown={"content_sim": second_content_sim, "keyword_overlap": 0.3},
                matched_by=["content_overlap"],
            )
        )
    return proposal, candidates, ambiguous, user_text


def test_bypass_fires_with_all_guards_met() -> None:
    proposal, candidates, ambiguous, user_text = _make_correction_scenario()
    resolution = try_deterministic_correction(
        proposal=proposal,
        candidates=candidates,
        ambiguous=ambiguous,
        user_text=user_text,
    )
    assert resolution is not None
    assert resolution.action == "confirm_supersede"
    assert resolution.target_id == candidates[0].memory.id
    assert resolution.transition_type == "corrected"
    assert resolution.reason == "deterministic_explicit_correction"


def test_returns_none_when_not_ambiguous() -> None:
    proposal, candidates, _, user_text = _make_correction_scenario(ambiguous=False)
    assert (
        try_deterministic_correction(
            proposal=proposal,
            candidates=candidates,
            ambiguous=False,
            user_text=user_text,
        )
        is None
    )


def test_returns_none_without_strong_marker() -> None:
    proposal, candidates, ambiguous, _ = _make_correction_scenario()
    assert (
        try_deterministic_correction(
            proposal=proposal,
            candidates=candidates,
            ambiguous=ambiguous,
            user_text="I'm not happy with 上海",
        )
        is None
    )
    assert (
        try_deterministic_correction(
            proposal=proposal,
            candidates=candidates,
            ambiguous=ambiguous,
            user_text="Please note that I moved to 北京",
        )
        is None
    )


def test_returns_none_when_topic_id_none() -> None:
    proposal, candidates, ambiguous, user_text = _make_correction_scenario(
        proposal_topic=None,
    )
    assert (
        try_deterministic_correction(
            proposal=proposal,
            candidates=candidates,
            ambiguous=ambiguous,
            user_text=user_text,
        )
        is None
    )


def test_returns_none_when_topic_mismatch() -> None:
    proposal, candidates, ambiguous, user_text = _make_correction_scenario(
        top_topic="user.profile.name",
    )
    assert (
        try_deterministic_correction(
            proposal=proposal,
            candidates=candidates,
            ambiguous=ambiguous,
            user_text=user_text,
        )
        is None
    )


def test_returns_none_when_type_mismatch() -> None:
    proposal, candidates, ambiguous, user_text = _make_correction_scenario(
        top_type="preference",
    )
    assert (
        try_deterministic_correction(
            proposal=proposal,
            candidates=candidates,
            ambiguous=ambiguous,
            user_text=user_text,
        )
        is None
    )


def test_returns_none_when_content_sim_below_absolute_threshold() -> None:
    proposal, candidates, ambiguous, user_text = _make_correction_scenario(
        top_content_sim=0.1,
        second_content_sim=0.05,
    )
    assert (
        try_deterministic_correction(
            proposal=proposal,
            candidates=candidates,
            ambiguous=ambiguous,
            user_text=user_text,
        )
        is None
    )


def test_returns_none_when_top_does_not_lead_top2() -> None:
    proposal, candidates, ambiguous, user_text = _make_correction_scenario(
        top_content_sim=0.30,
        second_content_sim=0.29,
    )
    assert (
        try_deterministic_correction(
            proposal=proposal,
            candidates=candidates,
            ambiguous=ambiguous,
            user_text=user_text,
        )
        is None
    )


def test_returns_none_when_top_missing_content_overlap_matched_by() -> None:
    proposal, candidates, ambiguous, user_text = _make_correction_scenario(
        top_matched_by=["type_exact", "topic_hint"],
    )
    assert (
        try_deterministic_correction(
            proposal=proposal,
            candidates=candidates,
            ambiguous=ambiguous,
            user_text=user_text,
        )
        is None
    )


def test_is_rollback_legal_rejects_when_newer_active_memory_exists() -> None:
    store = MemoryBase()
    scope = Scope(kind="personal")
    revoked = make_memory(
        content="我不能吃花生",
        topic_id="user.constraint.diet",
        scope=scope,
        memory_type="constraint",
        state="revoked",
        revoked_at=datetime(2026, 4, 1, tzinfo=UTC),
        ingest_time=datetime(2026, 3, 30, tzinfo=UTC),
    )
    newer_active = make_memory(
        content="我不能吃花生和坚果",
        topic_id="user.constraint.diet",
        scope=scope,
        memory_type="constraint",
        ingest_time=datetime(2026, 4, 2, tzinfo=UTC),
    )
    store.add(revoked)
    store.add(newer_active)

    assert (
        is_rollback_legal(
            memory_base=store,
            target=revoked,
            scope=scope,
            now=datetime(2026, 4, 3, tzinfo=UTC),
            window_days=30,
        )
        is False
    )
