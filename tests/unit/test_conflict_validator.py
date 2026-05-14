from datetime import UTC, datetime

from driftscope.agents.conflict_validator import ConflictValidator
from driftscope.agents.types import CandidateMatch, ConflictInput, ConflictResolution, UpdateProposal
from driftscope.core.schema import Scope, TopicQuery
from tests.unit.helpers import make_memory


def test_validator_accepts_valid_supersede_resolution() -> None:
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
    input_obj = ConflictInput(
        proposal=proposal,
        scope=scope,
        timestamp=datetime(2026, 4, 2, tzinfo=UTC),
        candidates=[CandidateMatch(memory=target, score=0.9)],
    )
    resolution = ConflictResolution(
        action="confirm_supersede",
        target_id=target.id,
        transition_type="corrected",
        confidence=0.9,
        reason="same topic and same scope",
    )

    result = ConflictValidator().validate(input_obj=input_obj, resolution=resolution)
    assert result.is_valid is True


def test_validator_rejects_illegal_transition_for_fact() -> None:
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
    input_obj = ConflictInput(
        proposal=proposal,
        scope=scope,
        timestamp=datetime(2026, 4, 2, tzinfo=UTC),
        candidates=[CandidateMatch(memory=target, score=0.9)],
    )
    resolution = ConflictResolution(
        action="confirm_supersede",
        target_id=target.id,
        transition_type="preference_shifted",
        confidence=0.9,
        reason="bad transition",
    )

    result = ConflictValidator().validate(input_obj=input_obj, resolution=resolution)
    assert result.is_valid is False
    assert "illegal" in result.errors[0]


def test_validator_accepts_add_proposal_resolved_as_supersede() -> None:
    scope = Scope(kind="personal")
    target = make_memory(
        content="我现在住在上海",
        topic_id="user.profile.location",
        scope=scope,
    )
    proposal = UpdateProposal(
        intent="add",
        candidate=make_memory(
            content="我搬到北京了",
            topic_id="user.profile.location",
            scope=scope,
        ),
    )
    input_obj = ConflictInput(
        proposal=proposal,
        scope=scope,
        timestamp=datetime(2026, 4, 2, tzinfo=UTC),
        candidates=[CandidateMatch(memory=target, score=0.9)],
    )
    resolution = ConflictResolution(
        action="confirm_supersede",
        target_id=target.id,
        transition_type="corrected",
        confidence=0.9,
        reason="same-attribute update; ConflictAgent escalated add to supersede",
    )

    result = ConflictValidator().validate(input_obj=input_obj, resolution=resolution)
    assert result.is_valid is True, result.errors


def test_validator_rejects_revoke_proposal_resolved_as_supersede() -> None:
    scope = Scope(kind="personal")
    target = make_memory(
        content="我现在住在上海",
        topic_id="user.profile.location",
        scope=scope,
    )
    proposal = UpdateProposal(
        intent="revoke",
        target_hint=TopicQuery(topic_id="user.profile.location", keywords=["上海"]),
        transition_type="user_revoked",
    )
    input_obj = ConflictInput(
        proposal=proposal,
        scope=scope,
        timestamp=datetime(2026, 4, 2, tzinfo=UTC),
        candidates=[CandidateMatch(memory=target, score=0.9)],
    )
    resolution = ConflictResolution(
        action="confirm_supersede",
        target_id=target.id,
        transition_type="corrected",
        confidence=0.9,
        reason="should not be allowed",
    )

    result = ConflictValidator().validate(input_obj=input_obj, resolution=resolution)
    assert result.is_valid is False
    assert any("confirm_supersede requires" in err for err in result.errors)


def test_validator_rejects_assistant_summary_as_conflict_target() -> None:
    scope = Scope(kind="personal")
    target = make_memory(
        content="你现在住在上海",
        topic_id="user.profile.location",
        scope=scope,
        origin_role="assistant",
        source_kind="summary",
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
    input_obj = ConflictInput(
        proposal=proposal,
        scope=scope,
        timestamp=datetime(2026, 4, 2, tzinfo=UTC),
        candidates=[CandidateMatch(memory=target, score=0.9)],
    )
    resolution = ConflictResolution(
        action="confirm_supersede",
        target_id=target.id,
        transition_type="corrected",
        confidence=0.9,
        reason="same topic",
    )

    result = ConflictValidator().validate(input_obj=input_obj, resolution=resolution)
    assert result.is_valid is False
    assert "user explicit memory" in result.errors[-1]
