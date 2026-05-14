from __future__ import annotations

from datetime import datetime, timedelta
import re

from driftscope.agents.types import CandidateMatch, ConflictResolution, UpdateProposal
from driftscope.core.memory_base import MemoryBase
from driftscope.core.schema import MemoryEntry, Scope, SupersedeLink, TopicQuery

_TOKEN_RE = re.compile(r"[a-z0-9_]+|[\u4e00-\u9fff]", re.IGNORECASE)

_STRONG_CORRECTION_MARKER_RE = re.compile(
    r"\b(actually|correction|my mistake|i meant|apologize|let me correct|scrap that)\b",
    re.IGNORECASE,
)
_STRONG_CORRECTION_MARKERS_ZH = ("其实", "不对", "更正", "应该是", "我说错")
_MIN_CONTENT_SIM_FOR_BYPASS = 0.2
_MIN_CONTENT_SIM_LEAD = 0.05


def apply_conflict_resolution(
    *,
    memory_base: MemoryBase,
    proposal: UpdateProposal,
    resolution: ConflictResolution,
    timestamp: datetime,
) -> bool:
    if resolution.action in {"request_clarification", "reject"}:
        return False

    if resolution.action == "apply_add":
        if proposal.candidate is None:
            return False
        memory_base.add(proposal.candidate)
        return True

    if resolution.action == "confirm_supersede":
        if proposal.candidate is None or resolution.target_id is None or resolution.transition_type is None:
            return False
        new_memory = proposal.candidate.model_copy(deep=True)
        supersedes = list(new_memory.supersedes)
        supersedes.append(
            SupersedeLink(
                target=resolution.target_id,
                transition_type=resolution.transition_type,
            )
        )
        new_memory.supersedes = supersedes
        new_memory.state = "active"
        new_memory.revoked_at = None
        with memory_base.transaction():
            memory_base.update_state(
                resolution.target_id,
                "superseded",
                valid_end=timestamp,
                commit=False,
            )
            memory_base.add(new_memory, commit=False)
        return True

    if resolution.action == "confirm_revoke":
        if resolution.target_id is None:
            return False
        memory_base.update_state(
            resolution.target_id,
            "revoked",
            revoked_at=timestamp,
            valid_end=timestamp,
        )
        return True

    return False


def locate_by_hint(candidates: list[MemoryEntry], target_hint: TopicQuery) -> MemoryEntry | None:
    if not candidates:
        return None

    scored: list[tuple[tuple[int, int, float, str], MemoryEntry]] = []
    hint_tokens = _tokenize(" ".join(target_hint.keywords))
    for memory in candidates:
        topic_score = 1 if target_hint.topic_id and memory.topic_id == target_hint.topic_id else 0
        keyword_overlap = len(hint_tokens.intersection(_tokenize(memory.summary_for_retrieval or memory.content)))
        recency_score = memory.revoked_at.timestamp() if memory.revoked_at is not None else 0.0
        scored.append(((topic_score, keyword_overlap, recency_score, memory.id), memory))

    scored.sort(key=lambda item: item[0], reverse=True)
    if len(scored) >= 2 and scored[0][0][:3] == scored[1][0][:3]:
        return None
    return scored[0][1]


def is_rollback_legal(
    *,
    memory_base: MemoryBase,
    target: MemoryEntry,
    scope: Scope,
    now: datetime,
    window_days: int,
) -> bool:
    if target.state != "revoked" or target.revoked_at is None:
        return False
    if target.scope != scope:
        return False
    if now - target.revoked_at > timedelta(days=window_days):
        return False

    visible = memory_base.query_visible(scope, now)
    for memory in visible:
        if memory.scope != scope:
            continue
        if memory.topic_id != target.topic_id:
            continue
        if memory.state != "active":
            continue
        if memory.ingest_time > target.revoked_at:
            return False
    return True


def try_deterministic_correction(
    *,
    proposal: UpdateProposal,
    candidates: list[CandidateMatch],
    ambiguous: bool,
    user_text: str,
) -> ConflictResolution | None:
    if proposal.intent != "supersede_full":
        return None
    if proposal.transition_type != "corrected":
        return None
    if not ambiguous:
        return None
    if proposal.candidate is None or not candidates:
        return None
    if proposal.candidate.topic_id is None:
        return None

    text = user_text or ""
    has_marker = bool(_STRONG_CORRECTION_MARKER_RE.search(text)) or any(
        marker in text for marker in _STRONG_CORRECTION_MARKERS_ZH
    )
    if not has_marker:
        return None

    top = candidates[0]
    if top.memory.type != proposal.candidate.type:
        return None
    if top.memory.topic_id != proposal.candidate.topic_id:
        return None

    top_sim = top.score_breakdown.get("content_sim", 0.0)
    if top_sim < _MIN_CONTENT_SIM_FOR_BYPASS:
        return None
    if "content_overlap" not in top.matched_by:
        return None
    if len(candidates) >= 2:
        second_sim = candidates[1].score_breakdown.get("content_sim", 0.0)
        if top_sim < second_sim + _MIN_CONTENT_SIM_LEAD:
            return None

    return ConflictResolution(
        action="confirm_supersede",
        target_id=top.memory.id,
        transition_type="corrected",
        confidence=1.0,
        reason="deterministic_explicit_correction",
    )


def _tokenize(text: str) -> set[str]:
    return {token.lower() for token in _TOKEN_RE.findall(text or "")}
