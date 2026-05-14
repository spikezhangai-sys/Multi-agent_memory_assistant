from __future__ import annotations

from datetime import datetime
import re

from driftscope.agents.types import CandidateMatch, CandidateSelection, CandidateSelectorConfig, UpdateProposal
from driftscope.config.loader import load_default_config
from driftscope.core.memory_base import MemoryBase
from driftscope.core.schema import MemoryEntry, Scope

_TOKEN_RE = re.compile(r"[a-z0-9_]+|[\u4e00-\u9fff]", re.IGNORECASE)
_TOPIC_HINT_BONUS = 0.15


class CandidateSelector:
    def __init__(self, config: CandidateSelectorConfig | None = None) -> None:
        if config is None:
            config = load_default_config().conflict_selector
        self.config = config

    def select(
        self,
        *,
        proposal: UpdateProposal,
        memory_base: MemoryBase,
        scope: Scope,
        timestamp: datetime,
    ) -> CandidateSelection:
        if proposal.intent in {"ignore", "rollback"}:
            return CandidateSelection()

        pool = [
            memory
            for memory in memory_base.query_visible(scope, timestamp)
            if memory_base.scope_rules.can_target(scope, memory.scope)
            and _is_conflict_eligible(memory)
        ]

        target_topic = self._target_topic(proposal)
        target_type = proposal.candidate.type if proposal.candidate else None
        if proposal.intent == "supersede_full" and target_type is not None:
            pool = [memory for memory in pool if memory.type == target_type]
            if not pool:
                return CandidateSelection()

        reference_text = self._reference_text(proposal)
        reference_tokens = _tokenize(reference_text)
        matches: list[CandidateMatch] = []
        for memory in pool:
            score_breakdown, matched_by = self._score_memory(
                proposal=proposal,
                reference_tokens=reference_tokens,
                memory=memory,
                timestamp=timestamp,
            )
            score = (
                self.config.content_sim_weight * score_breakdown["content_sim"]
                + self.config.keyword_overlap_weight * score_breakdown["keyword_overlap"]
                + self.config.time_proximity_weight * score_breakdown["time_proximity"]
                + self.config.confidence_weight * score_breakdown["confidence"]
                + _TOPIC_HINT_BONUS * score_breakdown["topic_hint"]
            )
            min_score = self.config.revoke_min_score if proposal.intent == "revoke" else self.config.min_score
            if score < min_score:
                continue
            matches.append(
                CandidateMatch(
                    memory=memory,
                    score=score,
                    score_breakdown=score_breakdown,
                    matched_by=matched_by,
                )
            )

        matches.sort(key=lambda item: (-item.score, -item.memory.ingest_time.timestamp(), item.memory.id))
        top_k = self.config.revoke_top_k if proposal.intent == "revoke" else self.config.top_k
        trimmed = matches[:top_k]

        ambiguous = False
        if len(trimmed) >= 2:
            ambiguous = abs(trimmed[0].score - trimmed[1].score) < self.config.ambiguity_margin
        return CandidateSelection(candidates=trimmed, ambiguous_candidates=ambiguous)

    def _target_topic(self, proposal: UpdateProposal) -> str | None:
        if proposal.candidate is not None:
            return proposal.candidate.topic_id
        if proposal.target_hint is not None:
            return proposal.target_hint.topic_id
        return None

    def _reference_text(self, proposal: UpdateProposal) -> str:
        if proposal.candidate is not None:
            return proposal.candidate.summary_for_retrieval or proposal.candidate.content
        if proposal.target_hint is not None:
            return " ".join(proposal.target_hint.keywords)
        return ""

    def _score_memory(
        self,
        *,
        proposal: UpdateProposal,
        reference_tokens: set[str],
        memory: MemoryEntry,
        timestamp: datetime,
    ) -> tuple[dict[str, float], list[str]]:
        memory_tokens = _tokenize(memory.summary_for_retrieval or memory.content)
        content_sim = _jaccard(reference_tokens, memory_tokens)

        keywords = set()
        if proposal.target_hint is not None:
            keywords = _tokenize(" ".join(proposal.target_hint.keywords))
        keyword_overlap = _jaccard(keywords or reference_tokens, memory_tokens)

        age_seconds = max((timestamp - memory.ingest_time).total_seconds(), 0.0)
        age_days = age_seconds / 86400.0
        time_proximity = 1.0 / (1.0 + age_days)
        confidence = memory.conf.combined
        target_topic = self._target_topic(proposal)
        # Three-way topic_hint:
        #   - both topics resolve and match: 1.0 (strong same-topic signal)
        #   - both topics are NULL: 0.5 (neutral — extraction failed for both,
        #     so we should not penalize as if the topics differ; this avoids
        #     dropping valid same-attribute KU updates whose canonicalizer
        #     happened to return None on both writes)
        #   - one is set and the other is not, or they differ: 0.0
        if target_topic is not None and memory.topic_id == target_topic:
            topic_hint = 1.0
        elif target_topic is None and memory.topic_id is None:
            topic_hint = 0.5
        else:
            topic_hint = 0.0

        matched_by: list[str] = []
        if proposal.candidate is not None and memory.type == proposal.candidate.type:
            matched_by.append("type_exact")
        if topic_hint > 0:
            matched_by.append("topic_hint")
        if content_sim > 0:
            matched_by.append("content_overlap")
        if keyword_overlap > 0:
            matched_by.append("keyword_overlap")

        return (
            {
                "content_sim": content_sim,
                "keyword_overlap": keyword_overlap,
                "time_proximity": time_proximity,
                "confidence": confidence,
                "topic_hint": topic_hint,
            },
            matched_by,
        )


def _tokenize(text: str) -> set[str]:
    return {token.lower() for token in _TOKEN_RE.findall(text or "")}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    denom = len(left.union(right))
    if denom == 0:
        return 0.0
    return len(left.intersection(right)) / denom


def _is_conflict_eligible(memory: MemoryEntry) -> bool:
    if memory.type == "raw_session":
        return False
    return memory.origin_role == "user" and memory.source_kind == "explicit"
