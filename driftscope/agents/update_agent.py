from __future__ import annotations

from datetime import datetime
import logging
import re
from typing import Callable, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from driftscope.agents.base import Agent
from driftscope.agents.types import IndexedUpdateProposal, UpdateInput, UpdateProposal
from driftscope.config.loader import DriftScopeConfig, load_default_config
from driftscope.core.schema import (
    Confidence,
    MemoryEntry,
    MemorySource,
    MemoryType,
    OriginRole,
    Scope,
    Sensitivity,
    SourceKind,
    TimeRange,
    TopicQuery,
    TransitionType,
)
from driftscope.core.topic_tree import TopicTree
from driftscope.llm.client import StructuredLLM

TopicCanonicalizer = Callable[[str, str], "str | None"]

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9_]+|[\u4e00-\u9fff]", re.IGNORECASE)
_SUPPORTED_UPDATE_INTENTS = {"add", "supersede_full", "revoke", "rollback"}
_SUPPORTED_MEMORY_TYPES = {"fact", "preference", "constraint", "episodic"}
_SUPPORTED_TRANSITION_TYPES = {"corrected", "preference_shifted", "user_revoked"}
_SUPPORTED_SENSITIVITY = {"low", "medium", "high"}


class HeuristicUpdateAgent(Agent):
    name = "update"

    def __init__(
        self,
        topic_tree: TopicTree | None = None,
        config: DriftScopeConfig | None = None,
    ) -> None:
        self.topic_tree = topic_tree or TopicTree.load_default()
        self.config = config or load_default_config()

    def run(self, input_obj: UpdateInput) -> UpdateProposal:
        normalized = input_obj.user_input.strip()
        if not normalized:
            return UpdateProposal(intent="ignore")

        topic_id = self.topic_tree.match(normalized)
        if topic_id is None:
            return UpdateProposal(intent="ignore")

        default_type = self.topic_tree.default_type_for_topic(topic_id) or "fact"
        candidate = self._build_candidate(
            content=normalized,
            topic_id=topic_id,
            memory_type=default_type,
            timestamp=input_obj.timestamp,
            scope=input_obj.scope,
            origin_role=input_obj.origin_role,
        )

        return UpdateProposal(intent="add", candidate=candidate)

    def run_many(self, input_obj: UpdateInput) -> list[UpdateProposal]:
        proposal = self.run(input_obj)
        return [] if proposal.intent == "ignore" else [proposal]

    def run_batch(self, input_objs: list[UpdateInput]) -> list[IndexedUpdateProposal]:
        proposals: list[IndexedUpdateProposal] = []
        for index, input_obj in enumerate(input_objs):
            proposal = self.run(input_obj)
            if proposal.intent == "ignore":
                continue
            proposals.append(IndexedUpdateProposal(source_turn_index=index, proposal=proposal))
        return proposals

    def _build_candidate(
        self,
        *,
        content: str,
        topic_id: str | None,
        memory_type: str,
        timestamp: datetime,
        scope,
        origin_role: str = "user",
    ) -> MemoryEntry:
        src, source_kind, prior_key = _source_defaults(origin_role)
        prior = self.config.confidence.prior_table[prior_key]
        return MemoryEntry(
            content=content,
            type=memory_type,
            topic_id=topic_id,
            scope=scope,
            src=src,
            origin_role=origin_role,
            source_kind=source_kind,
            conf=Confidence(prior=prior, llm_self=None, combined=prior),
            valid_time=TimeRange(start=timestamp),
            ingest_time=timestamp,
        )


class UpdateDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Closed enum so OpenAI strict json_schema enforces the value server-side
    # and gpt-4o-mini cannot emit "ignore" as a soft-hedge — observed bug
    # where the LLM emitted intent="ignore" alongside fully populated
    # candidate fields, silently destroying ~75% of extracted facts.
    intent: Literal["add", "supersede_full", "revoke", "rollback"]
    candidate_content: str | None = None
    candidate_type: MemoryType | None = None
    category: str | None = None
    leaf_suffix: str | None = None
    topic_id: str | None = None
    keywords: list[str] = Field(default_factory=list)
    transition_type: TransitionType | None = None
    evidence: str
    importance: float
    sensitivity: Sensitivity
    ttl_days: int | None = None
    event_time: datetime | None = None

    @field_validator("intent", mode="before")
    @classmethod
    def normalize_intent(cls, value: object) -> str:
        return _normalize_update_intent(value)

    @field_validator("candidate_type", mode="before")
    @classmethod
    def normalize_candidate_type(cls, value: object) -> MemoryType | None:
        return _normalize_memory_type(value)

    @field_validator("candidate_content", mode="before")
    @classmethod
    def normalize_candidate_content(cls, value: object) -> str | None:
        return _normalize_candidate_content(value)

    @field_validator("keywords", mode="before")
    @classmethod
    def normalize_keywords(cls, value: object) -> list[str]:
        return _normalize_keywords(value)

    @field_validator("transition_type", mode="before")
    @classmethod
    def normalize_transition_type(cls, value: object) -> TransitionType | None:
        return _normalize_transition_type(value)

    @field_validator("category", "leaf_suffix", mode="before")
    @classmethod
    def normalize_topic_parts(cls, value: object) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            return None
        stripped = value.strip()
        return stripped or None


class BatchUpdateDecisionItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_turn_index: int
    # See UpdateDecision.intent — same closed enum so OpenAI strict mode
    # rejects "ignore" at the API boundary instead of letting it silently
    # nullify the candidate downstream.
    intent: Literal["add", "supersede_full", "revoke", "rollback"]
    candidate_content: str | None = None
    candidate_type: MemoryType | None = None
    category: str | None = None
    leaf_suffix: str | None = None
    topic_id: str | None = None
    keywords: list[str] = Field(default_factory=list)
    transition_type: TransitionType | None = None
    evidence: str
    importance: float
    sensitivity: Sensitivity
    ttl_days: int | None = None
    event_time: datetime | None = None

    @field_validator("intent", mode="before")
    @classmethod
    def normalize_intent(cls, value: object) -> str:
        return _normalize_update_intent(value)

    @field_validator("candidate_type", mode="before")
    @classmethod
    def normalize_candidate_type(cls, value: object) -> MemoryType | None:
        return _normalize_memory_type(value)

    @field_validator("candidate_content", mode="before")
    @classmethod
    def normalize_candidate_content(cls, value: object) -> str | None:
        return _normalize_candidate_content(value)

    @field_validator("keywords", mode="before")
    @classmethod
    def normalize_keywords(cls, value: object) -> list[str]:
        return _normalize_keywords(value)

    @field_validator("transition_type", mode="before")
    @classmethod
    def normalize_transition_type(cls, value: object) -> TransitionType | None:
        return _normalize_transition_type(value)

    @field_validator("category", "leaf_suffix", mode="before")
    @classmethod
    def normalize_topic_parts(cls, value: object) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            return None
        stripped = value.strip()
        return stripped or None

    @model_validator(mode="after")
    def validate_shape(self) -> "BatchUpdateDecisionItem":
        if self.source_turn_index < 0:
            raise ValueError("source_turn_index must be >= 0")
        return self


class BatchUpdateDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    proposals: list[BatchUpdateDecisionItem] = Field(default_factory=list)


class LLMUpdateAgent(Agent):
    name = "update"

    def __init__(
        self,
        llm: StructuredLLM,
        *,
        topic_tree: TopicTree | None = None,
        config: DriftScopeConfig | None = None,
        heuristic_fallback: bool = True,
        canonicalizer: TopicCanonicalizer | None = None,
    ) -> None:
        self.llm = llm
        self.topic_tree = topic_tree or TopicTree.load_default()
        self.config = config or load_default_config()
        self.canonicalizer = canonicalizer
        self._heuristic = (
            HeuristicUpdateAgent(topic_tree=self.topic_tree, config=self.config)
            if heuristic_fallback
            else None
        )
        # Per-call diagnostic of the most recent batch/run_many attempt. Set
        # by run_batch / run_many; consumed by the orchestrator to surface in
        # turn-log extras["batch_update"]. Distinguishes "LLM call failed"
        # (exception_type set), "LLM returned valid empty list" (raw_count=0,
        # exception_type=None), and "valid items but all dropped downstream"
        # (raw_count>0 but filtered to 0 by index/ignore filters).
        self._last_batch_diagnostic: dict | None = None

    def run(self, input_obj: UpdateInput) -> UpdateProposal:
        prompt = self._build_prompt(input_obj)
        try:
            decision = self.llm.generate_structured(
                system_prompt=_UPDATE_SYSTEM_PROMPT,
                user_prompt=prompt,
                response_model=UpdateDecision,
            )
            if not isinstance(decision, UpdateDecision):
                decision = UpdateDecision.model_validate(_normalize_update_decision_payload(decision))
            return self._to_update_proposal(decision, input_obj)
        except Exception as exc:
            logger.warning(
                "LLMUpdateAgent.run failed (%s): %s — input=%r",
                type(exc).__name__,
                exc,
                input_obj.user_input[:120],
            )
            return self._heuristic_fallback(input_obj)

    def run_many(self, input_obj: UpdateInput) -> list[UpdateProposal]:
        """Extract zero or more proposals from a single turn.

        Uses the batch prompt on a 1-turn payload so multiple distinct facts in the
        same turn can each produce a proposal. Falls back to the single-turn prompt
        (``run``) if the batch call yields nothing for a user turn.
        """
        if not input_obj.user_input.strip():
            return []

        diagnostic: dict[str, object] = {
            "batch_size": 1,
            "exception_type": None,
            "exception_message": None,
            "raw_decision_count": None,
            "dropped_index_oor": 0,
            "dropped_ignore_intent": 0,
            "kept_proposal_count": 0,
            "path": "run_many",
        }
        self._last_batch_diagnostic = diagnostic
        try:
            decision = self.llm.generate_structured(
                system_prompt=_BATCH_UPDATE_SYSTEM_PROMPT,
                user_prompt=self._build_batch_prompt([input_obj]),
                response_model=BatchUpdateDecision,
            )
            if not isinstance(decision, BatchUpdateDecision):
                decision = BatchUpdateDecision.model_validate(
                    _normalize_batch_update_decision_payload(decision)
                )
        except Exception as exc:
            diagnostic["exception_type"] = type(exc).__name__
            diagnostic["exception_message"] = str(exc)[:500]
            logger.warning(
                "LLMUpdateAgent.run_many batch prompt failed (%s): %s — input=%r",
                type(exc).__name__,
                str(exc)[:200],
                input_obj.user_input[:120],
            )
            decision = BatchUpdateDecision()

        diagnostic["raw_decision_count"] = len(decision.proposals)
        proposals: list[UpdateProposal] = []
        for item in decision.proposals:
            if item.source_turn_index != 0:
                diagnostic["dropped_index_oor"] = int(diagnostic["dropped_index_oor"]) + 1
                continue
            try:
                proposal = self._to_update_proposal(
                    self._item_to_decision(item),
                    input_obj,
                )
            except Exception as exc:
                logger.warning(
                    "LLMUpdateAgent.run_many proposal build failed (%s): %s",
                    type(exc).__name__,
                    exc,
                )
                continue
            if proposal.intent == "ignore":
                diagnostic["dropped_ignore_intent"] = int(diagnostic["dropped_ignore_intent"]) + 1
                continue
            if _proposal_is_distinct(proposal, proposals):
                proposals.append(proposal)

        diagnostic["kept_proposal_count"] = len(proposals)
        self._last_batch_diagnostic = diagnostic

        if (
            not proposals
            and input_obj.origin_role == "user"
            and self.config.update.batch_supplemental_enabled
        ):
            supplemental = self.run(input_obj)
            if supplemental.intent != "ignore":
                proposals.append(supplemental)

        return proposals

    def _item_to_decision(self, item: BatchUpdateDecisionItem) -> UpdateDecision:
        return UpdateDecision(
            intent=item.intent,
            candidate_content=item.candidate_content,
            candidate_type=item.candidate_type,
            category=item.category,
            leaf_suffix=item.leaf_suffix,
            topic_id=item.topic_id,
            keywords=item.keywords,
            transition_type=item.transition_type,
            evidence=item.evidence,
            importance=item.importance,
            sensitivity=item.sensitivity,
            ttl_days=item.ttl_days,
            event_time=item.event_time,
        )

    def _heuristic_fallback(self, input_obj: UpdateInput) -> UpdateProposal:
        if self._heuristic is None:
            return UpdateProposal(intent="ignore")
        try:
            return self._heuristic.run(input_obj)
        except Exception as exc:
            logger.warning(
                "Heuristic fallback also failed (%s): %s",
                type(exc).__name__,
                exc,
            )
            return UpdateProposal(intent="ignore")

    def run_batch(self, input_objs: list[UpdateInput]) -> list[IndexedUpdateProposal]:
        if not input_objs:
            return []
        if len(input_objs) == 1:
            proposal = self.run(input_objs[0])
            if proposal.intent == "ignore":
                return []
            return [IndexedUpdateProposal(source_turn_index=0, proposal=proposal)]

        diagnostic: dict[str, object] = {
            "batch_size": len(input_objs),
            "exception_type": None,
            "exception_message": None,
            "raw_decision_count": None,
            "dropped_index_oor": 0,
            "dropped_ignore_intent": 0,
            "kept_proposal_count": 0,
        }
        # Stash early so _record_ignore_reason (called inside the per-item
        # loop below) writes into THIS batch's diagnostic, not the previous one.
        self._last_batch_diagnostic = diagnostic
        prompt = self._build_batch_prompt(input_objs)
        try:
            decision = self.llm.generate_structured(
                system_prompt=_BATCH_UPDATE_SYSTEM_PROMPT,
                user_prompt=prompt,
                response_model=BatchUpdateDecision,
            )
            if not isinstance(decision, BatchUpdateDecision):
                decision = BatchUpdateDecision.model_validate(_normalize_batch_update_decision_payload(decision))
        except Exception as exc:
            diagnostic["exception_type"] = type(exc).__name__
            # truncate exception message — Pydantic ValidationError can be huge
            diagnostic["exception_message"] = str(exc)[:500]
            logger.warning(
                "LLMUpdateAgent.run_batch failed (%s): %s — batch_size=%d",
                type(exc).__name__,
                str(exc)[:200],
                len(input_objs),
            )
            decision = BatchUpdateDecision()

        diagnostic["raw_decision_count"] = len(decision.proposals)
        indexed: list[IndexedUpdateProposal] = []
        proposals_by_index: dict[int, list[UpdateProposal]] = {}
        for item in sorted(decision.proposals, key=lambda proposal: proposal.source_turn_index):
            if item.source_turn_index >= len(input_objs):
                diagnostic["dropped_index_oor"] = int(diagnostic["dropped_index_oor"]) + 1
                continue
            proposal = self._to_update_proposal(
                UpdateDecision(
                    intent=item.intent,
                    candidate_content=item.candidate_content,
                    candidate_type=item.candidate_type,
                    category=item.category,
                    leaf_suffix=item.leaf_suffix,
                    topic_id=item.topic_id,
                    keywords=item.keywords,
                    transition_type=item.transition_type,
                    evidence=item.evidence,
                    importance=item.importance,
                    sensitivity=item.sensitivity,
                    ttl_days=item.ttl_days,
                    event_time=item.event_time,
                ),
                input_objs[item.source_turn_index],
            )
            if proposal.intent == "ignore":
                diagnostic["dropped_ignore_intent"] = int(diagnostic["dropped_ignore_intent"]) + 1
                continue
            proposals_by_index.setdefault(item.source_turn_index, []).append(proposal)
            indexed.append(IndexedUpdateProposal(source_turn_index=item.source_turn_index, proposal=proposal))

        diagnostic["kept_proposal_count"] = len(indexed)
        self._last_batch_diagnostic = diagnostic

        if self.config.update.batch_supplemental_enabled:
            for index, input_obj in enumerate(input_objs):
                existing = proposals_by_index.get(index, [])
                if input_obj.origin_role == "user" and not existing:
                    for supplemental in self.run_many(input_obj):
                        if supplemental.intent == "ignore":
                            continue
                        if not _proposal_is_distinct(supplemental, existing):
                            continue
                        existing.append(supplemental)
                        indexed.append(
                            IndexedUpdateProposal(source_turn_index=index, proposal=supplemental)
                        )

        indexed.sort(key=lambda item: item.source_turn_index)
        return indexed

    def _build_prompt(self, input_obj: UpdateInput) -> str:
        allowed_topics = self._allowed_topics()
        payload = {
            "user_input": input_obj.user_input,
            "origin_role": input_obj.origin_role,
            "scope": input_obj.scope.model_dump(mode="json"),
            "timestamp": input_obj.timestamp.isoformat(),
            "allowed_topics": allowed_topics,
            "rules": [
                "Intent must be one of add, supersede_full, revoke, rollback, ignore.",
                "Choose `category` from allowed_topics[].category — this is a CLOSED set. Pick the one whose description best fits the fact.",
                "Emit `leaf_suffix` as a short lowercase snake_case slug (1-3 words, English a-z0-9_, ≤60 chars) that names the specific topic bucket (e.g. category=user.preference + leaf_suffix=books for a books preference). Re-use an existing seed_leaves[].leaf_suffix when the fact matches one; otherwise invent a consistent new one.",
                "Leave both `category` and `leaf_suffix` null only when no category fits at all.",
                "Use user_revoked only for revoke.",
                "Use preference_shifted only when candidate_type is preference and the user indicates changed preference.",
                "If no stable memory should be stored, return ignore.",
            ],
        }
        nearby_memories = self._nearby_memory_payload(input_obj.nearby_memories)
        if nearby_memories:
            payload["nearby_memories"] = nearby_memories
        import json

        return json.dumps(payload, ensure_ascii=False, indent=2)

    def _build_batch_prompt(self, input_objs: list[UpdateInput]) -> str:
        turns = []
        for index, input_obj in enumerate(input_objs):
            turn_payload = {
                "source_turn_index": index,
                "user_input": input_obj.user_input,
                "origin_role": input_obj.origin_role,
                "scope": input_obj.scope.model_dump(mode="json"),
                "timestamp": input_obj.timestamp.isoformat(),
            }
            nearby_memories = self._nearby_memory_payload(input_obj.nearby_memories)
            if nearby_memories:
                turn_payload["nearby_memories"] = nearby_memories
            turns.append(turn_payload)
        payload = {
            "turns": turns,
            "allowed_topics": self._compact_allowed_topics(),
            "rules": [
                "Return zero or more proposals in chronological order.",
                "Each proposal must include source_turn_index referencing turns[].",
                "You may return multiple proposals for one turn when the user states multiple stable memories.",
                "Choose `category` from allowed_topics[].category — this is a CLOSED set.",
                "Emit `leaf_suffix` as a short lowercase snake_case slug. Re-use a listed seed suffix when it fits; otherwise invent a stable one.",
                "Leave both `category` and `leaf_suffix` null only when no category fits at all.",
                "Use user_revoked only for revoke.",
                "Use preference_shifted only when the memory type is preference and the user indicates a changed preference.",
                "Allowed intent values: add | supersede_full | revoke | rollback. There is NO 'ignore' intent. If a turn has no fact to extract, simply omit it from the proposals array — do NOT emit a proposal with intent='ignore'.",
                "MANDATORY: every proposal MUST set evidence (verbatim substring), importance (0-1 float), sensitivity (low|medium|high). NEVER null.",
                "MANDATORY: for candidate_type='episodic', event_time MUST be an ISO-8601 datetime. Use the turn timestamp when user says 'today'/'yesterday'. If no date is available, switch to candidate_type='fact' — do not submit episodic with null event_time.",
                "For schedules/rosters binding person→day or person→shift, emit one proposal per binding (e.g. 'Admon is assigned to the Sunday day shift') rather than one aggregate list.",
            ],
        }
        import json

        return json.dumps(payload, ensure_ascii=False, indent=2)

    def _to_update_proposal(self, decision: UpdateDecision, input_obj: UpdateInput) -> UpdateProposal:
        normalized_intent = decision.intent
        topic_id = self._resolve_topic_id(decision)
        keywords = decision.keywords[:8] if decision.keywords else _tokenize(input_obj.user_input)[:8]
        target_hint = TopicQuery(topic_id=topic_id, keywords=keywords)

        if normalized_intent == "ignore":
            self._record_ignore_reason("llm_intent_ignore", decision, input_obj)
            return UpdateProposal(intent="ignore")
        if normalized_intent == "rollback":
            return UpdateProposal(intent="rollback", target_hint=target_hint)
        if normalized_intent == "revoke":
            return UpdateProposal(
                intent="revoke",
                target_hint=target_hint,
                transition_type="user_revoked",
            )

        candidate_type = _resolve_candidate_type(
            explicit_type=decision.candidate_type,
            topic_id=topic_id,
            topic_tree=self.topic_tree,
            text=input_obj.user_input,
        )
        candidate_content = (decision.candidate_content or input_obj.user_input).strip()
        evidence = _validate_evidence(decision.evidence, input_obj.user_input)
        gate_reason = self._write_gate_reason(
            candidate_type=candidate_type,
            origin_role=input_obj.origin_role,
            importance=decision.importance,
            sensitivity=decision.sensitivity,
            evidence=evidence,
        )
        if gate_reason is not None:
            self._record_ignore_reason(gate_reason, decision, input_obj, candidate_type=candidate_type)
            return UpdateProposal(intent="ignore")
        candidate = self._build_candidate(
            content=candidate_content,
            topic_id=topic_id,
            memory_type=candidate_type,
            timestamp=input_obj.timestamp,
            scope=input_obj.scope,
            origin_role=input_obj.origin_role,
            evidence=evidence,
            importance=decision.importance,
            sensitivity=decision.sensitivity,
            ttl_days=decision.ttl_days,
            event_time=decision.event_time,
        )

        if normalized_intent == "supersede_full":
            transition_type = decision.transition_type
            if transition_type not in {"corrected", "preference_shifted"}:
                transition_type = "preference_shifted" if candidate_type == "preference" else "corrected"
            return UpdateProposal(
                intent="supersede_full",
                candidate=candidate,
                target_hint=target_hint,
                transition_type=transition_type,
            )

        return UpdateProposal(intent="add", candidate=candidate)

    def _write_gate_reason(
        self,
        *,
        candidate_type: MemoryType,
        origin_role: OriginRole,
        importance: float | None,
        sensitivity: Sensitivity | None,
        evidence: str | None,
    ) -> str | None:
        """Mirror of _passes_write_gate that returns the rejection reason or None.

        Used by _to_update_proposal so the diagnostic can attribute drops to
        the specific gate condition that fired (high sensitivity, missing
        evidence, score-below-threshold), rather than collapsing every
        rejection into a generic "ignore".
        """
        gate = self.config.write_gate
        if not gate.enabled:
            return None
        if gate.drop_high_sensitivity and sensitivity == "high":
            return "gate_high_sensitivity"
        if gate.require_evidence and not evidence:
            return "gate_missing_evidence"
        if origin_role == "assistant":
            imp = importance if importance is not None else gate.default_importance
            if imp < gate.assistant_min_importance:
                return f"gate_assistant_imp_low(imp={imp:.2f},need>={gate.assistant_min_importance:.2f})"
            return None
        _, _, prior_key = _source_defaults(origin_role)
        confidence = self.config.confidence.prior_table.get(prior_key, 0.5)
        imp = importance if importance is not None else gate.default_importance
        sens_key = sensitivity or "low"
        penalty = gate.sensitivity_penalty.get(sens_key, 0.0)
        score = gate.alpha_confidence * confidence + gate.beta_importance * imp - penalty
        threshold = gate.threshold_by_type.get(candidate_type, 0.55)
        if score < threshold:
            return (
                f"gate_score_low(score={score:.3f},threshold={threshold:.2f},"
                f"imp={imp:.2f},sens={sens_key},type={candidate_type})"
            )
        return None

    def _record_ignore_reason(
        self,
        reason: str,
        decision: UpdateDecision,
        input_obj: UpdateInput,
        *,
        candidate_type: MemoryType | None = None,
    ) -> None:
        """Append a per-proposal ignore reason to the running diagnostic.

        Bounded to the first 20 entries to keep the diagnostic small even
        when an entire 16-turn batch fails the gate.
        """
        diag = self._last_batch_diagnostic
        if diag is None:
            return
        reasons = diag.setdefault("ignore_reasons", [])
        if not isinstance(reasons, list) or len(reasons) >= 20:
            return
        reasons.append({
            "reason": reason,
            "intent": decision.intent,
            "candidate_type": candidate_type or decision.candidate_type,
            "importance": decision.importance,
            "sensitivity": decision.sensitivity,
            "origin_role": input_obj.origin_role,
            "preview": (decision.candidate_content or "")[:80],
        })

    def _passes_write_gate(
        self,
        *,
        candidate_type: MemoryType,
        origin_role: OriginRole,
        importance: float | None,
        sensitivity: Sensitivity | None,
        evidence: str | None,
    ) -> bool:
        gate = self.config.write_gate
        if not gate.enabled:
            return True
        if gate.drop_high_sensitivity and sensitivity == "high":
            return False
        if gate.require_evidence and not evidence:
            return False
        if origin_role == "assistant":
            imp = importance if importance is not None else gate.default_importance
            return imp >= gate.assistant_min_importance
        _, _, prior_key = _source_defaults(origin_role)
        confidence = self.config.confidence.prior_table.get(prior_key, 0.5)
        imp = importance if importance is not None else gate.default_importance
        sens_key = sensitivity or "low"
        penalty = gate.sensitivity_penalty.get(sens_key, 0.0)
        score = gate.alpha_confidence * confidence + gate.beta_importance * imp - penalty
        threshold = gate.threshold_by_type.get(candidate_type, 0.55)
        return score >= threshold

    def _allowed_topics(self) -> list[dict[str, object]]:
        """Return closed-set categories + seed leaf examples for each one.

        The LLM must pick ``category`` from this closed set and invent a
        ``leaf_suffix`` — a short lowercase snake_case slug that specializes
        the category (e.g. category=user.preference + leaf_suffix=books).
        Seed leaves are surfaced as calibration examples so the LLM re-uses
        the same suffix when the case matches an established leaf.

        Compact form: ``default_type`` is dropped (LLM picks type from the
        candidate_type field) and ``seed_leaves`` is omitted when empty so
        no per-call tokens are spent on placeholders.
        """
        payload: list[dict[str, object]] = []
        for category in self.topic_tree.categories():
            seeds = [
                {
                    "leaf_suffix": leaf.path.rsplit(".", 1)[-1],
                    "description": leaf.description,
                }
                for leaf in self.topic_tree.seeds_in_category(category.id)
            ]
            entry: dict[str, object] = {
                "category": category.id,
                "description": category.description,
            }
            if seeds:
                entry["seed_leaves"] = seeds
            payload.append(entry)
        return payload

    def _compact_allowed_topics(self) -> list[dict[str, object]]:
        """Return the smaller topic payload used by batch extraction.

        Compact form mirrors ``_allowed_topics``: ``default_type`` is dropped
        (no prompt instruction references it; ``candidate_type`` is chosen
        independently) and ``seed_leaf_suffixes`` is omitted when empty so no
        per-call tokens are spent on placeholders.
        """
        payload: list[dict[str, object]] = []
        for category in self.topic_tree.categories():
            seed_suffixes = [
                leaf.path.rsplit(".", 1)[-1]
                for leaf in self.topic_tree.seeds_in_category(category.id)
            ]
            entry: dict[str, object] = {"category": category.id}
            if seed_suffixes:
                entry["seed_leaf_suffixes"] = seed_suffixes
            payload.append(entry)
        return payload

    def _resolve_topic_id(self, decision: UpdateDecision) -> str | None:
        """Turn LLM output into a canonical topic_id.

        Three input shapes are accepted, in preference order:

        1. Explicit (category, leaf_suffix) pair — the new format. Routed
           through the canonicalizer (embedding-based dedup) or, if no
           canonicalizer is wired, accepted only when it resolves to a seed.
        2. Legacy ``topic_id`` in full-path form ``category.suffix`` — split
           and routed through the canonicalizer. Lets models that fall back
           to the old field name still land runtime leaves under the right
           bucket instead of getting dropped.
        3. Legacy ``topic_id`` matching a known seed path verbatim.

        Returns None when none of the above yields a valid topic.
        """
        composed = self._canonicalize_pair(decision.category, decision.leaf_suffix)
        if composed is not None:
            return composed

        legacy = decision.topic_id
        if legacy:
            if self.topic_tree.has_topic(legacy):
                return legacy
            category = self.topic_tree.category_for(legacy)
            if category is not None and "." in legacy:
                suffix = legacy.rsplit(".", 1)[-1]
                composed = self._canonicalize_pair(category, suffix)
                if composed is not None:
                    return composed

        if decision.category or decision.leaf_suffix or decision.topic_id:
            logger.warning(
                "topic resolution dropped to None: category=%r leaf_suffix=%r topic_id=%r",
                decision.category,
                decision.leaf_suffix,
                decision.topic_id,
            )
        return None

    def _canonicalize_pair(self, category: str | None, leaf_suffix: str | None) -> str | None:
        effective_category, effective_suffix = self._normalize_topic_parts(category, leaf_suffix)
        if effective_category is None or effective_suffix is None:
            return None
        if self.canonicalizer is not None:
            try:
                resolved = self.canonicalizer(effective_category, effective_suffix)
            except Exception as exc:
                logger.warning(
                    "canonicalizer failed for (%s, %s): %s",
                    effective_category,
                    effective_suffix,
                    exc,
                )
                resolved = None
            if resolved:
                return resolved
        composed = self.topic_tree.compose_leaf_path(effective_category, effective_suffix)
        if composed and self.topic_tree.has_topic(composed):
            return composed
        return None

    def _normalize_topic_parts(
        self,
        category: str | None,
        leaf_suffix: str | None,
    ) -> tuple[str | None, str | None]:
        """Coerce LLM-emitted topic parts into (category, leaf_suffix).

        Handles the common LLM mistake of packing the full leaf path into the
        ``category`` field. Three observed shapes are mapped to the same
        clean pair:

        - category="user.activity.cultural_visit", leaf_suffix="cultural_visit"
          → ("user.activity", "cultural_visit")  # dup → use one copy
        - category="user.possessions.home",       leaf_suffix=None
          → ("user.possessions", "home")         # infer suffix from category
        - category="user.possessions.vehicle",    leaf_suffix="oil_change"
          → ("user.possessions", "vehicle_oil_change")  # preserve both signals

        Returns (None, None) when no known category prefix can be recovered.
        """
        if not category:
            return None, None

        if self.topic_tree.has_category(category):
            return category, leaf_suffix or None

        parts = category.split(".")
        effective_category: str | None = None
        embedded_suffix: str | None = None
        for i in range(len(parts) - 1, 1, -1):
            candidate = ".".join(parts[:i])
            if self.topic_tree.has_category(candidate):
                effective_category = candidate
                embedded_suffix = "_".join(parts[i:])
                break
        if effective_category is None:
            return None, None

        if leaf_suffix and embedded_suffix and leaf_suffix != embedded_suffix:
            effective_suffix: str | None = f"{embedded_suffix}_{leaf_suffix}"
        else:
            effective_suffix = leaf_suffix or embedded_suffix
        return effective_category, effective_suffix or None

    def _nearby_memory_payload(self, memories: list[MemoryEntry]) -> list[dict[str, object]]:
        if self.config.update.nearby_k <= 0:
            return []
        return [
            {
                "id": memory.id,
                "content": memory.summary_for_retrieval or memory.content,
                "type": memory.type,
                "topic_id": memory.topic_id,
                "scope": memory.scope.model_dump(mode="json"),
                "state": memory.state,
            }
            for memory in memories[: self.config.update.nearby_k]
        ]

    def _build_candidate(
        self,
        *,
        content: str,
        topic_id: str | None,
        memory_type: MemoryType,
        timestamp: datetime,
        scope: Scope,
        origin_role: OriginRole = "user",
        evidence: str | None = None,
        importance: float | None = None,
        sensitivity: Sensitivity | None = None,
        ttl_days: int | None = None,
        event_time: datetime | None = None,
    ) -> MemoryEntry:
        src, source_kind, prior_key = _source_defaults(origin_role)
        prior = self.config.confidence.prior_table[prior_key]
        return MemoryEntry(
            content=content,
            type=memory_type,
            topic_id=topic_id,
            scope=scope,
            src=src,
            origin_role=origin_role,
            source_kind=source_kind,
            conf=Confidence(prior=prior, llm_self=None, combined=prior),
            valid_time=TimeRange(start=timestamp),
            ingest_time=timestamp,
            evidence=evidence,
            importance=importance,
            sensitivity=sensitivity,
            ttl_days=ttl_days,
            event_time=event_time,
        )


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(text)]


def _source_defaults(origin_role: OriginRole) -> tuple[MemorySource, SourceKind, str]:
    if origin_role == "assistant":
        return "inferred", "summary", "inferred"
    return "user_explicit", "explicit", "user_explicit"


def _resolve_candidate_type(
    *,
    explicit_type: MemoryType | None,
    topic_id: str | None,
    topic_tree: TopicTree,
    text: str,
) -> MemoryType:
    if explicit_type in _SUPPORTED_MEMORY_TYPES:
        return explicit_type
    if topic_id is not None:
        category = topic_tree.category_for(topic_id)
        if category is not None:
            category_type = cast(MemoryType, topic_tree.default_type_for_category(category))
            if category_type in _SUPPORTED_MEMORY_TYPES:
                return category_type
    return "fact"


def _proposal_is_distinct(proposal: UpdateProposal, existing: list[UpdateProposal]) -> bool:
    if proposal.candidate is None:
        return proposal.intent not in {item.intent for item in existing}
    candidate_content = proposal.candidate.content.strip().lower()
    for item in existing:
        if item.candidate is None:
            continue
        if item.candidate.content.strip().lower() == candidate_content:
            return False
    return True


_UNMAPPED_INTENT_SAMPLES: list[str] = []


_INTENT_SYNONYMS: dict[str, str] = {
    # supersede_full synonyms — most common LLM emissions for "update an existing fact"
    "update": "supersede_full",
    "supersede": "supersede_full",
    "supersedes": "supersede_full",
    "replace": "supersede_full",
    "modify": "supersede_full",
    "edit": "supersede_full",
    "correct": "supersede_full",
    "amend": "supersede_full",
    "overwrite": "supersede_full",
    "change": "supersede_full",
    # add synonyms
    "create": "add",
    "new": "add",
    "insert": "add",
    "store": "add",
    "save": "add",
    "record": "add",
    "remember": "add",
    # revoke synonyms
    "delete": "revoke",
    "remove": "revoke",
    "forget": "revoke",
    "drop": "revoke",
    "erase": "revoke",
    # NOTE: "skip"/"noop"/"pass"/"discard" intentionally NOT mapped to anything.
    # When the LLM emits one of these, _normalize_update_intent falls back to
    # "add" (see below) so the populated candidate is preserved instead of
    # being silently dropped. The LLM is supposed to OMIT proposals it doesn't
    # want stored, not tag them ignore — see _BATCH_UPDATE_SYSTEM_PROMPT.
}


def _normalize_update_intent(value: object) -> str:
    """Coerce LLM-emitted intent strings to one of the supported intents.

    Unknown / non-string / "ignore" / "skip"-style values fall back to "add"
    rather than "ignore" — when the LLM emits a proposal with full candidate
    fields, dropping the candidate (the prior "fall back to ignore" default)
    silently destroys the LLM's work. The LLM is told to OMIT unwanted
    proposals entirely, not to tag them ignore. If the LLM ignores that
    instruction and emits ignore-tagged proposals anyway, "add" preserves
    the candidate so the downstream write_gate can still filter on quality.
    """
    if not isinstance(value, str):
        return "add"
    normalized = _normalize_label(value)
    if normalized in _SUPPORTED_UPDATE_INTENTS:
        return normalized
    if normalized in _INTENT_SYNONYMS:
        return _INTENT_SYNONYMS[normalized]
    if len(_UNMAPPED_INTENT_SAMPLES) < 100 and normalized not in _UNMAPPED_INTENT_SAMPLES:
        _UNMAPPED_INTENT_SAMPLES.append(normalized)
        logger.info("unknown update intent %r coerced to 'add'", normalized)
    return "add"


def _normalize_memory_type(value: object) -> MemoryType | None:
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    normalized = _normalize_label(value)
    return normalized if normalized in _SUPPORTED_MEMORY_TYPES else None


def _normalize_transition_type(value: object) -> TransitionType | None:
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    normalized = _normalize_label(value)
    return normalized if normalized in _SUPPORTED_TRANSITION_TYPES else None


def _normalize_keywords(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return _tokenize(value)[:8]
    if isinstance(value, (list, tuple, set)):
        normalized: list[str] = []
        for item in value:
            text = str(item).strip()
            if text:
                normalized.append(text)
        return normalized[:8]
    return []


def _normalize_update_decision_payload(payload: object) -> dict[str, object]:
    if isinstance(payload, BaseModel):
        payload = payload.model_dump(mode="json")
    if not isinstance(payload, dict):
        return {"intent": "add"}

    sensitivity = payload.get("sensitivity")
    if sensitivity not in _SUPPORTED_SENSITIVITY:
        sensitivity = "low"
    importance = payload.get("importance")
    if not isinstance(importance, (int, float)) or not 0.0 <= float(importance) <= 1.0:
        importance = 0.5
    evidence = payload.get("evidence") if isinstance(payload.get("evidence"), str) else ""
    return {
        "intent": payload.get("intent"),
        "candidate_content": payload.get("candidate_content"),
        "candidate_type": payload.get("candidate_type"),
        "category": payload.get("category"),
        "leaf_suffix": payload.get("leaf_suffix"),
        "topic_id": payload.get("topic_id"),
        "keywords": payload.get("keywords", []),
        "transition_type": payload.get("transition_type"),
        "evidence": evidence,
        "importance": float(importance),
        "sensitivity": sensitivity,
        "ttl_days": payload.get("ttl_days"),
        "event_time": _sanitize_event_time(payload.get("event_time")),
    }


def _sanitize_event_time(value: object) -> object:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return None
    lowered = stripped.lower()
    if lowered in {"null", "none", "n/a", "na", "unknown"} or stripped.startswith("0000"):
        return None
    normalized = stripped.replace("Z", "+00:00")
    try:
        datetime.fromisoformat(normalized)
    except ValueError:
        return None
    return stripped


def _validate_evidence(evidence: str | None, user_input: str) -> str | None:
    if not evidence or not isinstance(evidence, str):
        return None
    stripped = evidence.strip()
    if not stripped:
        return None
    if stripped.lower() in user_input.lower():
        return stripped
    return None


def _normalize_batch_update_decision_payload(payload: object) -> dict[str, object]:
    if isinstance(payload, BaseModel):
        payload = payload.model_dump(mode="json")
    proposals = payload.get("proposals", []) if isinstance(payload, dict) else []
    if not isinstance(proposals, list):
        return {"proposals": []}

    normalized_items = [
        _normalize_batch_update_decision_item(item, fallback_index=index)
        for index, item in enumerate(proposals)
    ]
    return {"proposals": normalized_items}


def _normalize_batch_update_decision_item(item: object, *, fallback_index: int) -> dict[str, object]:
    if isinstance(item, BaseModel):
        item = item.model_dump(mode="json")
    if not isinstance(item, dict):
        return {"source_turn_index": fallback_index, "intent": "add"}
    return {
        "source_turn_index": _coerce_source_turn_index(item.get("source_turn_index"), fallback=fallback_index),
        **_normalize_update_decision_payload(item),
    }


def _normalize_candidate_content(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip()
        return normalized or None
    if isinstance(value, (list, tuple, set)):
        parts: list[str] = []
        for item in value:
            normalized = _normalize_candidate_content(item)
            if normalized:
                parts.append(normalized)
        if not parts:
            return None
        return " ".join(parts)

    normalized = str(value).strip()
    return normalized or None


def _coerce_source_turn_index(value: object, *, fallback: int) -> int:
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            return int(stripped)
    return fallback


def _normalize_label(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


_UPDATE_SYSTEM_PROMPT = """You are the Update Agent for a personal memory system.
Decide whether the user input should add, supersede, revoke, rollback, or ignore memory.
Return structured JSON only.

# Topic assignment (two fields — BOTH required together, or BOTH null)

Every stored proposal is indexed by (`category`, `leaf_suffix`). These are downstream retrieval/conflict keys — be consistent.

- `category` MUST be exactly one of the ids in allowed_topics[].category — e.g. "user.preference" or "user.possessions". Do NOT append the leaf to it: "user.possessions.home" is WRONG (that is a full path, not a category). Never invent a category outside the closed set.
- `leaf_suffix` is a short lowercase snake_case slug (English a-z/0-9/_, 1-3 words, ≤60 chars) naming the specific bucket within the category. It is NOT a full path — do not include dots or the category prefix.
- Re-use an existing seed `leaf_suffix` from allowed_topics[].seed_leaves when the fact matches one. Only invent a new suffix when none fit, and keep it stable across turns (use the SAME suffix for the SAME kind of fact — don't alternate between near-synonyms like "books" / "reading" / "literature").
- If no category fits at all, set BOTH to null. Do not emit one without the other.

Do not emit the legacy `topic_id` field; use `category` + `leaf_suffix` instead.

# Extraction & recall (this agent is a fallback for batch misses — err toward extracting)

When in doubt, EXTRACT. A redundant proposal is far less costly than a missed fact (dedup happens downstream).

#1 failure mode is "First Topic Dominance": a turn has a dominant topic (the question, the main activity) AND a secondary fact introduced as an aside; both get ignored, signal lost. Identify asides by their function — a self-contained side disclosure that does not advance the main topic — not by matching specific marker phrases. Common forms: topic-shift discourse markers, parenthetical disclosures, concrete details that refine a generic prior statement, and updates to prior facts (changed quantity, switched preference, revised plan). Updates are HIGH PRIORITY — they should supersede the older value, not be ignored because the main turn is about something else.

A single turn may contain multiple independent stable facts; the run_many path supports emitting one proposal per fact.

What counts as a stable fact:
- Concrete actions the user has done with named objects/places/people
- Identity / relationship / location / occupation / possessions
- Preferences and constraints that will plausibly matter later
- Plans with concrete anchors (dates, places, people)

What to skip:
- Pure questions with no self-disclosure
- Hypotheticals, wishes, counterfactuals
- Generic small talk
- Third-party facts not anchored to the user's life

Self-check before returning ignore: did the input contain (a) a concrete detail refining a generic prior statement, (b) an updated fact hidden inside a turn about something else, (c) a self-contained aside, or (d) a preference / constraint revealed through a request? If yes to any, return a proposal.

# Required fields (every non-ignore proposal — NEVER null)

- `evidence`: verbatim contiguous substring of user_input. Prefer the shortest snippet that includes the key noun + qualifier. If you cannot find a literal substring, return ignore — fabricated evidence is a hard failure.
- `importance` (0-1): >=0.7 for identity / constraints / long-lived preferences; ~0.5 for habitual preferences and ongoing plans; <0.3 for casual observations. Default 0.5 if unsure.
- `sensitivity`: "high" for health / medical / finance / precise address / anything the user marks private; "medium" for political / religious / relationship status; "low" otherwise.
- `event_time`: REQUIRED when candidate_type == "episodic" — ISO-8601 datetime for when the event happened (not when the user mentioned it). Use the user-stated date if present; otherwise infer from relative phrasing ("last week", "yesterday") using the turn timestamp. If no date anchor exists, use type "fact" instead — never leave event_time null on episodic.
- `ttl_days`: null for stable facts / preferences / constraints; small integer (1-30) only for transient states.

# candidate_content vs evidence (OPPOSITE rules — do not confuse)

- `evidence` = verbatim substring of the user turn. No edits.
- `candidate_content` = REWRITTEN self-contained sentence. Not a copy of the user turn.

Self-containment rules for candidate_content:
1. Resolve pronouns ("her", "him", "it", "that") to their named referents from surrounding conversation. A stored sentence with unresolved pronouns is unfindable later by keyword retrieval.
2. Include the entity / occasion / topic nouns (person name, occasion, place, trip name) even if the user omitted them in chat-implicit context.
3. Preserve proper nouns and concrete descriptors verbatim (names, colors, quantities, brand names).
4. A reader seeing ONLY candidate_content (no surrounding turns) must be able to answer who/what/where/when about the fact.

Anti-pattern: user "I got her a [item]" referring to a previously mentioned recipient → WRONG to store "I got her a [item]" (pronoun unresolved). RIGHT to store "For [recipient]'s [occasion], got a [item]".

# Type selection

Use "episodic" (not "fact") when the memory is a discrete past event with a when/where anchor — visits, workouts, races, errands completed, cultural outings. Use "fact" for stable traits (location, education, possessions). Use "preference" for likes/dislikes. Use "constraint" for hard limits (allergies, schedule bans).

# Compound sentences: prefer the habitual attribute, not the main-clause action

A single turn often qualifies a transient plan with a subordinate clause that actually states a habitual attribute (a recurring time, a usual place, a named activity). This agent may only emit ONE proposal, so when such a pair appears, prefer the habitual attribute as the candidate_content: a stable fact is retrievable long after the plan is stale. Rewrite so only the attribute remains — do not carry vestigial planning framing ("before the meeting", "while waiting") into the stored sentence.
"""

_BATCH_UPDATE_SYSTEM_PROMPT = """You are the Batch Update Agent for a personal memory system.
Extract stable user memories from a chronological batch of turns. Return strict structured JSON only.

# Output policy (READ FIRST)

- Return a JSON object with `proposals: [...]`. The array MAY be empty.
- Allowed `intent`: add | supersede_full | revoke | rollback. NO "ignore" intent — if a turn yields no stable user fact, simply do not emit a proposal for that turn.
- Turn-to-proposal mapping is NOT 1-to-1: a turn with no fact contributes zero items; a turn with three facts contributes three. Each proposal must reference the right `source_turn_index`.
- Skip entirely (do NOT emit a proposal): pure questions, generic small talk, hypotheticals / wishes / counterfactuals, assistant-only facts unrelated to the user, third-party facts not anchored to the user's life.

# Topic fields

- `category` must be one of allowed_topics[].category.
- `leaf_suffix` is a short lowercase snake_case bucket name; reuse allowed_topics[].seed_leaf_suffixes when they fit.
- Set both to null only when no category fits. Do not emit legacy `topic_id`.

# What to extract (recall matters — "First Topic Dominance" is the #1 failure mode)

- Stable facts about the user, preferences, constraints, plans with concrete anchors, discrete past events.
- Updates that replace older facts (changed quantities, switched preferences, revised plans).
- Asides introduced by topic-shift discourse markers — a self-contained side disclosure that does not advance the main topic, identified by function not by exact phrase.
- Multiple independent facts from one turn → separate proposals.

# Required fields per proposal

- `candidate_content`: REWRITTEN self-contained sentence — resolve pronouns to named referents from prior turns, preserve proper nouns / titles / quantities verbatim, include the topic / occasion noun even if the user omitted it in chat-implicit context.
- `evidence`: verbatim substring from the source turn — copy/paste, no edits. If you cannot find a literal anchor, omit the proposal.
- `candidate_type`: fact | preference | constraint | episodic.
- `importance`: 0-1 float; 0.5 if unsure.
- `sensitivity`: low | medium | high.
- `event_time`: ISO-8601 datetime when candidate_type is episodic. If no date anchor exists, use fact instead.

# Intent semantics

- `supersede_full` + `transition_type=corrected` for factual corrections (changed quantity / location / status).
- `supersede_full` + `transition_type=preference_shifted` for changed preferences.
- `revoke` + `transition_type=user_revoked` only when the user asks to remove/forget a memory.

# Cumulative updates (CRITICAL — commonly missed)

When a turn says "added / got N more X", "went N more times", "another N", "now I have ...", combine with any matching prior fact in `nearby_memories` and emit ONE supersede_full proposal whose candidate_content states the NEW TOTAL. Set `transition_type=corrected`. Do NOT emit a separate add proposal that stores only the increment — the old quantity must not survive. If no prior count is available, emit add with the total/increment as stated and let downstream handle it.

Worked pattern — UPDATE hidden in a planning turn:
  Input shape: "I'm planning [trip A]. By the way, [updated fact about prior memory B]."
  Correct: TWO proposals — (a) add for the trip plan, (b) supersede_full transition_type=corrected for fact B, where candidate_content states the NEW value of B.
"""
