from __future__ import annotations

import json

from driftscope.agents.base import Agent
from driftscope.agents.conflict_validator import ConflictValidator
from driftscope.agents.types import ConflictAgentResult, ConflictInput, ConflictResolution
from driftscope.llm.client import StructuredLLM


class ConflictAgent(Agent):
    name = "conflict"

    def __init__(self, llm: StructuredLLM, validator: ConflictValidator | None = None) -> None:
        self.llm = llm
        self.validator = validator or ConflictValidator()

    def run(self, input_obj: ConflictInput) -> ConflictAgentResult:
        prompt = self._build_prompt(input_obj)
        try:
            raw = self.llm.generate_structured(
                system_prompt=_SYSTEM_PROMPT,
                user_prompt=prompt,
                response_model=ConflictResolution,
            )
            raw_resolution = self._coerce_resolution(raw, input_obj)
        except Exception as exc:
            healed = self._heal_from_exception(exc, input_obj)
            if healed is not None:
                raw_resolution = healed
            else:
                fallback = self._fallback_resolution(
                    input_obj=input_obj,
                    reason=f"LLM failure: {exc}",
                )
                return ConflictAgentResult(
                    resolution=fallback,
                    used_fallback=True,
                    validation_errors=[str(exc)],
                )

        validation = self.validator.validate(input_obj=input_obj, resolution=raw_resolution)
        if validation.is_valid:
            return ConflictAgentResult(resolution=raw_resolution)

        fallback = self._fallback_resolution(
            input_obj=input_obj,
            reason="; ".join(validation.errors),
        )
        return ConflictAgentResult(
            resolution=fallback,
            raw_resolution=raw_resolution,
            used_fallback=True,
            validation_errors=validation.errors,
        )

    def _build_prompt(self, input_obj: ConflictInput) -> str:
        proposal_payload = input_obj.proposal.model_dump(mode="json", exclude_none=True)
        candidate_payload = [
            {
                "id": match.memory.id,
                "content": match.memory.summary_for_retrieval or match.memory.content,
                "type": match.memory.type,
                "topic_id": match.memory.topic_id,
                "scope": match.memory.scope.model_dump(mode="json"),
                "state": match.memory.state,
                "ingest_time": match.memory.ingest_time.isoformat(),
                "confidence": match.memory.conf.combined,
                "selector_score": match.score,
                "matched_by": match.matched_by,
            }
            for match in input_obj.candidates
        ]
        payload = {
            "proposal": proposal_payload,
            "current_scope": input_obj.scope.model_dump(mode="json"),
            "timestamp": input_obj.timestamp.isoformat(),
            "ambiguous_candidates": input_obj.ambiguous_candidates,
            "candidates": candidate_payload,
            "rules": [
                "Only choose target_id values that appear in candidates.",
                "Do not cross scopes.",
                "If candidates are ambiguous, prefer request_clarification.",
                "If no compatible target exists for supersede_full, prefer apply_add.",
                "Use user_revoked only for revoke decisions.",
            ],
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def _coerce_resolution(self, raw, input_obj: ConflictInput) -> ConflictResolution:
        data = raw if isinstance(raw, dict) else (raw.model_dump() if hasattr(raw, "model_dump") else dict(raw))
        return ConflictResolution.model_validate(self._heal_payload(data, input_obj))

    def _heal_from_exception(self, exc: Exception, input_obj: ConflictInput) -> ConflictResolution | None:
        payload = getattr(exc, "raw_payload", None)
        if not isinstance(payload, dict):
            return None
        try:
            return ConflictResolution.model_validate(self._heal_payload(dict(payload), input_obj))
        except Exception:
            return None

    def _heal_payload(self, data: dict, input_obj: ConflictInput) -> dict:
        action = data.get("action")
        if action == "apply_add":
            data["target_id"] = None
            data["transition_type"] = None
        elif action in {"confirm_supersede", "confirm_revoke"}:
            candidate_ids = {match.memory.id for match in input_obj.candidates}
            target = data.get("target_id")
            if target not in candidate_ids:
                if not target and len(candidate_ids) == 1:
                    data["target_id"] = next(iter(candidate_ids))
                # else: leave invalid so the validator records the error and the caller falls back;
                # a hallucinated id is a signal the LLM was confused, not that we should paper over it.
        elif action == "reject":
            data["target_id"] = None
        return data

    def _fallback_resolution(self, *, input_obj: ConflictInput, reason: str) -> ConflictResolution:
        if input_obj.proposal.intent == "add":
            return ConflictResolution(
                action="apply_add",
                confidence=0.0,
                reason=f"Fallback apply_add: {reason}",
            )
        if input_obj.proposal.intent == "supersede_full" and not input_obj.candidates:
            return ConflictResolution(
                action="apply_add",
                confidence=0.0,
                reason=f"Fallback apply_add due to empty candidate set: {reason}",
            )
        return ConflictResolution(
            action="request_clarification",
            confidence=0.0,
            reason=f"Fallback clarification: {reason}",
            clarification_question="你是想覆盖之前那条记忆，还是新增一条新的记忆？",
        )


_SYSTEM_PROMPT = """You resolve memory conflicts. Return ONLY structured output.

Hard field rules (the schema will reject violations):
- action=apply_add            -> target_id MUST be null, transition_type MUST be null.
- action=confirm_supersede    -> target_id MUST be one of the provided candidate ids, transition_type MUST be set.
- action=confirm_revoke       -> target_id MUST be one of the provided candidate ids, transition_type MUST be "user_revoked".
- action=request_clarification -> clarification_question MUST be non-empty.
- action=reject               -> target_id MUST be null.

When to choose confirm_supersede (CRITICAL — drives knowledge-update correctness):
- Pick confirm_supersede when the proposal restates the SAME ATTRIBUTE as an existing candidate with a DIFFERENT VALUE — different quantity, different count, different location, different status, different preference. The old value must NOT survive alongside the new one.
- Quantity / count updates ("17 postcards" -> "25 postcards", "3 sessions" -> "5 sessions", "$350,000 pre-approval" -> "$400,000 pre-approval") MUST supersede. Coexistence here is wrong.
- Choose apply_add only when proposal and candidates describe ORTHOGONAL facets that legitimately coexist (e.g., "owns a Honda Civic" plus "owns a Specialized bike"; "loves Italian food" plus "loves Thai food"). Different attributes of the user's life — not different values of the same attribute.

transition_type for confirm_supersede:
- "corrected"          -> factual updates: quantities, counts, locations, statuses, identity facts, plans (the most common case for KU updates).
- "preference_shifted" -> use ONLY when both candidate.type=preference and the user has changed their preference (e.g., switched favorite brand).
- "user_revoked"       -> NEVER use for supersede; reserved for confirm_revoke.

Decision heuristics:
- Never invent a target_id. If no candidate matches, use apply_add (for add/supersede_full) or request_clarification.
- If multiple candidates are plausible AND the proposal does NOT clearly target one of them, prefer request_clarification over guessing.
- If multiple candidates are plausible but ONE is clearly the same-attribute prior value, pick that one for confirm_supersede; do not bail to clarification just because other candidates exist.
- Do not cross scopes.
"""

