from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from driftscope.agents.types import ConflictInput, ConflictResolution
from driftscope.core.scope_compat import ScopeRules
from driftscope.core.schema import MemoryType, TransitionType

_ALLOWED_TRANSITIONS: dict[MemoryType, set[TransitionType]] = {
    "fact": {"corrected", "user_revoked"},
    "preference": {"corrected", "preference_shifted", "user_revoked"},
    "constraint": {"corrected", "user_revoked"},
    "episodic": {"corrected", "user_revoked"},
}


class ConflictValidationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    is_valid: bool
    errors: list[str] = Field(default_factory=list)


class ConflictValidator:
    def __init__(self, scope_rules: ScopeRules | None = None) -> None:
        self.scope_rules = scope_rules or ScopeRules.load_default()

    def validate(self, *, input_obj: ConflictInput, resolution: ConflictResolution) -> ConflictValidationResult:
        errors: list[str] = []
        candidate_by_id = {match.memory.id: match.memory for match in input_obj.candidates}

        if resolution.action == "apply_add":
            if input_obj.proposal.intent not in {"add", "supersede_full"}:
                errors.append("apply_add is only valid for add/supersede_full proposals")

        elif resolution.action == "confirm_supersede":
            if input_obj.proposal.intent not in {"add", "supersede_full"}:
                errors.append("confirm_supersede requires add or supersede_full proposal")
            errors.extend(self._validate_target(resolution.target_id, candidate_by_id, input_obj))
            target = candidate_by_id.get(resolution.target_id or "")
            if target is not None:
                if resolution.transition_type not in _ALLOWED_TRANSITIONS[target.type]:
                    errors.append(
                        f"transition_type {resolution.transition_type} is illegal for memory type {target.type}"
                    )

        elif resolution.action == "confirm_revoke":
            if input_obj.proposal.intent != "revoke":
                errors.append("confirm_revoke requires revoke proposal")
            errors.extend(self._validate_target(resolution.target_id, candidate_by_id, input_obj))

        elif resolution.action == "request_clarification":
            if input_obj.proposal.intent == "ignore":
                errors.append("ignore proposal should not reach conflict resolution")

        elif resolution.action == "reject":
            if input_obj.proposal.intent == "add":
                errors.append("add proposal should not be rejected without validation reason")

        return ConflictValidationResult(is_valid=not errors, errors=errors)

    def _validate_target(self, target_id: str | None, candidate_by_id: dict[str, object], input_obj: ConflictInput) -> list[str]:
        errors: list[str] = []
        if target_id is None:
            errors.append("target_id is required")
            return errors

        target = candidate_by_id.get(target_id)
        if target is None:
            errors.append("target_id must be selected from provided candidates")
            return errors

        if target.state != "active":
            errors.append("target memory must be active")
        if not self.scope_rules.can_target(input_obj.scope, target.scope):
            errors.append("target scope is incompatible with current scope")
        if target.origin_role != "user" or target.source_kind != "explicit":
            errors.append("target memory must be a user explicit memory")
        return errors
