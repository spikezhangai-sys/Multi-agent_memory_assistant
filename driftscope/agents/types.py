from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from driftscope.core.schema import MemoryEntry, OriginRole, Scope, TopicQuery, TransitionType


class UpdateInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_input: str
    origin_role: OriginRole = "user"
    scope: Scope
    timestamp: datetime
    nearby_memories: list[MemoryEntry] = Field(default_factory=list)


class UpdateProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent: Literal["add", "supersede_full", "revoke", "rollback", "ignore"]
    candidate: MemoryEntry | None = None
    target_hint: TopicQuery | None = None
    transition_type: TransitionType | None = None

    @model_validator(mode="after")
    def validate_intent_shape(self) -> "UpdateProposal":
        if self.intent == "ignore":
            if any([self.candidate, self.target_hint, self.transition_type]):
                raise ValueError("ignore intent must not carry candidate/target/transition")
        elif self.intent == "add":
            if self.candidate is None:
                raise ValueError("add intent requires candidate")
            if self.target_hint is not None or self.transition_type is not None:
                raise ValueError("add intent must not carry target_hint/transition_type")
        elif self.intent == "supersede_full":
            if self.candidate is None or self.target_hint is None or self.transition_type is None:
                raise ValueError("supersede_full requires candidate, target_hint, transition_type")
        elif self.intent == "revoke":
            if self.candidate is not None or self.target_hint is None or self.transition_type != "user_revoked":
                raise ValueError("revoke requires target_hint and user_revoked transition")
        elif self.intent == "rollback":
            if self.candidate is not None or self.target_hint is None or self.transition_type is not None:
                raise ValueError("rollback requires target_hint only")
        return self


class IndexedUpdateProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_turn_index: int
    proposal: UpdateProposal

    @field_validator("source_turn_index")
    @classmethod
    def validate_source_turn_index(cls, value: int) -> int:
        if value < 0:
            raise ValueError("source_turn_index must be >= 0")
        return value


class CandidateMatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    memory: MemoryEntry
    score: float
    score_breakdown: dict[str, float] = Field(default_factory=dict)
    matched_by: list[str] = Field(default_factory=list)

    @field_validator("score")
    @classmethod
    def validate_score(cls, value: float) -> float:
        if value < 0:
            raise ValueError("score must be >= 0")
        return value


class CandidateSelectorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    top_k: int = 5
    revoke_top_k: int = 3
    min_score: float = 0.40
    revoke_min_score: float = 0.25
    ambiguity_margin: float = 0.08
    content_sim_weight: float = 0.5
    keyword_overlap_weight: float = 0.2
    time_proximity_weight: float = 0.2
    confidence_weight: float = 0.1


class CandidateSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidates: list[CandidateMatch] = Field(default_factory=list)
    ambiguous_candidates: bool = False


class ConflictInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    proposal: UpdateProposal
    scope: Scope
    timestamp: datetime
    candidates: list[CandidateMatch] = Field(default_factory=list)
    ambiguous_candidates: bool = False


class ConflictResolution(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal[
        "apply_add",
        "confirm_supersede",
        "confirm_revoke",
        "request_clarification",
        "reject",
    ]
    target_id: str | None = None
    transition_type: TransitionType | None = None
    confidence: float
    reason: str
    clarification_question: str | None = None

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("confidence must be within [0, 1]")
        return value

    @model_validator(mode="after")
    def validate_shape(self) -> "ConflictResolution":
        if self.action == "apply_add":
            if self.target_id is not None or self.transition_type is not None:
                raise ValueError("apply_add must not set target_id/transition_type")
        elif self.action == "confirm_supersede":
            if self.target_id is None or self.transition_type is None:
                raise ValueError("confirm_supersede requires target_id and transition_type")
        elif self.action == "confirm_revoke":
            if self.target_id is None or self.transition_type != "user_revoked":
                raise ValueError("confirm_revoke requires target_id and user_revoked")
        elif self.action == "request_clarification":
            if not self.clarification_question:
                raise ValueError("request_clarification requires clarification_question")
        elif self.action == "reject":
            if self.target_id is not None:
                raise ValueError("reject must not set target_id")
        return self


class ConflictAgentResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    resolution: ConflictResolution
    raw_resolution: ConflictResolution | None = None
    used_fallback: bool = False
    validation_errors: list[str] = Field(default_factory=list)


class RetrievalInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    scope: Scope
    timestamp: datetime
    allow_sensitive_raw: bool = False


class RetrievalResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ranked_memories: list[CandidateMatch] = Field(default_factory=list)
    injected_constraints: list[MemoryEntry] = Field(default_factory=list)
    gating_stats: dict[str, int] = Field(default_factory=dict)
    predicted_topic: str | None = None


class ResponseInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    retrieval: RetrievalResult
    allow_sensitive_raw: bool = False


class ResponseOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str
    cited_memory_ids: list[str] = Field(default_factory=list)
    context_only_ids: list[str] = Field(default_factory=list)
    abstained: bool = False
    abstain_reason: str | None = None
