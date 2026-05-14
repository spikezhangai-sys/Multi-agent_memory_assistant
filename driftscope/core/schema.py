from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

TopicID = str

ScopeKind = Literal["global", "personal", "project", "session"]
MemoryType = Literal["fact", "preference", "constraint", "episodic", "raw_session"]
Sensitivity = Literal["low", "medium", "high"]
MemoryState = Literal["active", "superseded", "revoked"]
MemorySource = Literal["user_explicit", "user_implicit", "inferred", "external"]
OriginRole = Literal["user", "assistant"]
SourceKind = Literal["explicit", "derived", "summary"]
TransitionType = Literal["corrected", "preference_shifted", "user_revoked"]


class Scope(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: ScopeKind
    ref: str | None = None

    @model_validator(mode="after")
    def validate_ref(self) -> "Scope":
        if self.kind in {"global", "personal"} and self.ref is not None:
            raise ValueError("global/personal scopes must not set ref")
        if self.kind in {"project", "session"} and not self.ref:
            raise ValueError("project/session scopes require a non-empty ref")
        return self

    def to_key(self) -> tuple[str, str | None]:
        return self.kind, self.ref


class Confidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prior: float
    llm_self: float | None = None
    combined: float

    @field_validator("prior", "llm_self", "combined")
    @classmethod
    def validate_score_range(cls, value: float | None) -> float | None:
        if value is None:
            return value
        if not 0.0 <= value <= 1.0:
            raise ValueError("confidence values must be within [0, 1]")
        return value


class TimeRange(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    start: datetime
    end: datetime | None = None

    @model_validator(mode="after")
    def validate_order(self) -> "TimeRange":
        if self.end is not None and self.start > self.end:
            raise ValueError("start must be <= end")
        return self

    def covers(self, timestamp: datetime) -> bool:
        if timestamp < self.start:
            return False
        if self.end is None:
            return True
        return timestamp <= self.end


class SupersedeLink(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target: str
    mode: Literal["full"] = "full"
    transition_type: TransitionType

    @field_validator("target")
    @classmethod
    def validate_target(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("target must be non-empty")
        return value


class MemoryEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: str(uuid4()))
    content: str
    type: MemoryType
    topic_id: TopicID | None = None
    scope: Scope
    src: MemorySource
    origin_role: OriginRole = "user"
    source_kind: SourceKind = "explicit"
    conf: Confidence
    valid_time: TimeRange
    ingest_time: datetime
    state: MemoryState = "active"
    revoked_at: datetime | None = None
    supersedes: list[SupersedeLink] = Field(default_factory=list)
    sensitive: bool = False
    summary_for_retrieval: str | None = None
    event_time: datetime | None = None
    evidence: str | None = None
    importance: float | None = None
    sensitivity: Sensitivity | None = None
    ttl_days: int | None = None

    @field_validator("importance")
    @classmethod
    def validate_importance(cls, value: float | None) -> float | None:
        if value is not None and not 0.0 <= value <= 1.0:
            raise ValueError("importance must be within [0, 1]")
        return value

    @field_validator("ttl_days")
    @classmethod
    def validate_ttl_days(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError("ttl_days must be positive")
        return value

    @field_validator("id", "content")
    @classmethod
    def validate_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("field must be non-empty")
        return value

    @model_validator(mode="after")
    def validate_memory_state(self) -> "MemoryEntry":
        if self.sensitive and not self.summary_for_retrieval:
            raise ValueError("sensitive memories require summary_for_retrieval")
        if not self.sensitive and self.summary_for_retrieval is not None:
            raise ValueError("summary_for_retrieval is only allowed for sensitive memories")
        if self.state == "revoked" and self.revoked_at is None:
            raise ValueError("revoked memories require revoked_at")
        if self.state != "revoked" and self.revoked_at is not None:
            raise ValueError("only revoked memories may set revoked_at")
        return self


class TopicQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    topic_id: TopicID | None = None
    keywords: list[str] = Field(default_factory=list)


class ScoredMemory(BaseModel):
    model_config = ConfigDict(extra="forbid")

    memory: MemoryEntry
    score: float
    score_breakdown: dict[str, float]
    gating_trace: dict[str, str]


class TurnInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scope: Scope
    timestamp: datetime
    origin_role: OriginRole = "user"
    user_input: str | None = None
    query: str | None = None

    @model_validator(mode="after")
    def validate_payload(self) -> "TurnInput":
        if not self.user_input and not self.query:
            raise ValueError("at least one of user_input/query must be provided")
        return self


class TurnResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str | None = None
    cited_memory_ids: list[str] = Field(default_factory=list)
    context_only_ids: list[str] = Field(default_factory=list)
    agents_called: list[str] = Field(default_factory=list)
    write_applied: bool = False
    write_only: bool = False
    query_only: bool = False
    abstained: bool = False
    errors: list[str] = Field(default_factory=list)
