from __future__ import annotations

from importlib import resources
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from driftscope.agents.types import CandidateSelectorConfig


class UpdateConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    nearby_k: int = 10
    route_add_through_conflict: bool = True
    batch_token_budget: int = 90000   # cap per-batch input tokens; turns are packed greedily under this. 90k leaves ~30k headroom under gpt-4o-mini's 128k window for output + system prompt + schema.
    batch_per_turn_overhead_tokens: int = 600   # rough per-turn overhead for nearby_memories+metadata (used by token estimator)
    # When False, skip per-turn run_many fallback inside run_batch AND the run()
    # fallback inside run_many. The batch prompt is treated as authoritative.
    # Defaults to False because empirical eval shows the supplemental path
    # duplicates 30-70% of LLM calls with near-identical (often empty) results.
    batch_supplemental_enabled: bool = False


class ConfidenceConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    w_p: float = 0.7
    w_l: float = 0.3
    prior_table: dict[str, float] = Field(
        default_factory=lambda: {
            "user_explicit": 0.9,
            "user_implicit": 0.6,
            "inferred": 0.4,
            "external": 0.5,
        }
    )


class RetentionConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    rollback_window_days: int = 30


class RetrievalConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    top_k: int = 10
    lambda_r: float = 0.5
    lambda_f: float = 0.3
    lambda_c: float = 0.2
    tau_pref: float = 30.0
    tau_episodic: float = 60.0
    tau_fact: float = 90.0
    embedding_backend: str = "mock"
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536
    rrf_k: int = 60
    dense_top_n: int = 30
    sparse_top_n: int = 30
    lambda_time: float = 0.15
    tau_time: float = 7.0
    query_time_parse_enabled: bool = False
    lambda_quoted: float = 0.5
    lambda_person: float = 0.3
    topic_canonicalize_threshold: float = 0.85
    topic_soft_hint_threshold: float = 0.3
    topic_soft_hint_floor: float = 0.5
    topic_sibling_floor: float = 0.25
    topic_query_predict_threshold: float = 0.5
    multi_topic_retrieval_enabled: bool = False
    multi_topic_max_topics: int = 4


class WriteGateConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    enabled: bool = True
    alpha_confidence: float = 0.55
    beta_importance: float = 0.45
    default_importance: float = 0.5
    sensitivity_penalty: dict[str, float] = Field(
        default_factory=lambda: {"low": 0.0, "medium": 0.1, "high": 0.25}
    )
    threshold_by_type: dict[str, float] = Field(
        default_factory=lambda: {
            "fact": 0.55,
            "preference": 0.55,
            "constraint": 0.5,
            "episodic": 0.55,
        }
    )
    drop_high_sensitivity: bool = True
    assistant_min_importance: float = 0.4
    require_evidence: bool = False


class LLMConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    default_model: str = "deepseek/deepseek-v4-flash"
    timeout_sec: int = 30
    max_retries: int = 3


class DriftScopeConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    update: UpdateConfig = Field(default_factory=UpdateConfig)
    confidence: ConfidenceConfig = Field(default_factory=ConfidenceConfig)
    retention: RetentionConfig = Field(default_factory=RetentionConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    conflict_selector: CandidateSelectorConfig = Field(default_factory=CandidateSelectorConfig)
    write_gate: WriteGateConfig = Field(default_factory=WriteGateConfig)


def load_config_from_path(path: str | Path) -> DriftScopeConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return DriftScopeConfig.model_validate(payload)


def load_default_config() -> DriftScopeConfig:
    resource = resources.files("driftscope.config").joinpath("default.yaml")
    return load_config_from_path(Path(resource))
