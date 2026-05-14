from __future__ import annotations

import os

from dotenv import load_dotenv

from driftscope.config.loader import DriftScopeConfig, load_default_config
from driftscope.llm.deepseek import DeepSeekStructuredLLM
from driftscope.llm.client import StructuredLLM
from driftscope.llm.mock import RuleBasedConflictLLM
from driftscope.llm.openrouter import OpenRouterStructuredLLM


def build_structured_llm(
    *,
    config: DriftScopeConfig | None = None,
    backend: str | None = None,
    model: str | None = None,
) -> StructuredLLM:
    load_dotenv()
    cfg = config or load_default_config()
    resolved_backend = (backend or os.getenv("DRIFTSCOPE_CONFLICT_LLM", "mock")).strip().lower()
    if resolved_backend == "openrouter":
        return OpenRouterStructuredLLM.from_env(config=cfg, model=model)
    if resolved_backend in {"deepseek", "deepseek-official"}:
        return DeepSeekStructuredLLM.from_env(config=cfg, model=model)
    return RuleBasedConflictLLM()


def build_conflict_llm(
    *,
    config: DriftScopeConfig | None = None,
    backend: str | None = None,
    model: str | None = None,
) -> StructuredLLM:
    return build_structured_llm(config=config, backend=backend, model=model)
