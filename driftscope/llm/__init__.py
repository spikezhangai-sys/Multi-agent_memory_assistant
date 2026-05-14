"""LLM client interfaces."""

from driftscope.llm.client import StructuredLLM
from driftscope.llm.deepseek import DeepSeekStructuredLLM
from driftscope.llm.factory import build_conflict_llm, build_structured_llm
from driftscope.llm.mock import RuleBasedConflictLLM
from driftscope.llm.openrouter import OpenRouterStructuredLLM

__all__ = [
    "DeepSeekStructuredLLM",
    "OpenRouterStructuredLLM",
    "RuleBasedConflictLLM",
    "StructuredLLM",
    "build_conflict_llm",
    "build_structured_llm",
]
