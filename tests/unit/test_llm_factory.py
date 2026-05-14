from driftscope.llm.factory import build_conflict_llm
from driftscope.llm.mock import RuleBasedConflictLLM


def test_build_conflict_llm_defaults_to_mock(monkeypatch) -> None:
    monkeypatch.setenv("DRIFTSCOPE_CONFLICT_LLM", "mock")
    llm = build_conflict_llm()
    assert isinstance(llm, RuleBasedConflictLLM)


def test_build_conflict_llm_uses_openrouter_when_requested(monkeypatch) -> None:
    monkeypatch.setenv("DRIFTSCOPE_CONFLICT_LLM", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_MODEL", "gpt4omini")

    llm = build_conflict_llm()
    assert llm.__class__.__name__ == "OpenRouterStructuredLLM"
    assert llm.model == "openai/gpt-4o-mini"


def test_build_conflict_llm_accepts_explicit_model_override(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    llm = build_conflict_llm(backend="openrouter", model="openai/gpt-4o-mini")
    assert llm.__class__.__name__ == "OpenRouterStructuredLLM"
    assert llm.model == "openai/gpt-4o-mini"


def test_build_conflict_llm_defaults_to_deepseek_v4_flash(monkeypatch) -> None:
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    llm = build_conflict_llm(backend="openrouter")

    assert llm.__class__.__name__ == "OpenRouterStructuredLLM"
    assert llm.model == "deepseek/deepseek-v4-flash"


def test_build_conflict_llm_uses_deepseek_official_when_requested(monkeypatch) -> None:
    monkeypatch.setenv("DRIFTSCOPE_CONFLICT_LLM", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseekv4flash")

    llm = build_conflict_llm()

    assert llm.__class__.__name__ == "DeepSeekStructuredLLM"
    assert llm.model == "deepseek-v4-flash"


def test_build_conflict_llm_accepts_deepseek_official_explicit_model(monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")

    llm = build_conflict_llm(backend="deepseek", model="deepseek/deepseek-v4-flash")

    assert llm.__class__.__name__ == "DeepSeekStructuredLLM"
    assert llm.model == "deepseek-v4-flash"
