# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

DriftScope-MA: a versioned personal-assistant memory system prototype targeting the LongMemEval benchmark. The authoritative spec is `DriftScope-MA_program_requirements.md` (Chinese); it supersedes any older v1–v5 drafts. When spec, code, or tests disagree, the spec prose wins.

## Commands

Install (editable) and run tests:

```bash
pip install -e .
pytest                                          # full suite
pytest tests/unit/test_orchestrator.py          # single file
pytest tests/unit/test_orchestrator.py::test_name -q
```

Run the LongMemEval harness (console script registered in `pyproject.toml`):

```bash
driftscope-longmemeval <dataset.json> <output_dir> [--backend env|mock|openrouter] [--model deepseekv4flash] [--limit N]
# equivalent: python -m driftscope.eval.longmemeval <args>
```

`--backend env` reads `DRIFTSCOPE_CONFLICT_LLM` from `.env` (see `.env.example`). `mock` uses the rule-based stub in `driftscope/llm/mock.py`; `openrouter` hits OpenRouter with strict JSON schema response formatting and requires `OPENROUTER_API_KEY`.
`deepseek` hits DeepSeek's official OpenAI-compatible API and requires `DEEPSEEK_API_KEY`.

Run Mem0 as an external LongMemEval baseline through the same output contract:

```bash
pip install -e '.[mem0-baseline,local-embeddings]'
# .env: OPENROUTER_API_KEY=...
driftscope-longmemeval <dataset.json> <output_dir> --baseline mem0 --mem0-mode oss --mem0-config driftscope/config/mem0_oss_openai_qdrant.yaml --backend openrouter --model deepseekv4flash
driftscope-longmemeval <dataset.json> <output_dir> --baseline mem0 --mem0-mode cloud --mem0-api-key "$MEM0_API_KEY" --backend openrouter --model deepseekv4flash
driftscope-longmemeval <dataset.json> <output_dir> --baseline mem0 --mem0-mode oss --mem0-config driftscope/config/mem0_oss_deepseek_qdrant.yaml --backend deepseek --model deepseekv4flash
```

The Mem0 baseline writes isolated per-question `user_id`s, emits the same `predictions.jsonl` shape (`question_id`, `hypothesis`), and logs Mem0 add/search turns to `turns.jsonl`.

## Architecture

Core data flow is a 4-agent pipeline orchestrated per turn by [TurnProcessor](driftscope/pipeline/orchestrator.py):

1. **UpdateAgent** ([agents/update_agent.py](driftscope/agents/update_agent.py)) — consumes `TurnInput.user_input` plus `nearby_memories` and emits an `UpdateProposal` (intent: write / rollback / ignore).
2. **CandidateSelector** ([agents/candidate_selector.py](driftscope/agents/candidate_selector.py)) — deterministic pre-filter that finds conflict candidates in the MemoryBase for the proposal's scope.
3. **ConflictAgent** ([agents/conflict_agent.py](driftscope/agents/conflict_agent.py)) — validated by [agents/conflict_validator.py](driftscope/agents/conflict_validator.py); produces a `ConflictResolution` (supersede / coexist / revoke / etc). Falls back deterministically when the LLM output fails validation (`used_fallback=True`, errors surfaced in logs).
4. **RetrieverAgent → ResponseAgent** ([agents/retriever_agent.py](driftscope/agents/retriever_agent.py), [agents/response_agent.py](driftscope/agents/response_agent.py)) — run only when the turn has a `query`. Two-stage: scope/time gating in MemoryBase, then scoring in the agent.

`TurnProcessor.process_turn` branches on whether the turn is write-only, query-only, or both. `process_replay_batch` is a batched write-only path that calls `update_agent.run_batch` if available.

State lives in [MemoryBase](driftscope/core/memory_base.py) — a SQLite-backed store (default `:memory:`) keyed by `Scope` and validated via [TopicTree](driftscope/core/topic_tree.py) and [ScopeRules](driftscope/core/scope_compat.py) loaded from YAML in `driftscope/config/`. Transitions (supersede, rollback, revoke) go through [pipeline/transitions.py](driftscope/pipeline/transitions.py); rollbacks require `is_rollback_legal` inside `retention.rollback_window_days`.

Config is Pydantic ([config/loader.py](driftscope/config/loader.py)) loaded from [config/default.yaml](driftscope/config/default.yaml). Keys like `retrieval.top_k`, `conflict_selector.*`, `retention.*`, `llm.*` are read by the agents/selectors; don't hardcode values that already live there.

LLM integration: all agents that need an LLM depend on the `StructuredLLM` interface ([llm/client.py](driftscope/llm/client.py)). [llm/factory.py](driftscope/llm/factory.py) chooses between `RuleBasedConflictLLM` (mock) and `OpenRouterStructuredLLM` based on env. `OpenRouterStructuredLLM.from_env` reads `OPENROUTER_*` vars.

Evaluation: [eval/longmemeval/runner.py](driftscope/eval/longmemeval/runner.py) drives the full pipeline over a LongMemEval dataset; [eval/instrumentation.py](driftscope/eval/instrumentation.py) writes per-turn JSONL logs (update proposal, candidate selection, conflict resolution, retrieval, response) into the output dir — use these logs to debug pipeline decisions.

## Conventions

- Python 3.10+; agents and schemas use Pydantic v2 with `from __future__ import annotations`.
- Agent inputs/outputs are the Pydantic models in [agents/types.py](driftscope/agents/types.py) and [core/schema.py](driftscope/core/schema.py). New agent behavior should extend these rather than passing dicts.
- `tests/unit/helpers.py` holds shared fixtures; prefer reusing them over constructing MemoryBase/TurnInput by hand.



## Never hard code
