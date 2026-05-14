# DriftScope

A versioned personal-assistant memory system for the LongMemEval benchmark.

## Headline

On a LongMemEval subset, DriftScope reaches **0.803** using:

- **LLM:** `gpt-4o-mini`
- **Embeddings:** `BAAI/bge-large-en-v1.5`
- **Replay batch size:** `10`

Same models, same subset, the Mem0 OSS baseline scores in the mid-0.5s — DriftScope wins by **20+ points**.

## Reproduce

```bash
pip install -e '.[mem0-baseline,local-embeddings]'
export OPENROUTER_API_KEY=...

# DriftScope
driftscope-longmemeval <dataset.json> runs/driftscope \
    --backend openrouter --model gpt4omini --batch-size 10

# Mem0 baseline (same LLM + embeddings)
driftscope-longmemeval <dataset.json> runs/mem0 \
    --baseline mem0 --mem0-mode oss \
    --mem0-config driftscope/config/mem0_oss_gpt4omini_qdrant.yaml \
    --backend openrouter --model gpt4omini
```

Both runs emit `predictions.jsonl` with the same `(question_id, hypothesis)` shape; score with the LongMemEval judge.

See `CLAUDE.md` and `DriftScope-MA_program_requirements.md` for architecture and spec.
