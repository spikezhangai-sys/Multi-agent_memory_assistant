from __future__ import annotations

import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable

from driftscope.agents.conflict_agent import ConflictAgent
from driftscope.agents.response_agent import LLMResponseAgent
from driftscope.agents.retriever_agent import HybridRetrieverAgent
from driftscope.agents.topic_predictor import LLMTopicPredictor
from driftscope.agents.update_agent import LLMUpdateAgent
from driftscope.config.loader import load_config_from_path, load_default_config
from driftscope.core.memory_base import MemoryBase
from driftscope.embeddings import build_embedder
from driftscope.eval.instrumentation import JsonlTurnLogger
from driftscope.eval.longmemeval.adapter import LongMemEvalAdapter
from driftscope.eval.longmemeval.metrics import build_summary_from_turn_logs
from driftscope.llm import build_structured_llm
from driftscope.pipeline.orchestrator import TurnProcessor
from driftscope.retrieval.rule_time_parser import RuleBasedQueryTimeParser


@dataclass(frozen=True)
class RunArtifacts:
    predictions_path: Path
    turns_path: Path
    summary_path: Path


@dataclass(frozen=True)
class RunSummary:
    num_instances: int
    num_predictions: int
    predictions_path: Path
    turns_path: Path
    summary_path: Path


class LongMemEvalRunner:
    def __init__(
        self,
        *,
        adapter: LongMemEvalAdapter | None = None,
        processor_factory: Callable[[Path, str | None], TurnProcessor] | None = None,
        llm_backend: str | None = None,
        llm_model: str | None = None,
        replay_batch_size: int = 16,
        include_assistant_turns: bool = True,
        workers: int = 1,
        persist_memory: bool = True,
        config_path: str | Path | None = None,
    ) -> None:
        self.include_assistant_turns = include_assistant_turns
        self.adapter = adapter or LongMemEvalAdapter(user_turns_only=not include_assistant_turns)
        self.processor_factory = processor_factory or self._default_processor_factory
        self.llm_backend = llm_backend
        self.llm_model = llm_model
        self.replay_batch_size = max(1, replay_batch_size)
        self.workers = max(1, workers)
        self.persist_memory = persist_memory
        self.config_path = config_path

    def run(
        self,
        dataset_path: str | Path,
        output_dir: str | Path,
        *,
        limit: int | None = None,
    ) -> RunSummary:
        output_root = Path(output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        artifacts = RunArtifacts(
            predictions_path=output_root / "predictions.jsonl",
            turns_path=output_root / "turns.jsonl",
            summary_path=output_root / "summary.json",
        )
        artifacts.predictions_path.write_text("", encoding="utf-8")
        artifacts.turns_path.write_text("", encoding="utf-8")

        db_dir: Path | None = None
        if self.persist_memory:
            db_dir = output_root / "db"
            if db_dir.exists():
                shutil.rmtree(db_dir)
            db_dir.mkdir(parents=True, exist_ok=True)

        instances = self.adapter.load(dataset_path, limit=limit)

        def _process_one(instance):
            db_path = str(db_dir / f"{instance.question_id}.db") if db_dir is not None else None
            processor = self.processor_factory(artifacts.turns_path, db_path)
            self._process_replay_turns(processor, instance.replay_turns)
            result = processor.process_turn(instance.question_turn)
            return {
                "question_id": instance.question_id,
                "hypothesis": result.answer or "",
            }

        with artifacts.predictions_path.open("a", encoding="utf-8") as handle:
            if self.workers <= 1:
                for instance in instances:
                    payload = _process_one(instance)
                    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                    handle.flush()
            else:
                with ThreadPoolExecutor(max_workers=self.workers) as pool:
                    for payload in pool.map(_process_one, instances):
                        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                        handle.flush()

        summary = build_summary_from_turn_logs(artifacts.turns_path, num_questions=len(instances))
        artifacts.summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return RunSummary(
            num_instances=len(instances),
            num_predictions=len(instances),
            predictions_path=artifacts.predictions_path,
            turns_path=artifacts.turns_path,
            summary_path=artifacts.summary_path,
        )

    def _default_processor_factory(self, turns_path: Path, db_path: str | None) -> TurnProcessor:
        config = (
            load_config_from_path(self.config_path)
            if self.config_path
            else load_default_config()
        )
        embedder = build_embedder(
            backend=config.retrieval.embedding_backend,
            model=config.retrieval.embedding_model,
        )
        memory_base = MemoryBase(db_path=db_path or ":memory:", embedder=embedder)
        llm = build_structured_llm(backend=self.llm_backend, model=self.llm_model)
        query_time_parser = (
            RuleBasedQueryTimeParser() if config.retrieval.query_time_parse_enabled else None
        )
        topic_predictor = (
            LLMTopicPredictor(llm, max_topics=config.retrieval.multi_topic_max_topics)
            if config.retrieval.multi_topic_retrieval_enabled
            else None
        )
        retriever = HybridRetrieverAgent(
            memory_base=memory_base,
            config=config,
            embedder=embedder,
            query_time_parser=query_time_parser,
            topic_predictor=topic_predictor,
        )
        return TurnProcessor(
            memory_base=memory_base,
            update_agent=LLMUpdateAgent(
                llm,
                topic_tree=memory_base.topic_tree,
                canonicalizer=memory_base.canonicalize_topic,
            ),
            conflict_agent=ConflictAgent(llm),
            retriever_agent=retriever,
            response_agent=LLMResponseAgent(llm),
            turn_logger=JsonlTurnLogger(str(turns_path)),
            ingest_assistant_turns=self.include_assistant_turns,
            config=config,
        )

    def _process_replay_turns(self, processor: TurnProcessor, replay_turns) -> None:
        """Pack replay turns into batches under a token budget.

        The fixed-count batching that came before this could pile a 19k-token
        Wikipedia-paste turn into the same prompt as 15 normal turns and blow
        past the model's context window — which silently degraded UpdateAgent
        attention even when it didn't crash. Now: greedy pack until adding the
        next turn would exceed `batch_token_budget`; a single turn that alone
        exceeds the budget gets its own batch (still attempted, may fail at
        the LLM, but at least isolated).

        `replay_batch_size` remains as a hard upper bound on count, to avoid
        accumulating hundreds of trivially-short turns into one batch.
        """
        budget = processor.config.update.batch_token_budget
        per_turn_overhead = processor.config.update.batch_per_turn_overhead_tokens
        max_count = self.replay_batch_size

        buffer: list = []
        buffer_tokens = 0

        def _estimate_turn_tokens(turn) -> int:
            text = turn.user_input or ""
            return len(text) // 4 + per_turn_overhead

        for turn in replay_turns:
            turn_tokens = _estimate_turn_tokens(turn)
            would_overflow_budget = buffer and (buffer_tokens + turn_tokens > budget)
            would_overflow_count = len(buffer) >= max_count
            if would_overflow_budget or would_overflow_count:
                processor.process_replay_batch(buffer)
                buffer = []
                buffer_tokens = 0
            buffer.append(turn)
            buffer_tokens += turn_tokens
        if buffer:
            processor.process_replay_batch(buffer)


if __name__ == "__main__":
    from driftscope.eval.longmemeval.cli import main

    raise SystemExit(main())
