from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Callable, Literal, Protocol

from dotenv import load_dotenv
from pydantic import BaseModel
import yaml

from driftscope.core.schema import TurnInput, TurnResult
from driftscope.eval.instrumentation import JsonlTurnLogger
from driftscope.eval.longmemeval.adapter import LongMemEvalAdapter, LongMemEvalInstance
from driftscope.eval.longmemeval.metrics import build_summary_from_turn_logs
from driftscope.eval.longmemeval.runner import RunArtifacts, RunSummary
from driftscope.llm import build_structured_llm
from driftscope.llm.client import StructuredLLM


class Mem0Client(Protocol):
    def add(self, messages: Any, **kwargs: Any) -> Any:
        ...

    def search(self, query: str, **kwargs: Any) -> Any:
        ...

    def delete_all(self, **kwargs: Any) -> Any:
        ...


class Mem0AnswerDecision(BaseModel):
    answer: str


@dataclass(frozen=True)
class Mem0BaselineConfig:
    mode: Literal["oss", "cloud"] = "oss"
    config_path: Path | None = None
    api_key: str | None = None
    top_k: int = 20
    threshold: float = 0.1
    rerank: bool = False
    infer: bool = True
    answer_mode: Literal["llm", "concat"] = "llm"
    batch_size: int = 16
    user_prefix: str = "driftscope_lme_mem0"
    run_id: str | None = "longmemeval"
    clean_start: bool = True
    llm_backend: str | None = None
    llm_model: str | None = None


class Mem0BaselineRunner:
    """Run LongMemEval through Mem0 as an external memory baseline."""

    def __init__(
        self,
        *,
        adapter: LongMemEvalAdapter | None = None,
        client_factory: Callable[[], Mem0Client] | None = None,
        config: Mem0BaselineConfig | None = None,
        workers: int = 1,
    ) -> None:
        self.adapter = adapter or LongMemEvalAdapter()
        self.config = config or Mem0BaselineConfig()
        self.client_factory = client_factory or self._default_client_factory
        self.workers = max(1, workers)
        self.llm: StructuredLLM | None = None

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

        instances = self.adapter.load(dataset_path, limit=limit)
        shared_client = self.client_factory() if self.workers <= 1 else None
        if self.config.answer_mode == "llm":
            self.llm = build_structured_llm(
                backend=self.config.llm_backend,
                model=self.config.llm_model,
            )

        def _process_one(instance: LongMemEvalInstance) -> dict[str, str]:
            client = shared_client or self.client_factory()
            logger = JsonlTurnLogger(str(artifacts.turns_path))
            return self._process_instance(client, logger, instance)

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
        summary.update(
            {
                "baseline": "mem0",
                "mem0_mode": self.config.mode,
                "mem0_top_k": self.config.top_k,
                "mem0_threshold": self.config.threshold,
                "mem0_rerank": self.config.rerank,
                "mem0_infer": self.config.infer,
                "mem0_answer_mode": self.config.answer_mode,
            }
        )
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

    def _process_instance(
        self,
        client: Mem0Client,
        logger: JsonlTurnLogger,
        instance: LongMemEvalInstance,
    ) -> dict[str, str]:
        filters = self._filters_for_instance(instance)
        if self.config.clean_start:
            self._delete_existing_scope(client, filters)

        for batch_index, turns in enumerate(_chunks(instance.replay_turns, self.config.batch_size)):
            messages = [
                {"role": turn.origin_role, "content": turn.user_input or ""}
                for turn in turns
                if turn.user_input
            ]
            if not messages:
                continue
            metadata = {
                "baseline": "mem0",
                "question_id": instance.question_id,
                "batch_index": batch_index,
                "start_timestamp": turns[0].timestamp.isoformat(),
                "end_timestamp": turns[-1].timestamp.isoformat(),
            }
            add_response = self._add(client, messages, filters=filters, metadata=metadata)
            add_count = len(_extract_results(add_response))
            for turn in turns:
                logger.log_turn(
                    turn,
                    TurnResult(
                        agents_called=["mem0_add"],
                        write_applied=True,
                        write_only=True,
                    ),
                    extras={
                        "baseline": "mem0",
                        "user_id": filters["user_id"],
                        "run_id": filters.get("run_id"),
                        "batch_index": batch_index,
                        "add_result_count": add_count,
                    },
                )

        raw_search = self._search(client, instance.question, filters=filters)
        mem0_results = _normalize_results(raw_search)
        answer, cited_ids, context_only_ids, abstained, errors = self._answer(
            query=instance.question,
            mem0_results=mem0_results,
        )
        logger.log_turn(
            instance.question_turn,
            TurnResult(
                answer=answer,
                cited_memory_ids=cited_ids,
                context_only_ids=context_only_ids,
                agents_called=["mem0_search"] + (["response"] if self.config.answer_mode == "llm" else []),
                query_only=True,
                abstained=abstained,
                errors=errors,
            ),
            extras={
                "baseline": "mem0",
                "user_id": filters["user_id"],
                "run_id": filters.get("run_id"),
                "mem0_result_count": len(mem0_results),
                "mem0_results": [
                    {
                        "id": item["id"],
                        "memory": item["memory"],
                        "score": item["score"],
                    }
                    for item in mem0_results
                ],
            },
        )
        return {
            "question_id": instance.question_id,
            "hypothesis": answer,
        }

    def _default_client_factory(self) -> Mem0Client:
        load_dotenv(dotenv_path=Path.cwd() / ".env")
        os.environ.setdefault("MEM0_DIR", "/tmp/driftscope-mem0")
        try:
            if self.config.mode == "cloud":
                from mem0 import MemoryClient

                return MemoryClient(api_key=self.config.api_key)

            from mem0 import Memory

            if self.config.config_path is not None:
                return Memory.from_config(_load_config_file(self.config.config_path))
            return Memory()
        except ImportError as exc:
            raise RuntimeError(
                "Mem0 baseline requires the mem0ai package. Install it with "
                "`pip install mem0ai`, or install the local mem0 checkout with `pip install -e ./mem0`."
            ) from exc

    def _filters_for_instance(self, instance: LongMemEvalInstance) -> dict[str, str]:
        filters = {"user_id": _scoped_user_id(self.config.user_prefix, instance.question_id)}
        if self.config.run_id:
            filters["run_id"] = self.config.run_id
        return filters

    def _add(
        self,
        client: Mem0Client,
        messages: list[dict[str, str]],
        *,
        filters: dict[str, str],
        metadata: dict[str, Any],
    ) -> Any:
        top_level_kwargs = {
            **filters,
            "metadata": metadata,
            "infer": self.config.infer,
        }
        filter_kwargs = {
            "filters": filters,
            "metadata": metadata,
            "infer": self.config.infer,
        }
        return _call_with_fallbacks(
            lambda kwargs: client.add(messages, **kwargs),
            [top_level_kwargs, filter_kwargs],
        )

    def _search(self, client: Mem0Client, query: str, *, filters: dict[str, str]) -> Any:
        filter_kwargs = {
            "filters": filters,
            "top_k": self.config.top_k,
            "threshold": self.config.threshold,
            "rerank": self.config.rerank,
        }
        top_level_kwargs = {
            **filters,
            "top_k": self.config.top_k,
            "threshold": self.config.threshold,
            "rerank": self.config.rerank,
        }
        return _call_with_fallbacks(
            lambda kwargs: client.search(query, **kwargs),
            [filter_kwargs, top_level_kwargs],
        )

    def _delete_existing_scope(self, client: Mem0Client, filters: dict[str, str]) -> None:
        top_level_kwargs = dict(filters)
        filter_kwargs = {"filters": filters}
        last_error: Exception | None = None
        for kwargs in (top_level_kwargs, filter_kwargs):
            try:
                client.delete_all(**{key: value for key, value in kwargs.items() if value is not None})
                return
            except (TypeError, ValueError) as exc:
                if _looks_like_empty_scope(exc):
                    return
                last_error = exc
            except Exception as exc:
                if _looks_like_empty_scope(exc):
                    return
                raise
        if last_error is not None:
            raise last_error

    def _answer(
        self,
        *,
        query: str,
        mem0_results: list[dict[str, Any]],
    ) -> tuple[str, list[str], list[str], bool, list[str]]:
        if not mem0_results:
            return "我目前没有足够信息来回答这个问题。", [], [], True, []

        if self.config.answer_mode == "concat":
            return _concat_answer(mem0_results), [mem0_results[0]["id"]], [
                item["id"] for item in mem0_results[1:]
            ], False, []

        evidence = [
            {
                "id": item["id"],
                "kind": "ranked",
                "content": item["memory"],
                "type": "mem0",
                "score": item["score"],
                "event_time": item.get("metadata", {}).get("timestamp"),
            }
            for item in mem0_results
            if item["memory"]
        ]
        if not evidence:
            return "我目前没有足够信息来回答这个问题。", [], [], True, []

        prompt = json.dumps(
            {
                "query": query,
                "evidence": evidence,
                "rules": [
                    "Answer using only the provided Mem0 search results.",
                    "If the evidence does not answer the question, abstain.",
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        try:
            if self.llm is None:
                self.llm = build_structured_llm(
                    backend=self.config.llm_backend,
                    model=self.config.llm_model,
                )
            generate_text = getattr(self.llm, "generate_text", None)
            if callable(generate_text):
                answer = generate_text(
                    system_prompt=_MEM0_RESPONSE_SYSTEM_PROMPT,
                    user_prompt=prompt,
                )
            else:
                decision = self.llm.generate_structured(
                    system_prompt=_MEM0_RESPONSE_SYSTEM_PROMPT_STRUCTURED,
                    user_prompt=prompt,
                    response_model=Mem0AnswerDecision,
                )
                answer = _stringify_mem0_llm_output(decision)
        except Exception as exc:
            return (
                _concat_answer(mem0_results),
                [mem0_results[0]["id"]],
                [item["id"] for item in mem0_results[1:]],
                False,
                [f"mem0_response_fallback: {exc}"],
            )

        cited = [evidence[0]["id"]]
        context_only = [item["id"] for item in evidence[1:]]
        return answer.strip(), cited, context_only, False, []


def _load_config_file(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        payload = yaml.safe_load(text)
    else:
        payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("Mem0 config file must contain a JSON/YAML object")
    return payload


def _call_with_fallbacks(call: Callable[[dict[str, Any]], Any], attempts: list[dict[str, Any]]) -> Any:
    last_error: Exception | None = None
    for kwargs in attempts:
        try:
            return call({key: value for key, value in kwargs.items() if value is not None})
        except (TypeError, ValueError) as exc:
            last_error = exc
            continue
    if last_error is not None:
        raise last_error
    raise RuntimeError("no Mem0 call attempts were provided")


def _extract_results(payload: Any) -> list[Any]:
    if payload is None:
        return []
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("results", "memories", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
        return []
    return []


def _normalize_results(payload: Any) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(_extract_results(payload)):
        text = _memory_text(item)
        if not text:
            continue
        normalized.append(
            {
                "id": _memory_id(item, text=text, index=index),
                "memory": text,
                "score": _memory_score(item),
                "metadata": _memory_metadata(item),
                "raw": item,
            }
        )
    return normalized


def _memory_text(item: Any) -> str:
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, dict):
        value = item.get("memory") or item.get("text") or item.get("content") or item.get("value")
        if isinstance(value, dict):
            value = value.get("memory") or value.get("text") or value.get("content")
        return str(value or "").strip()
    value = getattr(item, "memory", None) or getattr(item, "text", None) or getattr(item, "content", None)
    return str(value or "").strip()


def _memory_id(item: Any, *, text: str, index: int) -> str:
    if isinstance(item, dict):
        raw_id = item.get("id") or item.get("memory_id")
    else:
        raw_id = getattr(item, "id", None) or getattr(item, "memory_id", None)
    if raw_id:
        return str(raw_id)
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    return f"mem0_{index}_{digest}"


def _memory_score(item: Any) -> float:
    value: Any
    if isinstance(item, dict):
        value = item.get("score") or item.get("similarity") or item.get("relevance")
    else:
        value = getattr(item, "score", None)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _memory_metadata(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        metadata = item.get("metadata")
        return metadata if isinstance(metadata, dict) else {}
    metadata = getattr(item, "metadata", None)
    return metadata if isinstance(metadata, dict) else {}


def _concat_answer(mem0_results: list[dict[str, Any]]) -> str:
    return "\n".join(item["memory"] for item in mem0_results if item["memory"]).strip()


def _stringify_mem0_llm_output(payload: Any) -> str:
    if isinstance(payload, Mem0AnswerDecision):
        return payload.answer
    if isinstance(payload, BaseModel):
        return json.dumps(payload.model_dump(exclude_none=True), ensure_ascii=False)
    if isinstance(payload, dict):
        if "answer" in payload:
            return _stringify_answer(payload["answer"])
        if len(payload) == 1:
            return _stringify_answer(next(iter(payload.values())))
        return json.dumps(payload, ensure_ascii=False)
    return _stringify_answer(payload)


def _stringify_answer(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _scoped_user_id(prefix: str, question_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.:-]+", "_", question_id).strip("_")[:80] or "question"
    digest = hashlib.sha1(question_id.encode("utf-8")).hexdigest()[:10]
    return f"{prefix}_{safe}_{digest}"


def _chunks(items: list[TurnInput], size: int) -> list[list[TurnInput]]:
    chunk_size = max(1, size)
    return [items[index : index + chunk_size] for index in range(0, len(items), chunk_size)]


def _looks_like_empty_scope(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "404",
            "not found",
            "no memories",
            "no entities",
            "does not exist",
            "doesn't exist",
        )
    )


_MEM0_RESPONSE_SYSTEM_PROMPT = """You are evaluating a Mem0 memory baseline.
Answer the user's query using only the Mem0 search results in the provided JSON.
Return the answer directly as plain text. Keep it concise.
If the search results do not contain enough evidence, say that there is not enough information."""


_MEM0_RESPONSE_SYSTEM_PROMPT_STRUCTURED = """You are evaluating a Mem0 memory baseline.
Answer the user's query using only the Mem0 search results in the provided JSON.
Return structured JSON only with exactly one top-level key: answer.
If the search results do not contain enough evidence, answer that there is not enough information."""
