import json
from pathlib import Path
from typing import Any

from driftscope.eval.longmemeval.mem0_baseline import Mem0BaselineConfig, Mem0BaselineRunner


class FakeMem0Client:
    def __init__(self) -> None:
        self.memories: dict[tuple[str, str | None], list[dict[str, Any]]] = {}
        self.deleted_scopes: list[tuple[str, str | None]] = []

    def add(self, messages, **kwargs):
        filters = self._filters(kwargs)
        key = (filters["user_id"], filters.get("run_id"))
        rows = self.memories.setdefault(key, [])
        results = []
        for message in messages:
            memory = {
                "id": f"m{len(rows) + 1}",
                "memory": message["content"],
                "metadata": kwargs.get("metadata", {}),
                "score": 1.0,
            }
            rows.append(memory)
            results.append(memory)
        return {"results": results}

    def search(self, query, **kwargs):
        filters = self._filters(kwargs)
        key = (filters["user_id"], filters.get("run_id"))
        results = list(self.memories.get(key, []))
        return {"results": results[: kwargs.get("top_k", 5)]}

    def delete_all(self, **kwargs):
        filters = self._filters(kwargs)
        key = (filters["user_id"], filters.get("run_id"))
        self.deleted_scopes.append(key)
        self.memories.pop(key, None)
        return {"message": "ok"}

    def _filters(self, kwargs):
        return kwargs.get("filters") or {key: kwargs[key] for key in ("user_id", "run_id") if key in kwargs}


class FakeTextLLM:
    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.calls: list[dict[str, str]] = []

    def generate_text(self, *, system_prompt: str, user_prompt: str) -> str:
        self.calls.append({"system_prompt": system_prompt, "user_prompt": user_prompt})
        return self.answer

    def generate_structured(self, **kwargs):
        raise AssertionError("Mem0 baseline should prefer raw text generation")


def test_mem0_baseline_writes_predictions_turns_and_summary(tmp_path: Path) -> None:
    dataset_path = tmp_path / "longmemeval.json"
    dataset_path.write_text(
        json.dumps(
            [
                {
                    "question_id": "q1",
                    "question": "我住在哪？",
                    "haystack_sessions": [
                        [
                            {"role": "user", "content": "我现在住在上海"},
                            {"role": "assistant", "content": "我记住了"},
                        ]
                    ],
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "mem0_run"
    fake_client = FakeMem0Client()

    summary = Mem0BaselineRunner(
        client_factory=lambda: fake_client,
        config=Mem0BaselineConfig(answer_mode="concat", batch_size=8, run_id="test-run"),
    ).run(dataset_path, output_dir)

    assert summary.num_instances == 1
    predictions = [
        json.loads(line)
        for line in summary.predictions_path.read_text(encoding="utf-8").splitlines()
    ]
    assert predictions == [
        {
            "question_id": "q1",
            "hypothesis": "我现在住在上海\n我记住了",
        }
    ]

    turns = [
        json.loads(line)
        for line in summary.turns_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [turn["result"]["agents_called"] for turn in turns] == [
        ["mem0_add"],
        ["mem0_add"],
        ["mem0_search"],
    ]
    assert turns[-1]["result"]["query_only"] is True
    assert turns[-1]["extras"]["mem0_result_count"] == 2

    summary_payload = json.loads(summary.summary_path.read_text(encoding="utf-8"))
    assert summary_payload["baseline"] == "mem0"
    assert summary_payload["num_questions"] == 1
    assert summary_payload["num_turns"] == 3
    assert summary_payload["agent_call_counts"]["mem0_add"] == 2
    assert summary_payload["agent_call_counts"]["mem0_search"] == 1
    assert fake_client.deleted_scopes


def test_mem0_baseline_uses_per_question_isolation(tmp_path: Path) -> None:
    dataset_path = tmp_path / "longmemeval.json"
    dataset_path.write_text(
        json.dumps(
            [
                {"question_id": "q1", "question": "one?", "haystack_sessions": ["first"]},
                {"question_id": "q2", "question": "two?", "haystack_sessions": ["second"]},
            ]
        ),
        encoding="utf-8",
    )
    fake_client = FakeMem0Client()

    summary = Mem0BaselineRunner(
        client_factory=lambda: fake_client,
        config=Mem0BaselineConfig(answer_mode="concat", user_prefix="lme_test", run_id="run"),
    ).run(dataset_path, tmp_path / "run")

    predictions = [
        json.loads(line)
        for line in summary.predictions_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [item["hypothesis"] for item in predictions] == ["first", "second"]
    assert len(fake_client.memories) == 2
    assert len({scope[0] for scope in fake_client.memories}) == 2


def test_mem0_baseline_uses_raw_llm_text_for_answer() -> None:
    runner = Mem0BaselineRunner(
        client_factory=FakeMem0Client,
        config=Mem0BaselineConfig(answer_mode="llm"),
    )
    runner.llm = FakeTextLLM('{"to_watch_list_count": 25}')

    answer, cited, context_only, abstained, errors = runner._answer(
        query="How many titles are currently on my to-watch list?",
        mem0_results=[
            {
                "id": "m1",
                "memory": "User has been meaning to organize their to-watch list, which currently contains 25 titles.",
                "score": 0.9,
                "metadata": {},
            },
            {
                "id": "m2",
                "memory": "User has a long to-watch list with 20 titles waiting to be checked off.",
                "score": 0.8,
                "metadata": {},
            },
        ],
    )

    assert answer == '{"to_watch_list_count": 25}'
    assert cited == ["m1"]
    assert context_only == ["m2"]
    assert abstained is False
    assert errors == []
