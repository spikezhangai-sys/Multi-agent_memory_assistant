import json
from pathlib import Path

from driftscope.agents.base import Agent
from driftscope.agents.conflict_agent import ConflictAgent
from driftscope.agents.types import IndexedUpdateProposal, UpdateInput, UpdateProposal
from driftscope.agents.update_agent import HeuristicUpdateAgent
from driftscope.core.memory_base import MemoryBase
from driftscope.core.schema import Scope
from driftscope.eval.instrumentation import JsonlTurnLogger
from driftscope.eval.longmemeval.runner import LongMemEvalRunner
from driftscope.llm.mock import RuleBasedConflictLLM
from driftscope.pipeline.orchestrator import TurnProcessor


class CountingBatchUpdateAgent(Agent):
    name = "update"

    def __init__(self) -> None:
        self.run_batch_calls = 0

    def run(self, input_obj):
        raise AssertionError("runner should use process_replay_batch")

    def run_batch(self, input_objs: list[UpdateInput]):
        self.run_batch_calls += 1
        proposals: list[IndexedUpdateProposal] = []
        for index, input_obj in enumerate(input_objs):
            proposals.append(
                IndexedUpdateProposal(
                    source_turn_index=index,
                    proposal=UpdateProposal(
                        intent="add",
                        candidate=HeuristicUpdateAgent()._build_candidate(
                            content=input_obj.user_input,
                            topic_id="user.profile.location",
                            memory_type="fact",
                            timestamp=input_obj.timestamp,
                            scope=input_obj.scope,
                        ),
                    ),
                )
            )
        return proposals


def test_runner_writes_predictions_and_summary(tmp_path: Path) -> None:
    dataset_path = tmp_path / "longmemeval.json"
    dataset_path.write_text(
        json.dumps(
            [
                {
                    "question_id": "q1",
                    "question": "我住在哪？",
                    "haystack_sessions": ["我现在住在上海"],
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "run"
    turns_path = output_dir / "turns.jsonl"

    def processor_factory(path: Path, db_path: str | None) -> TurnProcessor:
        memory_base = MemoryBase(db_path=db_path or ":memory:")
        return TurnProcessor(
            memory_base=memory_base,
            update_agent=HeuristicUpdateAgent(topic_tree=memory_base.topic_tree),
            conflict_agent=ConflictAgent(RuleBasedConflictLLM()),
            turn_logger=JsonlTurnLogger(str(path)),
        )

    summary = LongMemEvalRunner(processor_factory=processor_factory).run(dataset_path, output_dir, limit=1)

    assert summary.num_instances == 1
    predictions = [json.loads(line) for line in summary.predictions_path.read_text(encoding="utf-8").splitlines()]
    assert predictions == [{"question_id": "q1", "hypothesis": "我现在住在上海"}]

    turns = summary.turns_path.read_text(encoding="utf-8").splitlines()
    assert len(turns) == 2

    summary_payload = json.loads(summary.summary_path.read_text(encoding="utf-8"))
    assert summary_payload["num_questions"] == 1
    assert summary_payload["num_turns"] == 2


def test_runner_batches_replay_turns(tmp_path: Path) -> None:
    dataset_path = tmp_path / "longmemeval.json"
    dataset_path.write_text(
        json.dumps(
            [
                {
                    "question_id": "q1",
                    "question": "我住在哪？",
                    "haystack_sessions": [
                        [
                            {"role": "user", "content": "我住在上海"},
                            {"role": "user", "content": "我住在北京"},
                        ]
                    ],
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "run_batch"
    update_agent = CountingBatchUpdateAgent()

    def processor_factory(path: Path, db_path: str | None) -> TurnProcessor:
        memory_base = MemoryBase(db_path=db_path or ":memory:")
        return TurnProcessor(
            memory_base=memory_base,
            update_agent=update_agent,
            conflict_agent=ConflictAgent(RuleBasedConflictLLM()),
            turn_logger=JsonlTurnLogger(str(path)),
        )

    LongMemEvalRunner(processor_factory=processor_factory, replay_batch_size=8).run(dataset_path, output_dir, limit=1)

    assert update_agent.run_batch_calls == 1


def test_runner_isolates_oversized_turn_into_its_own_batch(tmp_path: Path) -> None:
    """Adaptive batching: a turn whose token estimate alone exceeds the
    budget should flush the current buffer and go into a batch by itself,
    instead of being concatenated with neighbors and pushing the prompt past
    the model's context window. This is what makes ku_15-style Wikipedia-paste
    turns survive at all under gpt-4o-mini's 128k limit.
    """
    from driftscope.config.loader import load_default_config

    huge = "X" * 1200  # ~300 tokens via the chars//4 heuristic
    dataset_path = tmp_path / "longmemeval_oversized.json"
    dataset_path.write_text(
        json.dumps(
            [
                {
                    "question_id": "q1",
                    "question": "ok?",
                    "haystack_sessions": [
                        [
                            {"role": "user", "content": "short1"},
                            {"role": "user", "content": "short2"},
                            {"role": "user", "content": huge},
                            {"role": "user", "content": "short4"},
                        ]
                    ],
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "run_oversized"
    update_agent = CountingBatchUpdateAgent()

    def processor_factory(path: Path, db_path: str | None) -> TurnProcessor:
        # Tight budget so the synthetic huge turn alone exceeds it.
        config = load_default_config()
        config.update.batch_token_budget = 200
        config.update.batch_per_turn_overhead_tokens = 10
        memory_base = MemoryBase(db_path=db_path or ":memory:")
        return TurnProcessor(
            memory_base=memory_base,
            update_agent=update_agent,
            conflict_agent=ConflictAgent(RuleBasedConflictLLM()),
            turn_logger=JsonlTurnLogger(str(path)),
            config=config,
        )

    LongMemEvalRunner(processor_factory=processor_factory, replay_batch_size=16).run(
        dataset_path, output_dir, limit=1
    )

    # Expected packing for [short1, short2, HUGE, short4] under budget=200:
    #   batch1 = [short1, short2]  (small total)
    #   batch2 = [HUGE]            (single oversized turn isolated)
    #   batch3 = [short4]          (post-HUGE accumulation)
    assert update_agent.run_batch_calls == 3


def test_runner_defaults_to_ingesting_assistant_replay_turns(tmp_path: Path) -> None:
    dataset_path = tmp_path / "longmemeval_assistant_default.json"
    dataset_path.write_text(
        json.dumps(
            [
                {
                    "question_id": "q1",
                    "question": "我喜欢什么？",
                    "haystack_sessions": [
                        [
                            {"role": "user", "content": "我喜欢日料"},
                            {"role": "assistant", "content": "好的，我记住了"},
                        ]
                    ],
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    runner = LongMemEvalRunner()
    instances = runner.adapter.load(dataset_path)

    assert len(instances) == 1
    assert len(instances[0].replay_turns) == 2
    assert instances[0].replay_turns[0].origin_role == "user"
    assert instances[0].replay_turns[1].origin_role == "assistant"
