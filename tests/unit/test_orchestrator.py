from datetime import UTC, datetime
import json
from pathlib import Path

from pydantic import BaseModel

from driftscope.agents.conflict_agent import ConflictAgent
from driftscope.agents.base import Agent
from driftscope.agents.types import (
    CandidateMatch,
    CandidateSelection,
    IndexedUpdateProposal,
    UpdateInput,
    UpdateProposal,
)
from driftscope.agents.update_agent import HeuristicUpdateAgent
from driftscope.core.memory_base import MemoryBase
from driftscope.core.schema import Scope, TopicQuery, TurnInput
from driftscope.eval.instrumentation import JsonlTurnLogger
from driftscope.pipeline.orchestrator import TurnProcessor
from tests.unit.helpers import make_memory


class FakeLLM:
    def __init__(self, payload):
        self.payload = payload

    def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[BaseModel],
    ):
        return self.payload


class CountingBatchUpdateAgent(Agent):
    name = "update"

    def __init__(self) -> None:
        self.run_calls = 0
        self.run_batch_calls = 0

    def run(self, input_obj):
        self.run_calls += 1
        raise AssertionError("process_replay_batch should not call run() when run_batch() is available")

    def run_batch(self, input_objs: list[UpdateInput]):
        self.run_batch_calls += 1
        proposals = []
        for index, input_obj in enumerate(input_objs):
            proposals.append(
                IndexedUpdateProposal(
                    source_turn_index=index,
                    proposal=UpdateProposal(
                        intent="add",
                        candidate=make_memory(
                            content=input_obj.user_input,
                            topic_id="user.profile.location",
                            scope=input_obj.scope,
                            ingest_time=input_obj.timestamp,
                        ),
                    ),
                )
            )
        return proposals


def test_turn_processor_handles_write_only_add() -> None:
    store = MemoryBase()
    processor = TurnProcessor(
        memory_base=store,
        update_agent=HeuristicUpdateAgent(),
        conflict_agent=ConflictAgent(
            FakeLLM(
                {
                    "action": "apply_add",
                    "confidence": 0.9,
                    "reason": "new memory",
                }
            )
        ),
        raw_session_sidecar=False,
    )
    result = processor.process_turn(
        TurnInput(
            user_input="我现在住在上海",
            scope={"kind": "personal"},
            timestamp=datetime(2026, 4, 1, tzinfo=UTC),
        )
    )

    assert result.write_only is True
    assert result.write_applied is True
    memories = store.query_visible(Scope(kind="personal"), datetime(2026, 4, 2, tzinfo=UTC))
    assert len(memories) == 1


def test_turn_processor_answers_query_only_turn() -> None:
    store = MemoryBase()
    store.add(
        make_memory(
            content="我现在住在上海",
            topic_id="user.profile.location",
            scope=Scope(kind="personal"),
        )
    )
    processor = TurnProcessor(
        memory_base=store,
        update_agent=HeuristicUpdateAgent(),
        conflict_agent=ConflictAgent(
            FakeLLM(
                {
                    "action": "apply_add",
                    "confidence": 0.9,
                    "reason": "unused",
                }
            )
        ),
    )
    result = processor.process_turn(
        TurnInput(
            query="我住在哪？",
            scope={"kind": "personal"},
            timestamp=datetime(2026, 4, 1, tzinfo=UTC),
        )
    )

    assert result.query_only is True
    assert result.answer == "我现在住在上海"
    assert result.cited_memory_ids


def test_turn_processor_logs_turn_when_logger_is_set(tmp_path: Path) -> None:
    store = MemoryBase()
    log_path = tmp_path / "turns.jsonl"
    processor = TurnProcessor(
        memory_base=store,
        update_agent=HeuristicUpdateAgent(),
        conflict_agent=ConflictAgent(
            FakeLLM(
                {
                    "action": "apply_add",
                    "confidence": 0.9,
                    "reason": "new memory",
                }
            )
        ),
        turn_logger=JsonlTurnLogger(str(log_path)),
    )

    processor.process_turn(
        TurnInput(
            user_input="我现在住在上海",
            scope={"kind": "personal"},
            timestamp=datetime(2026, 4, 1, tzinfo=UTC),
        )
    )

    assert log_path.exists()
    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert lines
    payload = json.loads(lines[0])
    assert payload["extras"]["update_proposal"]["intent"] == "add"
    assert payload["extras"]["conflict_resolution"]["resolution"]["action"] == "apply_add"


def test_turn_processor_processes_replay_batch_with_single_update_call() -> None:
    store = MemoryBase()
    update_agent = CountingBatchUpdateAgent()
    processor = TurnProcessor(
        memory_base=store,
        update_agent=update_agent,
        conflict_agent=ConflictAgent(
            FakeLLM(
                {
                    "action": "apply_add",
                    "confidence": 0.9,
                    "reason": "new memory",
                }
            )
        ),
    )

    results = processor.process_replay_batch(
        [
            TurnInput(
                user_input="我现在住在上海",
                scope={"kind": "personal"},
                timestamp=datetime(2026, 4, 1, tzinfo=UTC),
            ),
            TurnInput(
                user_input="我住在北京",
                scope={"kind": "personal"},
                timestamp=datetime(2026, 4, 2, tzinfo=UTC),
            ),
        ]
    )

    assert update_agent.run_batch_calls == 1
    assert update_agent.run_calls == 0
    assert len(results) == 2
    # Turn 0 is a fresh add (empty memory base) — fast-path, no conflict_agent.
    # Turn 1's add proposal collides with turn 0's same-topic memory, so
    # candidate_selector now surfaces it as a conflict candidate and
    # conflict_agent is invoked (returns apply_add via FakeLLM in this test).
    assert results[0].agents_called == ["update"]
    assert results[1].agents_called == ["conflict"]


def test_turn_processor_kill_switch_bypasses_conflict_for_add() -> None:
    """When route_add_through_conflict=False, `add` proposals skip the conflict
    pipeline entirely and apply directly — even if a same-topic conflicting
    memory already exists. This is the rollback path for the new add-routing.
    """
    from driftscope.config.loader import load_default_config

    config = load_default_config()
    config.update.route_add_through_conflict = False

    store = MemoryBase()
    update_agent = CountingBatchUpdateAgent()
    processor = TurnProcessor(
        memory_base=store,
        update_agent=update_agent,
        conflict_agent=ConflictAgent(
            FakeLLM({"action": "apply_add", "confidence": 0.9, "reason": "x"})
        ),
        config=config,
    )

    results = processor.process_replay_batch(
        [
            TurnInput(
                user_input="我现在住在上海",
                scope={"kind": "personal"},
                timestamp=datetime(2026, 4, 1, tzinfo=UTC),
            ),
            TurnInput(
                user_input="我住在北京",
                scope={"kind": "personal"},
                timestamp=datetime(2026, 4, 2, tzinfo=UTC),
            ),
        ]
    )

    # Both adds bypass conflict_agent under the kill switch — no "conflict" in agents_called.
    assert results[0].agents_called == ["update"]
    assert results[1].agents_called == []


def test_turn_processor_skips_assistant_authored_replay_writes() -> None:
    store = MemoryBase()
    processor = TurnProcessor(
        memory_base=store,
        update_agent=HeuristicUpdateAgent(),
        conflict_agent=ConflictAgent(
            FakeLLM(
                {
                    "action": "apply_add",
                    "confidence": 0.9,
                    "reason": "unused",
                }
            )
        ),
    )

    result = processor.process_turn(
        TurnInput(
            user_input="好的，我记住了",
            origin_role="assistant",
            scope={"kind": "personal"},
            timestamp=datetime(2026, 4, 1, tzinfo=UTC),
        )
    )

    assert result.write_applied is False
    assert result.agents_called == []
    assert store.query_visible(Scope(kind="personal"), datetime(2026, 4, 2, tzinfo=UTC)) == []


def test_turn_processor_can_ingest_assistant_authored_replay_writes_when_enabled() -> None:
    store = MemoryBase()
    processor = TurnProcessor(
        memory_base=store,
        update_agent=HeuristicUpdateAgent(),
        conflict_agent=ConflictAgent(
            FakeLLM(
                {
                    "action": "apply_add",
                    "confidence": 0.9,
                    "reason": "assistant memory",
                }
            )
        ),
        ingest_assistant_turns=True,
        raw_session_sidecar=False,
    )

    result = processor.process_turn(
        TurnInput(
            user_input="I graduated with a degree in Business Administration.",
            origin_role="assistant",
            scope={"kind": "personal"},
            timestamp=datetime(2026, 4, 1, tzinfo=UTC),
        )
    )

    assert result.write_applied is True
    assert result.agents_called == ["update"]
    memories = store.query_visible(Scope(kind="personal"), datetime(2026, 4, 2, tzinfo=UTC))
    assert len(memories) == 1
    assert memories[0].origin_role == "assistant"


class MultiProposalUpdateAgent(Agent):
    """Agent that emits multiple proposals per turn via run_many."""

    name = "update"

    def __init__(self) -> None:
        self.run_many_calls = 0

    def run(self, input_obj: UpdateInput) -> UpdateProposal:
        raise AssertionError("process_turn should call run_many when available")

    def run_many(self, input_obj: UpdateInput) -> list[UpdateProposal]:
        self.run_many_calls += 1
        return [
            UpdateProposal(
                intent="add",
                candidate=make_memory(
                    content="aunt retirement gifts: teal blazer and matching brooch",
                    topic_id="user.profile.location",
                    scope=input_obj.scope,
                    ingest_time=input_obj.timestamp,
                ),
            ),
            UpdateProposal(
                intent="add",
                candidate=make_memory(
                    content="visiting aunt Lorena this weekend",
                    topic_id="user.profile.location",
                    scope=input_obj.scope,
                    ingest_time=input_obj.timestamp,
                ),
            ),
        ]


def test_turn_processor_applies_multiple_proposals_from_single_turn(tmp_path: Path) -> None:
    store = MemoryBase()
    log_path = tmp_path / "turns.jsonl"
    update_agent = MultiProposalUpdateAgent()
    processor = TurnProcessor(
        memory_base=store,
        update_agent=update_agent,
        conflict_agent=ConflictAgent(
            FakeLLM(
                {
                    "action": "apply_add",
                    "confidence": 0.9,
                    "reason": "new memory",
                }
            )
        ),
        turn_logger=JsonlTurnLogger(str(log_path)),
        raw_session_sidecar=False,
    )

    result = processor.process_turn(
        TurnInput(
            user_input="For my aunt Lorena's retirement I bought a teal blazer and a matching brooch, visiting this weekend",
            scope={"kind": "personal"},
            timestamp=datetime(2026, 4, 1, tzinfo=UTC),
        )
    )

    assert update_agent.run_many_calls == 1
    assert result.write_applied is True
    memories = store.query_visible(Scope(kind="personal"), datetime(2026, 4, 2, tzinfo=UTC))
    assert len(memories) == 2

    payload = json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])
    assert "update_proposals" in payload["extras"]
    assert len(payload["extras"]["update_proposals"]) == 2
    assert "update_executions" in payload["extras"]
    assert len(payload["extras"]["update_executions"]) == 2


class RaisingConflictLLM:
    def __init__(self) -> None:
        self.call_count = 0

    def generate_structured(self, *, system_prompt, user_prompt, response_model):
        self.call_count += 1
        raise AssertionError("ConflictAgent LLM must not be called when deterministic bypass fires")


class ExplicitCorrectionUpdateAgent(Agent):
    name = "update"

    def __init__(self, candidate_memory) -> None:
        self.candidate_memory = candidate_memory

    def run(self, input_obj: UpdateInput) -> UpdateProposal:
        return UpdateProposal(
            intent="supersede_full",
            candidate=self.candidate_memory,
            target_hint=TopicQuery(
                topic_id=self.candidate_memory.topic_id,
                keywords=["Main", "Street"],
            ),
            transition_type="corrected",
        )


class AmbiguousStubSelector:
    def __init__(self, candidates: list[CandidateMatch]) -> None:
        self._candidates = candidates

    def select(self, *, proposal, memory_base, scope, timestamp) -> CandidateSelection:
        return CandidateSelection(candidates=self._candidates, ambiguous_candidates=True)


def test_explicit_correction_bypasses_conflict_llm_when_ambiguous(tmp_path: Path) -> None:
    scope = Scope(kind="personal")
    topic = "user.profile.location"
    stored_target = make_memory(
        content="I live at 400 Main Street in Shanghai",
        topic_id=topic,
        scope=scope,
    )
    stored_other = make_memory(
        content="I visit Central Park with my main friend on weekends",
        topic_id=topic,
        scope=scope,
    )
    store = MemoryBase()
    store.add(stored_target)
    store.add(stored_other)

    new_candidate = make_memory(
        content="I moved to 500 Main Street in Beijing",
        topic_id=topic,
        scope=scope,
    )
    candidates = [
        CandidateMatch(
            memory=stored_target,
            score=0.7,
            score_breakdown={"content_sim": 0.45, "keyword_overlap": 0.5},
            matched_by=["content_overlap", "type_exact", "topic_hint"],
        ),
        CandidateMatch(
            memory=stored_other,
            score=0.68,
            score_breakdown={"content_sim": 0.2, "keyword_overlap": 0.4},
            matched_by=["content_overlap", "type_exact", "topic_hint"],
        ),
    ]

    raising_llm = RaisingConflictLLM()
    log_path = tmp_path / "turns.jsonl"
    processor = TurnProcessor(
        memory_base=store,
        update_agent=ExplicitCorrectionUpdateAgent(new_candidate),
        conflict_agent=ConflictAgent(raising_llm),
        candidate_selector=AmbiguousStubSelector(candidates),
        turn_logger=JsonlTurnLogger(str(log_path)),
    )

    result = processor.process_turn(
        TurnInput(
            user_input="Actually, I moved to 500 Main Street in Beijing, not 400 Main Street",
            scope={"kind": "personal"},
            timestamp=datetime(2026, 4, 3, tzinfo=UTC),
        )
    )

    assert result.write_applied is True
    assert raising_llm.call_count == 0
    assert "conflict" not in result.agents_called
    assert store.get(stored_target.id).state == "superseded"

    payload = json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])
    cr = payload["extras"]["conflict_resolution"]
    assert cr["source"] == "deterministic_correction"
    assert cr["resolution"]["action"] == "confirm_supersede"


class _ParaphrasingUpdateAgent(Agent):
    """Update agent that buries the brand inside an unrelated paraphrase.

    Mirrors the d682f1a2 failure mode: the LLM extracts a recipe-preference
    fact and loses the literal brand name from the user's sentence.
    """

    name = "update"

    def run(self, input_obj: UpdateInput) -> UpdateProposal:
        return UpdateProposal(
            intent="add",
            candidate=make_memory(
                content="The user is looking for quick weeknight recipe ideas.",
                topic_id=None,
                scope=input_obj.scope,
                ingest_time=input_obj.timestamp,
            ),
        )


def test_raw_session_sidecar_preserves_verbatim_brand_after_paraphrase() -> None:
    from driftscope.agents.retriever_agent import HybridRetrieverAgent
    from driftscope.agents.types import RetrievalInput

    store = MemoryBase()
    processor = TurnProcessor(
        memory_base=store,
        update_agent=_ParaphrasingUpdateAgent(),
        conflict_agent=ConflictAgent(
            FakeLLM({"action": "apply_add", "confidence": 0.9, "reason": "new"})
        ),
    )

    user_sentence = (
        "Looking for quick recipes — my weekends have been all about Vermillion Table lately."
    )
    processor.process_turn(
        TurnInput(
            user_input=user_sentence,
            scope={"kind": "personal"},
            timestamp=datetime(2026, 4, 1, tzinfo=UTC),
        )
    )

    scope = Scope(kind="personal")
    visible = store.query_visible(scope, datetime(2026, 4, 2, tzinfo=UTC))
    raw = [m for m in visible if m.type == "raw_session"]
    assert len(raw) == 1
    assert raw[0].content == user_sentence

    retriever = HybridRetrieverAgent(memory_base=store)
    result = retriever.run(
        RetrievalInput(
            query="What brand has the user relied on for weekend meals?",
            scope=scope,
            timestamp=datetime(2026, 4, 2, tzinfo=UTC),
        )
    )
    top_contents = [m.memory.content for m in result.ranked_memories]
    assert any("Vermillion Table" in c for c in top_contents), (
        "raw_session sidecar must surface the verbatim brand name even when "
        "the extracted fact paraphrased it away"
    )


def test_raw_session_sidecar_can_be_disabled() -> None:
    store = MemoryBase()
    processor = TurnProcessor(
        memory_base=store,
        update_agent=HeuristicUpdateAgent(),
        conflict_agent=ConflictAgent(
            FakeLLM({"action": "apply_add", "confidence": 0.9, "reason": "new"})
        ),
        raw_session_sidecar=False,
    )
    processor.process_turn(
        TurnInput(
            user_input="I moved to the Caldermere district last month.",
            scope={"kind": "personal"},
            timestamp=datetime(2026, 4, 1, tzinfo=UTC),
        )
    )
    visible = store.query_visible(Scope(kind="personal"), datetime(2026, 4, 2, tzinfo=UTC))
    assert all(m.type != "raw_session" for m in visible)


def test_raw_session_sidecar_writes_for_replay_batch() -> None:
    store = MemoryBase()
    processor = TurnProcessor(
        memory_base=store,
        update_agent=HeuristicUpdateAgent(),
        conflict_agent=ConflictAgent(
            FakeLLM({"action": "apply_add", "confidence": 0.9, "reason": "new"})
        ),
    )
    processor.process_replay_batch(
        [
            TurnInput(
                user_input="Took the kids to Pinemoor Park.",
                scope={"kind": "personal"},
                timestamp=datetime(2026, 4, 1, 10, 0, tzinfo=UTC),
            ),
            TurnInput(
                user_input="Picked up groceries from RestoWorks.",
                scope={"kind": "personal"},
                timestamp=datetime(2026, 4, 1, 18, 0, tzinfo=UTC),
            ),
        ]
    )
    visible = store.query_visible(Scope(kind="personal"), datetime(2026, 4, 2, tzinfo=UTC))
    raw = [m for m in visible if m.type == "raw_session"]
    assert len(raw) == 2
    raw_texts = sorted(m.content for m in raw)
    assert raw_texts == sorted(
        [
            "Took the kids to Pinemoor Park.",
            "Picked up groceries from RestoWorks.",
        ]
    )