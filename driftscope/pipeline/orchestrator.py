from __future__ import annotations

from driftscope.agents.base import Agent
from driftscope.agents.candidate_selector import CandidateSelector
from driftscope.agents.conflict_agent import ConflictAgent
from driftscope.agents.response_agent import HeuristicResponseAgent
from driftscope.agents.retriever_agent import HeuristicRetrieverAgent
from driftscope.agents.types import (
    ConflictInput,
    IndexedUpdateProposal,
    ResponseInput,
    RetrievalInput,
    UpdateInput,
    UpdateProposal,
)
from driftscope.config.loader import DriftScopeConfig, load_default_config
from driftscope.core.memory_base import MemoryBase
from driftscope.core.schema import (
    Confidence,
    MemoryEntry,
    TimeRange,
    TurnInput,
    TurnResult,
)
from driftscope.eval.instrumentation import JsonlTurnLogger
from driftscope.pipeline.transitions import (
    apply_conflict_resolution,
    is_rollback_legal,
    locate_by_hint,
    try_deterministic_correction,
)


class TurnProcessor:
    def __init__(
        self,
        *,
        memory_base: MemoryBase,
        update_agent: Agent,
        conflict_agent: Agent,
        retriever_agent: Agent | None = None,
        response_agent: Agent | None = None,
        candidate_selector: CandidateSelector | None = None,
        turn_logger: JsonlTurnLogger | None = None,
        config: DriftScopeConfig | None = None,
        ingest_assistant_turns: bool = False,
        raw_session_sidecar: bool = True,
    ) -> None:
        self.memory_base = memory_base
        self.update_agent = update_agent
        self.conflict_agent = conflict_agent
        self.retriever_agent = retriever_agent or HeuristicRetrieverAgent(memory_base=memory_base)
        self.response_agent = response_agent or HeuristicResponseAgent()
        self.candidate_selector = candidate_selector or CandidateSelector()
        self.turn_logger = turn_logger
        self.config = config or load_default_config()
        self.ingest_assistant_turns = ingest_assistant_turns
        self.raw_session_sidecar = raw_session_sidecar

    def process_turn(self, turn: TurnInput) -> TurnResult:
        result = TurnResult(
            write_only=turn.user_input is not None and turn.query is None,
            query_only=turn.user_input is None and turn.query is not None,
        )
        log_extras: dict[str, object] = {}

        if turn.user_input is not None and self._should_update_turn(turn):
            raw_id = self._write_raw_session(turn)
            if raw_id is not None:
                log_extras["raw_session_id"] = raw_id
            result.agents_called.append("update")
            update_input = UpdateInput(
                user_input=turn.user_input,
                origin_role=turn.origin_role,
                scope=turn.scope,
                timestamp=turn.timestamp,
                nearby_memories=self._nearby_memories(turn),
            )
            proposals = self._run_update_many(update_input)
            first_proposal = proposals[0] if proposals else UpdateProposal(intent="ignore")
            log_extras["update_proposal"] = first_proposal.model_dump(mode="json", exclude_none=True)
            if len(proposals) > 1:
                log_extras["update_proposals"] = [
                    proposal.model_dump(mode="json", exclude_none=True) for proposal in proposals
                ]

            any_applied = False
            first_exec_extras: dict[str, object] = {}
            exec_extras_list: list[dict[str, object]] = []
            total_conflict_calls = 0
            effective_proposals = proposals if proposals else [first_proposal]
            for index, proposal in enumerate(effective_proposals):
                write_applied, exec_errors, exec_extras, conflict_calls = self._execute_update_proposal(
                    proposal=proposal,
                    turn=turn,
                )
                any_applied = any_applied or write_applied
                result.errors.extend(exec_errors)
                total_conflict_calls += conflict_calls
                if index == 0:
                    first_exec_extras = exec_extras
                exec_extras_list.append(exec_extras)

            result.write_applied = any_applied
            result.agents_called.extend(["conflict"] * total_conflict_calls)
            log_extras.update(first_exec_extras)
            if len(effective_proposals) > 1:
                log_extras["update_executions"] = exec_extras_list

        if turn.query is not None:
            result.agents_called.append("retriever")
            retrieval = self.retriever_agent.run(
                RetrievalInput(
                    query=turn.query,
                    scope=turn.scope,
                    timestamp=turn.timestamp,
                )
            )
            log_extras["retrieval"] = retrieval.model_dump(mode="json", exclude_none=True)
            result.agents_called.append("response")
            response = self.response_agent.run(
                ResponseInput(
                    query=turn.query,
                    retrieval=retrieval,
                )
            )
            log_extras["response"] = response.model_dump(mode="json", exclude_none=True)
            result.answer = response.answer
            result.cited_memory_ids = response.cited_memory_ids
            result.context_only_ids = response.context_only_ids
            result.abstained = response.abstained

        if self.turn_logger is not None:
            self.turn_logger.log_turn(turn, result, extras=log_extras)

        return result

    def process_replay_batch(self, turns: list[TurnInput]) -> list[TurnResult]:
        if not turns:
            return []
        for turn in turns:
            if turn.user_input is None or turn.query is not None:
                raise ValueError("process_replay_batch only accepts write-only replay turns")

        results = [TurnResult(write_only=True) for _ in turns]
        extras_by_turn: list[dict[str, object]] = [{} for _ in turns]

        for index, turn in enumerate(turns):
            if not self._should_update_turn(turn):
                continue
            raw_id = self._write_raw_session(turn)
            if raw_id is not None:
                extras_by_turn[index]["raw_session_id"] = raw_id

        indexed_update_inputs = [
            (
                index,
                UpdateInput(
                    user_input=turn.user_input or "",
                    origin_role=turn.origin_role,
                    scope=turn.scope,
                    timestamp=turn.timestamp,
                    nearby_memories=self._nearby_memories(turn),
                ),
            )
            for index, turn in enumerate(turns)
            if self._should_update_turn(turn)
        ]
        update_inputs = [item[1] for item in indexed_update_inputs]
        indexed_proposals = self._run_update_batch(update_inputs)

        if indexed_update_inputs:
            results[indexed_update_inputs[0][0]].agents_called.append("update")
            batch_extras: dict[str, object] = {
                "batch_size": len(turns),
                "proposal_count": len(indexed_proposals),
            }
            # If the agent stashed a per-batch diagnostic (LLM exception type,
            # raw decision count, drop reasons), surface it so silent batch
            # failures show up in turns.jsonl instead of being invisible.
            diagnostic = getattr(self.update_agent, "_last_batch_diagnostic", None)
            if diagnostic:
                batch_extras["diagnostic"] = diagnostic
            extras_by_turn[indexed_update_inputs[0][0]]["batch_update"] = batch_extras

        input_index_to_turn_index = {
            update_index: source_turn_index
            for update_index, (source_turn_index, _) in enumerate(indexed_update_inputs)
        }

        for indexed in indexed_proposals:
            proposal = indexed.proposal
            source_index = input_index_to_turn_index[indexed.source_turn_index]
            source_turn = turns[source_index]
            extras_by_turn[source_index].setdefault("update_proposals", []).append(
                proposal.model_dump(mode="json", exclude_none=True)
            )
            write_applied, exec_errors, exec_extras, conflict_calls = self._execute_update_proposal(
                proposal=proposal,
                turn=source_turn,
            )
            results[source_index].write_applied = results[source_index].write_applied or write_applied
            results[source_index].errors.extend(exec_errors)
            results[source_index].agents_called.extend(["conflict"] * conflict_calls)
            for key, value in exec_extras.items():
                if key in {"candidate_selection", "conflict_resolution"}:
                    extras_by_turn[source_index].setdefault(key, []).append(value)
                else:
                    extras_by_turn[source_index][key] = value

        if self.turn_logger is not None:
            for turn, result, extras in zip(turns, results, extras_by_turn, strict=False):
                self.turn_logger.log_turn(turn, result, extras=extras or None)

        return results

    def _write_raw_session(self, turn: TurnInput) -> str | None:
        """Persist the verbatim user/assistant utterance as a raw_session memory.

        Bypasses the LLM extraction pipeline entirely so the original wording is
        searchable even when update_agent paraphrases the fact away (e.g. burying
        a brand name inside a recipe-preference clause).
        """
        if not self.raw_session_sidecar:
            return None
        text = (turn.user_input or "").strip()
        if not text:
            return None
        is_user = turn.origin_role == "user"
        memory = MemoryEntry(
            content=text,
            type="raw_session",
            topic_id=None,
            scope=turn.scope,
            src="user_explicit" if is_user else "external",
            origin_role=turn.origin_role,
            source_kind="explicit",
            conf=Confidence(prior=1.0, combined=1.0),
            valid_time=TimeRange(start=turn.timestamp),
            ingest_time=turn.timestamp,
            event_time=turn.timestamp,
            evidence=text,
            importance=0.5,
            sensitivity="low",
        )
        self.memory_base.add(memory)
        return memory.id

    def _nearby_memories(self, turn: TurnInput):
        if self.config.update.nearby_k <= 0:
            return []
        visible = self.memory_base.query_visible(turn.scope, turn.timestamp)
        user_authored = [
            memory
            for memory in visible
            if memory.origin_role == "user" and memory.source_kind == "explicit"
        ]
        return user_authored[: self.config.update.nearby_k]

    def _run_update_many(self, update_input: UpdateInput) -> list[UpdateProposal]:
        run_many = getattr(self.update_agent, "run_many", None)
        if callable(run_many):
            proposals = run_many(update_input)
            return [p for p in proposals if p.intent != "ignore"]
        single = self.update_agent.run(update_input)
        return [] if single.intent == "ignore" else [single]

    def _run_update_batch(self, update_inputs: list[UpdateInput]) -> list[IndexedUpdateProposal]:
        run_batch = getattr(self.update_agent, "run_batch", None)
        if callable(run_batch):
            return run_batch(update_inputs)

        proposals: list[IndexedUpdateProposal] = []
        for index, input_obj in enumerate(update_inputs):
            proposal = self.update_agent.run(input_obj)
            if proposal.intent == "ignore":
                continue
            proposals.append(IndexedUpdateProposal(source_turn_index=index, proposal=proposal))
        return proposals

    def _execute_update_proposal(
        self,
        *,
        proposal,
        turn: TurnInput,
    ) -> tuple[bool, list[str], dict[str, object], int]:
        if proposal.intent == "rollback":
            applied = self._handle_rollback(proposal.target_hint, turn)
            errors = [] if applied else ["rollback was not legal or target was not found"]
            return applied, errors, {}, 0
        if proposal.intent == "ignore":
            return False, [], {}, 0
        if proposal.intent == "add" and proposal.candidate is None:
            return False, ["add proposal did not include a candidate"], {}, 0

        # Kill switch: when route_add_through_conflict is disabled, `add` proposals
        # go straight to memory_base.add() like the pre-fix behavior. Lets ops
        # roll back the new conflict-routing without redeploying.
        if proposal.intent == "add" and not self.config.update.route_add_through_conflict:
            self.memory_base.add(proposal.candidate)
            return (
                True,
                [],
                {
                    "conflict_resolution": {
                        "resolution": {
                            "action": "apply_add",
                            "confidence": proposal.candidate.conf.combined,
                            "reason": "Direct add path (route_add_through_conflict disabled).",
                        },
                        "used_fallback": False,
                        "validation_errors": [],
                    }
                },
                0,
            )

        extras: dict[str, object] = {}
        selection = self.candidate_selector.select(
            proposal=proposal,
            memory_base=self.memory_base,
            scope=turn.scope,
            timestamp=turn.timestamp,
        )
        extras["candidate_selection"] = {
            "ambiguous_candidates": selection.ambiguous_candidates,
            "candidate_ids": [match.memory.id for match in selection.candidates],
        }

        # Fast path: `add` proposal with no conflict candidates — apply directly,
        # skip ConflictAgent. Preserves O(1) write-cost for the common case where
        # the user states a brand-new fact unrelated to anything in memory.
        if (
            proposal.intent == "add"
            and not selection.candidates
            and not selection.ambiguous_candidates
        ):
            self.memory_base.add(proposal.candidate)
            extras["conflict_resolution"] = {
                "resolution": {
                    "action": "apply_add",
                    "confidence": proposal.candidate.conf.combined,
                    "reason": "No conflict candidates found; direct add.",
                },
                "used_fallback": False,
                "validation_errors": [],
            }
            return True, [], extras, 0
        deterministic = try_deterministic_correction(
            proposal=proposal,
            candidates=selection.candidates,
            ambiguous=selection.ambiguous_candidates,
            user_text=turn.user_input or "",
        )
        if deterministic is not None:
            extras["conflict_resolution"] = {
                "resolution": deterministic.model_dump(mode="json", exclude_none=True),
                "used_fallback": False,
                "validation_errors": [],
                "source": "deterministic_correction",
            }
            applied = apply_conflict_resolution(
                memory_base=self.memory_base,
                proposal=proposal,
                resolution=deterministic,
                timestamp=turn.timestamp,
            )
            return applied, [], extras, 0
        conflict_out = self.conflict_agent.run(
            ConflictInput(
                proposal=proposal,
                scope=turn.scope,
                timestamp=turn.timestamp,
                candidates=selection.candidates,
                ambiguous_candidates=selection.ambiguous_candidates,
            )
        )
        extras["conflict_resolution"] = {
            "resolution": conflict_out.resolution.model_dump(mode="json", exclude_none=True),
            "used_fallback": conflict_out.used_fallback,
            "validation_errors": conflict_out.validation_errors,
        }
        if conflict_out.raw_resolution is not None:
            extras["conflict_resolution"]["raw_resolution"] = conflict_out.raw_resolution.model_dump(
                mode="json",
                exclude_none=True,
            )
        applied = apply_conflict_resolution(
            memory_base=self.memory_base,
            proposal=proposal,
            resolution=conflict_out.resolution,
            timestamp=turn.timestamp,
        )
        errors = conflict_out.validation_errors if conflict_out.used_fallback else []
        return applied, errors, extras, 1

    def _should_update_turn(self, turn: TurnInput) -> bool:
        if turn.origin_role == "user":
            return True
        return self.ingest_assistant_turns and turn.origin_role == "assistant"

    def _handle_rollback(self, target_hint, turn: TurnInput) -> bool:
        if target_hint is None:
            return False
        candidates = self.memory_base.query_revoked_within(
            window_days=self.config.retention.rollback_window_days,
            scope=turn.scope,
            time=turn.timestamp,
        )
        target = locate_by_hint(candidates, target_hint)
        if target is None:
            return False
        if not is_rollback_legal(
            memory_base=self.memory_base,
            target=target,
            scope=turn.scope,
            now=turn.timestamp,
            window_days=self.config.retention.rollback_window_days,
        ):
            return False
        return self.memory_base.rollback(target.id)
