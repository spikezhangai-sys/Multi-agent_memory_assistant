from __future__ import annotations

import json

from pydantic import BaseModel, ConfigDict, Field

from driftscope.agents.base import Agent
from driftscope.agents.types import ResponseInput, ResponseOutput
from driftscope.core.schema import MemoryEntry
from driftscope.llm.client import StructuredLLM


def _episodic_anchor(memory: MemoryEntry) -> str | None:
    if memory.event_time is not None:
        return memory.event_time.isoformat()
    if memory.type == "episodic":
        return memory.ingest_time.isoformat()
    return None


def _evidence_payload(
    memory: MemoryEntry,
    input_obj: "ResponseInput",
    *,
    kind: str,
    score: float | None,
) -> dict:
    content = memory.summary_for_retrieval if memory.sensitive and not input_obj.allow_sensitive_raw else memory.content
    return {
        "id": memory.id,
        "kind": kind,
        "content": content,
        "type": memory.type,
        "score": score,
        "event_time": _episodic_anchor(memory),
        "ingest_time": memory.ingest_time.isoformat(),
        "origin_role": memory.origin_role,
        "src": memory.src,
        "source_kind": memory.source_kind,
        "state": memory.state,
        "supersedes": [link.target for link in memory.supersedes],
        "evidence": memory.evidence,
    }


class HeuristicResponseAgent(Agent):
    name = "response"

    def run(self, input_obj: ResponseInput) -> ResponseOutput:
        ranked = input_obj.retrieval.ranked_memories
        constraints = input_obj.retrieval.injected_constraints

        if not ranked and not constraints:
            return ResponseOutput(
                answer="我目前没有足够信息来回答这个问题。",
                abstained=True,
                abstain_reason="no_evidence",
            )

        cited_ids: list[str] = []
        context_only_ids: list[str] = []
        answer_parts: list[str] = []

        if ranked:
            primary = ranked[0].memory
            primary_text = primary.summary_for_retrieval if primary.sensitive and not input_obj.allow_sensitive_raw else primary.content
            answer_parts.append(primary_text or primary.content)
            cited_ids.append(primary.id)
            context_only_ids.extend(match.memory.id for match in ranked[1:])

        if constraints:
            constraint_texts: list[str] = []
            for item in constraints:
                text = item.summary_for_retrieval if item.sensitive and not input_obj.allow_sensitive_raw else item.content
                constraint_texts.append(text or item.content)
                cited_ids.append(item.id)
            answer_parts.append("需要注意：" + "；".join(constraint_texts))

        return ResponseOutput(
            answer=" ".join(part for part in answer_parts if part).strip(),
            cited_memory_ids=cited_ids,
            context_only_ids=context_only_ids,
            abstained=False,
        )


class ResponseDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str
    cited_memory_ids: list[str] = Field(default_factory=list)
    context_only_ids: list[str] = Field(default_factory=list)
    abstained: bool = False
    abstain_reason: str | None = None


class LLMResponseAgent(Agent):
    name = "response"

    def __init__(self, llm: StructuredLLM) -> None:
        self.llm = llm

    def run(self, input_obj: ResponseInput) -> ResponseOutput:
        evidence = []
        allowed_ids: set[str] = set()
        for match in input_obj.retrieval.ranked_memories:
            memory = match.memory
            allowed_ids.add(memory.id)
            evidence.append(_evidence_payload(memory, input_obj, kind="ranked", score=match.score))
        for memory in input_obj.retrieval.injected_constraints:
            allowed_ids.add(memory.id)
            evidence.append(_evidence_payload(memory, input_obj, kind="constraint", score=None))

        if not evidence:
            return ResponseOutput(
                answer="我目前没有足够信息来回答这个问题。",
                abstained=True,
                abstain_reason="no_evidence",
            )

        prompt = json.dumps(
            {
                "query": input_obj.query,
                "evidence": evidence,
                "rules": [
                    "Only cite ids that appear in evidence.",
                    "Use constraint evidence as hard requirements.",
                    "Consider ALL evidence items, not only the top-ranked one. Ranking is an approximate prior; the actual answer may live in a lower-ranked item, or may require combining multiple items.",
                    "If the query asks for a current/latest/most recent attribute, first collect all evidence items about the same entity and attribute. Do not choose an older literal item if a newer item updates, revises, replaces, or contradicts it. Cite the latest user-stated item even if it is lower-ranked or raw_session.",
                    "For non-current factual recall, a literal match is useful; for current/latest questions, recency and update semantics override literalness.",
                    "If no single item fully answers the query but several items together do (e.g. 'user did X' + 'user frequently shops at Y' and the query asks where X happened), you may synthesize across them and cite all the items you relied on. Do not invent details not supported by evidence.",
                    "Abstain only when no combination of evidence supports the answer. Partial information with the key attribute missing is still an abstention.",
                    "When the query asks about time elapsed / days between / when-did-X, use each episodic item's event_time field (ISO-8601) — compute deltas from event_time, not from ingest order. If event_time is null, the item cannot anchor a date-arithmetic answer.",
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        try:
            decision = self.llm.generate_structured(
                system_prompt=_RESPONSE_SYSTEM_PROMPT,
                user_prompt=prompt,
                response_model=ResponseDecision,
            )
            if not isinstance(decision, ResponseDecision):
                decision = ResponseDecision.model_validate(decision)
        except Exception as exc:
            return _heuristic_fallback(input_obj, evidence, exc)

        cited = [item for item in decision.cited_memory_ids if item in allowed_ids]
        context_only = [item for item in decision.context_only_ids if item in allowed_ids and item not in cited]
        if not cited and not decision.abstained:
            return ResponseOutput(
                answer="我目前没有足够信息来回答这个问题。",
                abstained=True,
                abstain_reason="invalid_citations",
            )
        return ResponseOutput(
            answer=decision.answer,
            cited_memory_ids=cited,
            context_only_ids=context_only,
            abstained=decision.abstained,
            abstain_reason=decision.abstain_reason,
        )


def _heuristic_fallback(
    input_obj: ResponseInput,
    evidence: list[dict],
    exc: Exception,
) -> ResponseOutput:
    ranked = input_obj.retrieval.ranked_memories
    constraints = input_obj.retrieval.injected_constraints
    if not ranked and not constraints:
        return ResponseOutput(
            answer="我目前没有足够信息来回答这个问题。",
            context_only_ids=[item["id"] for item in evidence],
            abstained=True,
            abstain_reason=f"llm_parse_failure_no_evidence: {exc}",
        )

    cited_ids: list[str] = []
    answer_parts: list[str] = []
    if ranked:
        primary = ranked[0].memory
        text = primary.summary_for_retrieval if primary.sensitive and not input_obj.allow_sensitive_raw else primary.content
        answer_parts.append(text or primary.content)
        cited_ids.append(primary.id)
    constraint_texts: list[str] = []
    for item in constraints:
        text = item.summary_for_retrieval if item.sensitive and not input_obj.allow_sensitive_raw else item.content
        constraint_texts.append(text or item.content)
        cited_ids.append(item.id)
    if constraint_texts:
        answer_parts.append("需要注意：" + "；".join(constraint_texts))
    context_only_ids = [match.memory.id for match in ranked[1:]]
    return ResponseOutput(
        answer=" ".join(part for part in answer_parts if part).strip(),
        cited_memory_ids=cited_ids,
        context_only_ids=context_only_ids,
        abstained=False,
        abstain_reason=f"llm_parse_failure_fallback_heuristic: {exc}",
    )


_RESPONSE_SYSTEM_PROMPT = '''
You are the Response Agent. Answer the user's query using only the provided
evidence items. Return structured JSON only.

# Core rules
1. Treat evidence as a SET — the answer may live in a lower-ranked item or
   require combining several. Don't auto-pick rank 1.
2. For current/latest-state queries ("how many X do I have now", "what brand
   am I using", "most recent", "currently"), first collect all evidence items
   about the same entity and attribute. Prefer the latest user-stated item by
   event_time / ingest_time when multiple items state different values. Do not
   choose an older literal item if a newer item updates, revises, replaces, or
   contradicts it.
3. For past-state queries ("previously", "back when", "before X"), pick the
   item whose time anchor matches the query's temporal scope.
4. Abstain when no evidence literally states the answer. For factual recall
   ("how often do I see Dr. X"), the named entity must appear literally in
   evidence — partial / similar entities don't count.
5. origin_role="user" outweighs origin_role="assistant" stating the same
   attribute. Assistant echoes ("congrats on N!") are not user-stated facts.
6. fact / preference / constraint items outweigh raw_session items when
   they share topic_id and state the same attribute.
7. Preserve proper nouns verbatim from evidence. Cite ids actually used.

'''
