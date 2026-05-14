from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel


class RuleBasedConflictLLM:
    """Deterministic stand-in for update/conflict/response during local eval."""

    def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[BaseModel],
    ) -> BaseModel | dict[str, Any]:
        payload = json.loads(user_prompt)
        if "proposal" in payload:
            return self._handle_conflict(payload, response_model)
        if "turns" in payload and "allowed_topics" in payload:
            return self._handle_batch_update(payload, response_model)
        if "user_input" in payload and "allowed_topics" in payload:
            return self._handle_update(payload, response_model)
        if "query" in payload and "evidence" in payload:
            return self._handle_response(payload, response_model)
        raise ValueError("Unsupported mock payload shape")

    def _handle_conflict(self, payload: dict[str, Any], response_model: type[BaseModel]) -> BaseModel:
        proposal = payload["proposal"]
        candidates = payload["candidates"]
        ambiguous = payload["ambiguous_candidates"]
        intent = proposal["intent"]

        if intent == "add":
            return response_model.model_validate(
                {
                    "action": "apply_add",
                    "confidence": 0.95,
                    "reason": "rule-based add",
                }
            )

        if intent == "supersede_full":
            if ambiguous:
                return response_model.model_validate(
                    {
                        "action": "request_clarification",
                        "confidence": 0.6,
                        "reason": "ambiguous candidates",
                        "clarification_question": "你是想覆盖哪一条之前的记忆？",
                    }
                )
            if not candidates:
                return response_model.model_validate(
                    {
                        "action": "apply_add",
                        "confidence": 0.7,
                        "reason": "no compatible target found",
                    }
                )
            return response_model.model_validate(
                {
                    "action": "confirm_supersede",
                    "target_id": candidates[0]["id"],
                    "transition_type": proposal["transition_type"],
                    "confidence": 0.9,
                    "reason": "highest selector score",
                }
            )

        if intent == "revoke":
            if ambiguous or not candidates:
                return response_model.model_validate(
                    {
                        "action": "request_clarification",
                        "confidence": 0.6,
                        "reason": "revoke target is ambiguous",
                        "clarification_question": "你是想撤销哪一条之前的记忆？",
                    }
                )
            return response_model.model_validate(
                {
                    "action": "confirm_revoke",
                    "target_id": candidates[0]["id"],
                    "transition_type": "user_revoked",
                    "confidence": 0.9,
                    "reason": "single revoke target",
                }
            )

        return response_model.model_validate(
            {
                "action": "reject",
                "confidence": 0.0,
                "reason": f"unsupported intent: {intent}",
            }
        )

    def _handle_update(self, payload: dict[str, Any], response_model: type[BaseModel]) -> BaseModel:
        return response_model.model_validate(
            self._predict_update_payload(
                user_input=payload["user_input"],
                allowed_topics=payload["allowed_topics"],
                nearby_memories=payload.get("nearby_memories", []),
            )
        )

    def _handle_batch_update(self, payload: dict[str, Any], response_model: type[BaseModel]) -> BaseModel:
        proposals: list[dict[str, Any]] = []
        for turn in payload.get("turns", []):
            single_payload = self._predict_update_payload(
                user_input=turn["user_input"],
                allowed_topics=payload["allowed_topics"],
                nearby_memories=payload.get("nearby_memories", []),
            )
            if single_payload.get("intent") == "ignore":
                continue
            single_payload["source_turn_index"] = turn["source_turn_index"]
            proposals.append(single_payload)
        return response_model.model_validate({"proposals": proposals})

    def _predict_update_payload(
        self,
        *,
        user_input: str,
        allowed_topics: list[dict[str, Any]],
        nearby_memories: list[dict[str, Any]],
    ) -> dict[str, Any]:
        lowered = user_input.lower()

        if any(token in lowered for token in ["恢复", "rollback", "undo"]):
            return {
                "intent": "rollback",
                "keywords": _tokenize(user_input),
            }

        if any(token in lowered for token in ["不要记", "删掉", "删除", "取消"]):
            topic_id = _guess_topic_id(user_input, allowed_topics)
            return {
                "intent": "revoke",
                "topic_id": topic_id,
                "keywords": _tokenize(user_input),
                "transition_type": "user_revoked",
            }

        topic_id = _guess_topic_id(user_input, allowed_topics)
        if topic_id is None:
            return {"intent": "ignore"}

        candidate_type = _default_type_for_topic(topic_id, allowed_topics)
        nearby_same_topic = [item for item in nearby_memories if item.get("topic_id") == topic_id]
        intent = "supersede_full" if nearby_same_topic and any(
            token in lowered for token in ["现在", "搬到", "改成", "换成", "最近", "不再"]
        ) else "add"
        transition_type = None
        if intent == "supersede_full":
            transition_type = "preference_shifted" if candidate_type == "preference" else "corrected"

        return {
            "intent": intent,
            "candidate_content": user_input,
            "candidate_type": candidate_type,
            "topic_id": topic_id,
            "keywords": _tokenize(user_input),
            "transition_type": transition_type,
        }

    def _handle_response(self, payload: dict[str, Any], response_model: type[BaseModel]) -> BaseModel:
        evidence = payload.get("evidence", [])
        if not evidence:
            return response_model.model_validate(
                {
                    "answer": "我目前没有足够信息来回答这个问题。",
                    "cited_memory_ids": [],
                    "context_only_ids": [],
                    "abstained": True,
                    "abstain_reason": "no_evidence",
                }
            )

        ranked = [item for item in evidence if item.get("kind") == "ranked"]
        constraints = [item for item in evidence if item.get("kind") == "constraint"]
        answer_parts: list[str] = []
        cited: list[str] = []
        context_only: list[str] = []

        if ranked:
            answer_parts.append(ranked[0]["content"])
            cited.append(ranked[0]["id"])
            context_only.extend(item["id"] for item in ranked[1:])

        if constraints:
            answer_parts.append("需要注意：" + "；".join(item["content"] for item in constraints))
            cited.extend(item["id"] for item in constraints)

        return response_model.model_validate(
            {
                "answer": " ".join(part for part in answer_parts if part).strip(),
                "cited_memory_ids": cited,
                "context_only_ids": context_only,
                "abstained": False,
                "abstain_reason": None,
            }
        )


def _guess_topic_id(user_input: str, allowed_topics: list[dict[str, Any]]) -> str | None:
    lowered = user_input.lower()
    best_topic: str | None = None
    best_score = 0
    for topic in allowed_topics:
        topic_id = _topic_id_from_allowed_topic(topic)
        if topic_id is None:
            continue
        score = 0
        for keyword in topic.get("keywords", []):
            if keyword.lower() in lowered:
                score += 2
        for suffix in topic.get("seed_leaf_suffixes", []):
            if str(suffix).lower().replace("_", " ") in lowered:
                score += 2
        for part in topic_id.split("."):
            if part.lower() in lowered:
                score += 1
        if score > best_score:
            best_score = score
            best_topic = topic_id
    return best_topic


def _default_type_for_topic(topic_id: str, allowed_topics: list[dict[str, Any]]) -> str:
    for topic in allowed_topics:
        allowed_topic_id = _topic_id_from_allowed_topic(topic)
        category = topic.get("category")
        if allowed_topic_id == topic_id or (category and str(topic_id).startswith(f"{category}.")):
            return topic["default_type"]
    return "fact"


def _topic_id_from_allowed_topic(topic: dict[str, Any]) -> str | None:
    topic_id = topic.get("topic_id")
    if topic_id:
        return str(topic_id)
    category = topic.get("category")
    if not category:
        return None
    suffixes = topic.get("seed_leaf_suffixes") or []
    if suffixes:
        return f"{category}.{suffixes[0]}"
    return f"{category}.other"


def _tokenize(text: str) -> list[str]:
    tokens = [char for char in text.strip() if not char.isspace()]
    return tokens[:8]
