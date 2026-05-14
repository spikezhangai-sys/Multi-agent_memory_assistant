from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import json
from pathlib import Path
from typing import Any

from driftscope.core.schema import Scope, TurnInput


@dataclass(frozen=True)
class LongMemEvalInstance:
    question_id: str
    question: str
    replay_turns: list[TurnInput]
    question_turn: TurnInput
    gold_answer: str | None = None


class LongMemEvalAdapter:
    def __init__(self, *, default_scope: Scope | None = None, user_turns_only: bool = False) -> None:
        self.default_scope = default_scope or Scope(kind="personal")
        self.user_turns_only = user_turns_only

    def load(self, path: str | Path, *, limit: int | None = None) -> list[LongMemEvalInstance]:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError("LongMemEval dataset must be a JSON array")
        if limit is not None:
            raw = raw[:limit]
        return [self._normalize_instance(item, index) for index, item in enumerate(raw)]

    def _normalize_instance(self, item: dict[str, Any], index: int) -> LongMemEvalInstance:
        question_id = str(item.get("question_id") or item.get("id") or f"q_{index}")
        question = str(item.get("question") or item.get("query") or "").strip()
        if not question:
            raise ValueError(f"instance {question_id} is missing question text")

        raw_turns = item.get("haystack_sessions") or item.get("haystack") or item.get("context") or []
        if not isinstance(raw_turns, list):
            raise ValueError(f"instance {question_id} has invalid haystack_sessions")
        haystack_dates = item.get("haystack_dates") or []
        if haystack_dates and (not isinstance(haystack_dates, list) or len(haystack_dates) != len(raw_turns)):
            raise ValueError(f"instance {question_id} has invalid haystack_dates")

        base_time = self._coerce_timestamp(item.get("timestamp")) or datetime(2026, 1, 1, tzinfo=UTC)
        replay_turns: list[TurnInput] = []
        for session_idx, raw_session in enumerate(raw_turns):
            session_time = self._coerce_timestamp(haystack_dates[session_idx]) if haystack_dates else None
            session_time = session_time or self._coerce_turn_timestamp(raw_session, base_time, session_idx)
            session_turns = raw_session if isinstance(raw_session, list) else [raw_session]
            if not isinstance(session_turns, list):
                raise ValueError(f"instance {question_id} has invalid session payload")
            for turn_idx, raw_turn in enumerate(session_turns):
                if self.user_turns_only and not self._should_ingest_turn(raw_turn):
                    continue
                turn_text = self._extract_turn_text(raw_turn)
                if turn_text is None:
                    continue
                replay_turns.append(
                    TurnInput(
                        origin_role=self._turn_role(raw_turn),
                        user_input=turn_text,
                        query=None,
                        scope=self.default_scope,
                        timestamp=session_time + timedelta(seconds=turn_idx),
                    )
                )

        question_time = self._coerce_timestamp(item.get("question_timestamp"))
        if question_time is None:
            question_time = base_time + timedelta(seconds=max(len(replay_turns), 1))

        return LongMemEvalInstance(
            question_id=question_id,
            question=question,
            replay_turns=replay_turns,
            question_turn=TurnInput(
                user_input=None,
                query=question,
                scope=self.default_scope,
                timestamp=question_time,
            ),
            gold_answer=item.get("answer"),
        )

    def _should_ingest_turn(self, raw_turn: Any) -> bool:
        if not isinstance(raw_turn, dict):
            return True
        role = raw_turn.get("role")
        if role is None:
            return True
        return str(role).strip().lower() == "user"

    def _turn_role(self, raw_turn: Any) -> str:
        if not isinstance(raw_turn, dict):
            return "user"
        role = raw_turn.get("role")
        if role is None:
            return "user"
        normalized = str(role).strip().lower()
        return "assistant" if normalized == "assistant" else "user"

    def _extract_turn_text(self, raw_turn: Any) -> str | None:
        if isinstance(raw_turn, str):
            text = raw_turn.strip()
        elif isinstance(raw_turn, dict):
            text = str(
                raw_turn.get("text")
                or raw_turn.get("content")
                or raw_turn.get("user_input")
                or raw_turn.get("utterance")
                or ""
            ).strip()
        else:
            raise ValueError("haystack turn has unsupported payload type")
        if not text:
            return None
        return text

    def _coerce_turn_timestamp(self, raw_turn: Any, base_time: datetime, offset: int) -> datetime:
        if isinstance(raw_turn, dict):
            parsed = self._coerce_timestamp(raw_turn.get("timestamp"))
            if parsed is not None:
                return parsed
        return base_time + timedelta(seconds=offset)

    def _coerce_timestamp(self, value: Any) -> datetime | None:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(value, tz=UTC)
        if isinstance(value, str):
            normalized = value.strip()
            if not normalized:
                return None
            normalized = normalized.replace("Z", "+00:00")
            for fmt in ("%Y/%m/%d (%a) %H:%M", "%Y/%m/%d %H:%M"):
                try:
                    return datetime.strptime(normalized, fmt).replace(tzinfo=UTC)
                except ValueError:
                    pass
            try:
                parsed = datetime.fromisoformat(normalized)
            except ValueError:
                return None
            return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)
        return None
