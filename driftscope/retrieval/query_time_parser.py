from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QueryTimeHint:
    """Parsed temporal anchor for a query.

    ``center`` is the instant used for time-proximity scoring. ``start``/``end``
    are kept for future range-based retrieval; a point-in-time hint sets both
    to the center.
    """

    center: datetime
    start: datetime
    end: datetime


class _TimeParserLLM(Protocol):
    def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[BaseModel],
    ) -> BaseModel | dict:
        ...


class _QueryTimeDecision(BaseModel):
    model_config = ConfigDict(extra="ignore")

    has_time_hint: bool = False
    start: datetime | None = None
    end: datetime | None = None
    reason: str = Field(default="")


class QueryTimeParser:
    """LLM-backed extractor that pulls a temporal anchor out of a query.

    Returns ``None`` when the query carries no temporal intent. The LLM is
    invoked with the current timestamp so it can resolve relative phrases like
    "last week" or "tomorrow".
    """

    def __init__(self, llm: _TimeParserLLM) -> None:
        self.llm = llm

    def parse(self, *, query: str, now: datetime) -> QueryTimeHint | None:
        if not query.strip():
            return None
        prompt = (
            f"current_time: {now.isoformat()}\n"
            f"query: {query}\n"
            "Extract a temporal anchor if and only if the query refers to a specific time window "
            "(yesterday, last week, tomorrow, next month, a named month, etc.). "
            "Return JSON with has_time_hint=true and ISO-8601 start/end only when confident."
        )
        try:
            decision = self.llm.generate_structured(
                system_prompt=_SYSTEM_PROMPT,
                user_prompt=prompt,
                response_model=_QueryTimeDecision,
            )
            if not isinstance(decision, _QueryTimeDecision):
                decision = _QueryTimeDecision.model_validate(decision)
        except Exception as exc:
            logger.warning("QueryTimeParser.parse failed (%s): %s", type(exc).__name__, exc)
            return None

        if not decision.has_time_hint or decision.start is None:
            return None
        start = decision.start
        end = decision.end or start
        if end < start:
            end = start
        center = start + (end - start) / 2
        return QueryTimeHint(center=center, start=start, end=end)


_SYSTEM_PROMPT = """You are a query time parser for a personal memory system.
Decide whether the user's query anchors its answer to a specific date or date range.
Return JSON only. Set has_time_hint=false when the query is time-agnostic.
When you set has_time_hint=true, start/end must be ISO-8601 timestamps in UTC.
Prefer narrow windows: "yesterday" is a 24h window, "last week" is 7 days, "last month" is ~30 days.
Leave end=null for truly instant anchors; the caller will treat it as a point-in-time.
"""
