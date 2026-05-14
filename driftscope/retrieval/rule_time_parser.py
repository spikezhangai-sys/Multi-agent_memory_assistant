from __future__ import annotations

import re
from datetime import datetime, timedelta

from driftscope.retrieval.query_time_parser import QueryTimeHint

_PATTERNS: list[tuple[re.Pattern, "callable"]] = [
    (re.compile(r"(\d+)\s+days?\s+ago", re.IGNORECASE), lambda m: (int(m.group(1)), 2)),
    (re.compile(r"a\s+couple\s+(?:of\s+)?days?\s+ago", re.IGNORECASE), lambda m: (2, 2)),
    (re.compile(r"\byesterday\b", re.IGNORECASE), lambda m: (1, 1)),
    (re.compile(r"\btoday\b", re.IGNORECASE), lambda m: (0, 1)),
    (re.compile(r"\btomorrow\b", re.IGNORECASE), lambda m: (-1, 1)),
    (re.compile(r"a\s+week\s+ago", re.IGNORECASE), lambda m: (7, 3)),
    (re.compile(r"(\d+)\s+weeks?\s+ago", re.IGNORECASE), lambda m: (int(m.group(1)) * 7, 5)),
    (re.compile(r"\blast\s+week\b", re.IGNORECASE), lambda m: (7, 3)),
    (re.compile(r"a\s+month\s+ago", re.IGNORECASE), lambda m: (30, 7)),
    (re.compile(r"(\d+)\s+months?\s+ago", re.IGNORECASE), lambda m: (int(m.group(1)) * 30, 10)),
    (re.compile(r"\blast\s+month\b", re.IGNORECASE), lambda m: (30, 7)),
    (re.compile(r"\blast\s+year\b", re.IGNORECASE), lambda m: (365, 30)),
    (re.compile(r"a\s+year\s+ago", re.IGNORECASE), lambda m: (365, 30)),
    (re.compile(r"\brecently\b", re.IGNORECASE), lambda m: (14, 14)),
]


class RuleBasedQueryTimeParser:
    """Regex-only temporal anchor extractor.

    Ported from mempalace `parse_time_offset_days`
    (benchmarks/longmemeval_bench.py:1540). Returns a `QueryTimeHint` whose
    center is `now - offset_days` and whose half-window matches the unit's
    granularity. No LLM call.
    """

    def parse(self, *, query: str, now: datetime) -> QueryTimeHint | None:
        if not query.strip():
            return None
        for pattern, extractor in _PATTERNS:
            match = pattern.search(query)
            if not match:
                continue
            offset_days, half_window = extractor(match)
            center = now - timedelta(days=offset_days)
            half = timedelta(days=half_window)
            return QueryTimeHint(center=center, start=center - half, end=center + half)
        return None
