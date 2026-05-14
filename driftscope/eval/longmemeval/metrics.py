from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def build_summary_from_turn_logs(turns_path: str | Path, *, num_questions: int) -> dict[str, Any]:
    path = Path(turns_path)
    if not path.exists():
        return {
            "num_turns": 0,
            "num_questions": num_questions,
            "agent_call_counts": {},
            "write_apply_rate": 0.0,
            "abstain_rate": 0.0,
            "avg_agents_per_turn": 0.0,
        }

    rows: list[dict[str, Any]] = []
    malformed = 0
    for line_no, line in enumerate(path.read_text(encoding="utf-8").split("\n"), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            malformed += 1
            logger.warning("turns.jsonl line %d malformed (skipped): %s", line_no, exc)
    if malformed:
        logger.warning("turns.jsonl skipped %d malformed line(s)", malformed)
    agent_counts: dict[str, int] = {}
    write_applied = 0
    write_attempted = 0
    abstained = 0
    response_turns = 0
    total_agents = 0
    for row in rows:
        result = row.get("result", {})
        extras = row.get("extras") or {}
        agents = result.get("agents_called", [])
        total_agents += len(agents)
        for agent in agents:
            agent_counts[agent] = agent_counts.get(agent, 0) + 1

        proposals = extras.get("update_proposals")
        if proposals is None and extras.get("update_proposal") is not None:
            proposals = [extras["update_proposal"]]
        proposals = proposals or []
        non_ignore = [p for p in proposals if p and p.get("intent") != "ignore"]
        if non_ignore:
            write_attempted += 1
            if result.get("write_applied"):
                write_applied += 1

        if "response" in agents or result.get("answer") is not None:
            response_turns += 1
            if result.get("abstained"):
                abstained += 1

    num_turns = len(rows)
    summary: dict[str, Any] = {
        "num_turns": num_turns,
        "num_questions": num_questions,
        "num_response_turns": response_turns,
        "num_write_attempts": write_attempted,
        "agent_call_counts": agent_counts,
        "write_apply_rate": write_applied / write_attempted if write_attempted else 0.0,
        "abstain_rate": abstained / response_turns if response_turns else 0.0,
        "avg_agents_per_turn": total_agents / num_turns if num_turns else 0.0,
    }
    if malformed:
        summary["num_malformed_lines"] = malformed
    return summary

