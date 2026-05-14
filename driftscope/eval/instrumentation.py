from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from driftscope.core.schema import TurnInput, TurnResult


class JsonlTurnLogger:
    _path_locks: dict[Path, threading.Lock] = {}
    _registry_lock = threading.Lock()

    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        resolved = self.path.resolve()
        with JsonlTurnLogger._registry_lock:
            lock = JsonlTurnLogger._path_locks.get(resolved)
            if lock is None:
                lock = threading.Lock()
                JsonlTurnLogger._path_locks[resolved] = lock
        self._lock = lock

    def log_turn(self, turn: TurnInput, result: TurnResult, extras: dict[str, Any] | None = None) -> None:
        payload = {
            "timestamp": turn.timestamp.isoformat(),
            "scope": turn.scope.model_dump(mode="json"),
            "user_input": turn.user_input,
            "query": turn.query,
            "result": result.model_dump(mode="json"),
        }
        if extras:
            payload["extras"] = extras
        line = json.dumps(payload, ensure_ascii=False) + "\n"
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line)

