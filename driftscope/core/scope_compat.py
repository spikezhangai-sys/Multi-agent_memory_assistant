from __future__ import annotations

from importlib import resources
from pathlib import Path

import yaml

from driftscope.core.schema import Scope


class ScopeRules:
    def __init__(self, visibility: dict[str, list[str]], compat: dict[str, list[str]]) -> None:
        self.visibility = visibility
        self.compat = compat

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ScopeRules":
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        return cls(
            visibility=payload.get("visibility", {}),
            compat=payload.get("compat", {}),
        )

    @classmethod
    def load_default(cls) -> "ScopeRules":
        resource = resources.files("driftscope.config").joinpath("scope_compat.yaml")
        return cls.from_yaml(resource)

    def can_read(self, current: Scope, candidate: Scope) -> bool:
        rules = self.visibility.get(current.kind, [])
        return any(self._rule_matches(rule, current, candidate) for rule in rules)

    def can_target(self, current: Scope, target: Scope) -> bool:
        rules = self.compat.get(current.kind, [])
        return any(self._rule_matches(rule, current, target) for rule in rules)

    def _rule_matches(self, rule: str, current: Scope, other: Scope) -> bool:
        if rule == "global":
            return other.kind == "global"
        if rule == "personal":
            return other.kind == "personal"
        if rule == "project_same_ref":
            return current.kind == "project" and other.kind == "project" and current.ref == other.ref
        if rule == "session_same_ref":
            return current.kind == "session" and other.kind == "session" and current.ref == other.ref
        raise ValueError(f"unknown scope rule token: {rule}")

