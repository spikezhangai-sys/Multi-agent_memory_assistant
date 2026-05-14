from driftscope.core.scope_compat import ScopeRules
from driftscope.core.schema import Scope


def test_personal_visibility() -> None:
    rules = ScopeRules.load_default()
    current = Scope(kind="personal")
    assert rules.can_read(current, Scope(kind="global"))
    assert rules.can_read(current, Scope(kind="personal"))
    assert not rules.can_read(current, Scope(kind="project", ref="alpha"))


def test_project_visibility_and_targeting_are_same_ref_only() -> None:
    rules = ScopeRules.load_default()
    current = Scope(kind="project", ref="alpha")
    assert rules.can_read(current, Scope(kind="project", ref="alpha"))
    assert not rules.can_read(current, Scope(kind="project", ref="beta"))
    assert rules.can_target(current, Scope(kind="project", ref="alpha"))
    assert not rules.can_target(current, Scope(kind="personal"))

