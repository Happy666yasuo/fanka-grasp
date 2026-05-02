from __future__ import annotations

import sys
import os
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../embodied_agent/src')))
from embodied_agent.contracts import PlannerStep, ExecutorResult, ContactState, FailureRecord

from .new_skills import PressSkill, PushSkill, PullSkill, RotateSkill, BaseScriptedSkill


class _BuiltinSkill(BaseScriptedSkill):
    """Placeholder for built-in simulator skills (pick, place) handled by UnifiedSkillExecutor."""

    def __init__(self, name: str) -> None:
        self.skill_name = name

    def execute(self, simulation, **kwargs):  # pragma: no cover
        raise NotImplementedError(f"_BuiltinSkill '{self.skill_name}' is handled by UnifiedSkillExecutor directly.")


class SkillRegistry:
    """Unified registry for all available skills."""

    def __init__(self) -> None:
        self._skills: dict[str, BaseScriptedSkill] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        self.register("pick", _BuiltinSkill("pick"))
        self.register("place", _BuiltinSkill("place"))
        self.register("press", PressSkill())
        self.register("push", PushSkill())
        self.register("pull", PullSkill())
        self.register("rotate", RotateSkill())

    def register(self, name: str, skill: BaseScriptedSkill) -> None:
        if not name or not isinstance(name, str):
            raise ValueError(f"Skill name must be a non-empty string, got: {name!r}")
        self._skills[name] = skill

    def get_skill(self, name: str) -> BaseScriptedSkill:
        if name not in self._skills:
            available = ", ".join(sorted(self._skills.keys()))
            raise KeyError(f"Skill '{name}' not registered. Available: {available}")
        return self._skills[name]

    def list_skills(self) -> list[str]:
        return sorted(self._skills.keys())

    def has_skill(self, name: str) -> bool:
        return name in self._skills

    def __len__(self) -> int:
        return len(self._skills)

    def __contains__(self, name: str) -> bool:
        return self.has_skill(name)
