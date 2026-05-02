from __future__ import annotations

import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.skills.skill_registry import SkillRegistry
from src.skills.new_skills import PressSkill, PushSkill, PullSkill, RotateSkill


class TestSkillRegistry(unittest.TestCase):
    def setUp(self):
        self.registry = SkillRegistry()

    def test_registry_initialized_with_default_skills(self):
        skills = self.registry.list_skills()
        self.assertIn("pick", skills)
        self.assertIn("place", skills)
        self.assertIn("press", skills)
        self.assertIn("push", skills)
        self.assertIn("pull", skills)
        self.assertIn("rotate", skills)
        self.assertEqual(len(skills), 6)

    def test_get_skill_returns_correct_type(self):
        press = self.registry.get_skill("press")
        self.assertIsInstance(press, PressSkill)
        push = self.registry.get_skill("push")
        self.assertIsInstance(push, PushSkill)
        pull = self.registry.get_skill("pull")
        self.assertIsInstance(pull, PullSkill)
        rotate = self.registry.get_skill("rotate")
        self.assertIsInstance(rotate, RotateSkill)

    def test_get_skill_unknown_raises_key_error(self):
        with self.assertRaises(KeyError):
            self.registry.get_skill("nonexistent_skill")

    def test_register_new_skill(self):
        self.registry.register("custom_press", PressSkill())
        self.assertIn("custom_press", self.registry)
        self.assertEqual(len(self.registry), 7)

    def test_register_empty_name_raises(self):
        with self.assertRaises(ValueError):
            self.registry.register("", PressSkill())

    def test_has_skill(self):
        self.assertTrue(self.registry.has_skill("press"))
        self.assertFalse(self.registry.has_skill("imaginary"))

    def test_list_skills_returns_sorted(self):
        skills = self.registry.list_skills()
        self.assertEqual(skills, sorted(skills))

    def test_contains_operator(self):
        self.assertIn("press", self.registry)
        self.assertNotIn("magic", self.registry)


if __name__ == "__main__":
    unittest.main()
