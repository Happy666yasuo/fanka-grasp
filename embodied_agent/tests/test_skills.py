from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embodied_agent.rl_support import policy_spec_from_dict
from embodied_agent.types import PlanStep, PostCondition
from embodied_agent.skills import SkillLibrary, _stabilize_pick_action


class _FakeLearnedPolicy:
    def __init__(self, skill_name: str, use_staging: bool = True) -> None:
        self.spec = SimpleNamespace(skill_name=skill_name, use_staging=use_staging)
        self.calls: list[tuple[str, str, str]] = []

    def execute(self, simulation, object_name: str, zone_name: str, gui_delay: float = 0.0) -> bool:
        self.calls.append((simulation.__class__.__name__, object_name, zone_name))
        return True


class _FakeSimulation:
    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []
        self.captured_states: list[dict[str, object]] = []
        self.held_object_name: str | None = "red_block"
        self.object_ids = {"red_block": 1}
        self.zone_positions = {"green_zone": (0.5, -0.1, 0.623), "blue_zone": (0.6, 0.1, 0.623)}

    def prepare_pick_staging_pose(self, object_name: str) -> None:
        self.events.append(("pick_stage", object_name))

    def normalize_held_object_for_place(self, object_name: str) -> None:
        self.events.append(("place_carry", object_name))

    def prepare_place_staging_pose(self, zone_name: str) -> None:
        self.events.append(("place_stage", zone_name))

    def canonicalize_held_object_for_place(self, object_name: str, zone_name: str) -> None:
        self.events.append(("place_snap", f"{object_name}:{zone_name}"))

    def pick_object(self, object_name: str) -> None:
        self.held_object_name = object_name
        self.events.append(("scripted_pick", object_name))

    def is_object_held(self, object_name: str) -> bool:
        self.events.append(("holding_check", object_name))
        return self.held_object_name == object_name

    def is_pick_success(self, object_name: str) -> bool:
        self.events.append(("pick_check", object_name))
        return object_name == "red_block"

    def is_place_success(self, object_name: str, zone_name: str) -> bool:
        self.events.append(("place_check", f"{object_name}:{zone_name}"))
        return object_name == "red_block" and zone_name == "green_zone"

    def capture_skill_state(self, object_name: str, zone_name: str) -> dict[str, object]:
        snapshot = {
            "object_name": object_name,
            "zone_name": zone_name,
            "held_object_name": self.held_object_name,
            "holding_target_object": self.held_object_name == object_name,
            "gripper_open_ratio": 0.0,
            "end_effector_position": [0.5, 0.0, 0.7],
            "end_effector_orientation": [0.0, 0.0, 0.0, 1.0],
            "object_position": [0.55, 0.0, 0.63],
            "object_orientation": [0.0, 0.0, 0.0, 1.0],
            "zone_position": list(self.zone_positions[zone_name]),
            "ee_to_object": [0.05, 0.0, -0.07],
            "object_to_zone": [-0.05, -0.1, -0.01],
            "held_local_position": [0.0, 0.0, -0.06],
            "held_local_orientation": [0.0, 0.0, 0.0, 1.0],
            "metrics": {
                "ee_object_distance": 0.09,
                "object_zone_distance_xy": 0.11,
                "object_height": 0.63,
                "holding_flag": 1.0,
                "lift_progress": 0.01,
                "gripper_open_ratio": 0.0,
            },
        }
        self.captured_states.append(snapshot)
        return snapshot


class SkillLibraryStagingTests(unittest.TestCase):
    def test_pick_stabilization_closes_gripper_and_blocks_downward_motion(self) -> None:
        delta_position, gripper_command, stabilized = _stabilize_pick_action(
            skill_name="pick",
            holding_object=True,
            delta_position=(0.01, -0.02, -0.03),
            gripper_command=0.7,
        )

        self.assertTrue(stabilized)
        self.assertEqual(delta_position, (0.01, -0.02, 0.0))
        self.assertEqual(gripper_command, -0.8)

    def test_pick_stabilization_does_not_modify_non_holding_action(self) -> None:
        delta_position, gripper_command, stabilized = _stabilize_pick_action(
            skill_name="pick",
            holding_object=False,
            delta_position=(0.01, -0.02, -0.03),
            gripper_command=0.7,
        )

        self.assertFalse(stabilized)
        self.assertEqual(delta_position, (0.01, -0.02, -0.03))
        self.assertEqual(gripper_command, 0.7)

    def test_pick_policy_stages_before_execution(self) -> None:
        simulation = _FakeSimulation()
        policy = _FakeLearnedPolicy(skill_name="pick", use_staging=True)
        library = SkillLibrary(simulation=simulation, learned_skill_policies={"pick": policy})

        library._execute_pick("red_block", "green_zone")

        self.assertEqual(simulation.events, [("pick_stage", "red_block")])
        self.assertEqual(policy.calls, [("_FakeSimulation", "red_block", "green_zone")])

    def test_place_policy_stages_before_execution(self) -> None:
        simulation = _FakeSimulation()
        policy = _FakeLearnedPolicy(skill_name="place", use_staging=True)
        library = SkillLibrary(simulation=simulation, learned_skill_policies={"place": policy})

        library._execute_place(object_name="red_block", zone_name="green_zone")

        self.assertEqual(simulation.events, [("place_stage", "green_zone")])
        self.assertEqual(policy.calls, [("_FakeSimulation", "red_block", "green_zone")])
        snapshot = library.debug_snapshot()
        self.assertEqual(
            [record["capture_stage"] for record in snapshot["runtime_alignment"]["place_entry_states"]],
            ["pre_place_runtime_raw", "pre_place_policy_entry"],
        )

    def test_staging_can_be_disabled_per_policy(self) -> None:
        simulation = _FakeSimulation()
        policy = _FakeLearnedPolicy(skill_name="pick", use_staging=False)
        library = SkillLibrary(simulation=simulation, learned_skill_policies={"pick": policy})

        library._execute_pick("red_block", "green_zone")

        self.assertEqual(simulation.events, [])
        self.assertEqual(policy.calls, [("_FakeSimulation", "red_block", "green_zone")])

    def test_execute_pick_uses_zone_parameter_from_step(self) -> None:
        simulation = _FakeSimulation()
        policy = _FakeLearnedPolicy(skill_name="pick", use_staging=True)
        library = SkillLibrary(simulation=simulation, learned_skill_policies={"pick": policy})

        action = library.execute(
            PlanStep(action="pick", target="red_block", parameters={"zone": "blue_zone"})
        )

        self.assertEqual(action, "pick:red_block")
        self.assertEqual(policy.calls, [("_FakeSimulation", "red_block", "blue_zone")])

    def test_scripted_pick_handoff_normalizes_carry_state_for_learned_place(self) -> None:
        simulation = _FakeSimulation()
        place_policy = _FakeLearnedPolicy(skill_name="place", use_staging=True)
        library = SkillLibrary(simulation=simulation, learned_skill_policies={"place": place_policy})

        library._execute_pick("red_block", "green_zone")
        library._execute_place(object_name="red_block", zone_name="green_zone")

        self.assertEqual(
            simulation.events,
            [
                ("scripted_pick", "red_block"),
                ("place_carry", "red_block"),
                ("place_stage", "green_zone"),
                ("place_snap", "red_block:green_zone"),
            ],
        )
        self.assertEqual(library.last_pick_controller, "scripted")
        snapshot = library.debug_snapshot()
        self.assertEqual(
            snapshot["runtime_alignment"]["post_pick_states"][0]["capture_stage"],
            "post_pick_success",
        )
        self.assertEqual(
            snapshot["runtime_alignment"]["place_entry_states"][0]["pick_controller"],
            "scripted",
        )

    def test_learned_pick_does_not_force_scripted_place_handoff_normalization(self) -> None:
        simulation = _FakeSimulation()
        pick_policy = _FakeLearnedPolicy(skill_name="pick", use_staging=True)
        place_policy = _FakeLearnedPolicy(skill_name="place", use_staging=True)
        library = SkillLibrary(
            simulation=simulation,
            learned_skill_policies={"pick": pick_policy, "place": place_policy},
        )

        library._execute_pick("red_block", "green_zone")
        simulation.events.clear()
        library._execute_place(object_name="red_block", zone_name="green_zone")

        self.assertEqual(simulation.events, [("place_stage", "green_zone")])
        self.assertEqual(library.last_pick_controller, "learned")
        snapshot = library.debug_snapshot()
        self.assertEqual(
            snapshot["runtime_alignment"]["post_pick_states"][0]["pick_controller"],
            "learned",
        )

    def test_post_condition_checks_delegate_to_simulation(self) -> None:
        simulation = _FakeSimulation()
        library = SkillLibrary(simulation=simulation)

        self.assertTrue(
            library.check_post_condition(PostCondition(kind="holding", object_name="red_block"))
        )
        self.assertTrue(
            library.check_post_condition(
                PostCondition(kind="placed", object_name="red_block", zone_name="green_zone")
            )
        )
        self.assertEqual(
            simulation.events,
            [
                ("holding_check", "red_block"),
                ("place_check", "red_block:green_zone"),
            ],
        )

    def test_holding_post_condition_does_not_require_pick_lift_success(self) -> None:
        simulation = _FakeSimulation()
        library = SkillLibrary(simulation=simulation)

        simulation.held_object_name = "red_block"
        self.assertTrue(
            library.check_post_condition(PostCondition(kind="holding", object_name="red_block"))
        )
        self.assertEqual(simulation.events, [("holding_check", "red_block")])


class SkillPolicySpecParsingTests(unittest.TestCase):
    def test_manifest_use_staging_defaults_to_true(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_dir = Path(tmp_dir)
            manifest_path = temp_dir / "manifest.json"
            model_path = temp_dir / "best_model.zip"
            model_path.write_bytes(b"")
            manifest_path.write_text(
                json.dumps(
                    {
                        "algorithm": "sac",
                        "model_path": str(model_path),
                        "max_steps": 48,
                        "action_repeat": 24,
                        "action_scale": 0.04,
                        "deterministic": True,
                    }
                ),
                encoding="utf-8",
            )

            spec = policy_spec_from_dict("pick", {"manifest_path": str(manifest_path)}, base_dir=temp_dir)

            self.assertTrue(spec.use_staging)

    def test_config_can_override_manifest_use_staging(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_dir = Path(tmp_dir)
            manifest_path = temp_dir / "manifest.json"
            model_path = temp_dir / "best_model.zip"
            model_path.write_bytes(b"")
            manifest_path.write_text(
                json.dumps(
                    {
                        "algorithm": "sac",
                        "model_path": str(model_path),
                        "max_steps": 48,
                        "action_repeat": 24,
                        "action_scale": 0.04,
                        "deterministic": True,
                        "use_staging": True,
                    }
                ),
                encoding="utf-8",
            )

            spec = policy_spec_from_dict(
                "pick",
                {"manifest_path": str(manifest_path), "use_staging": False},
                base_dir=temp_dir,
            )

            self.assertFalse(spec.use_staging)


if __name__ == "__main__":
    unittest.main()