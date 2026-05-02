from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "build_place_runtime_reset_dataset.py"


def _load_dataset_builder() -> ModuleType:
    spec = importlib.util.spec_from_file_location("build_place_runtime_reset_dataset", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class RuntimeResetDatasetBuilderPresetTests(unittest.TestCase):
    def test_resolves_strict_preset_as_default_thresholds(self) -> None:
        module = _load_dataset_builder()

        thresholds = module._resolve_thresholds("strict")

        self.assertEqual(thresholds.min_object_height, 0.705)
        self.assertEqual(thresholds.min_lift_progress, 0.065)
        self.assertEqual(thresholds.max_lift_progress, 0.09)
        self.assertEqual(thresholds.min_object_zone_distance_xy, 0.35)
        self.assertEqual(thresholds.max_object_zone_distance_xy, 0.60)
        self.assertEqual(thresholds.max_ee_object_distance, 0.16)
        self.assertEqual(thresholds.min_held_local_z, 0.08)

    def test_resolves_broadened_v1_thresholds(self) -> None:
        module = _load_dataset_builder()

        thresholds = module._resolve_thresholds("broadened_v1")

        self.assertEqual(thresholds.min_object_height, 0.64)
        self.assertEqual(thresholds.min_lift_progress, 0.0)
        self.assertEqual(thresholds.max_lift_progress, 0.20)
        self.assertEqual(thresholds.min_object_zone_distance_xy, 0.05)
        self.assertEqual(thresholds.max_object_zone_distance_xy, 0.60)
        self.assertEqual(thresholds.max_ee_object_distance, 0.50)
        self.assertEqual(thresholds.min_held_local_z, -0.20)
        self.assertEqual(thresholds.to_dict()["preset"], "broadened_v1")

    def test_rejects_unknown_preset(self) -> None:
        module = _load_dataset_builder()

        with self.assertRaisesRegex(ValueError, "Unknown threshold preset"):
            module._resolve_thresholds("loose")


if __name__ == "__main__":
    unittest.main()
