from __future__ import annotations

from typing import Any


class IsaacLabPickPlaceSimulation:
    """Inspectable skeleton for the future IsaacLab training backend.

    The simulator factory exposes this backend name now so callers can depend
    on a stable contract while the real IsaacLab environment is implemented.
    It intentionally does not import or start IsaacLab, because this repository
    must remain testable on machines that only have the MuJoCo validation stack.
    """

    backend_name = "isaaclab"
    observation_contract = (
        "object_pose",
        "zone_pose",
        "end_effector_pose",
        "held_object_name",
    )
    action_contract = (
        "delta_position",
        "gripper_command",
        "action_steps",
        "object_name",
    )
    result_contract = (
        "success",
        "reward",
        "final_state",
        "contact_state",
        "error_code",
        "failure_history",
    )

    def __init__(self, *_args: Any, allow_unconfigured: bool = False, **_kwargs: Any) -> None:
        if allow_unconfigured:
            self.config = None
            self.object_names: tuple[str, ...] = ()
            self.zone_names: tuple[str, ...] = ()
            self.object_ids: dict[str, Any] = {}
            self.zone_ids: dict[str, Any] = {}
            self.zone_positions: dict[str, Any] = {}
            self.held_object_name: str | None = None
            return
        raise RuntimeError(
            "IsaacLab pick/place backend is registered but not implemented yet. "
            "Use backend='mujoco' for current simulator-backed validation. "
            "The future IsaacLab adapter must implement the shared "
            "PickPlaceSimulationProtocol observation/action contract before it "
            "can run training."
        )

    @classmethod
    def contract_summary(cls) -> dict[str, tuple[str, ...]]:
        return {
            "observations": cls.observation_contract,
            "actions": cls.action_contract,
            "results": cls.result_contract,
        }

    def reset(self) -> None:
        self._raise_not_implemented()

    def observe_scene(self, instruction: str = "") -> Any:
        self._raise_not_implemented()

    def apply_skill_action(
        self,
        delta_position: tuple[float, float, float],
        gripper_command: float,
        action_steps: int = 24,
        object_name: str = "red_block",
    ) -> Any:
        self._raise_not_implemented()

    def pick_object(self, object_name: str) -> None:
        self._raise_not_implemented()

    def place_object(self, zone_name: str) -> None:
        self._raise_not_implemented()

    def shutdown(self) -> None:
        return None

    def _raise_not_implemented(self) -> None:
        raise NotImplementedError(
            "IsaacLab pick/place adapter skeleton is inspectable but not runnable. "
            "Implement the shared observation/action contract before using it as "
            "a training backend."
        )
