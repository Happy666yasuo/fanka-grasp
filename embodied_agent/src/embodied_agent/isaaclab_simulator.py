from __future__ import annotations

from typing import Any


class IsaacLabPickPlaceSimulation:
    """Placeholder for the future IsaacLab training backend.

    The simulator factory exposes this backend name now so callers can depend
    on a stable contract while the real IsaacLab environment is implemented.
    """

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError(
            "IsaacLab pick/place backend is registered but not implemented yet. "
            "Use backend='mujoco' for current simulator-backed validation."
        )
