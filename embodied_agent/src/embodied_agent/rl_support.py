from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any

import numpy as np
import torch
from stable_baselines3 import PPO, SAC


ALGORITHM_CLASSES = {
    "ppo": PPO,
    "sac": SAC,
    "bc": "bc",  # sentinel, handled separately
}


@dataclass(frozen=True)
class SkillPolicySpec:
    skill_name: str
    algorithm: str
    model_path: Path
    max_steps: int = 48
    action_repeat: int = 24
    action_scale: float = 0.04
    deterministic: bool = True
    use_staging: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_name": self.skill_name,
            "algorithm": self.algorithm,
            "model_path": str(self.model_path),
            "max_steps": self.max_steps,
            "action_repeat": self.action_repeat,
            "action_scale": self.action_scale,
            "deterministic": self.deterministic,
            "use_staging": self.use_staging,
        }


def normalize_algorithm_name(algorithm_name: str) -> str:
    normalized_name = algorithm_name.strip().lower()
    if normalized_name not in ALGORITHM_CLASSES:
        supported = ", ".join(sorted(ALGORITHM_CLASSES.keys()))
        raise ValueError(f"Unsupported RL algorithm '{algorithm_name}'. Supported values: {supported}")
    return normalized_name


def get_algorithm_class(algorithm_name: str):
    return ALGORITHM_CLASSES[normalize_algorithm_name(algorithm_name)]


class _BCModelWrapper:
    """Wraps a BCPolicy to match the SB3 model.predict() interface."""

    def __init__(self, bc_policy):
        self.policy = bc_policy

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> tuple[np.ndarray, None]:
        action = self.policy.predict(obs, deterministic=deterministic)
        return action, None


def load_policy_model(algorithm_name: str, model_path: Path | str):
    normalized = normalize_algorithm_name(algorithm_name)
    if normalized == "bc":
        from embodied_agent.bc_training import BCPolicy
        model_path = Path(model_path)
        # Load metadata to reconstruct architecture
        meta_path = model_path.parent / "bc_model_meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            obs_dim = int(meta["obs_dim"])
            act_dim = int(meta["act_dim"])
            hidden = meta.get("hidden", [128, 128])
        else:
            obs_dim, act_dim, hidden = 17, 4, [128, 128]
        policy = BCPolicy(obs_dim, act_dim, hidden)
        policy.load_state_dict(torch.load(str(model_path), map_location="cpu", weights_only=True))
        policy.eval()
        return _BCModelWrapper(policy)
    algorithm_class = get_algorithm_class(algorithm_name)
    return algorithm_class.load(str(model_path), device="cpu")


def policy_spec_from_dict(
    skill_name: str,
    config: dict[str, object],
    base_dir: Path | None = None,
) -> SkillPolicySpec:
    if "manifest_path" in config:
        manifest_path = _resolve_path(Path(str(config["manifest_path"])), base_dir)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return SkillPolicySpec(
            skill_name=skill_name,
            algorithm=normalize_algorithm_name(str(manifest["algorithm"])),
            model_path=_resolve_path(Path(str(manifest["model_path"])), manifest_path.parent),
            max_steps=int(manifest["max_steps"]),
            action_repeat=int(manifest["action_repeat"]),
            action_scale=float(manifest["action_scale"]),
            deterministic=bool(manifest.get("deterministic", True)),
            use_staging=bool(config.get("use_staging", manifest.get("use_staging", True))),
        )

    if "model_path" not in config:
        raise ValueError(f"Missing model_path for skill '{skill_name}'.")

    model_path = _resolve_path(Path(str(config["model_path"])), base_dir)
    return SkillPolicySpec(
        skill_name=skill_name,
        algorithm=normalize_algorithm_name(str(config.get("algorithm", "sac"))),
        model_path=model_path,
        max_steps=int(config.get("max_steps", 48)),
        action_repeat=int(config.get("action_repeat", 24)),
        action_scale=float(config.get("action_scale", 0.04)),
        deterministic=bool(config.get("deterministic", True)),
        use_staging=bool(config.get("use_staging", True)),
    )


def _resolve_path(path: Path, base_dir: Path | None) -> Path:
    windows_path = _resolve_windows_style_path(str(path), base_dir)
    if windows_path is not None:
        return windows_path

    if path.is_absolute():
        return path
    if base_dir is None:
        return path.resolve()
    return (base_dir / path).resolve()


def _resolve_windows_style_path(raw_path: str, base_dir: Path | None) -> Path | None:
    if "\\" not in raw_path or "embodied_agent" not in raw_path.lower():
        return None

    project_root = _find_named_parent(base_dir, "embodied_agent")
    if project_root is None:
        return None

    windows_path = PureWindowsPath(raw_path)
    lowered_parts = [part.lower() for part in windows_path.parts]
    try:
        project_index = lowered_parts.index("embodied_agent")
    except ValueError:
        return None

    relative_parts = windows_path.parts[project_index + 1 :]
    return project_root.joinpath(*relative_parts).resolve()


def _find_named_parent(base_dir: Path | None, target_name: str) -> Path | None:
    if base_dir is None:
        return None

    current = base_dir.resolve()
    while True:
        if current.name == target_name:
            return current
        if current.parent == current:
            return None
        current = current.parent