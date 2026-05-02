"""Expert warm-start: collect scripted demonstrations and pre-fill SAC replay buffer."""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

from embodied_agent.rl_envs import PickPlaceSkillEnv


def expert_pick_action(obs: np.ndarray, action_scale: float) -> np.ndarray:
    """Deterministic expert pick: approach object with open gripper, close once near, then lift."""
    holding = obs[16] > 0.5

    if holding:
        return np.array([0.0, 0.0, 1.0, -1.0], dtype=np.float32)

    ee_to_obj = obs[9:12].copy()
    dist = float(np.linalg.norm(ee_to_obj))

    direction = ee_to_obj / max(action_scale, 1e-6)
    direction = np.clip(direction, -1.0, 1.0)

    if dist > 0.12:
        gripper_cmd = 1.0
    else:
        gripper_cmd = -1.0

    return np.array([direction[0], direction[1], direction[2], gripper_cmd], dtype=np.float32)


def expert_place_action(obs: np.ndarray, action_scale: float) -> np.ndarray:
    """Deterministic expert place: align over zone, lower, release."""
    holding = obs[16] > 0.5

    if not holding:
        return np.array([0.0, 0.0, 0.5, 1.0], dtype=np.float32)

    obj_to_zone = obs[12:15].copy()
    obj_zone_xy_dist = float(np.linalg.norm(obj_to_zone[:2]))
    object_z = float(obs[5])
    table_top_z = 0.62

    if obj_zone_xy_dist > 0.02:
        direction = obj_to_zone / action_scale
        direction[2] = -0.3
        direction = np.clip(direction, -1.0, 1.0)
        return np.array([direction[0], direction[1], direction[2], -1.0], dtype=np.float32)

    if object_z > table_top_z + 0.055:
        return np.array([0.0, 0.0, -1.0, -1.0], dtype=np.float32)

    return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)


def get_expert_policy(skill_name: str) -> Callable[[np.ndarray, float], np.ndarray]:
    if skill_name == "pick":
        return expert_pick_action
    if skill_name == "place":
        return expert_place_action
    raise ValueError(f"No expert policy for skill: {skill_name}")


def collect_expert_episodes(
    env: PickPlaceSkillEnv,
    expert_fn: Callable[[np.ndarray, float], np.ndarray],
    n_episodes: int,
    action_scale: float,
    noise_std: float = 0.05,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Collect expert demonstrations. Returns (transitions, stats)."""
    transitions: list[dict[str, Any]] = []
    successes = 0
    total_reward = 0.0
    rng = np.random.default_rng(42)

    for ep in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0.0
        done = False

        while not done:
            action = expert_fn(obs, action_scale)
            if noise_std > 0:
                noise = rng.normal(0, noise_std, size=3).astype(np.float32)
                action = action.copy()
                action[:3] = np.clip(action[:3] + noise, -1.0, 1.0)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            transitions.append({
                "obs": obs.copy(),
                "next_obs": next_obs.copy(),
                "action": action.copy(),
                "reward": np.array([reward], dtype=np.float32),
                "done": np.array([terminated], dtype=np.float32),
                "infos": [info],
            })

            episode_reward += reward
            obs = next_obs

        if info.get("is_success", False) or info.get("success", False):
            successes += 1
        total_reward += episode_reward

    stats = {
        "n_episodes": float(n_episodes),
        "n_transitions": float(len(transitions)),
        "success_rate": successes / max(n_episodes, 1),
        "mean_reward": total_reward / max(n_episodes, 1),
    }
    return transitions, stats


def prefill_replay_buffer(model: Any, transitions: list[dict[str, Any]]) -> int:
    """Add collected transitions to the model's replay buffer."""
    buffer = model.replay_buffer
    for t in transitions:
        buffer.add(
            obs=t["obs"],
            next_obs=t["next_obs"],
            action=t["action"],
            reward=t["reward"],
            done=t["done"],
            infos=t["infos"],
        )
    return len(transitions)
