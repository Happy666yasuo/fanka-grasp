"""Behavior Cloning (BC) training: supervised learning from expert demonstrations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from embodied_agent.experiment import ensure_training_run_dir, load_yaml_config, save_json, save_yaml
from embodied_agent.expert_warmstart import collect_expert_episodes, get_expert_policy
from embodied_agent.rl_envs import PickPlaceSkillEnv, SkillEnvSettings


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class BCPolicy(nn.Module):
    """Simple MLP policy for behavior cloning."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: list[int] | None = None):
        super().__init__()
        hidden = hidden or [128, 128]
        layers: list[nn.Module] = []
        prev_dim = obs_dim
        for h in hidden:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, act_dim))
        layers.append(nn.Tanh())  # actions in [-1, 1]
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            action = self.net(obs_t).squeeze(0).numpy()
        return action


def train_bc(
    config_path: Path,
    total_epochs: int | None = None,
    run_name: str | None = None,
) -> dict[str, Any]:
    config = load_yaml_config(config_path)
    skill_name = str(config.get("skill", "pick"))
    seed = int(config.get("seed", 7))
    n_epochs = int(total_epochs or config.get("epochs", 200))
    output_root = _resolve_path(config.get("output_root", "outputs"), PROJECT_ROOT)
    run_dir = ensure_training_run_dir(output_root, skill_name, "bc", run_name=run_name)

    env_settings = SkillEnvSettings.from_config(skill_name, config.get("env"))
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Collect expert data
    expert_episodes = int(config.get("expert_episodes", 500))
    noise_std = float(config.get("expert_noise_std", 0.05))
    expert_env = PickPlaceSkillEnv(settings=env_settings, gui=False)
    expert_fn = get_expert_policy(skill_name)
    transitions, expert_stats = collect_expert_episodes(
        expert_env, expert_fn, expert_episodes,
        action_scale=env_settings.action_scale, noise_std=noise_std,
    )
    expert_env.close()
    print(f"[BC] Expert data: {expert_stats}")

    # Build dataset
    obs_array = np.array([t["obs"] for t in transitions], dtype=np.float32)
    act_array = np.array([t["action"] for t in transitions], dtype=np.float32)
    obs_t = torch.from_numpy(obs_array)
    act_t = torch.from_numpy(act_array)
    dataset = TensorDataset(obs_t, act_t)
    batch_size = int(config.get("batch_size", 256))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Build model
    obs_dim = obs_array.shape[1]
    act_dim = act_array.shape[1]
    hidden = config.get("hidden", [128, 128])
    policy = BCPolicy(obs_dim, act_dim, hidden)
    lr = float(config.get("learning_rate", 1e-3))
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # Training loop
    eval_env = PickPlaceSkillEnv(settings=env_settings, gui=False)
    eval_freq = int(config.get("eval_freq", 20))
    eval_episodes = int(config.get("eval_episodes", 20))
    best_success = -1.0
    best_epoch = -1

    save_yaml(run_dir / "resolved_config.yaml", {**config, "epochs": n_epochs, "skill": skill_name})

    for epoch in range(n_epochs):
        policy.train()
        epoch_loss = 0.0
        n_batches = 0
        for obs_batch, act_batch in loader:
            pred = policy(obs_batch)
            loss = loss_fn(pred, act_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        if (epoch + 1) % eval_freq == 0 or epoch == 0:
            success_rate, mean_reward = evaluate_bc(policy, eval_env, eval_episodes)
            print(f"[BC] Epoch {epoch+1:4d}/{n_epochs}: loss={avg_loss:.4f}, eval_success={success_rate:.2%}, eval_reward={mean_reward:.2f}")
            if success_rate > best_success:
                best_success = success_rate
                best_epoch = epoch + 1
                torch.save(policy.state_dict(), run_dir / "best_policy.pt")
                print(f"[BC] New best! success={success_rate:.2%}")

    # Save final
    torch.save(policy.state_dict(), run_dir / "final_policy.pt")
    eval_env.close()

    # Save model metadata for loading
    bc_meta = {"obs_dim": obs_dim, "act_dim": act_dim, "hidden": hidden}
    save_json(run_dir / "bc_model_meta.json", bc_meta)

    # Save manifest for evaluation pipeline
    manifest = {
        "algorithm": "bc",
        "model_path": "best_policy.pt",
        "max_steps": env_settings.max_steps,
        "action_repeat": env_settings.action_repeat,
        "action_scale": env_settings.action_scale,
        "deterministic": True,
        "use_staging": env_settings.use_staging,
    }
    save_json(run_dir / "best_policy_manifest.json", manifest)

    summary = {
        "skill": skill_name,
        "algorithm": "bc",
        "seed": seed,
        "epochs": n_epochs,
        "expert_stats": expert_stats,
        "best_success_rate": best_success,
        "best_epoch": best_epoch,
        "run_dir": str(run_dir.resolve()),
    }
    save_json(run_dir / "training_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def evaluate_bc(
    policy: BCPolicy,
    env: PickPlaceSkillEnv,
    n_episodes: int,
) -> tuple[float, float]:
    """Evaluate BC policy, return (success_rate, mean_reward)."""
    policy.eval()
    successes = 0
    total_reward = 0.0
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            action = policy.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward
        if info.get("is_success", False) or info.get("success", False):
            successes += 1
        total_reward += ep_reward
    return successes / max(n_episodes, 1), total_reward / max(n_episodes, 1)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train BC policy from expert demonstrations.")
    parser.add_argument("--config", required=True, help="YAML config path.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args(argv)
    train_bc(Path(args.config).resolve(), total_epochs=args.epochs, run_name=args.run_name)
    return 0


def _resolve_path(raw_path: object, base_dir: Path) -> Path:
    path = Path(str(raw_path))
    return path if path.is_absolute() else (base_dir / path).resolve()


if __name__ == "__main__":
    raise SystemExit(main())
