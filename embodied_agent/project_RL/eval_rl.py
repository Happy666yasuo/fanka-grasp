"""
Evaluate a trained RL model on the pick task.

Usage
-----
python eval_rl.py --model outputs/rl_training/.../best_model.zip --algo ppo
python eval_rl.py --model outputs/rl_training/.../best_model.zip --algo sac --episodes 200 --randomize
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
from stable_baselines3 import PPO, SAC, TD3

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dense_pick_env import DensePickEnv  # noqa: E402

ALGO_MAP: dict[str, type] = {"ppo": PPO, "sac": SAC, "td3": TD3}


def evaluate(model, env: DensePickEnv, n_episodes: int = 100, slow: float = 0.0):
    successes = 0
    rewards: list[float] = []
    steps: list[int] = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        if slow > 0:
            time.sleep(slow)  # pause after reset so viewer can see initial state
        done = False
        ep_r = 0.0
        ep_s = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_r += reward
            ep_s += 1
            done = terminated or truncated
            if slow > 0:
                time.sleep(slow)
        rewards.append(ep_r)
        steps.append(ep_s)
        suc = info.get("is_success", False)
        if suc:
            successes += 1
        if slow > 0:
            print(f"  Episode {ep+1}: {'SUCCESS' if suc else 'FAIL'}  "
                  f"steps={ep_s}  reward={ep_r:.1f}")
            time.sleep(slow * 3)  # longer pause between episodes

    sr = successes / n_episodes
    mr = float(np.mean(rewards))
    ms = float(np.mean(steps))

    print(f"Episodes      : {n_episodes}")
    print(f"Success rate  : {sr * 100:.1f}%  ({successes}/{n_episodes})")
    print(f"Mean reward   : {mr:.2f}")
    print(f"Mean steps    : {ms:.1f}")
    return {"success_rate": sr, "mean_reward": mr, "mean_steps": ms}


def main():
    parser = argparse.ArgumentParser(description="Evaluate RL pick policy")
    parser.add_argument("--model", required=True, help="Path to .zip model file")
    parser.add_argument(
        "--algo", choices=list(ALGO_MAP.keys()), required=True
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--randomize", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--slow", type=float, default=0.0,
                        help="Seconds to pause between steps for GUI viewing (e.g. 0.5)")
    args = parser.parse_args()

    model = ALGO_MAP[args.algo].load(args.model)
    render_mode = "human" if args.render else None
    env = DensePickEnv(randomize=args.randomize, render_mode=render_mode,
                       action_repeat=60 if args.slow > 0 else 30)

    evaluate(model, env, n_episodes=args.episodes, slow=args.slow)
    env.close()


if __name__ == "__main__":
    main()
