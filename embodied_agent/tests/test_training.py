from __future__ import annotations

import sys
import unittest
from pathlib import Path

import gymnasium as gym
import torch
from gymnasium import spaces
from stable_baselines3 import SAC


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embodied_agent.bc_training import BCPolicy
from embodied_agent.training import _initialize_sac_actor_from_bc_policy


class DummyContinuousEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self) -> None:
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(17,), dtype=float)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=float)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        del seed, options
        return self.observation_space.sample() * 0.0, {}

    def step(self, action):
        del action
        return self.observation_space.sample() * 0.0, 0.0, False, False, {}


class TrainingUtilsTests(unittest.TestCase):
    def test_initialize_sac_actor_from_bc_policy_copies_matching_layers(self) -> None:
        env = DummyContinuousEnv()
        model = SAC(
            "MlpPolicy",
            env,
            seed=7,
            verbose=0,
            device="cpu",
            policy_kwargs={"net_arch": [128, 128]},
        )
        bc_policy = BCPolicy(obs_dim=17, act_dim=4, hidden=[128, 128])
        with torch.no_grad():
            for index, parameter in enumerate(bc_policy.parameters()):
                parameter.fill_(0.1 * (index + 1))

        stats = _initialize_sac_actor_from_bc_policy(model.actor, bc_policy, log_std_value=-2.5)
        actor_state = model.actor.state_dict()
        bc_state = bc_policy.state_dict()

        self.assertTrue(torch.allclose(actor_state["latent_pi.0.weight"], bc_state["net.0.weight"]))
        self.assertTrue(torch.allclose(actor_state["latent_pi.0.bias"], bc_state["net.0.bias"]))
        self.assertTrue(torch.allclose(actor_state["latent_pi.2.weight"], bc_state["net.2.weight"]))
        self.assertTrue(torch.allclose(actor_state["latent_pi.2.bias"], bc_state["net.2.bias"]))
        self.assertTrue(torch.allclose(actor_state["mu.weight"], bc_state["net.4.weight"]))
        self.assertTrue(torch.allclose(actor_state["mu.bias"], bc_state["net.4.bias"]))
        self.assertTrue(torch.allclose(actor_state["log_std.weight"], torch.zeros_like(actor_state["log_std.weight"])))
        self.assertTrue(torch.allclose(actor_state["log_std.bias"], torch.full_like(actor_state["log_std.bias"], -2.5)))
        self.assertTrue(stats["applied"])
        self.assertTrue(stats["log_std_initialized"])

    def test_initialize_sac_actor_from_bc_policy_rejects_mismatched_hidden_dims(self) -> None:
        env = DummyContinuousEnv()
        model = SAC(
            "MlpPolicy",
            env,
            seed=7,
            verbose=0,
            device="cpu",
            policy_kwargs={"net_arch": [128, 128]},
        )
        bc_policy = BCPolicy(obs_dim=17, act_dim=4, hidden=[64, 64])

        with self.assertRaises(ValueError):
            _initialize_sac_actor_from_bc_policy(model.actor, bc_policy)


if __name__ == "__main__":
    unittest.main()