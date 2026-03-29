---
name: franka-cube-grasp
description: |
  IsaacLab 2.1.0 Franka Panda robotic arm cube grasping project.
  Use this skill when working on: SAC training, reward shaping, HER integration,
  IsaacLab environment configuration, Sim2Sim (IsaacSim → MuJoCo) transfer,
  or any task related to the Franka manipulation project in this workspace.
---

# Franka Cube Grasp — IsaacLab SAC Project

## Project Goal

Train a **Franka Emika Panda** (7-DOF + gripper) to pick up a cube from a table
using **SAC (Soft Actor-Critic)** in **IsaacLab 2.1.0 / IsaacSim 4.5.0**.
Solve the sparse-reward challenge via **Reward Shaping**, **PBRS**, **HER**, and
**Curriculum Learning**. Validate with **Sim2Sim** transfer to **MuJoCo**.

---

## Environment Matrix (STRICT — never mix)

| Purpose | Conda Env | Python | Key Packages |
|---------|-----------|--------|--------------|
| **Training & scene** | `beyondmimic` | 3.10 | isaacsim 4.5.0, isaaclab 0.36.21 (2.1.0), SB3 2.7.1, mujoco 3.4.0, torch 2.5.1 |
| Sim2Sim MuJoCo | `unitree-rl` | 3.8 | mujoco 3.2.3, torch 2.3.1 |
| Reference only | `env_isaaclab` | 3.11 | isaaclab 0.47.2, SB3 2.7.0, torch 2.7.0+cu128 |

- IsaacLab path: `~/Desktop/isaac_workspace/IsaacLab-2.1.0/`
- Project path: `~/Desktop/beyondmimic/my_practice/franka_cube_grasp/`

---

## Hard Rules

1. **Never** run `pip uninstall` or `conda remove` on any package.
2. **Always** specify `conda activate <env>` before every shell command.
3. Before any `rm -rf`, print ⚠️ WARNING and **wait for user confirmation**.
4. Install new packages only after checking they don't already exist.
5. Use **Manager-Based** pattern (not Direct) for IsaacLab environments.
6. All reward / observation code must be **GPU-vectorized** (PyTorch tensors, no Python loops over envs).
7. **ALL IsaacSim/IsaacLab scripts MUST use `--headless`** — GUI rendering will exhaust VRAM.
8. **VRAM is limited** — see "VRAM Constraints" section below. Never blindly set `num_envs ≥ 256`.

---

## VRAM Constraints (⚠️ STRICT)

| Scenario | `num_envs` | `buffer_size` | Notes |
|----------|-----------|---------------|-------|
| Smoke test | ≤ 4 | — | Quick sanity check |
| Short verification | ≤ 16 | 100,000 | Verify pipeline works |
| **Production training** | **64 – 128** | **100,000** | Adjust by actual VRAM headroom |
| OOM recovery | halve `num_envs` first | then halve `batch_size` | Never increase blindly |

- `--headless` is **mandatory** for every training / eval / smoke-test command.
- SAC `buffer_size` defaults to `100_000` (not 1 M) to avoid CPU/GPU memory overflow.
- If OOM occurs: `num_envs` ↓ first → `batch_size` ↓ second → `buffer_size` ↓ last.

---

## Code Conventions

- Python 3.10, full type annotations (`from __future__ import annotations`).
- IsaacLab style: `@configclass` decorator, `MISSING` placeholders, `configclass` inheritance.
- Naming: `snake_case` variables, `PascalCase` classes, `UPPER_SNAKE_CASE` constants.
- Every file header: purpose + target conda environment.
- Docstrings: Google style, include math for reward terms.

---

## Project Layout

```
my_practice/franka_cube_grasp/
├── envs/
│   ├── __init__.py
│   ├── franka_grasp_env_cfg.py      # Scene + MDP config
│   └── mdp/
│       ├── observations.py
│       ├── rewards.py               # Sparse / Shaped / PBRS
│       └── terminations.py
├── agents/
│   ├── sac_cfg.py                   # SB3 SAC hyper-parameters
│   └── her_wrapper.py               # GoalEnv adapter for HER
├── scripts/
│   ├── train.py                     # Main training entry
│   ├── eval.py
│   └── visualize.py
├── sim2sim/
│   ├── export_onnx.py
│   ├── mujoco_eval.py
│   └── franka_table.xml             # MuJoCo MJCF scene
├── configs/                          # Hydra overrides
├── logs/                             # TensorBoard / WandB
└── checkpoints/
```

---

## Key References

| Resource | Location |
|----------|----------|
| Official Lift task | `IsaacLab-2.1.0/source/isaaclab_tasks/.../manipulation/lift/` |
| Franka config | `.../lift/config/franka/joint_pos_env_cfg.py` |
| BeyondMimic Sim2Sim | `~/Desktop/beyondmimic/Beyondmimic_Deploy_G1/deploy_mujoco_1.py` |
| IsaacSim 4.5.0 docs | https://docs.isaacsim.omniverse.nvidia.com/4.5.0/index.html |
| IsaacLab ecosystem | https://isaac-sim.github.io/IsaacLab/main/source/setup/ecosystem.html |
| SB3 SAC | https://stable-baselines3.readthedocs.io/en/master/modules/sac.html |
| SB3 HER | https://stable-baselines3.readthedocs.io/en/master/modules/her.html |

---

## Workflow Checklist

1. After any env config change → smoke-test with `num_envs=4 --headless`.
2. Production training → `num_envs=64–128 --headless` (VRAM limited; never blindly use 256+).
3. Save checkpoint every 10 k steps.
4. Log full hyper-parameters before each training run.
5. Compare experiments: sparse → shaped → PBRS → HER → curriculum.
6. **Every** IsaacSim/IsaacLab command **must** include `--headless`.

---

## MDP Summary

**Observation (~35-D):** joint pos (7), joint vel (7), gripper (2),
EE pose (7), object pose (7), target pos (3), EE→object relative (3).

**Action (8-D):** 7 joint-position targets + 1 gripper open/close.

**Reward strategies:**
- *Sparse:* +1 if cube height > threshold, else 0.
- *Shaped:* reach distance + grasp bonus + lift height + stability + action penalty.
- *PBRS:* γΦ(s') − Φ(s), Φ = −‖EE−obj‖ − ‖obj−target‖.
- *HER:* GoalEnv with achieved/desired goal relabelling (future strategy).

**Termination:** timeout (500 steps) | object dropped | success held 10 steps.

---

## SAC Defaults

```yaml
algorithm: SAC
policy: MlpPolicy
net_arch: [256, 256, 256]
learning_rate: 3e-4
buffer_size: 100_000        # ⚠️ 降至 100K（本机内存/显存有限，勿用 1M）
batch_size: 256
tau: 0.005
gamma: 0.99
ent_coef: auto
train_freq: 1
gradient_steps: 1
num_envs: 64               # ⚠️ 正式训练推荐 64~128，冒烟测试用 4
headless: true              # ⚠️ 必须 headless，GUI 会耗尽显存
```
