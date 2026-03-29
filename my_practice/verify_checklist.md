# 验证清单 — 每阶段自检

> **用途**: AI 每完成一个阶段，必须逐项执行本清单中对应阶段的检查命令，
>           全部通过后才能记录 checkpoint 并进入下一阶段。

---

## Phase 0: 环境验证

```bash
# 所有命令在 beyondmimic 环境执行
conda activate beyondmimic

# [V0-1] IsaacSim 可导入
python -c "import isaacsim; print('OK:', isaacsim.__version__)"

# [V0-2] IsaacLab 可导入
python -c "import isaaclab; print('OK:', isaaclab.__version__)"

# [V0-3] SB3 + SAC 可导入
python -c "from stable_baselines3 import SAC; print('OK: SAC ready')"

# [V0-4] PyTorch + CUDA
python -c "import torch; print('OK:', torch.__version__, 'CUDA:', torch.cuda.is_available())"

# [V0-5] 项目目录存在
ls ~/Desktop/beyondmimic/my_practice/franka_cube_grasp/
```

通过标准: V0-1 ~ V0-4 全部打印 OK，V0-5 显示目录内容。

---

## Phase 1: 场景搭建

```bash
conda activate beyondmimic
cd ~/Desktop/beyondmimic/my_practice/franka_cube_grasp

# [V1-1] Python 语法检查
python -m py_compile envs/franka_grasp_env_cfg.py && echo "OK: syntax"

# [V1-2] 环境能实例化 (num_envs=2, headless)
python scripts/smoke_test.py --num_envs 2 --headless

# [V1-3] 观测维度正确
python -c "
from scripts.smoke_test import get_obs_dim
dim = get_obs_dim()
assert dim > 0, f'obs dim = {dim}'
print(f'OK: obs_dim={dim}')
"

# [V1-4] 随机动作不崩溃 (跑 100 步)
python scripts/smoke_test.py --num_envs 2 --steps 100 --headless
```

通过标准: V1-1 ~ V1-4 全部无错误退出。

---

## Phase 2: SAC Baseline

```bash
conda activate beyondmimic
cd ~/Desktop/beyondmimic/my_practice/franka_cube_grasp

# [V2-1] 训练脚本语法正确
python -m py_compile scripts/train.py && echo "OK: syntax"

# [V2-2] 短训练不崩溃 (1000 steps, 4 envs, headless)
python scripts/train.py --num_envs 4 --total_timesteps 1000 --reward_type sparse --headless --log_dir logs/test_run

# [V2-3] Checkpoint 文件生成
ls logs/test_run/checkpoints/*.zip 2>/dev/null && echo "OK: checkpoint saved"

# [V2-4] TensorBoard 日志存在
ls logs/test_run/tb_logs/ 2>/dev/null && echo "OK: TB logs exist"

# [V2-5] 评估脚本可运行
python scripts/eval.py --checkpoint logs/test_run/checkpoints/latest.zip --num_episodes 5 --headless
```

通过标准: V2-1 ~ V2-5 全部无错误。

---

## Phase 3: Reward Shaping

```bash
conda activate beyondmimic
cd ~/Desktop/beyondmimic/my_practice/franka_cube_grasp

# [V3-1] 奖励模块语法检查
python -m py_compile envs/mdp/rewards.py && echo "OK: syntax"

# [V3-2] shaped 奖励可运行
python scripts/train.py --num_envs 4 --total_timesteps 1000 --reward_type shaped --headless --log_dir logs/test_shaped

# [V3-3] PBRS 奖励可运行
python scripts/train.py --num_envs 4 --total_timesteps 1000 --reward_type pbrs --headless --log_dir logs/test_pbrs

# [V3-4] 奖励值范围合理 (不是 NaN / Inf)
python scripts/check_reward_range.py --reward_type shaped --steps 200
```

通过标准: V3-1 ~ V3-4 无错误，V3-4 报告奖励在 [-100, 100] 范围内。

---

## Phase 4: HER 集成

```bash
conda activate beyondmimic
cd ~/Desktop/beyondmimic/my_practice/franka_cube_grasp

# [V4-1] GoalEnv wrapper 语法检查
python -m py_compile agents/her_wrapper.py && echo "OK: syntax"

# [V4-2] HER + SAC 短训练
python scripts/train.py --num_envs 4 --total_timesteps 1000 --algo sac_her --headless --log_dir logs/test_her

# [V4-3] HER replay buffer 正常填充
python scripts/check_her_buffer.py --steps 500
```

通过标准: V4-1 ~ V4-3 无错误。

---

## Phase 5: Sim2Sim

```bash
# [V5-1] ONNX 导出成功 (beyondmimic 环境)
conda activate beyondmimic
cd ~/Desktop/beyondmimic/my_practice/franka_cube_grasp
python sim2sim/export_onnx.py --checkpoint checkpoints/best_model.zip

# [V5-2] ONNX 文件存在
ls sim2sim/policy_franka_grasp.onnx && echo "OK: ONNX exists"

# [V5-3] MuJoCo 推理 (unitree-rl 环境)
conda activate unitree-rl
cd ~/Desktop/beyondmimic/my_practice/franka_cube_grasp
python sim2sim/mujoco_eval.py --episodes 10 --render false

# [V5-4] 成功率输出
# 预期输出格式: "Success rate: XX.X% (X/10)"
```

通过标准: V5-1 ~ V5-3 无错误，V5-4 有成功率输出。
