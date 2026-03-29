# Checkpoint Log — 版本节点记录

> **用途**: AI 在完成每个关键阶段后，在此文件记录 git tag / commit hash / 状态摘要，
>           方便人类随时回退到任意节点。
> **规则**: 只追加，不删除已有记录。

---

## 节点格式

```
### CP-<编号>: <简短标题>
- **时间**: YYYY-MM-DD HH:MM
- **Git Tag**: cp-<编号>-<关键词>
- **Commit**: <hash 前 8 位>
- **分支**: franka-grasp
- **状态**: ✅ 通过 / ⚠️ 部分通过 / ❌ 失败
- **内容**: <做了什么>
- **验证**: <跑了什么测试，结果如何>
- **回退命令**: `git checkout cp-<编号>-<关键词>`
```

---

## 节点记录

（AI 将在此处追加记录，请勿手动删除）

### CP-0: 环境验证与项目初始化
- **时间**: 2026-03-29 17:15
- **Git Tag**: cp-0-env-init
- **Commit**: 590b576
- **分支**: franka-grasp
- **状态**: ✅ 通过
- **内容**:
  - 验证 `beyondmimic` 环境: IsaacSim 可导入, IsaacLab 0.36.21, SB3 SAC ready, PyTorch 2.5.1+cu124 CUDA:True
  - 验证 `unitree-rl` 环境: MuJoCo 3.2.3
  - GPU 确认: NVIDIA RTX 4060 Laptop 8GB, Driver 580.95.05, CUDA 13.0
  - 从 master 创建 `franka-grasp` 分支
  - 创建 `my_practice/franka_cube_grasp/` 完整目录结构 (16 文件)
  - 编写 `scripts/smoke_test.py` (Phase 0 版本, 仅验证导入)
  - 创建 `.gitignore` 排除 isaacgym 等大目录
- **验证**: V0-1 ~ V0-5 全部通过, smoke_test.py 输出 "ALL CHECKS PASSED ✅"
- **回退命令**: `git checkout cp-0-env-init`

### CP-1: 场景与环境配置
- **时间**: 2026-03-29 18:10
- **Git Tag**: cp-1-scene-setup
- **Commit**: 34ec480
- **分支**: franka-grasp
- **状态**: ✅ 通过
- **内容**:
  - 完整阅读官方 Lift 任务源码 (lift_env_cfg.py, joint_pos_env_cfg.py, franka.py, mdp/*)
  - 编写 `envs/mdp/observations.py`: 3 个自定义观测函数 (object_pos, ee_pos, ee_object_rel)
  - 编写 `envs/mdp/rewards.py`: 3 个奖励函数 (sparse lift, dense reach, goal tracking)
  - 编写 `envs/mdp/terminations.py`: 1 个终止条件 (object_dropped_below_table)
  - 编写 `envs/mdp/__init__.py`: 重导出所有 MDP 函数
  - 编写 `envs/franka_grasp_env_cfg.py`: 完整场景 + MDP 配置
    - FrankaGraspSceneCfg: Franka Panda + table + cube + EE frame + ground + light
    - FrankaGraspEnvCfg: obs(32D) + action(8D) + 5 reward terms + 2 terminations + 2 events
    - 使用 AppLauncher 标准启动模式
  - 编写 `envs/__init__.py`: gymnasium 环境注册 (Isaac-Grasp-Cube-Franka-v0)，使用字符串引用避免提前导入
  - 重写 `scripts/smoke_test.py`: Phase 1 版本，AppLauncher 启动 + 环境实例化 + 随机 rollout
- **验证**:
  - V1-1: 12 个 .py 文件全部 py_compile 通过 ✅
  - V1-2: headless, 2 envs, 50 步随机 rollout ✅
    - obs_space: `Dict('policy': Box(-inf, inf, (2, 32), float32))`
    - act_space: `Box(-inf, inf, (2, 8), float32)`
    - Observation: joint_pos(9) + joint_vel(9) + object_pos(3) + ee_object_rel(3) + actions(8) = 32D
    - Action: arm(7) + gripper(1) = 8D
    - Reward: reaching(1.0) + lifting(15.0) + goal_tracking(10.0) + action_rate(-1e-4) + joint_vel(-1e-4)
    - "ALL CHECKS PASSED ✅"
- **回退命令**: `git checkout cp-1-scene-setup`
