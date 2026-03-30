# SKILL — 项目技能树与 AI 协作规则（人类阅读版）

> **项目名称**: 基于 IsaacLab 的 Franka 机械臂抓取（SAC + Reward Shaping / HER）
> **维护者**: happywindman
> **最后更新**: 2026-03-30
>
> ⚠️ **注意**: 标准 Agent Skills 文件（供 AI 编辑器自动发现使用）位于：
> `<项目根目录>/.agents/skills/franka-cube-grasp/SKILL.md`
> 本文件为人类可读的扩展版本。

---

## 0. 项目概述

在 IsaacLab 2.1.0（基于 IsaacSim 4.5.0）中搭建 Franka Emika Panda 机械臂 + 桌面方块场景，
使用 **SAC（Soft Actor-Critic）** 算法完成 **"从桌上抓起方块"** 的稀疏奖励任务。
核心技术挑战：稀疏奖励下的探索效率，需要结合 **Reward Shaping** 和/或 **HER（Hindsight Experience Replay）**。
最终通过 **Sim2Sim**（IsaacSim → MuJoCo）验证策略的可迁移性。

---

## 1. 环境与依赖矩阵（⚠️ 严格遵守，不得混用）

| 用途 | Conda 环境名 | Python | 关键包 | IsaacLab 路径 |
|------|-------------|--------|--------|---------------|
| **主开发环境** | `beyondmimic` | 3.10 | isaacsim 4.5.0, isaaclab 2.1.0 (0.36.21), stable_baselines3 2.7.1, mujoco 3.4.0, torch 2.5.1 | `~/Desktop/isaac_workspace/IsaacLab-2.1.0/` |
| 最新 IsaacLab（参考） | `env_isaaclab` | 3.11 | isaaclab 0.47.2, stable_baselines3 2.7.0, torch 2.7.0+cu128（无 mujoco） | `~/Desktop/isaac_workspace/IsaacLab/` |
| MuJoCo Sim2Sim 验证 | `unitree-rl` | 3.8 | mujoco 3.2.3, torch 2.3.1（无 IsaacLab） | N/A |
| RL 算法参考 | `spinningup` | — | OpenAI SpinningUp（参考用） | N/A |

### 🔴 硬性规则
1. **绝不擅自删除任何 conda 环境中的软件包**
2. **使用 `rm -rf` 之前必须提醒用户并获得确认**
3. 训练与场景搭建 → 使用 `beyondmimic` 环境
4. Sim2Sim MuJoCo 验证 → 使用 `unitree-rl`（mujoco 3.2.3）或 `beyondmimic`（mujoco 3.4.0）
5. 切换环境时必须显式 `conda activate <env_name>`

### 🟡 显存约束（⚠️ 严格遵守）
- **所有 IsaacSim/IsaacLab 脚本必须使用 `--headless`**，GUI 渲染会耗尽显存
- 冒烟测试: `num_envs ≤ 4`
- 短训练验证: `num_envs ≤ 16`
- **正式训练: `num_envs = 64~128`**（绝不盲目开 256+）
- SAC `buffer_size` 默认 `100_000`（非 1M），避免 CPU/GPU 内存溢出
- OOM 时：优先降 `num_envs` → 其次降 `batch_size` → 最后降 `buffer_size`

---

## 2. 技能树（Skill Tree）

### 2.1 仿真与场景搭建
- [x] **IsaacSim 基础**: USD 场景格式、PhysX GPU 物理引擎、Omniverse 渲染管线
- [x] **IsaacLab Manager-Based 环境**: `ManagerBasedRLEnvCfg`、`InteractiveSceneCfg`
- [x] **机器人模型**: Franka Panda URDF/USD 导入、关节配置（7-DOF + 2 gripper）
- [x] **场景搭建**: 桌面 + 方块 (`RigidObjectCfg`) + 灯光 + 地面
- [x] **传感器**: `FrameTransformerCfg`（末端执行器位姿）
- [x] **域随机化**: 物体初始位姿随机化 (x/y offset)

### 2.2 强化学习算法
- [x] **SAC 理论**: 最大熵 RL、自动温度调节 α、soft Q-function
- [x] **SAC 实现**: `stable_baselines3.SAC` 框架集成
- [x] **稀疏奖励问题**: 理解探索困难的根因
- [x] **Reward Shaping**: 距离引导、阶段性奖励 (shaped_multi_stage)、势函数方法 (PBRS)
- [x] **HER**: 目标重标注策略（future）、与 SAC 结合、GoalEnv wrapper
- [x] **课程学习**: 由易到难逐步增加任务难度 (curriculum_reward)

### 2.3 观测-动作-奖励 设计（MDP）
- [x] **观测空间**: joint_pos_rel(9D) + joint_vel_rel(9D) + object_pos_b(3D) + ee_object_rel(3D) + last_action(8D) = 32D
- [x] **动作空间**: 关节位置增量(7D, scale=0.5) + 二值夹爪(1D) = 8D
- [x] **奖励函数**: Sparse / Shaped(4阶段) / PBRS / Curriculum — 4种可切换
- [x] **终止条件**: 超时(5s/250步) + 物体掉落

### 2.4 Sim2Sim 迁移
- [x] **IsaacSim → MuJoCo**: MuJoCo Menagerie Franka Panda、物理参数对齐 (dt/gravity/contact)
- [x] **策略部署**: ONNX 导出 (export_onnx.py)、MuJoCo 推理脚本 (mujoco_eval.py)
- [x] **性能对比**: 两个仿真器中的成功率对比、差异分析文档 (REPORT.md)

### 2.5 工程实践
- [x] **实验管理**: TensorBoard 日志 + SuccessRateCallback
- [ ] **超参搜索**: 学习率、batch size、buffer size、α、γ (未深入调优)
- [x] **代码组织**: @configclass 配置分离、envs/agents/scripts/sim2sim 模块化
- [x] **版本控制**: Git 分阶段管理 (cp-0 ~ cp-6)、checkpoint.md 记录

---

## 3. 关键参考资源

### 3.1 官方文档
- IsaacSim 4.5.0: https://docs.isaacsim.omniverse.nvidia.com/4.5.0/index.html
- IsaacLab 生态: https://isaac-sim.github.io/IsaacLab/main/source/setup/ecosystem.html
- IsaacLab RL 框架对比: https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/rl_frameworks.html

### 3.2 本地参考代码
- **IsaacLab Lift 任务**: `~/Desktop/isaac_workspace/IsaacLab-2.1.0/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/`
- **IsaacLab Franka 配置**: `~/Desktop/isaac_workspace/IsaacLab-2.1.0/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/config/franka/`
- **BeyondMimic 项目（迁移参考）**: `~/Desktop/beyondmimic/Beyondmimic_Deploy_G1/`
- **BeyondMimic GitHub**: https://github.com/Happy666yasuo/BeyondMimic_G1_project

### 3.3 论文与理论
- SAC: Haarnoja et al., "Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL with a Stochastic Actor" (2018)
- HER: Andrychowicz et al., "Hindsight Experience Replay" (NeurIPS 2017)
- Reward Shaping (PBRS): Ng et al., "Policy Invariance Under Reward Transformations" (ICML 1999)

---

## 4. AI 协作规则（给 AI 编程助手的指令）

### 4.1 代码风格
- Python 3.10 语法，类型注解必须完整（typing / dataclass）
- 遵循 IsaacLab 编码规范：`@configclass` 装饰器、`MISSING` 占位符
- 文件头注释包含用途说明和环境要求
- 变量命名：`snake_case`；类名：`PascalCase`；常量：`UPPER_SNAKE_CASE`

### 4.2 环境安全
- **永远不要建议 `pip uninstall` 或 `conda remove` 任何包**
- **需要 `rm -rf` 时必须在输出中用 ⚠️ 显著提醒**
- 安装新包前先检查是否已存在
- 始终指明在哪个 conda 环境中执行命令

### 4.3 项目结构约定
```
my_practice/
├── SKILL.md                     # 本文件
├── process.md                   # 项目流程文档
├── prompts.md                   # 提示词文档
├── verify_checklist.md          # 验证清单
├── checkpoint.md                # 版本节点记录
├── Embodied_Intelligence_Advance_Plan.md  # 进阶计划
├── daily_log_*.md               # 每日工作日志
└── franka_cube_grasp/           # 项目主目录
    ├── REPORT.md                # 完整实验报告 (~460行)
    ├── envs/                    # 自定义环境
    │   ├── __init__.py
    │   ├── franka_grasp_env_cfg.py     # 场景 + MDP (5种奖励配置)
    │   └── mdp/                        # 观测/奖励/终止
    │       ├── observations.py         # object_pos, ee_pos, ee_object_rel
    │       ├── rewards.py              # sparse, shaped, pbrs, curriculum
    │       └── terminations.py         # object_dropped_below_table
    ├── agents/                  # RL 算法配置
    │   ├── sac_cfg.py           # SAC 超参数 (dataclass)
    │   └── her_wrapper.py       # HERGoalVecEnvWrapper (218行)
    ├── sim2sim/                 # Sim2Sim 迁移
    │   ├── export_onnx.py       # SB3 → ONNX 导出
    │   ├── mujoco_eval.py       # MuJoCo 推理评估 (450行)
    │   ├── policy_franka_grasp.onnx  # 部署策略 (312KB)
    │   └── franka_emika_panda/  # MuJoCo Menagerie 模型
    │       ├── franka_table.xml # Sim2Sim 场景
    │       ├── panda.xml        # Franka Panda MJCF
    │       └── hand.xml         # 夹爪 MJCF
    ├── scripts/                 # 训练/评估/诊断脚本
    │   ├── train.py             # SAC/SAC+HER 训练 (307行)
    │   ├── eval.py              # 评估脚本
    │   ├── run_experiments.sh   # 6组实验批量运行
    │   ├── plot_results.py      # TensorBoard 对比绘图
    │   ├── check_reward_range.py  # 奖励范围诊断
    │   ├── check_her_buffer.py  # HER buffer 诊断
    │   └── smoke_test.py        # 冒烟测试
    ├── configs/                 # 配置文件 (预留)
    ├── logs/                    # TensorBoard 日志
    └── checkpoints/             # 模型 checkpoint
```

### 4.4 工作流约定
1. 每次修改环境配置后，先用 `num_envs=4 --headless` 做冒烟测试
2. 正式训练使用 `num_envs=64~128 --headless`（本机显存有限，勿开 256+）
3. 每 10k steps 保存 checkpoint
4. 训练前记录完整超参到日志
5. **所有 IsaacSim/IsaacLab 命令必须包含 `--headless`**

---

## 5. 当前进度追踪

| 阶段 | 状态 | 备注 |
|------|------|------|
| 环境调研 | ✅ 完成 | conda 环境已确认, GPU RTX 4060 8GB |
| 文档规划 | ✅ 完成 | SKILL / process / prompts / verify_checklist / checkpoint |
| 场景搭建 | ✅ 完成 | Franka + table + DexCube + EE frame, 32D obs, 8D action |
| SAC 训练 | ✅ 完成 | SB3 SAC, MlpPolicy, buffer=100K, net=[256,256] |
| Reward Shaping | ✅ 完成 | Sparse / Shaped(4阶段) / PBRS / Curriculum 四种 |
| HER 集成 | ✅ 完成 | GoalEnv wrapper + HerReplayBuffer(future, n=4) |
| Sim2Sim | ✅ 完成 | ONNX 导出 + MuJoCo Menagerie Franka + 推理评估 |
| 实验报告 | ✅ 完成 | REPORT.md ~350行, 含参数对齐表/差异分析 |
