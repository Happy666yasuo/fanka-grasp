# Franka 机械臂桌面抓取实验报告

> **项目**: Franka Cube Grasp — 基于 IsaacLab + SAC 的强化学习抓取
> **作者**: happywindman
> **日期**: 2026-03-30
> **环境**: IsaacSim 4.5.0 / IsaacLab 2.1.0 / SB3 2.7.1 / MuJoCo 3.2.3

---

## 1. 问题定义与动机

### 1.1 任务描述

使用 Franka Emika Panda 7-DOF 机械臂在桌面场景中抓取并举起一个方块（DexCube）。
这是机器人操作（Robot Manipulation）领域最经典的 benchmark 之一。

### 1.2 核心挑战

1. **稀疏奖励问题**: 抓取成功是极低概率事件，纯随机探索几乎不可能在合理时间内发现成功轨迹
2. **高维连续控制**: 8D 动作空间（7 关节 + 1 夹爪），32D 观测空间
3. **接触动力学**: 物体抓取涉及复杂的接触力学，对物理模拟器精度有较高要求
4. **Sim-to-Sim Gap**: IsaacSim (PhysX) 与 MuJoCo 之间的物理引擎差异

### 1.3 项目目标

- 搭建完整的 RL 训练管线（环境 → 训练 → 评估 → 部署）
- 实现并对比多种解决稀疏奖励的方法
- 完成 Sim2Sim 迁移验证（IsaacSim → MuJoCo）

---

## 2. 环境与场景设计

### 2.1 仿真平台

| 组件 | 配置 |
|------|------|
| 物理引擎 | NVIDIA IsaacSim 4.5.0 (PhysX 5) |
| RL 框架 | IsaacLab 2.1.0 (Manager-Based) |
| 算法库 | Stable-Baselines3 2.7.1 |
| GPU | NVIDIA RTX 4060 Laptop 8GB |
| 仿真频率 | 100 Hz (dt=0.01s) |
| 控制频率 | 50 Hz (decimation=2) |
| Episode 长度 | 5.0s (250 steps) |

### 2.2 场景组成

```
FrankaGraspSceneCfg:
├── Robot: Franka Panda (panda_instanceable.usd)
│   ├── 7 arm joints (revolute) + 2 finger joints (prismatic)
│   ├── ImplicitActuator: stiffness=80, damping=4 (shoulder/forearm)
│   └── ImplicitActuator: stiffness=2000, damping=100 (hand)
├── Object: DexCube (scale=0.8, mass~0.1kg)
│   └── init_pos = [0.5, 0, 0.055]
├── Table: SeattleLabTable
├── EE Frame: FrameTransformer (panda_hand + offset [0,0,0.1034])
├── Ground Plane
└── Dome Light
```

### 2.3 观测空间 (32D)

| 索引 | 名称 | 维度 | 说明 |
|------|------|------|------|
| 0-8 | joint_pos_rel | 9D | 关节位置（相对默认值）|
| 9-17 | joint_vel_rel | 9D | 关节速度 |
| 18-20 | object_pos_b | 3D | 方块在机器人基座系中的位置 |
| 21-23 | ee_object_rel | 3D | 末端执行器→方块相对向量 |
| 24-31 | last_action | 8D | 上一步动作 |

### 2.4 动作空间 (8D)

| 索引 | 名称 | 说明 |
|------|------|------|
| 0-6 | arm_action | 关节位置增量 (scale=0.5, offset=default) |
| 7 | gripper_action | 二值夹爪 (>0: open 0.04m, ≤0: close 0m) |

### 2.5 事件（Domain Randomization）

- **方块位置随机化**: x ∈ [-0.1, +0.1], y ∈ [-0.25, +0.25]（相对初始位置）
- **场景重置**: 每 episode 结束时 reset_scene_to_default

---

## 3. 算法选择与稀疏奖励挑战

### 3.1 算法: SAC (Soft Actor-Critic)

选择 SAC 的原因:
- **样本效率**: Off-policy 算法，可充分利用 replay buffer 中的历史经验
- **自动温度调节**: `ent_coef="auto"` 自动平衡探索与利用
- **连续动作**: 天然适合连续控制任务
- **稳定性**: 相比 DDPG/TD3，SAC 的训练更稳定

### 3.2 SAC 超参数

```python
SACConfig:
    policy: "MlpPolicy"
    buffer_size: 100,000  (显存约束)
    batch_size: 256
    learning_rate: 3e-4
    gamma: 0.99
    tau: 0.005
    net_arch: [256, 256]
    ent_coef: "auto"
    train_freq: 1
    gradient_steps: 1
```

### 3.3 稀疏奖励的困难

纯稀疏奖励（仅在方块被举起时给 +1）下，SAC 面临严重的探索困难：
- 需要依次完成 **伸手 → 接近 → 对准 → 夹取 → 举起** 五个子步骤
- 随机动作下完成整个序列的概率极低
- 1000 步冒烟测试中成功率 = 0%

---

## 4. 解决方案

### 4.1 Reward Shaping（奖励工程）

#### 4.1.1 Shaped Multi-Stage Reward

四阶段密集奖励：

$$R_{shaped} = \begin{cases}
1.0 \cdot R_{reach} & \text{if } d_{ee \to obj} > 0.1 \\
0.5 \cdot R_{reach} + 2.0 \cdot R_{grasp} & \text{if } d_{ee \to obj} \leq 0.1 \\
3.0 \cdot R_{grasp} + 5.0 \cdot R_{lift} & \text{if grasping} \\
10.0 \cdot R_{hold} & \text{if } z_{obj} > z_{thresh}
\end{cases}$$

其中：
- $R_{reach} = 1 - \tanh(5 \cdot d_{ee \to obj})$：到达奖励
- $R_{grasp}$：基于手指距离的夹取奖励
- $R_{lift} = \max(0, z_{obj} - z_{init}) \cdot 10$：举起奖励
- $R_{hold}$：保持高度奖励

#### 4.1.2 PBRS (Potential-Based Reward Shaping)

基于势函数的奖励整形，保证最优策略不变：

$$F(s, s') = \gamma \Phi(s') - \Phi(s)$$

势函数设计：

$$\Phi(s) = -5.0 \cdot d_{ee \to obj} + 10.0 \cdot \max(0, z_{obj} - z_{init})$$

#### 4.1.3 Curriculum Reward

三级课程学习：
1. **Easy** (0-50K steps): 仅到达奖励
2. **Medium** (50K-150K): 到达 + 夹取 + 举起
3. **Hard** (150K+): 稀疏成功 + 精确控制惩罚

### 4.2 HER (Hindsight Experience Replay)

#### 4.2.1 核心思想

HER 通过回顾性地替换失败轨迹的目标为实际到达的状态，将失败经验转化为成功经验。

#### 4.2.2 GoalEnv 包装器

```
Flat obs (32D) → Dict obs:
├── observation (29D): joint_pos[0:18] + ee_object_rel[21:32]
├── achieved_goal (3D): object_pos_b[18:21]
└── desired_goal (3D): [0.0, 0.0, 0.2] (固定高度目标)
```

#### 4.2.3 HER 参数

| 参数 | 值 | 说明 |
|------|-----|------|
| n_sampled_goal | 4 | 每个 transition 采样 4 个虚拟目标 |
| goal_selection_strategy | "future" | 从同一 episode 后续状态采样 |
| distance_threshold | 0.05 | 距离阈值，< 0.05m 视为达到目标 |
| learning_starts | 250 × num_envs | 至少等 1 个完整 episode 结束 |

#### 4.2.4 关键实现细节

1. **terminal_observation 转换**: SB3 的 `_store_transition` 要求 `infos[i]["terminal_observation"]` 为 Dict 格式，需在 wrapper 中拦截转换
2. **compute_reward 路由**: HER buffer 调用 `env.env_method("compute_reward", ...)` 需在 wrapper 中拦截，避免传递到不支持该方法的 IsaacLab 内部环境
3. **稀疏奖励**: `R = 0` if $\|g_{achieved} - g_{desired}\| < \epsilon$, else `R = -1`

### 4.3 方法对比总结

| 方法 | 优点 | 缺点 |
|------|------|------|
| Sparse | 最优策略保证 | 极难收敛 |
| Shaped | 快速引导学习 | 可能引入局部最优 |
| PBRS | 保持最优策略 + 加速 | 势函数设计需要领域知识 |
| HER | 适用于稀疏目标达成 | 内存开销大，goal 定义需合理 |

---

## 5. 实验设计与结果

### 5.1 实验矩阵

为系统对比不同算法和奖励策略的效果，设计了 6 组消融实验：

| ID | 算法 | 奖励类型 | HER | 课程学习 | num_envs | 训练步数 |
|----|------|----------|-----|----------|----------|----------|
| exp-01 | SAC | sparse | No | No | 64 | 500K |
| exp-02 | SAC | shaped | No | No | 64 | 500K |
| exp-03 | SAC | curriculum | No | Yes | 64 | 500K |
| exp-04 | SAC | PBRS | No | No | 64 | 500K |
| exp-05 | SAC+HER | sparse | Yes | No | 64 | 500K |
| exp-06 | SAC+HER | shaped | Yes | No | 64 | 500K |

**实验基础设施**：
- 批量执行脚本: `scripts/run_experiments.sh`（6 组实验顺序执行）
- 对比图表工具: `scripts/plot_results.py`（从 TensorBoard 日志生成对比图）
- 所有实验固定 `--headless`, `num_envs=64`, `buffer_size=100_000`, `seed=42`

**运行方式**:
```bash
conda activate beyondmimic
cd ~/Desktop/beyondmimic/my_practice/franka_cube_grasp
bash scripts/run_experiments.sh           # 启动全部 6 组实验
python scripts/plot_results.py            # 生成对比图表
```

### 5.2 冒烟测试验证

在正式 500K 实验之前，所有变体在 1000 步冒烟测试中验证通过：

| 变体 | 状态 | 备注 |
|------|------|------|
| SAC + Sparse (exp-01) | ✅ | 基线，成功率 0% (预期) |
| SAC + Shaped (exp-02) | ✅ | 奖励范围 [0, 0.04]，无 NaN/Inf |
| SAC + Curriculum (exp-03) | ✅ | 课程阶段自动切换正常 |
| SAC + PBRS (exp-04) | ✅ | PBRS 项正常计算 |
| SAC+HER + Sparse (exp-05) | ✅ | Buffer 正确填充，HER 重标注生效 |
| SAC+HER + Shaped (exp-06) | ✅ | HER + 密集奖励组合正常 |

### 5.3 HER Buffer 诊断

```
Buffer: HerReplayBuffer, size=375, strategy=FUTURE, n_sampled_goal=4
Reward distribution after HER relabeling:
  min=-1.0, max=0.0, mean=-0.043
  Success (reward=0): 292/375 = 77.9%
```

HER 重标注成功地将大量失败经验转化为"成功"经验，显著提高了有效样本比例。

### 5.4 预期结果分析

基于奖励设计和算法特性，对 6 组实验的收敛性做如下预测：

| ID | 预期收敛速度 | 预期最终性能 | 分析依据 |
|----|------------|------------|---------|
| exp-01 | ❌ 极慢/不收敛 | 低 | 纯稀疏奖励，探索极其困难 |
| exp-02 | 🟡 中等 | 中高 | 4 阶段密集奖励提供持续梯度 |
| exp-03 | 🟡 中等 | 中高 | 课程学习渐进增加难度 |
| exp-04 | 🟢 较快 | 高 | PBRS 保最优策略 + 加速探索 |
| exp-05 | 🟡 中等 | 中 | HER 缓解稀疏问题但目标简单 |
| exp-06 | 🟢 较快 | 高 | HER + shaped 双重加速 |

**关键假设**:
- **exp-04 (PBRS)** 和 **exp-06 (HER+Shaped)** 预计表现最好，因为它们从两个不同角度解决稀疏奖励问题
- **exp-01 (Sparse)** 几乎不可能收敛，作为消融基线验证奖励工程的必要性
- **exp-03 (Curriculum)** 的效果取决于阶段切换阈值是否合理

### 5.5 观察与当前发现

**关键发现**:
1. **管线完整性**: 从 IsaacLab 场景定义到 MuJoCo Sim2Sim 的完整闭环已打通
2. **HER 有效性**: 即使在短训练中，HER 重标注机制已体现优势（77.9% 虚拟成功率）
3. **奖励设计**: shaped/PBRS/curriculum 奖励均产生了合理的梯度信号

**后续工作**: 使用 `scripts/run_experiments.sh` 执行完整 500K 实验后，
用 `scripts/plot_results.py` 生成学习曲线对比图以验证上述假设

---

## 6. Sim2Sim 迁移

### 6.1 MuJoCo 场景搭建

使用 MuJoCo Menagerie 官方 Franka Panda 模型:

| IsaacLab | MuJoCo | 对齐 |
|----------|--------|------|
| PhysX, dt=0.01s | Implicit fast, dt=0.002×5 | ✅ 等效 |
| decimation=2 | 10 substeps/action | ✅ 等效 |
| gravity=-9.81 | gravity=-9.81 | ✅ |
| DexCube (scale 0.8) | Box (half-size 0.02) | ≈ 近似 |
| panda_instanceable.usd | Menagerie panda.xml | ≈ 近似 |
| ImplicitActuator | General actuator (PD) | ≈ 不同增益 |
| SeattleLabTable | Box (简化) | ≈ 近似 |

### 6.2 观测空间对齐

32D 观测重建:
- `joint_pos_rel`: MuJoCo `qpos[:9] - default_qpos` ↔ IsaacLab `joint_pos_rel`
- `joint_vel_rel`: MuJoCo `qvel[:9]` ↔ IsaacLab `joint_vel_rel`
- `object_pos_b`: MuJoCo `R_base^T @ (cube_w - base_w)` ↔ IsaacLab `subtract_frame_transforms`
- `ee_object_rel`: MuJoCo `cube_w - (hand_w + R_hand @ offset)` ↔ IsaacLab `cube_w - ee_w`
- `last_action`: 直接复制上一步动作

### 6.3 动作空间映射

| IsaacLab | MuJoCo |
|----------|--------|
| `arm_action * 0.5 + default` | `ctrl[0:7] = target_pos` |
| `gripper > 0 → open(0.04)` | `ctrl[7] = 255 (open)` |
| `gripper ≤ 0 → close(0.0)` | `ctrl[7] = 0 (close)` |

### 6.4 迁移结果

使用冒烟测试 checkpoint（1000 步训练）:
- **IsaacLab 成功率**: 0% (eval.py, 5 episodes)
- **MuJoCo 成功率**: 0% (mujoco_eval.py, 10 episodes)
- **策略行为**: 输出非零动作（范围 [-0.92, 0.51]），但尚未形成有效行为模式

### 6.5 Sim2Sim 差异分析

预期的主要差异来源:
1. **接触模型**: PhysX 使用 TGS solver，MuJoCo 使用 complementarity solver
2. **执行器模型**: IsaacLab ImplicitActuator vs MuJoCo general actuator (不同 PD 增益)
3. **摩擦模型**: PhysX friction correlation distance vs MuJoCo solref/solimp
4. **几何近似**: USD 高精度网格 vs MJCF STL/OBJ 近似

**建议改进方向**:
- 在 MuJoCo 中 fine-tune 几步以弥补 sim gap
- 训练时增加域随机化（关节阻尼、摩擦系数扰动）
- 使用 system identification 对齐两侧物理参数

---

## 7. 结论与展望

### 7.1 项目成果

1. **完整训练管线**: 从零搭建了 IsaacLab Manager-Based 环境 → SB3 SAC → ONNX 导出 → MuJoCo 评估的端到端管线
2. **四种奖励策略**: 实现了 Sparse / Shaped / PBRS / Curriculum 四种奖励函数，支持灵活切换
3. **HER 集成**: 成功将 SB3 HER 与 IsaacLab VecEnv 环境对接，解决了多个兼容性问题
4. **Sim2Sim 管道**: 构建了 IsaacSim → MuJoCo 的完整迁移管道，包括场景、观测、动作对齐

### 7.2 技术亮点

- **HER Wrapper 设计**: 处理了 SB3 HER 与非标准 VecEnv 的三个关键兼容性问题（terminal_observation、compute_reward 路由、learning_starts 计算）
- **模块化架构**: `@configclass` + Manager-Based 模式，奖励/观测/动作全部可配置
- **物理参数对齐**: 详细的 IsaacSim ↔ MuJoCo 参数映射文档

### 7.3 局限与未来工作

1. **训练深度不足**: 由于时间限制，未进行大规模训练（500K+ steps），无法展示各方法的完整学习曲线
2. **超参数调优**: 未进行系统的超参数搜索（learning rate, net_arch, buffer_size 等）
3. **Sim2Sim 精度**: 执行器模型和接触参数的精确对齐需要更多标定工作
4. **真实世界迁移**: Sim2Real 需要额外的域随机化和系统辨识

### 7.4 经验总结

| 经验 | 详情 |
|------|------|
| 显存管理 | RTX 4060 8GB 下必须 `--headless`, `num_envs ≤ 128`, `buffer_size ≤ 100K` |
| HER 陷阱 | SB3 HER 对 VecEnv 有严格的接口要求，需仔细处理 Dict obs 转换 |
| Sim2Sim | 即使任务简单，两个物理引擎间的差异也不容忽视 |
| 工程规范 | 分阶段开发 + 验证清单 + git checkpoint 是高效的项目管理方式 |

---

## 附录

### A. 项目文件结构

```
franka_cube_grasp/
├── __init__.py
├── agents/
│   ├── __init__.py
│   ├── sac_cfg.py           # SAC 超参数配置
│   └── her_wrapper.py       # HER GoalEnv VecEnv 包装器
├── configs/                  # (预留)
├── envs/
│   ├── __init__.py           # Gymnasium 环境注册
│   ├── franka_grasp_env_cfg.py  # 场景 + MDP 完整配置
│   └── mdp/
│       ├── __init__.py       # MDP 函数重导出
│       ├── observations.py   # 自定义观测函数
│       ├── rewards.py        # 6 种奖励函数
│       └── terminations.py   # 终止条件
├── scripts/
│   ├── train.py              # SAC/SAC+HER 训练脚本
│   ├── eval.py               # 评估脚本
│   ├── smoke_test.py         # 环境冒烟测试
│   ├── check_reward_range.py # 奖励范围诊断
│   ├── check_her_buffer.py   # HER buffer 诊断
│   ├── run_experiments.sh    # 6 组实验批量执行脚本
│   └── plot_results.py       # TensorBoard 日志对比图表
├── sim2sim/
│   ├── export_onnx.py        # ONNX 导出
│   ├── mujoco_eval.py        # MuJoCo 推理评估
│   ├── policy_franka_grasp.onnx  # 导出的策略
│   └── franka_emika_panda/   # MuJoCo Menagerie Franka 模型
│       ├── franka_table.xml  # Sim2Sim 场景
│       ├── panda.xml         # Franka MJCF
│       └── assets/           # 网格文件
├── logs/                     # 训练日志 + checkpoints
└── checkpoints/              # (预留)
```

### B. 运行命令速查

```bash
# --- beyondmimic 环境 ---
conda activate beyondmimic

# 冒烟测试
python scripts/smoke_test.py --num_envs 2 --headless

# SAC 训练 (sparse)
python scripts/train.py --headless --num_envs 64 --total_timesteps 500000 --reward_type sparse

# SAC 训练 (shaped)
python scripts/train.py --headless --num_envs 64 --total_timesteps 500000 --reward_type shaped

# SAC + HER 训练
python scripts/train.py --headless --num_envs 64 --total_timesteps 500000 --algo sac_her

# SAC + Curriculum 训练
python scripts/train.py --headless --num_envs 64 --total_timesteps 500000 --reward_type curriculum

# 批量运行 6 组实验
bash scripts/run_experiments.sh

# 生成对比图表
python scripts/plot_results.py --log_dir logs/experiments

# 评估
python scripts/eval.py --checkpoint logs/xxx/checkpoints/latest.zip --headless --num_episodes 100

# ONNX 导出
python sim2sim/export_onnx.py --checkpoint logs/xxx/checkpoints/latest.zip

# --- unitree-rl 环境 ---
conda activate unitree-rl

# Sim2Sim 评估
python sim2sim/mujoco_eval.py --episodes 100 --render false
```

### C. Git Checkpoint 索引

| Tag | Commit | 阶段 | 日期 |
|-----|--------|------|------|
| cp-0-env-init | 590b576 | Phase 0: 环境验证 | 2026-03-29 |
| cp-1-scene-setup | 34ec480 | Phase 1: 场景搭建 | 2026-03-29 |
| cp-2-sac-baseline | 712c51d | Phase 2: SAC Baseline | 2026-03-29 |
| cp-3-reward-shaping | ff96763 | Phase 3: Reward Shaping | 2026-03-29 |
| cp-4-her | f6a80fc | Phase 4: HER 集成 | 2026-03-30 |
| cp-5-sim2sim | 41f3583 | Phase 5: Sim2Sim | 2026-03-30 |
| cp-6-report | fcc3026 | Phase 6: 实验报告 | 2026-03-30 |
