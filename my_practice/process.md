# 项目工作流程 — Franka 机械臂桌面方块抓取（SAC + IsaacLab）

> **项目代号**: franka_cube_grasp
> **核心目标**: 在 IsaacLab 2.1.0 中，使用 SAC 算法训练 Franka Panda 机械臂完成 "从桌上抓起方块" 任务，
>               解决稀疏奖励问题（Reward Shaping + HER），并通过 Sim2Sim 迁移至 MuJoCo 验证。

---

## 阶段总览

```
Phase 0 ──→ Phase 1 ──→ Phase 2 ──→ Phase 3 ──→ Phase 4 ──→ Phase 5 ──→ Phase 6
环境确认     场景搭建     Baseline     奖励工程     HER集成     Sim2Sim      总结报告
  ✅           ✅         SAC训练        ✅          ✅         迁移验证        ✅
                           ✅                                    ✅
```

**实际工期**: 2 天 (Day 1: Phase 0-3, Day 2: Phase 4-6)

---

## Phase 0: 环境验证与准备（Day 1）

### 0.1 确认 Conda 环境可用性

```bash
# 主开发环境
conda activate beyondmimic
python -c "import isaacsim; print('IsaacSim:', isaacsim.__version__)"
python -c "import isaaclab; print('IsaacLab:', isaaclab.__version__)"
python -c "import stable_baselines3; print('SB3:', stable_baselines3.__version__)"
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"

# Sim2Sim 验证环境
conda activate unitree-rl
python -c "import mujoco; print('MuJoCo:', mujoco.__version__)"
```

### 0.2 确认 GPU 与显存

```bash
nvidia-smi
# 记录：GPU 型号、显存大小、驱动版本
```

### 0.3 确认 IsaacLab 官方 Lift 任务可运行

```bash
conda activate beyondmimic
cd ~/Desktop/isaac_workspace/IsaacLab-2.1.0

# 列出可用环境
python -m isaaclab.scripts.environments.list_envs 2>/dev/null || \
  python source/isaaclab_tasks/isaaclab_tasks/utils/parse_cfg.py  # 备选

# 尝试运行 Franka Lift 的 random agent
python scripts/environments/random_agent.py --task Isaac-Lift-Cube-Franka-v0 --num_envs 4 --headless
```

### ✅ 验收标准
- [x] `beyondmimic` 环境中 IsaacSim 4.5.0 + IsaacLab 2.1.0 确认可用
- [ ] GPU 正常工作，显存 ≥ 8GB
- [ ] 官方 Lift 任务能以 4 个环境正常启动

---

## Phase 1: 场景搭建与自定义环境（Day 2-4）

### 1.1 分析官方 Lift 任务结构

官方代码位置:
```
~/Desktop/isaac_workspace/IsaacLab-2.1.0/source/isaaclab_tasks/
  isaaclab_tasks/manager_based/manipulation/lift/
  ├── __init__.py
  ├── lift_env_cfg.py          # 场景 + MDP 配置（核心文件）
  ├── config/
  │   └── franka/
  │       ├── __init__.py
  │       └── joint_pos_env_cfg.py   # Franka 具体配置
  └── mdp/
      ├── __init__.py
      ├── observations.py      # 自定义观测
      └── rewards.py           # 自定义奖励
```

**关键学习点:**
- `ObjectTableSceneCfg`: 如何定义桌子 + 物体 + 机器人
- `ObservationsCfg` / `RewardsCfg`: Manager-Based 架构的 MDP 定义
- `EventsCfg`: 域随机化（物体初始位姿、机器人关节噪声）

### 1.2 创建自定义环境

在 `my_practice/franka_cube_grasp/` 下创建项目结构：

```
franka_cube_grasp/
├── __init__.py
├── envs/
│   ├── __init__.py
│   ├── franka_grasp_env_cfg.py    # 基于 lift_env_cfg.py 修改
│   └── mdp/
│       ├── __init__.py
│       ├── observations.py        # 观测函数
│       ├── rewards.py             # 奖励函数（重点修改）
│       └── terminations.py        # 终止条件
├── agents/
│   └── __init__.py
└── scripts/
    └── __init__.py
```

### 1.3 定义场景配置

```python
# franka_grasp_env_cfg.py 核心内容设计
@configclass
class FrankaCubeGraspSceneCfg(InteractiveSceneCfg):
    """Franka + 桌面 + 方块 场景"""
    robot: ArticulationCfg = FRANKA_PANDA_CFG  # 7-DOF + gripper
    ee_frame: FrameTransformerCfg = ...        # 末端执行器跟踪
    object: RigidObjectCfg = ...               # 5cm 立方体
    table: AssetBaseCfg = ...                  # 实验桌
```

### 1.4 定义 MDP

**观测空间** (~35 维):
| 分量 | 维度 | 说明 |
|------|------|------|
| 关节角 | 7 | Franka 7 个关节 |
| 关节角速度 | 7 | 关节速度 |
| Gripper 状态 | 2 | 左右手指位置 |
| EE 位置 | 3 | 末端执行器世界坐标 |
| EE 朝向 | 4 | 四元数 |
| 物体位置 | 3 | 方块世界坐标 |
| 物体朝向 | 4 | 方块旋转 |
| 目标位置 | 3 | 抬起目标点 |
| EE→物体相对位姿 | 3 | 方向向量 |

**动作空间** (8 维):
- 7 个关节位置目标 + 1 个 gripper 开合

### ✅ 验收标准
- [ ] 自定义环境能正常初始化（`num_envs=4 --headless`）
- [ ] 机器人、桌子、方块在 headless 模式下物理仿真正常
- [ ] 随机动作不会导致崩溃
- [ ] 观测维度正确，能打印观测值

---

## Phase 2: Baseline — 稀疏奖励 SAC 训练（Day 5-7）

### 2.1 选择 RL 框架

**方案对比:**

| 框架 | SAC 支持 | HER 支持 | IsaacLab 集成度 | 推荐度 |
|------|---------|---------|----------------|--------|
| **Stable-Baselines3** | ✅ 原生 | ✅ 原生 | ✅ 有官方wrapper | ⭐⭐⭐⭐⭐ |
| SKRL | ✅ 原生 | ❌ 需自实现 | ✅ 有官方wrapper | ⭐⭐⭐⭐ |
| RL-Games | ❌ 仅 PPO | ❌ 无 | ✅ 深度集成 | ⭐⭐ |

**结论**: 使用 **Stable-Baselines3 (SB3)** 作为主框架（SAC + HER 原生支持）。

### 2.2 纯稀疏奖励 Baseline

```python
# 奖励设计 v0: 纯稀疏
def sparse_reward(env):
    """物体抬起到目标高度以上 → +1, 否则 → 0"""
    object_height = env.object.data.root_pos_w[:, 2]  # z 坐标
    success = object_height > TARGET_HEIGHT  # e.g., 0.2m above table
    return success.float()
```

### 2.3 SAC 配置

```python
from stable_baselines3 import SAC

model = SAC(
    policy="MlpPolicy",
    env=isaac_env_wrapper,
    learning_rate=3e-4,
    buffer_size=100_000,       # ⚠️ 本机显存/内存有限，不用 1M
    batch_size=256,
    tau=0.005,           # soft update 系数
    gamma=0.99,
    train_freq=1,
    gradient_steps=1,
    ent_coef="auto",     # 自动调节温度 α
    verbose=1,
    tensorboard_log="./logs/sac_baseline"
)
```

### 2.4 训练与记录

```bash
conda activate beyondmimic
cd ~/Desktop/beyondmimic/my_practice/franka_cube_grasp

# Baseline 训练（预期：纯稀疏奖励大概率不收敛或极慢）
python scripts/train.py \
    --algo sac \
    --reward sparse \
    --num_envs 64 \
    --total_timesteps 500000 \
    --headless \
    --log_dir logs/sac_sparse_baseline
```

### ✅ 验收标准
- [ ] SAC 训练能正常跑起来（无报错）
- [ ] TensorBoard 能看到 reward / loss 曲线
- [ ] 记录稀疏奖励下的成功率（预期很低）
- [ ] 形成 Baseline 对比基准

---

## Phase 3: 奖励工程 — Reward Shaping + 课程学习（Day 8-12）

### 3.1 Reward Shaping 策略（核心难点）

**分层奖励设计 v1:**

```python
def shaped_reward(env):
    """多阶段密集奖励"""
    reward = 0.0

    # Stage 1: 接近奖励 — EE 靠近物体
    dist_ee_to_obj = torch.norm(ee_pos - obj_pos, dim=-1)
    reward += -1.0 * dist_ee_to_obj  # 距离越近奖励越高

    # Stage 2: 抓握奖励 — Gripper 闭合且接触物体
    is_grasping = detect_grasp(env)
    reward += 2.0 * is_grasping

    # Stage 3: 抬升奖励 — 物体高度增加
    lift_height = obj_pos[:, 2] - table_height
    reward += 5.0 * torch.clamp(lift_height, 0, 0.3)

    # Stage 4: 稳定奖励 — 物体在目标高度保持稳定
    at_target = (lift_height > target_height).float()
    reward += 10.0 * at_target

    # 正则化惩罚
    reward -= 0.01 * torch.norm(actions, dim=-1)  # 动作平滑
    reward -= 0.1 * (obj_dropped).float()          # 掉落惩罚

    return reward
```

### 3.2 Potential-Based Reward Shaping (PBRS)

```python
# 保证最优策略不变性的势函数方法
def potential(state):
    """势函数: 基于到目标的距离"""
    dist = torch.norm(ee_pos - obj_pos, dim=-1) + \
           torch.norm(obj_pos - target_pos, dim=-1)
    return -dist

# F(s, s') = γ * Φ(s') - Φ(s)
shaping_reward = gamma * potential(next_state) - potential(state)
total_reward = sparse_reward + shaping_reward
```

### 3.3 课程学习（Curriculum Learning）

```
阶段 1 (0-100k steps):  物体在 gripper 正下方，只需闭合+抬起
阶段 2 (100k-300k):     物体在桌面随机位置，需要 reach + grasp + lift
阶段 3 (300k-500k):     增加物体初始朝向随机化
阶段 4 (500k+):         增加域随机化（摩擦、质量、噪声）
```

### 3.4 对比实验矩阵

| 实验编号 | 奖励类型 | 课程学习 | 预期成功率 |
|---------|---------|---------|-----------|
| exp_01 | 稀疏 | ❌ | < 5% |
| exp_02 | 密集 (shaped) | ❌ | 30-60% |
| exp_03 | 密集 (shaped) | ✅ | 50-80% |
| exp_04 | PBRS + 稀疏 | ❌ | 20-40% |
| exp_05 | 密集 + HER | ❌ | 60-90% |

### ✅ 验收标准
- [ ] Shaped reward 训练曲线明显优于 sparse baseline
- [ ] 成功率从 <5% 提升到 >50%
- [ ] 至少完成 3 组对比实验并记录到 TensorBoard
- [ ] 找到一组可稳定收敛的奖励权重

---

## Phase 4: HER 集成与优化（Day 13-15）

### 4.1 HER + SAC (SB3 原生支持)

```python
from stable_baselines3 import HerReplayBuffer, SAC

model = SAC(
    policy="MultiInputPolicy",  # HER 需要 dict observation
    env=goal_conditioned_env,   # 需要 GoalEnv 接口
    replay_buffer_class=HerReplayBuffer,
    replay_buffer_kwargs=dict(
        n_sampled_goal=4,                    # 每个 transition 重标注 4 个目标
        goal_selection_strategy="future",    # 从未来轨迹中采样目标
    ),
    learning_rate=1e-3,
    buffer_size=100_000,        # ⚠️ 本机显存/内存有限，不用 1M
    batch_size=256,
    gamma=0.95,
    verbose=1,
    tensorboard_log="./logs/sac_her"
)
```

### 4.2 Goal-Conditioned 环境适配

需要将环境改造为 `gymnasium.GoalEnv` 格式：
```python
observation_space = Dict({
    "observation": Box(...),     # 机器人状态
    "achieved_goal": Box(...),   # 当前物体位置
    "desired_goal": Box(...),    # 目标物体位置
})

def compute_reward(achieved_goal, desired_goal, info):
    """HER 兼容的奖励计算"""
    dist = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
    return -(dist > threshold).astype(np.float32)  # 稀疏: 达到目标 → 0, 否则 → -1
```

### 4.3 HER vs Shaped Reward 对比

```bash
# 实验: SAC + HER (稀疏奖励)
python scripts/train.py --algo sac_her --reward sparse --num_envs 64 --headless --total_timesteps 500000

# 实验: SAC + HER + Shaped Reward
python scripts/train.py --algo sac_her --reward shaped --num_envs 64 --headless --total_timesteps 500000
```

### ✅ 验收标准
- [ ] HER 环境适配完成（GoalEnv 接口）
- [ ] SAC+HER 在稀疏奖励下也能学到有效策略
- [ ] HER 对比实验结果记录
- [ ] 最佳配置成功率 > 70%

---

## Phase 5: Sim2Sim — IsaacSim → MuJoCo 迁移验证（Day 16-18）

### 5.1 模型导出

**步骤 A: 策略导出为 ONNX**
```python
import torch

# 加载训练好的 SAC 策略
model = SAC.load("checkpoints/best_model.zip")

# 导出 actor 网络
dummy_obs = torch.randn(1, obs_dim)
torch.onnx.export(
    model.policy.actor,
    dummy_obs,
    "sim2sim/policy_franka_grasp.onnx",
    input_names=["observation"],
    output_names=["action"],
    opset_version=11
)
```

**步骤 B: 创建 MuJoCo 场景（MJCF）**
```xml
<!-- sim2sim/franka_table.xml -->
<mujoco model="franka_grasp">
  <include file="franka_emika_panda.xml"/>
  <!-- 桌面 -->
  <body name="table" pos="0.5 0 0.4">
    <geom type="box" size="0.3 0.3 0.02" rgba="0.6 0.4 0.2 1"/>
  </body>
  <!-- 方块 -->
  <body name="cube" pos="0.5 0 0.44">
    <joint type="free"/>
    <geom type="box" size="0.025 0.025 0.025" mass="0.1" rgba="1 0 0 1"/>
  </body>
</mujoco>
```

### 5.2 MuJoCo 推理验证

```python
# sim2sim/mujoco_eval.py
import mujoco
import onnxruntime as ort

session = ort.InferenceSession("policy_franka_grasp.onnx")
model = mujoco.MjModel.from_xml_path("franka_table.xml")
data = mujoco.MjData(model)

# 推理循环
for episode in range(100):
    mujoco.mj_resetData(model, data)
    for step in range(500):
        obs = get_observation(model, data)
        action = session.run(None, {"observation": obs})[0]
        data.ctrl[:] = action[0]
        mujoco.mj_step(model, data)
    # 记录成功率
```

### 5.3 物理参数对齐

| 参数 | IsaacSim 值 | MuJoCo 值 | 对齐方法 |
|------|------------|-----------|---------|
| 重力 | (0, 0, -9.81) | (0, 0, -9.81) | 直接设置 |
| 时间步 | 0.01s (1/dt) | 0.002s (x5 substep) | timestep × nsubstep |
| 关节阻尼 | PhysX default | 需手动匹配 | 导出 URDF 属性 |
| 接触摩擦 | friction cfg | solref / solimp | 实验标定 |
| 关节限位 | URDF limits | MJCF limits | 保持一致 |

### 5.4 参考 BeyondMimic 项目的 Sim2Sim 经验

```
关键参考文件:
- Beyondmimic_Deploy_G1/csv_to_npz_with_Interpolation.py  # 动作插值
- Beyondmimic_Deploy_G1/deploy_mujoco_1.py                # MuJoCo 部署脚本
- Beyondmimic_Deploy_G1/complete_success/                  # 成功案例
```

### ✅ 验收标准
- [ ] ONNX 模型成功导出
- [ ] MuJoCo MJCF 场景搭建完成
- [ ] 策略在 MuJoCo 中能执行（不要求完美）
- [ ] 记录两个仿真器的成功率差异
- [ ] 分析迁移失败的原因（如有）

---

## Phase 6: 总结报告与复盘（Day 19-20）

### 6.1 实验报告内容

```markdown
# Franka 机械臂桌面抓取实验报告

## 1. 问题定义与动机
## 2. 环境与场景设计
## 3. 算法选择（SAC）与稀疏奖励挑战
## 4. 解决方案
   4.1 Reward Shaping 设计与效果
   4.2 HER 集成与效果
   4.3 课程学习策略
## 5. 实验结果
   5.1 训练曲线对比
   5.2 成功率对比表
   5.3 最佳超参配置
## 6. Sim2Sim 迁移
   6.1 物理参数对齐
   6.2 迁移成功率
   6.3 差异分析
## 7. 结论与展望
## 附录: 超参数表、环境配置
```

### 6.2 关键指标收集

| 指标 | 记录方式 |
|------|---------|
| 训练奖励曲线 | TensorBoard |
| 成功率 vs 步数 | 自定义 callback |
| 训练时间 | 日志时间戳 |
| 显存占用 | nvidia-smi |
| MuJoCo 迁移成功率 | 评估脚本 |

### ✅ 验收标准
- [ ] 完整实验报告
- [ ] 所有对比实验的 TensorBoard 日志
- [ ] 最佳模型 checkpoint + ONNX
- [ ] 可复现的训练脚本

---

## 常见问题预案

### Q1: IsaacSim 启动报错 / 卡住
```bash
# 检查 NVIDIA 驱动
nvidia-smi
# 检查 Vulkan
vulkaninfo | head -20
# 降低 num_envs
--num_envs 2
```

### Q2: SB3 与 IsaacLab 环境接口不兼容
- 需要编写 Gymnasium Wrapper 将 IsaacLab 环境包装为标准 Gym 接口
- 参考: `isaaclab_rl` 中的 `Sb3VecEnvWrapper`

### Q3: SAC 训练内存溢出
```bash
# ⚠️ 必须使用 --headless，GUI 渲染额外占用大量显存
# 减小 replay buffer（默认已设为 100K）
--buffer_size 50000
# 减少并行环境数（正式训练推荐 64~128，OOM 时继续降低）
--num_envs 32
# 减小 batch_size
--batch_size 128
```

### Q4: 奖励不收敛
1. 检查观测是否归一化
2. 检查奖励量级是否合理（建议 [-10, 10] 范围）
3. 降低学习率
4. 增加探索噪声

### Q5: Sim2Sim 策略完全失效
1. 检查观测空间是否完全一致
2. 检查动作空间范围
3. 检查物理参数（时间步、摩擦、阻尼）
4. 尝试在 MuJoCo 中 fine-tune 几步

---

## 日志规范

```
logs/
├── sac_sparse_baseline/     # exp_01: 稀疏奖励基线
├── sac_shaped_v1/           # exp_02: 密集奖励 v1
├── sac_shaped_curriculum/   # exp_03: 密集 + 课程学习
├── sac_pbrs/                # exp_04: PBRS
├── sac_her_sparse/          # exp_05: HER + 稀疏
├── sac_her_shaped/          # exp_06: HER + 密集
└── sim2sim_eval/            # Sim2Sim 评估结果
```
