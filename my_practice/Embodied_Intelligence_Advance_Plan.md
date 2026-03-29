# 具身智能进阶执行手册（MuJoCo + Isaac Lab）

> 目标：在你已有强化学习基础上，快速进入“可建模 + 可跑通 + 可扩展”的工程状态。

---

## 0. 总体路线（建议 4~6 周）

- **第 1~2 周（MuJoCo）**
  - 会写/改 MJCF（XML）模型
  - 能稳定跑 `HalfCheetah` / `Hopper` / `Walker2d`
  - 完成一次 PPO 训练 + 评估 + 可视化
- **第 3~4 周（Isaac Lab）**
  - 完成 Isaac Lab 环境安装
  - 跑通官方 `Anymal` 或 `Franka` 任务
  - 理解“单 GPU 并行数千环境”的配置入口
- **第 5~6 周（强化）**
  - 统一实验记录（WandB/TensorBoard）
  - 做一次从 MuJoCo 到 Isaac Lab 的迁移对比实验

---

## 1) MuJoCo 进阶：从 MJCF 到标准 Locomotion 任务

## 1.1 环境准备（Linux / bash）

建议使用 Python 3.10~3.11，单独虚拟环境。

```bash
cd /home/happywindman/Desktop/mypractice
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip wheel setuptools
pip install "gymnasium[mujoco]" mujoco stable-baselines3[extra] tensorboard lxml
```

快速自检：

```bash
python - <<'PY'
import gymnasium as gym
env = gym.make("HalfCheetah-v4")
obs, info = env.reset(seed=0)
for _ in range(10):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
print("MuJoCo + Gymnasium OK", obs.shape)
env.close()
PY
```

> 若 `-v4` 不存在，请先列出可用版本：

```bash
python - <<'PY'
import gymnasium as gym
ids = [spec.id for spec in gym.envs.registry.values()]
for name in ["HalfCheetah", "Hopper", "Walker2d"]:
    print(name, [i for i in ids if name in i])
PY
```

---

## 1.2 MJCF（XML）学习最短路径

按下面顺序学（每一步都做“小改动 + 运行验证”）：

1. **最小模型结构**：`<mujoco><worldbody><body>...</body></worldbody></mujoco>`
2. **刚体与关节**：`body / geom / joint`
3. **驱动器**：`<actuator><motor .../></actuator>`
4. **接触与摩擦**：`geom` 的 `friction`、`solref`、`solimp`
5. **传感器与观测**：`<sensor>`（jointpos、jointvel、framepos）
6. **默认参数复用**：`<default>`
7. **Include 组织工程**：`<include file="..."/>`

最小 MJCF 示例（可当模板）：

```xml
<mujoco model="two_link">
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <worldbody>
    <body name="base" pos="0 0 0.5">
      <geom type="capsule" fromto="0 0 0 0 0 0.3" size="0.03"/>
      <body name="link1" pos="0 0 0.3">
        <joint name="j1" type="hinge" axis="0 1 0" range="-90 90"/>
        <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="j1" gear="50" ctrllimited="true" ctrlrange="-1 1"/>
  </actuator>
</mujoco>
```

---

## 1.3 跑通 `HalfCheetah` / `Hopper` / `Walker2d`

### A. 随机策略冒烟测试（先确认环境正常）

```bash
python - <<'PY'
import gymnasium as gym
for task in ["HalfCheetah-v4", "Hopper-v4", "Walker2d-v4"]:
    env = gym.make(task)
    obs, info = env.reset(seed=0)
    ep_ret = 0.0
    for _ in range(1000):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        ep_ret += reward
        if terminated or truncated:
            obs, info = env.reset()
    print(task, "smoke_return=", round(ep_ret, 2))
    env.close()
PY
```

### B. PPO 训练（建议先从 HalfCheetah 开始）

```bash
python - <<'PY'
import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make("HalfCheetah-v4")
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./tb_logs")
model.learn(total_timesteps=200_000)
model.save("ppo_halfcheetah")
env.close()
PY
```

### C. 评估

```bash
python - <<'PY'
import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make("HalfCheetah-v4")
model = PPO.load("ppo_halfcheetah")
obs, info = env.reset(seed=42)
ret = 0.0
for _ in range(2000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    ret += reward
    if terminated or truncated:
        break
print("Eval return:", round(ret, 2))
env.close()
PY
```

---

## 1.4 MuJoCo 阶段验收标准（你要达成）

- 能手写一个双连杆或四足简化模型（含关节、执行器）
- 能解释观测、动作、奖励三者对应关系
- 三个 Locomotion 任务都能完成“创建环境 + rollout + 训练 + 评估”
- 能定位至少 3 种常见问题（版本、渲染、发散）

---

## 2) Isaac Lab：并行大规模仿真入门

> 核心价值：在单卡上并行大量环境，提高采样效率，非常适合机器人强化学习。

## 2.1 机器准备（先检查）

```bash
nvidia-smi
python3 --version
```

建议：
- NVIDIA GPU（显存越大越好）
- 驱动版本满足当前 Isaac Sim / Isaac Lab 要求
- 至少 16GB 内存（更高更稳）

---

## 2.2 Isaac Lab 安装（推荐按官方仓库 README）

```bash
cd /home/happywindman/Desktop/mypractice
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
./isaaclab.sh --help
```

根据你机器与官方文档，执行安装步骤（不同版本参数可能变化）：

```bash
./isaaclab.sh --install
```

> 如果你偏好 Conda / Docker，也可以按官方文档对应分支走，避免版本错配。

---

## 2.3 列出可用任务并筛选 `Anymal` / `Franka`

```bash
cd /home/happywindman/Desktop/mypractice/IsaacLab
./isaaclab.sh --python scripts/environments/list_envs.py | grep -Ei "anymal|franka"
```

先做一个最小环境启动（只启动，不训练）：

```bash
./isaaclab.sh --python scripts/environments/random_agent.py --task <YOUR_TASK_NAME> --num_envs 64
```

---

## 2.4 跑官方 RL 示例（以 rsl_rl 为例）

训练：

```bash
./isaaclab.sh --python scripts/reinforcement_learning/rsl_rl/train.py --task <YOUR_TASK_NAME> --num_envs 1024
```

回放/评估：

```bash
./isaaclab.sh --python scripts/reinforcement_learning/rsl_rl/play.py --task <YOUR_TASK_NAME> --checkpoint <CHECKPOINT_PATH>
```

> `num_envs` 可从 `256 -> 512 -> 1024 -> 2048` 逐步调，观察显存与吞吐。

---

## 2.5 Isaac Lab 阶段验收标准

- 成功安装并运行 `list_envs.py`
- 跑通一个 `Anymal` 或 `Franka` 官方任务（训练 + play）
- 能解释并行数 `num_envs` 对采样速度与显存占用影响
- 能保存并复现实验（同种子、同配置）

---

## 3) 常见坑与排障清单

1. **GPU/驱动不匹配**：先 `nvidia-smi`，再对照官方兼容矩阵。
2. **环境版本冲突**：MuJoCo 与 Isaac Lab 分开虚拟环境。
3. **任务名错误**：先 `list_envs.py`，再复制精确 task 名。
4. **显存不足（OOM）**：降低 `num_envs`、关掉高开销渲染。
5. **训练不收敛**：检查奖励尺度、动作范围、观测归一化。

---

## 4) 你现在就能执行的最小行动（今天）

1. 在本地完成 MuJoCo 环境安装 + `HalfCheetah` 冒烟测试。
2. 跑一次 20 万步 PPO，拿到第一个 checkpoint。
3. 克隆 Isaac Lab，执行 `list_envs.py` 并筛出 `Anymal/Franka`。
4. 跑一个 `random_agent.py` 小规模并行（64 env）验证 GPU pipeline。

---

## 5) 下一步我可以继续帮你做什么

- 生成一套你可直接运行的 `MuJoCo` 训练脚本模板（含日志/评估/保存）
- 按你显卡型号给出 `Isaac Lab` 的 `num_envs` 推荐表
- 给你做一个“从 MuJoCo 迁移到 Isaac Lab”的实验对照模板（统一指标）
