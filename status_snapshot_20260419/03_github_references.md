# 参考 GitHub 项目地址

> 2026-05-02 状态提示：本参考列表仍可用，但“当前项目还处在 PyBullet baseline 阶段”的判断已过期。当前默认 simulator-backed backend 已迁移到 MuJoCo；IsaacLab 仍是下一阶段训练平台参考。

下面这些仓库来自飞书 `AI / 大创` 文档里的调研和对比方向，并结合当前项目阶段做了筛选。它们不是都要立刻上手，但都适合作为后续路线参考。

## 一、仿真与训练主平台

| 项目 | 地址 | 作用 | 当前建议 |
| --- | --- | --- | --- |
| IsaacLab | https://github.com/isaac-sim/IsaacLab | 强化学习、模仿学习、运动规划的统一机器人学习框架 | 长期主平台重点参考 |
| IsaacSim | https://github.com/isaac-sim/IsaacSim | 高保真仿真、资产导入、传感器、ROS、数字孪生底座 | 长期仿真底座重点参考 |
| MetaIsaacGrasp | https://github.com/YitianShi/MetaIsaacGrasp | 基于 IsaacLab 的抓取学习 test bench | 适合参考抓取技能与数据生成 |

## 二、MuJoCo 与 benchmark 方向

| 项目 | 地址 | 作用 | 当前建议 |
| --- | --- | --- | --- |
| panda_mujoco_gym | https://github.com/zichunxx/panda_mujoco_gym | MuJoCo 上的 Franka Panda 操作环境 | 未来做 Panda/MuJoCo 对照时优先参考 |
| Metaworld | https://github.com/Farama-Foundation/Metaworld | 多任务与元强化学习 benchmark | 适合做 benchmark 和多任务扩展参考 |

## 三、LLM 规划与中层执行

| 项目 | 地址 | 作用 | 当前建议 |
| --- | --- | --- | --- |
| DELTA | https://github.com/boschresearch/DELTA | 基于 LLM 的长程任务规划与 PDDL 分解 | 适合参考高层任务分解与结构化规划 |
| KIOS | https://github.com/ProNeverFake/kios | LLM + 行为树 + 机器人技能执行系统 | 适合参考中层执行管理器和行为树 |
| ROSGPT | https://github.com/aniskoubaa/rosgpt | 自然语言到 ROS2 命令接口 | 适合参考自然语言接口层 |

## 四、sim2sim / sim2real 迁移参考

| 项目 | 地址 | 作用 | 当前建议 |
| --- | --- | --- | --- |
| kinova_isaaclab_sim2real | https://github.com/louislelay/kinova_isaaclab_sim2real | Isaac Lab 训练到 ROS2/真机部署的机械臂模板 | 适合参考训练到部署链路 |
| isaaclab_ur_reach_sim2real | https://github.com/louislelay/isaaclab_ur_reach_sim2real | UR10 reach 的 sim2sim / sim2real 模板 | 适合参考 ROS2 驱动与验证流程 |
| unitree_rl_lab | https://github.com/unitreerobotics/unitree_rl_lab | Isaac Lab 训练后经 MuJoCo 做 sim2sim，再做 sim2real 的工程样板 | 适合参考迁移管线思路 |
| frankapy | https://github.com/iamlab-cmu/frankapy | Franka Panda 的 Python 控制接口 | 真机阶段重点参考 |

## 五、VLA 与前沿相关工作

| 项目 | 地址 | 作用 | 当前建议 |
| --- | --- | --- | --- |
| OpenVLA | https://github.com/openvla/openvla | 开源 Vision-Language-Action 模型 | 适合相关工作和长期扩展参考 |
| OpenHelix | https://github.com/OpenHelix-Team/OpenHelix | 双系统 VLA 模型 | 适合相关工作和系统叙事参考 |

## 当前最相关的第一梯队

如果只保留最值得继续跟踪的项目，建议优先看这 6 个：

1. IsaacLab
2. IsaacSim
3. DELTA
4. KIOS
5. panda_mujoco_gym
6. frankapy

## 当前项目和这些仓库的关系

- 历史判断：2026-04-19 时本地项目还处在 `PyBullet baseline` 阶段；2026-05-02 后默认 simulator-backed backend 已迁移到 MuJoCo。
- IsaacLab / IsaacSim 对应的是后续“更强训练与仿真主平台”。
- DELTA / KIOS 对应的是后续“高层规划和中层调度”。
- panda_mujoco_gym 对应的是后续“MuJoCo 对照环境”。
- frankapy 对应的是后续“Franka 真机接口”。

## 当前不建议直接上手的方向

- 不建议立刻把 OpenVLA 或 OpenHelix 当成当前大创主线。
- 不建议在现有低层技能还没稳定前直接切换成端到端 VLA 路线。
- 当前更合理的策略仍然是：`高层规划 + 中层调度 + 低层技能库`。
