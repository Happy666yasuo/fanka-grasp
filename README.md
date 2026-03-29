# Franka Cube Grasp — IsaacLab SAC Project

> 基于 IsaacLab 的 Franka 机械臂桌面方块抓取（SAC + Reward Shaping + HER + Sim2Sim）

## 项目目标

在 **IsaacLab 2.1.0 / IsaacSim 4.5.0** 中，使用 **SAC (Soft Actor-Critic)** 算法训练
**Franka Emika Panda** (7-DOF + gripper) 完成"从桌上抓起方块"任务。

核心技术挑战：
- **稀疏奖励** → Reward Shaping + PBRS
- **HER (Hindsight Experience Replay)** → 目标重标注
- **课程学习** → 由易到难
- **Sim2Sim** → IsaacSim → MuJoCo 迁移验证

## 环境矩阵

| 用途 | Conda 环境 | Python | 关键包 |
|------|-----------|--------|--------|
| 训练 & 场景 | `beyondmimic` | 3.10 | isaacsim 4.5.0, isaaclab 2.1.0, SB3 2.7.1, torch 2.5.1 |
| Sim2Sim MuJoCo | `unitree-rl` | 3.8 | mujoco 3.2.3, torch 2.3.1 |

## 项目结构

```
.
├── AGENTS.md                          # AI 编辑器全局规则（自动加载）
├── .agents/skills/franka-cube-grasp/  # Agent Skills 标准文件
│   └── SKILL.md
├── my_practice/
│   ├── SKILL.md                       # 技能树（人类阅读版）
│   ├── process.md                     # 6 阶段项目流程
│   ├── prompts.md                     # AI 提示词库 v2.1
│   ├── verify_checklist.md            # 每阶段验证清单
│   ├── checkpoint.md                  # 版本节点记录
│   ├── Embodied_Intelligence_Advance_Plan.md  # 进阶学习计划
│   ├── daily_log_*.md                 # 每日工作日志
│   └── franka_cube_grasp/             # 项目代码
│       ├── envs/                      # 自定义 IsaacLab 环境
│       │   ├── franka_grasp_env_cfg.py    # 场景 + MDP 配置
│       │   └── mdp/                       # 观测/奖励/终止函数
│       ├── agents/                    # SAC / HER 配置
│       ├── scripts/                   # 训练 / 评估 / 诊断脚本
│       └── sim2sim/                   # Sim2Sim 迁移
└── .gitignore
```

## 项目阶段

| Phase | 内容 | 状态 | Tag |
|-------|------|------|-----|
| 0 | 环境验证与项目初始化 | ✅ | `cp-0-env-init` |
| 1 | 场景搭建与自定义环境 | ✅ | `cp-1-scene-setup` |
| 2 | SAC Baseline 训练 | ✅ | `cp-2-sac-baseline` |
| 3 | Reward Shaping + 课程学习 | ✅ | `cp-3-reward-shaping` |
| 4 | HER 集成 | ⬜ | |
| 5 | Sim2Sim 迁移验证 | ⬜ | |
| 6 | 实验对比与报告 | ⬜ | |

## ⚠️ 显存约束

- 本机显存有限，**所有训练/测试必须 `--headless`**
- 冒烟测试: `num_envs ≤ 4` | 正式训练: `num_envs = 64~128`
- SAC `buffer_size = 100_000`（非 1M）

## 参考

- [IsaacSim 4.5.0 Docs](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/index.html)
- [IsaacLab Ecosystem](https://isaac-sim.github.io/IsaacLab/main/source/setup/ecosystem.html)
- [SB3 SAC](https://stable-baselines3.readthedocs.io/en/master/modules/sac.html)
- [SB3 HER](https://stable-baselines3.readthedocs.io/en/master/modules/her.html)

## License

Personal learning project.
