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
│   └── franka_cube_grasp/             # 项目代码（核心）
│       ├── REPORT.md                      # 完整实验报告 (~460行)
│       ├── envs/                          # 自定义 IsaacLab 环境
│       │   ├── franka_grasp_env_cfg.py    # 场景 + MDP 配置 (5种奖励)
│       │   └── mdp/                       # 观测/奖励/终止函数
│       │       ├── observations.py        # 3个自定义观测函数
│       │       ├── rewards.py             # 4个奖励函数 (sparse/shaped/pbrs/curriculum)
│       │       └── terminations.py        # 物体掉落终止
│       ├── agents/                        # SAC / HER 配置
│       │   ├── sac_cfg.py                 # SAC 超参数
│       │   └── her_wrapper.py             # HER GoalEnv wrapper (218行)
│       ├── scripts/                       # 训练 / 评估 / 诊断脚本
│       │   ├── train.py                   # SAC/SAC+HER 训练主脚本
│       │   ├── eval.py                    # 评估脚本
│       │   ├── run_experiments.sh         # 6组实验批量运行
│       │   ├── plot_results.py            # TensorBoard日志对比绘图
│       │   ├── check_reward_range.py      # 奖励范围诊断
│       │   ├── check_her_buffer.py        # HER buffer诊断
│       │   └── smoke_test.py              # 冒烟测试
│       └── sim2sim/                       # Sim2Sim 迁移
│           ├── export_onnx.py             # ONNX 导出
│           ├── mujoco_eval.py             # MuJoCo 推理评估
│           ├── policy_franka_grasp.onnx   # 部署策略 (312KB)
│           └── franka_emika_panda/        # MuJoCo Menagerie 模型
│               ├── franka_table.xml       # Sim2Sim 场景
│               ├── panda.xml              # Franka Panda MJCF
│               └── hand.xml              # 夹爪 MJCF
└── .gitignore
```

## 项目阶段

| Phase | 内容 | 状态 | Tag |
|-------|------|------|-----|
| 0 | 环境验证与项目初始化 | ✅ | `cp-0-env-init` |
| 1 | 场景搭建与自定义环境 | ✅ | `cp-1-scene-setup` |
| 2 | SAC Baseline 训练 | ✅ | `cp-2-sac-baseline` |
| 3 | Reward Shaping + 课程学习 | ✅ | `cp-3-reward-shaping` |
| 4 | HER 集成 | ✅ | `cp-4-her` |
| 5 | Sim2Sim 迁移验证 | ✅ | `cp-5-sim2sim` |
| 6 | 实验对比与报告 | ✅ | `cp-6-complete` |

## 实验矩阵

| 编号 | 算法 | 奖励方案 | 说明 |
|------|------|----------|------|
| exp-01 | SAC | Sparse | 消融基线 |
| exp-02 | SAC | Shaped | 多阶段密集奖励 |
| exp-03 | SAC | Curriculum | 课程学习奖励 |
| exp-04 | SAC | PBRS | 势函数整形 |
| exp-05 | SAC+HER | Sparse | HER 加速稀疏探索 |
| exp-06 | SAC+HER | Shaped | HER + 密集奖励 |

## 快速开始

```bash
# 训练 (SAC + Shaped Reward, 64 envs, 500K steps)
conda activate beyondmimic
cd ~/Desktop/beyondmimic/my_practice/franka_cube_grasp
python scripts/train.py --reward_type shaped --num_envs 64 --total_timesteps 500000 --headless

# 评估
python scripts/eval.py --checkpoint logs/<run_name>/checkpoints/latest.zip --num_episodes 50 --headless

# 导出 ONNX + MuJoCo 验证
python sim2sim/export_onnx.py --checkpoint logs/<run_name>/checkpoints/latest.zip
conda activate unitree-rl
python sim2sim/mujoco_eval.py --model sim2sim/policy_franka_grasp.onnx --episodes 100

# 批量实验
bash scripts/run_experiments.sh

# 绘制对比图
python scripts/plot_results.py --log_dir logs
```

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
