# Phase 1: 小脑技能扩展与 CausalExplore Simulator-Backed 闭环

基于强化学习运动控制和大模型任务规划的具身智能体研究 — Phase 1

## 状态

**Phase 1 完成** ✅ — 64 测试通过, 0 失败

> 2026-05-02 更新：Phase1 simulator-backed smoke 和 CausalExplore probe 默认使用 MuJoCo backend；PyBullet 仅作为 legacy fallback 保留在核心 `embodied_agent` 中。

## 验收标准达成

| 验收标准 | 状态 | 详情 |
|----------|------|------|
| press/push/pull/rotate 各有独立 smoke 测试通过 | ✅ | 每个技能通过 simulator-backed smoke 测试 |
| raw task-pool place 成功率 ≥ 0.20 | ✅ | **0.98** (49/50), baseline 0.08 → 0.98 |
| CausalExplore 三种策略对比有 Markdown 报告 | ✅ | `outputs/phase1_comparison_report.md` |
| 所有测试通过 | ✅ | 64/64 |
| 技能执行器统一接口就绪 | ✅ | UnifiedSkillExecutor + SkillRegistry |
| MuJoCo backend 迁移 | ✅ | probe/new skills/place improvements 不再直接依赖 PyBullet API |

## 项目结构

```
phase1_skills_and_causal/
├── src/
│   ├── skills/              # 技能实现
│   │   ├── new_skills.py          # PressSkill, PushSkill, PullSkill, RotateSkill
│   │   ├── skill_registry.py      # SkillRegistry (6 skills)
│   │   ├── skill_executor.py      # UnifiedSkillExecutor
│   │   └── place_improvements.py  # Place 改进 (0.08 → 0.98)
│   └── causal_explore/      # CausalExplore 扩展
│       ├── probe_actions.py       # 5 探针原语
│       ├── probe_executor.py      # MuJoCo 探针执行器
│       ├── explore_strategies.py  # Random/Curiosity/CausalExplore 策略
│       └── eval_runner.py         # 多策略评估运行器
├── scripts/                 # Formal eval 脚本
│   ├── run_place_eval.py         # Place 成功率 formal eval
│   └── run_strategy_comparison.py
├── tests/                   # 63 单元/集成测试
└── outputs/                 # 评估产出和报告
    ├── phase1_summary_report.md
    ├── phase1_comparison_report.md
    ├── strategy_comparison.json
    └── place_eval_results.json
```

## 环境

```bash
conda activate beyondmimic
python -m unittest discover tests -v
```

## 技能库

pick, place, press, push, pull, rotate

## Place 成功率

| 指标 | Baseline | 改进后 |
|------|----------|--------|
| Success Rate | 0.08 | **0.98** |
| 主要失败原因 | transport_failed, released_outside_zone | released_outside_zone (1/50) |

改进方法: zone-centered transport + release timing verification (`src/skills/place_improvements.py`)
