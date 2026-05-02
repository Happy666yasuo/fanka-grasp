# Phase 3: 系统集成、对比实验、消融实验与论文材料

基于强化学习运动控制和大模型任务规划的具身智能体研究 — Phase 3（最终阶段）

## 状态

**Phase 3 完成** ✅ — 65 测试通过, 0 失败

> 2026-05-03 更新：simulator-backed 路径已从 PyBullet 迁移到 MuJoCo。PyBullet 仅作为 legacy backend 保留；当前测试和在线 probe 回归默认验证 MuJoCo。comparative、ablation、final demo、reports 和 paper summary 已按 MuJoCo 口径重新生成。

## 验收标准达成

| 验收标准 | 状态 | 详情 |
|----------|------|------|
| 对比实验：无 CausalExplore vs metadata-backed vs simulator-backed | ✅ | 2026-05-03 MuJoCo 刷新，3 条件均 4/4；simulator-backed 不再静默 fallback |
| 消融实验覆盖 3 个维度 | ✅ | 2026-05-03 MuJoCo 刷新，strategy (3) × uncertainty (2) × recovery (2) = 12 条件；失败显式记录 error |
| Markdown 报告 + matplotlib 图表 | ✅ | `outputs/reports/` |
| 论文主线草稿 + 答辩 PPT 大纲 | ✅ | `outputs/paper/` |
| 最终 demo 可复现 | ✅ | 2026-05-03 刷新，多方块 + 鼠标交互，4/4 PASS |
| 所有测试通过 | ✅ | 65/65 |
| MuJoCo simulator-backed 回归 | ✅ | 覆盖 `red->green`、`blue->yellow`、`yellow->blue` 非默认目标对象 |

## 项目结构

```
phase3_experiments/
├── src/
│   ├── experiments/           # 实验框架
│   │   ├── comparative_experiment.py  # 对比实验 (3 条件)
│   │   └── ablation_experiment.py     # 消融实验 (12 条件)
│   ├── reporting/             # 报告生成
│   │   ├── experiment_reporter.py     # Markdown 报告生成
│   │   └── chart_generator.py         # matplotlib 图表生成
│   └── demo/                  # 最终 Demo
│       └── final_demo.py             # 完整闭环演示脚本
├── tests/                     # 63 单元/集成/验收测试
└── outputs/                   # 实验产出
    ├── experiments/           # 实验 JSON + artifact catalog
    ├── reports/               # Markdown 报告 + PNG 图表
    └── paper/                 # 论文材料
        ├── paper_draft.md
        ├── presentation_outline.md
        ├── architecture_diagram.md
        └── experiment_summary.md
```

## 环境

```bash
conda activate beyondmimic
python -m unittest discover tests -v
```

## 实验设计

### 对比实验（3 条件 × N 任务）
| 条件 | 说明 |
|------|------|
| no_causal | 纯视觉描述，无 CausalExplore |
| metadata_backed | 离线 CausalExplore 预测 |
| simulator_backed | 在线 MuJoCo 交互 CausalExplore |

当前代码验证结果：
- no_causal / metadata_backed / simulator_backed 三条件均可运行。
- simulator_backed 已覆盖蓝/黄非默认目标对象，不再因只创建默认 `red_block` 而失败。
- 若在线 probe 失败，结果仍会记录 `fallback_used=true` 和 `error`，并不得伪装为 simulator-backed 成功。

说明：`outputs/experiments/comparative_results.json` 是静态实验产物；正式论文前应重新运行并刷新 JSON/图表。

### 消融实验（3 维度 × 12 条件）
| 维度 | 取值 |
|------|------|
| 探索策略 | random / curiosity / causal_explore |
| Uncertainty 使用 | on / off |
| Recovery 启用 | on / off |

当前代码验证结果：不再包含缺失 `evaluate` 的 AttributeError；`causal_explore` 条件已覆盖非默认目标对象；无 recovery 条件仍可能失败，作为消融对照保留。

## Demo 场景

```
[PASS] 把红色方块移到绿色区域 → observe → pick → place
[PASS] 把蓝色方块移到黄色区域 → observe → pick → place
[PASS] 把黄色方块移到蓝色区域 → observe → pick → place
[PASS] 探索鼠标功能 → observe → probe → press → push → rotate
```

## 依赖

- Phase 1: skills + causal_explore (64 tests)
- Phase 2: planner + demo (88 tests)
- embodied_agent: contracts, simulator, executor, MuJoCo backend (112 tests)
