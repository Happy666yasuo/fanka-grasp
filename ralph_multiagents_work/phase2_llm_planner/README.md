# Phase 2: 结构化 LLM Planner 与系统闭环

基于强化学习运动控制和大模型任务规划的具身智能体研究 — Phase 2

## 状态

**Phase 2 完成** ✅ — 88 测试通过, 0 失败

## 验收标准达成

| 验收标准 | 状态 | 详情 |
|----------|------|------|
| LLM planner 输出严格符合 PlannerStep schema | ✅ | 所有 step 使用 allowed skills |
| 高不确定性时优先触发 probe | ✅ | uncertainty >= 0.50 自动触发 |
| 技能失败后可重规划 | ✅ | bounded retry (max 3 replans, 2 skill retries) |
| 最小 demo 从自然语言任务走完整链路 | ✅ | 多方块 + 鼠标交互 |
| 所有测试通过 | ✅ | 88/88 |

## 项目结构

```
phase2_llm_planner/
├── src/
│   ├── planner/               # LLM Planner 核心
│   │   ├── llm_planner.py           # LLMPlanner (遵循 Planner 协议)
│   │   ├── prompt_templates.py      # 中/英文提示模板
│   │   ├── uncertainty_handler.py   # 不确定性处理 + probe 注入
│   │   └── replan_handler.py        # 失败分类 + bounded retry
│   └── demo/                  # 闭环 Demo
│       └── closed_loop_demo.py      # 自然语言 → 规划 → 探索 → 执行
├── tests/                     # 88 单元/集成测试
└── outputs/                   # 评估产出
```

## 环境

```bash
conda activate beyondmimic
python -m unittest discover tests -v
```

## 核心能力

- **LLMPlanner**: 结构化 JSON 输出，严格禁止连续控制
- **UncertaintyHandler**: 高不确定性 → probe → 低不确定性 → 执行，并提供结构化 `evaluate()` 决策接口
- **ReplanHandler**: execution_error → retry skill, planning_error → replan
- **ClosedLoopDemo**: 多方块放置 + 鼠标用途推断（按压→拖动→滚轮），中文颜色区域解析避免把目标区域误识别为额外物体

## 技能白名单

probe, observe, pick, place, press, push, pull, rotate

## 关键约束

- LLM 不得输出关节位置/力矩等连续控制参数
- 最大重规划次数: 3
- 最大技能重试: 2
