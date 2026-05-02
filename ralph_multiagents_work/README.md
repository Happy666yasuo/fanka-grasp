# 具身智能体研究 — Ralph 多智能体驱动实现

> 基于强化学习运动控制和大模型任务规划的具身智能体研究
> 实现方式：Ralph for Claude Code 自主开发循环

## 项目总览

```
大脑 (LLM Planner) ──→ 认知桥梁 (CausalExplore) ──→ 小脑 (RL Skills)
      ↑                        ↑                         ↑
  Phase 2                  Phase 1/3                  Phase 1
```

## 三阶段状态

| Phase | 目录 | 内容 | 测试 |
|-------|------|------|------|
| Phase 1 | `phase1_skills_and_causal/` | 技能库 + CausalExplore 闭环 | 64 OK |
| Phase 2 | `phase2_llm_planner/` | LLM Planner + 系统 Demo | 88 OK |
| Phase 3 | `phase3_experiments/` | 实验 + 论文材料 | 65 OK |
| **总计** | | **44+ Python 文件** | **214 OK** |

## 快速验证

```bash
conda activate beyondmimic

# Phase 1
cd phase1_skills_and_causal && python -m unittest discover tests -v

# Phase 2
cd ../phase2_llm_planner && python -m unittest discover tests -v

# Phase 3
cd ../phase3_experiments && python -m unittest discover tests -v
```

## 系统链路

```
自然语言任务输入
  → LLM Planner (Phase 2) 生成结构化步骤
  → CausalExplore (Phase 1) 物体属性/用途推断
  → RL Skills (Phase 1) 执行低层动作
  → 执行反馈 → 失败时重规划 (Phase 2)
  → 实验评估 (Phase 3)
```

## 当前实验口径

- 当前 simulator-backed 主路径已迁移到 MuJoCo；PyBullet 仅作为 legacy backend 保留。
- 2026-05-03 已重新运行 Phase 3 comparative、ablation 和 final demo，并刷新 JSON/report/paper summary 产物。
- Phase 3 mock/demo 规划已修复中文颜色歧义：`蓝色方块 -> 黄色区域` 不再误生成额外 `yellow_block` 操作。
- `simulator_backed` 实验不再静默降级为 mock；无法完成在线 probe 时会在 JSON 结果中标记 `fallback_used`/`error` 并计为失败。
- 非默认目标对象已覆盖：`blue_block -> yellow_zone`、`yellow_block -> blue_zone` 的 simulator-backed 回归均通过。

## 论文材料

位于 `phase3_experiments/outputs/paper/`:
- `paper_draft.md` — 论文主线草稿
- `presentation_outline.md` — 答辩 PPT 大纲
- `architecture_diagram.md` — 系统架构图描述
- `experiment_summary.md` — 实验摘要表

## Ralph 配置参考

- 安装: `~/.local/bin/ralph*`
- 启动: `cd <phase_dir> && ralph --monitor --calls 30 --timeout 60`
- 已知问题: 任务完成后可能空转（见全局记忆）
