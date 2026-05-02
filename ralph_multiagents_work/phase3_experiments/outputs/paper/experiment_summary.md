# 实验摘要表

> 2026-05-03 更新：本文件已根据 MuJoCo backend 迁移后的最新 comparative、ablation 和 final demo 输出刷新。结果来自 `outputs/experiments/*.json` 与 `outputs/demo_session_20260503_163112.json`。

## 实验环境

| 参数 | 值 |
|------|-----|
| 仿真环境 | MuJoCo (headless), PyBullet legacy backend retained |
| LLM 后端 | Mock (基于规则的任务解析) |
| 物体集合 | red_block, blue_block, yellow_block |
| 目标区域 | green_zone, blue_zone, yellow_zone |
| 探针动作 | lateral_push, top_press, side_pull, surface_tap, grasp_attempt |
| 最大重规划次数 | 3 |
| 随机种子 | 42 |

---

## 对比实验摘要

| 条件 | 试验数 | 成功率 | 平均探索步数 | 平均规划质量 | 平均重规划 | 平均耗时(s) |
|------|--------|--------|--------------|--------------|------------|-------------|
| no_causal | 4 | 100.00% | 0.0 | 0.730 | 0.0 | 0.00 |
| metadata_backed | 4 | 100.00% | 0.0 | 0.730 | 0.0 | 0.00 |
| simulator_backed | 4 | 100.00% | 5.0 | 0.702 | 0.0 | 0.00 |

本轮 comparative 结果中 `fallback_used=0`，`error=0`。

---

## 消融实验摘要

| 条件 | 策略 | U | R | 成功率 | 探索步数 | 不确定性 | 重规划 | 恢复 |
|------|------|---|---|--------|----------|----------|--------|------|
| random_U+_R+ | random | ✓ | ✓ | 100.00% | 3.0 | 0.550 | 0.0 | 0.0 |
| random_U+_R- | random | ✓ | ✗ | 0.00% | 3.0 | 0.550 | 0.0 | 0.0 |
| random_U-_R+ | random | ✗ | ✓ | 100.00% | 0.0 | 0.550 | 0.0 | 0.0 |
| random_U-_R- | random | ✗ | ✗ | 100.00% | 0.0 | 0.550 | 0.0 | 0.0 |
| curiosity_U+_R+ | curiosity | ✓ | ✓ | 100.00% | 3.0 | 0.400 | 0.0 | 0.0 |
| curiosity_U+_R- | curiosity | ✓ | ✗ | 100.00% | 3.0 | 0.400 | 0.0 | 0.0 |
| curiosity_U-_R+ | curiosity | ✗ | ✓ | 100.00% | 3.0 | 0.400 | 0.0 | 0.0 |
| curiosity_U-_R- | curiosity | ✗ | ✗ | 50.00% | 3.0 | 0.400 | 0.0 | 0.0 |
| causal_explore_U+_R+ | causal_explore | ✓ | ✓ | 100.00% | 3.0 | 0.446 | 0.0 | 0.0 |
| causal_explore_U+_R- | causal_explore | ✓ | ✗ | 0.00% | 3.0 | 0.446 | 0.0 | 0.0 |
| causal_explore_U-_R+ | causal_explore | ✗ | ✓ | 100.00% | 3.0 | 0.446 | 0.0 | 0.0 |
| causal_explore_U-_R- | causal_explore | ✗ | ✗ | 100.00% | 3.0 | 0.446 | 0.0 | 0.0 |

本轮 ablation 中 5 个失败均为显式 no-recovery 条件失败，`fallback_used=0`。

---

## 最终 Demo 场景结果

| 场景 | 任务 | 成功 | Actions | 探索步数 | 重规划 | 耗时(s) |
|------|------|------|---------|----------|--------|---------|
| multi_block | 红色方块→绿色区域 | ✓ | observe, pick, place | 0 | 0 | 0.00 |
| multi_block | 蓝色方块→黄色区域 | ✓ | observe, pick, place | 0 | 0 | 0.00 |
| multi_block | 黄色方块→蓝色区域 | ✓ | observe, pick, place | 0 | 0 | 0.00 |
| interactive | 鼠标功能推断 | ✓ | observe, probe, press, push, rotate | 1 | 0 | 0.00 |

---

## 关键指标定义

| 指标 | 定义 | 范围 |
|------|------|------|
| 成功率 (Success Rate) | 任务成功完成的试验比例 | [0, 1] |
| 探索步数 (Explore Steps) | CausalExplore 探针动作执行次数 | [0, ∞) |
| 规划质量 (Planning Quality) | 基于动作权重的启发式评分 | [0, 1] |
| 不确定性分数 (Uncertainty Score) | ProbeExecutor 计算的综合不确定性 | [0, 1] |
| 重规划次数 (Replan Count) | 执行失败触发的 LLM 重规划次数 | [0, MAX_REPLANS] |
| 恢复次数 (Recovery Count) | Recovery 机制成功恢复的次数 | [0, ∞) |
