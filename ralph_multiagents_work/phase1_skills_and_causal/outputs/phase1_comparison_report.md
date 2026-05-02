# CausalExplore 探索策略对比报告

## 概述

本报告对比了 **Random**、**Curiosity-Driven**、**CausalExplore** 三种探索策略在 2 个物体上的表现。

## 物体清单

- **red_block** (类别: block)
  - 预期属性: movable, pressable, graspable, rigid
  - 候选用途: pushable, pressable, pullable, graspable, tappable
- **blue_block** (类别: block)
  - 预期属性: movable, pressable, graspable, rigid
  - 候选用途: pushable, pressable, pullable, graspable, tappable

## 策略描述

| 策略 | 描述 |
|------|------|
| **Random** | 均匀随机选择 (探针, 物体) 对，无历史信息利用 |
| **Curiosity-Driven** | 基于位移幅度的启发式好奇心，优先采样产生大位移的探针-物体对 |
| **CausalExplore** | 利用不确定性估计驱动探索，优先降低 Property/Affordance 信念不确定性 |

## 定量对比

| 策略 | 物体 | 步数 | 属性推断准确率 | 用途推断准确率 | 平均位移(m) | 唯一探针数 | 耗时(s) |
|------|------|------|----------------|----------------|-------------|------------|---------|
| random | red_block | 8 | 0.303 | 0.450 | 0.0013 | 4 | 1.04 |
| curiosity | red_block | 8 | 0.251 | 0.410 | 0.0075 | 5 | 1.15 |
| causal_explore | red_block | 8 | 0.251 | 0.410 | 0.0032 | 5 | 0.95 |
| random | blue_block | 8 | 0.250 | 0.380 | 0.0000 | 3 | 0.20 |
| curiosity | blue_block | 8 | 0.000 | 0.340 | 0.0000 | 4 | 0.20 |
| causal_explore | blue_block | 8 | 0.000 | 0.300 | 0.0000 | 5 | 0.20 |

### 聚合指标（按策略平均）

| 策略 | 属性推断准确率 | 用途推断准确率 | 平均位移(m) | 平均步数 |
|------|----------------|----------------|-------------|----------|
| causal_explore | 0.125 | 0.355 | 0.0016 | 8.0 |
| curiosity | 0.125 | 0.375 | 0.0038 | 8.0 |
| random | 0.276 | 0.415 | 0.0007 | 8.0 |

## 结论

- **CausalExplore** 属性推断准确率: 0.125, 用途推断准确率: 0.355
- **Curiosity-Driven** 属性推断准确率: 0.125, 用途推断准确率: 0.375
- **Random** 属性推断准确率: 0.276, 用途推断准确率: 0.415

详细结果已保存至: `outputs/strategy_comparison.json`
