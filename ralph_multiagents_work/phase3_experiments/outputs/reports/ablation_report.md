# 消融实验报告

## 实验配置

- 消融维度: explore_strategy, uncertainty, recovery
- 条件数: 12
- 任务数: 2

## 消融维度说明

| 维度 | 取值 | 说明 |
|------|------|------|
| **探索策略** | random / curiosity / causal_explore | 不同探索策略对任务成功率的影响 |
| **Uncertainty** | U+ / U- | Planner 是否使用 uncertainty handler 进行不确定性感知规划 |
| **Recovery** | R+ / R- | Executor 是否启用失败恢复 (replan) 机制 |

## 详细结果

| 条件 | 策略 | U | R | 任务 | 成功 | 总步数 | 探索 | 不确定性 | 重规划 | 恢复 | 耗时(s) | Fallback | Error |
|------|------|---|---|------|------|--------|------|----------|--------|------|---------|----------|-------|
| random_U+_R+ | random | ✓ | ✓ | 把红色方块移到绿色区域 | ✓ | 4 | 3 | 0.550 | 0 | 0 | 0.00 | no | - |
| random_U+_R- | random | ✓ | ✗ | 把红色方块移到绿色区域 | ✗ | 4 | 3 | 0.550 | 0 | 0 | 0.00 | no | Step failed without recovery: place |
| random_U-_R+ | random | ✗ | ✓ | 把红色方块移到绿色区域 | ✓ | 4 | 0 | 0.550 | 0 | 0 | 0.00 | no | - |
| random_U-_R- | random | ✗ | ✗ | 把红色方块移到绿色区域 | ✓ | 4 | 0 | 0.550 | 0 | 0 | 0.00 | no | - |
| curiosity_U+_R+ | curiosity | ✓ | ✓ | 把红色方块移到绿色区域 | ✓ | 3 | 3 | 0.400 | 0 | 0 | 0.00 | no | - |
| curiosity_U+_R- | curiosity | ✓ | ✗ | 把红色方块移到绿色区域 | ✓ | 3 | 3 | 0.400 | 0 | 0 | 0.00 | no | - |
| curiosity_U-_R+ | curiosity | ✗ | ✓ | 把红色方块移到绿色区域 | ✓ | 3 | 3 | 0.400 | 0 | 0 | 0.00 | no | - |
| curiosity_U-_R- | curiosity | ✗ | ✗ | 把红色方块移到绿色区域 | ✗ | 3 | 3 | 0.400 | 0 | 0 | 0.00 | no | Step failed without recovery: place |
| causal_explore_U+_R+ | causal_explore | ✓ | ✓ | 把红色方块移到绿色区域 | ✓ | 4 | 3 | 0.446 | 0 | 0 | 0.00 | no | - |
| causal_explore_U+_R- | causal_explore | ✓ | ✗ | 把红色方块移到绿色区域 | ✗ | 4 | 3 | 0.446 | 0 | 0 | 0.00 | no | Step failed without recovery: place |
| causal_explore_U-_R+ | causal_explore | ✗ | ✓ | 把红色方块移到绿色区域 | ✓ | 4 | 3 | 0.446 | 0 | 0 | 0.00 | no | - |
| causal_explore_U-_R- | causal_explore | ✗ | ✗ | 把红色方块移到绿色区域 | ✓ | 4 | 3 | 0.446 | 0 | 0 | 0.00 | no | - |
| random_U+_R+ | random | ✓ | ✓ | 把蓝色方块移到黄色区域 | ✓ | 4 | 3 | 0.550 | 0 | 0 | 0.00 | no | - |
| random_U+_R- | random | ✓ | ✗ | 把蓝色方块移到黄色区域 | ✗ | 4 | 3 | 0.550 | 0 | 0 | 0.00 | no | Step failed without recovery: pick |
| random_U-_R+ | random | ✗ | ✓ | 把蓝色方块移到黄色区域 | ✓ | 4 | 0 | 0.550 | 0 | 0 | 0.00 | no | - |
| random_U-_R- | random | ✗ | ✗ | 把蓝色方块移到黄色区域 | ✓ | 4 | 0 | 0.550 | 0 | 0 | 0.00 | no | - |
| curiosity_U+_R+ | curiosity | ✓ | ✓ | 把蓝色方块移到黄色区域 | ✓ | 3 | 3 | 0.400 | 0 | 0 | 0.00 | no | - |
| curiosity_U+_R- | curiosity | ✓ | ✗ | 把蓝色方块移到黄色区域 | ✓ | 3 | 3 | 0.400 | 0 | 0 | 0.00 | no | - |
| curiosity_U-_R+ | curiosity | ✗ | ✓ | 把蓝色方块移到黄色区域 | ✓ | 3 | 3 | 0.400 | 0 | 0 | 0.00 | no | - |
| curiosity_U-_R- | curiosity | ✗ | ✗ | 把蓝色方块移到黄色区域 | ✓ | 3 | 3 | 0.400 | 0 | 0 | 0.00 | no | - |
| causal_explore_U+_R+ | causal_explore | ✓ | ✓ | 把蓝色方块移到黄色区域 | ✓ | 4 | 3 | 0.446 | 0 | 0 | 0.00 | no | - |
| causal_explore_U+_R- | causal_explore | ✓ | ✗ | 把蓝色方块移到黄色区域 | ✗ | 4 | 3 | 0.446 | 0 | 0 | 0.00 | no | Step failed without recovery: pick |
| causal_explore_U-_R+ | causal_explore | ✗ | ✓ | 把蓝色方块移到黄色区域 | ✓ | 4 | 3 | 0.446 | 0 | 0 | 0.00 | no | - |
| causal_explore_U-_R- | causal_explore | ✗ | ✗ | 把蓝色方块移到黄色区域 | ✓ | 4 | 3 | 0.446 | 0 | 0 | 0.00 | no | - |

## 汇总统计

| 条件 | 试验数 | 成功率 | 平均探索 | 平均不确定性 | 平均重规划 | 平均恢复 |
|------|--------|--------|----------|--------------|------------|----------|
| causal_explore_U+_R+ | 2 | 100.00% | 3.0 | 0.446 | 0.0 | 0.0 |
| causal_explore_U+_R- | 2 | 0.00% | 3.0 | 0.446 | 0.0 | 0.0 |
| causal_explore_U-_R+ | 2 | 100.00% | 3.0 | 0.446 | 0.0 | 0.0 |
| causal_explore_U-_R- | 2 | 100.00% | 3.0 | 0.446 | 0.0 | 0.0 |
| curiosity_U+_R+ | 2 | 100.00% | 3.0 | 0.400 | 0.0 | 0.0 |
| curiosity_U+_R- | 2 | 100.00% | 3.0 | 0.400 | 0.0 | 0.0 |
| curiosity_U-_R+ | 2 | 100.00% | 3.0 | 0.400 | 0.0 | 0.0 |
| curiosity_U-_R- | 2 | 50.00% | 3.0 | 0.400 | 0.0 | 0.0 |
| random_U+_R+ | 2 | 100.00% | 3.0 | 0.550 | 0.0 | 0.0 |
| random_U+_R- | 2 | 0.00% | 3.0 | 0.550 | 0.0 | 0.0 |
| random_U-_R+ | 2 | 100.00% | 0.0 | 0.550 | 0.0 | 0.0 |
| random_U-_R- | 2 | 100.00% | 0.0 | 0.550 | 0.0 | 0.0 |

## 消融分析

- **探索策略影响**: CausalExplore 最优成功率 100.0% vs Random 100.0%
- **Uncertainty 影响**: U+ 最优成功率 100.0% vs U- 100.0%
- **Recovery 影响**: R+ 最优成功率 100.0% vs R- 100.0%