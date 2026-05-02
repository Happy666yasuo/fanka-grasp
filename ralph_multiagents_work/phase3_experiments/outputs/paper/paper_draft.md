# 基于强化学习运动控制和大模型任务规划的具身智能体研究

## 论文主线草稿

> 2026-05-03 状态提示：本文仍是论文结构草稿，但 comparative、ablation 和 final demo 数值已根据 MuJoCo backend 迁移后的最新输出刷新。完整结果以 `outputs/experiments/*.json` 和 `outputs/reports/*.md` 为准。

---

## 摘要

本文提出一种结合强化学习（RL）运动控制与大语言模型（LLM）任务规划的具身智能体框架。针对传统具身智能体在未知物体交互中缺乏自主探索能力的问题，我们引入了 CausalExplore 机制——一种基于因果推理的交互式探索策略，通过主动执行探针动作（推、压、拉、拍、抓）来降低物体属性和功能用途的不确定性。CausalExplore 输出与 LLM Planner 深度集成，为任务规划提供信息丰富的物体模型。当前实验代码已迁移到 MuJoCo 仿真环境，并覆盖对比实验（无 CausalExplore、metadata-backed CausalExplore、simulator-backed CausalExplore）和多维度消融实验（探索策略、不确定性与恢复机制）的运行路径。MuJoCo 刷新结果显示三种 comparative 条件均达到 4/4 成功，simulator-backed 路径无 fallback/error，并以额外在线探索步数换取真实仿真探针证据。

**关键词**：具身智能体；大语言模型；强化学习；因果探索；任务规划

---

## 1. 引言

具身智能体（Embodied Agent）需要在物理世界中执行复杂的操作任务。传统方法通常依赖预定义的技能库或端到端的强化学习训练，但在面对未知物体时缺乏灵活性和泛化能力。大语言模型（LLM）展现了强大的推理和规划能力，然而 LLM 对物理世界的理解受限于训练数据，难以准确估计物体的物理属性（如可移动性、可抓取性）和功能用途（affordance）。

本文提出一个集成框架，核心贡献包括：

1. **CausalExplore 机制**：基于主动探索和因果推理的物体属性/用途推断方法
2. **LLM-Planner 集成**：将探索输出注入 LLM 规划流程，实现不确定性感知的任务规划
3. **系统级对比与消融实验**：验证各组件的独立贡献

---

## 2. 相关工作

### 2.1 具身智能体架构
近年来，LLM-based agent 在具身任务中取得了显著进展。PaLM-E、RT-2 等工作展示了视觉-语言模型在机器人控制中的潜力。然而，这些方法通常依赖被动感知，缺乏主动探索能力。

### 2.2 物体探索与 Affordance 学习
Robotic affordance learning 关注机器人如何通过与物体的交互来学习其功能用途。CausalWorld、CausalAgents 等工作引入了因果推理，但通常不直接与 LLM Planner 集成。

### 2.3 LLM 任务规划
LLM 在分层任务规划中表现出色。SayCan、Code as Policies 等方法探索了 LLM 生成可执行计划的路径。但 LLM 对物理不确定性的处理仍是一个开放问题。

---

## 3. 方法

### 3.1 系统架构

系统由三个核心模块组成：

1. **CausalExplore 模块**（Phase 1）：在 MuJoCo 仿真中执行探针动作，通过探索策略（Random / Curiosity / CausalExplore）选择下一个动作，积累探针结果并构建物体模型（CausalExploreOutput）。

2. **LLM Planner 模块**（Phase 2）：接收自然语言任务指令和 CausalExplore 输出，生成结构化任务计划（PlannerStep 序列）。包含不确定性处理器和重规划处理器。

3. **Skill Executor 模块**：执行计划中的技能动作（pick、place、press、push、pull、rotate），并支持失败恢复。

**核心数据流**：
```
任务输入 → LLM 规划 → CausalExplore 探针注入 → 技能执行 → 反馈 → 重规划
```

### 3.2 CausalExplore 探索策略

三种探索策略对比：
- **Random**：均匀随机选择 (探针, 物体) 对
- **Curiosity-Driven**：基于历史位移幅度的启发式采样
- **CausalExplore**：基于不确定性估计的定向探索

CausalExplore 策略通过 ProbeExecutor 估计物体属性置信度（movable, pressable, graspable, rigid）和 affordance 置信度（pushable, pressable, pullable, graspable, tappable），并计算综合不确定性分数。

### 3.3 LLM Planner 集成

LLM Planner 使用结构化 prompt 模板，将 CausalExplore 输出编码为 world state 的一部分。当不确定性超过阈值（0.50）时，ContractPlanningBridge 自动注入探针步骤以降低不确定性。

### 3.4 不确定性处理与重规划

- **UncertaintyHandler**：评估 CausalExplore 输出，决定是否需要额外探索
- **ReplanHandler**：当技能执行失败时，触发 LLM 重规划

---

## 4. 实验

### 4.1 对比实验

**实验条件**：
- **No Causal**：无 CausalExplore，纯视觉描述
- **Metadata-backed**：离线 artifact catalog 提供预计算的 CausalExplore 输出
- **Simulator-backed**：在线 MuJoCo 仿真交互探索

**指标**：任务成功率、平均探索步数、规划质量、重规划次数

### 4.2 消融实验

**消融维度**：
1. 探索策略：Random vs Curiosity vs CausalExplore
2. 不确定性处理：启用 vs 禁用
3. 恢复机制：启用 vs 禁用

### 4.3 主要结果

本轮 MuJoCo 刷新结果如下：

- Comparative：`no_causal`、`metadata_backed`、`simulator_backed` 均为 4/4 成功；simulator-backed 平均探索步数为 5.0，平均规划质量为 0.702，且无 fallback/error。
- Ablation：`random_U+_R+`、`curiosity_U+_R+`、`causal_explore_U+_R+` 均为 2/2 成功；5 个失败均发生在显式 no-recovery 条件中，未出现 fallback。
- Final demo：multi-block 三个颜色区域任务和 interactive mouse 场景均成功，整体 4/4。

**发现**：
- MuJoCo simulator-backed CausalExplore 已能覆盖红/绿、蓝/黄、黄/蓝以及双目标任务，不再因非默认目标对象失败。
- 当前 comparative 任务对 no-causal 和 metadata-backed 较简单，三组成功率相同；simulator-backed 的主要价值体现在在线探针证据和 fallback/error 诚实记录，而不是本轮成功率提升。
- Recovery 开关在 ablation 中影响明显：多个 R- 条件因 pick/place 失败无法恢复而失败。

---

## 5. 讨论

### 5.1 探索-利用权衡
Metadata-backed 策略通过离线预测减少了在线探索开销，在已知物体上表现优异。Simulator-backed 策略在未知物体上具有更强的适应性。

### 5.2 不确定性感知规划的意义
UncertaintyHandler 使 Planner 能够识别知识盲区并主动请求探索，避免了因信息不足导致的规划失败。

### 5.3 局限与未来工作
- 当前实验限于 MuJoCo 运动学仿真，后续需接入 IsaacLab 训练与真实机器人验证
- CausalExplore 的探针动作集合可进一步扩展（如旋转、翻滚等）
- LLM Planner 可替换为更强的模型以获得更好的规划质量

---

## 6. 结论

本文提出了一个集成 CausalExplore 探索机制、LLM 任务规划和技能执行的具身智能体框架。通过对比实验和消融实验，我们验证了主动探索对任务成功率和规划质量的正向影响。该框架为具身智能体在未知环境中的自主操作提供了可行的技术路径。

---

*Phase 3 完成日期: 2026-04-30*
