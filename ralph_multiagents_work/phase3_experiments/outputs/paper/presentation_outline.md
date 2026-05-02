# 答辩 PPT 大纲

## 基于强化学习运动控制和大模型任务规划的具身智能体研究

---

## Slide 1: 标题页
- 论文题目
- 作者 / 导师
- 日期

## Slide 2: 研究背景与动机
- 具身智能体的核心挑战：未知物体交互
- LLM 的局限性：缺乏物理世界经验
- 研究问题：如何让智能体主动探索并利用探索结果进行规划？

## Slide 3: 系统架构概览
- 三大核心模块图：
  ```
  ┌─────────────┐     ┌──────────────┐     ┌──────────────┐
  │ CausalExplore│ ──▶ │  LLM Planner  │ ──▶ │Skill Executor│
  │   (Phase 1)  │     │   (Phase 2)   │     │  (Core)      │
  └─────────────┘     └──────────────┘     └──────────────┘
         │                     │                     │
         └─────────────────────┴─────────────────────┘
                       反馈 / 重规划
  ```

## Slide 4: CausalExplore 机制详解
- 探针动作集：lateral_push, top_press, side_pull, surface_tap, grasp_attempt
- 三种探索策略：Random / Curiosity-Driven / CausalExplore
- 输出：PropertyBelief + AffordanceCandidate + Uncertainty Score

## Slide 5: LLM Planner 集成
- 结构化 Prompt 模板
- CausalExplore 输出注入
- UncertaintyHandler + ReplanHandler
- ContractPlanningBridge 探针自动注入

## Slide 6: 对比实验设计
- 三种条件：
  1. No CausalExplore（基线）
  2. Metadata-backed CausalExplore（离线）
  3. Simulator-backed CausalExplore（在线）
- 评估指标：成功率、探索步数、规划质量、重规划次数

## Slide 7: 对比实验结果
- 成功率对比柱状图
- 探索步数分布图
- 关键发现：在线探索显著提升成功率

## Slide 8: 消融实验设计
- 三维消融矩阵：
  - 探索策略（3 种）
  - Uncertainty 处理（开/关）
  - Recovery 机制（开/关）
- 共 12 种条件组合

## Slide 9: 消融实验结果
- 各维度独立贡献分析
- 交互效应分析
- 关键发现：三个组件均正向贡献，Recovery × CausalExplore 存在正向交互

## Slide 10: 最终系统 Demo
- 场景 1: 多方块放置（red→green, blue→yellow, yellow→blue）
- 场景 2: 鼠标用途推断（按压 + 拖动 + 滚轮）
- 完整流程展示

## Slide 11: 创新点总结
1. CausalExplore：基于因果推理的主动探索
2. 深度集成：探索-规划-执行闭环
3. 系统验证：对比 + 消融双实验维度
4. 可复现：Mock + Simulator 双模式

## Slide 12: 局限与展望
- 当前局限：限于仿真、探针集合有限
- 未来方向：真实机器人迁移、探针动作扩展、更强的 LLM 模型

## Slide 13: 致谢
- 感谢导师
- 感谢实验室
- Q&A

---

*建议每页 2-3 分钟，总时长约 30-40 分钟*
