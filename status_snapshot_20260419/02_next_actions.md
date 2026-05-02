# 建议下一步

> 2026-05-02 状态提示：本文是 2026-04-19/20 的历史建议，当时“不建议切 MuJoCo”是基于低层技能诊断阶段的判断。当前项目已完成 MuJoCo simulator-backed v1 迁移；下一步应转向 IsaacLab adapter、MuJoCo robot dynamics/controller 和 Phase3 正式实验刷新。

## 总体建议

当前不建议切平台，也不建议继续重复“只扩布局随机化”的同类训练。最划算的做法是先把多物体 benchmark 里已经暴露出来的两个核心问题分别打穿：

- learned place 的 runtime alignment
- learned pick 的 representation / objective

## 近期优先级

### P0：先修 learned place 的 runtime mismatch

1. 抽样真实 `post-pick` 状态，和当前 `use_staging=true` 的 place 训练起始状态做分布对比。
2. 增加一组从真实 `post-pick` 状态起步的 place 评估配置，确认 learned place 为什么单技能强、全链路弱。
3. 如果分布差异明显，就补一版新的 place 训练或微调：从 post-pick state 或更贴近多物体 runtime 的 staged state 起步。
4. 验收标准不是单技能继续维持 `1.00`，而是把 `learned_place_multi_object_smoke` 从 `0.50` 往上推，并显著降低 place 主导的 `execution_error`。

### P0：再做 pick 的 representation / objective 改造

1. 不再继续只做 broad-uniform randomization，因为这条路线已经证明没有突破当前上限。
2. 优先尝试更强的相对量表征、对 lateral staging 偏移更鲁棒的输入设计，以及更聚焦的 expert 数据覆盖。
3. 训练目标上优先考虑更强的 supervised / BC-SAC 混合路线，而不是继续期望仅靠原有 SAC 目标自己跨过流形偏移。
4. 验收标准是先超过当前 `0.50` 的多物体 smoke 上限，再看 full hybrid 是否同步改善。

### P1：等 pick/place 都换新结果后，再回跑 full hybrid

1. 用新的 pick manifest 和新的 place manifest 回跑 `batch_eval_12_multi_object_hybrid_smoke.yaml`。
2. 同时跑 recovery/failure 汇总，确认 full hybrid 的失败主因是否真的发生迁移。
3. 这一步才算真正检验“这一轮多物体 benchmark 收敛没有”。

### P1：同步答辩材料和项目文档

1. 把当前关键结果整理成一页对比表：single-skill formal vs multi-object smoke。
2. 强调这轮最大的工程成果是 benchmark 和 failure-driven 诊断链，而不是单纯多训了几轮模型。
3. 用当时结论解释为什么下一步要先做 runtime alignment 和 representation/objective，而不是直接切 MuJoCo 或接 LLM planner。

## 当前不建议优先做的事

- 当时不建议立即切 MuJoCo 或 Isaac 平台；该判断已被 2026-05-02 的 MuJoCo v1 迁移取代。
- 不建议现在把主要精力转去高层 LLM planner。
- 不建议继续做更多“同一种随机化策略”的 pick sweep。
- 不建议在 pick 和 place 这两个低层瓶颈还没拆清前，同时换规划器、换感知、换仿真器。

## 一个务实的执行顺序

### 第一阶段

- 先做 place 的 runtime 对齐诊断和回归配置。
- 确认 learned place 在真实 post-pick state 下的失效模式。

### 第二阶段

- 再做 pick 的 observation / objective 改造。
- 优先追求多物体 smoke 从 `0.50` 继续往上走，而不是追求训练曲线更好看。

### 第三阶段

- pick 与 place 都换新 manifest 后，再回跑 full hybrid 和 failure analysis。
- 如果 full hybrid 仍卡住，再考虑是否需要动 planner 或更高层策略。

## 当前决策建议

最合适的短期决策是：

- `先修 place 的 runtime mismatch`
- `再改 pick 的 representation / objective`
- `最后再回跑 full hybrid`
