# Embodied Agent Mainline Plan

更新时间: 2026-04-20

## 目标

当前主线固定为 `my_new_practice/大创/projects/embodied_agent`。

- 不将 `my_practice/franka_cube_grasp` 的代码并入当前主线。
- `my_practice/franka_cube_grasp` 仅保留为后续 IsaacLab / MuJoCo / Sim2Sim 技术路线参考。
- 当前第一优先级不再是泛泛的“把 RL pick 收回主线”，而是先解决多物体 benchmark 里已经定位清楚的两个低层瓶颈：place 的 runtime alignment 和 pick 的 representation / objective。
- 在这两个问题没有打穿前，不切换仿真平台，也不进入高层 LLM planner 主线。

## 当前判断

### 已经可靠的部分

- `scripted_multi_object_smoke = 1.00`，说明多物体 benchmark 本身可靠。
- 当前主力 pick manifest 的单技能 formal eval 仍然很强：fixed `0.94`、random `0.92`。
- 当前 learned place 在新 settling 语义下的单技能 formal eval 仍然很强：fixed `1.00`、random `1.00`。

### 当前主线问题

- learned place 在单技能里强，但在多物体 full-task runtime 里只有 `0.50`，说明当前主要问题是 post-pick runtime state 和训练时 staged state 不一致。
- learned pick 在多物体里主要卡在 pre-grasp approach / acquisition，coverage 扩展型新训练没有超过旧 conservative baseline 的 `0.50`。
- 当前没有一个比旧 conservative SAC pick 更好的 ready-made 多物体 pick manifest。

## 执行顺序

### Phase A: Place Runtime Alignment

1. 在运行时记录真实 post-pick state，和当前 place 训练使用的 staged state 做对比。
2. 新增一组 place 评估配置，从真实 post-pick state 起步，而不是默认理想 staging pose。
3. 如果分布差异明显，就新增一版 place 训练或微调配置，让 place 从更接近真实 runtime 的状态起步。
4. 先把 `learned_place_multi_object_smoke` 从 `0.50` 往上推，再讨论 full hybrid。

### Phase B: Pick Representation / Objective Redesign

1. 不再继续做 broad-uniform randomization 的同类 sweep。
2. 优先改 pick observation 和训练目标，让策略对 lateral staging 偏移更鲁棒。
3. 补更聚焦的 expert 数据或 BC/SAC 混合路线，专门覆盖当前多物体里最易失败的 approach 区域。
4. 只有当新的 pick manifest 超过当前 `0.50` baseline 后，才进入下一轮 full hybrid。

### Phase C: Full Hybrid Re-validation

1. 用新的 pick manifest 和新的 place manifest 回跑 `batch_eval_12_multi_object_hybrid_smoke.yaml`。
2. 同步跑 recovery/failure 汇总，确认 full hybrid 的主导失败源是否发生迁移。
3. 若 full hybrid 仍然卡住，再考虑 planner 层或更高层结构化策略。

## 优先改动点

### Place 方向

- `src/embodied_agent/rl_envs.py`
- `src/embodied_agent/pick_evaluation.py`
- `src/embodied_agent/simulator.py`
- `configs/eval/place_formal_eval.yaml`
- `configs/skills/place_*`

### Pick 方向

- `src/embodied_agent/rl_envs.py`
- `src/embodied_agent/simulator.py`
- `src/embodied_agent/bc_training.py`
- `src/embodied_agent/training.py`
- `configs/skills/pick_*`

## 验收标准

### 第一阶段

- learned place 在真实 post-pick runtime 下的失效模式被量化清楚。
- 新的 place 配置能把 `learned_place_multi_object_smoke` 稳定推高到当前 `0.50` 以上。

### 第二阶段

- 新的 pick 路线必须超过当前 `0.50` 的多物体 smoke 上限。
- recovery/failure 表中 pick 主导的 `execution_error` 要明显下降，而不是只看训练曲线更平滑。

### 第三阶段

- full hybrid 多物体 smoke 明显高于当前 `0.25`。
- failure analysis 里不再由 pick 或 place 单边完全主导失败。

## 备注

- 当前阶段不切换仿真平台。
- 当前阶段不引入 LLM planner。
- 当前阶段不把 fallback 结果当成纯 learned full chain 结果。
