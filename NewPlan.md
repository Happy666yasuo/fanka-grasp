# NewPlan：具身智能体项目总计划

> 目标读者：三人项目组后续开发、实验、论文和答辩协作。
> 定位：本文件是后续主工作蓝图，不替代 `EMBODIED_AGENT_TEAM_DIVISION.md`，而是在当前代码状态、CausalExplore 状态、阶段路线和验收口径之间建立统一执行计划。

## 1. 总体定位

本项目不把“大模型规划”“因果认知”“强化学习控制”拆成三个互不相干的小题，而是构建一个可解释、可执行、可验证的具身智能体闭环：

```text
自然语言任务
  -> 大脑：结构化 LLM Planner
  -> 认知桥梁：CausalExplore 物体属性/用途推断
  -> 小脑：RL / scripted 技能执行器
  -> 执行反馈、失败记录、重试或重规划
```

核心路线是：

1. 已完成早期 PyBullet 桌面操作 baseline 和多物体 benchmark 的基础稳定。
2. 已将当前 simulator-backed 主路径迁移到 MuJoCo，并保留 PyBullet 作为 legacy backend。
3. 下一阶段接入 IsaacLab 技能训练环境，让 IsaacLab 负责训练，MuJoCo 负责 sim2sim/sim2real 验证。
4. 结构化 LLM planner 保持只输出技能级步骤，让系统在“信息不足时先探索、技能失败时可重规划”的条件下完成多对象、多任务演示和实验。

## 2. 当前状态基线

### 2.1 `embodied_agent`

当前已有 MuJoCo 默认 simulator backend 和 legacy PyBullet backend，主链路为：

```text
evaluation -> planner -> executor -> skills -> simulator
```

已经具备：

- 多物体 `task_pool`、显式 `object_name/zone_name` 和多实体场景。
- `create_pick_place_simulation(backend="mujoco")` 默认创建 MuJoCo kinematic pick/place scene。
- `create_pick_place_simulation(backend=None)` 会读取 `EMBODIED_SIM_BACKEND`，显式 `backend=` 参数优先级更高。
- `PickPlaceSimulationProtocol` 已定义，便于后续 IsaacLab adapter 复用同一 observation/action contract。
- `backend="isaaclab"` 已注册为占位 backend；当前会明确报错，不代表 IsaacLab 训练已接入。
- scripted baseline、learned pick、learned place、full hybrid 的批量评估。
- executor post-condition 检查、`failure_history`、`replan_count` 和 recovery telemetry。
- 单技能 formal eval 与链路 staged regression 的基本验收口径。

历史状态快照显示，2026-04-19 时 learned pick/place 在多物体 runtime 下仍明显掉点：

| 项目 | 结果 |
| --- | --- |
| scripted multi-object smoke | `1.00` |
| learned pick multi-object smoke | `0.50` |
| learned place multi-object smoke | `0.50` |
| learned pick-place multi-object smoke | `0.25` |

但后续 checkpoint 已经推进出更强的 staged chain baseline。当前应以 `embodied_agent/CHECKPOINTS.md` 和 `embodied_agent/EVAL_STANDARD.md` 中的最新记录为准：

| 验证项 | 当前接受基线 |
| --- | --- |
| 20-episode staged chain | `scripted_baseline_20 = 1.00`, `learned_pick_hybrid_20 = 1.00`, `learned_place_hybrid_20 = 0.95`, `learned_pick_place_full_20 = 1.00` |
| 50-episode staged chain | `scripted_baseline_50 = 1.00`, `learned_pick_hybrid_50 = 1.00`, `learned_place_hybrid_50 = 0.96`, `learned_pick_place_full_50 = 1.00` |
| recovery confirmation | 50-episode run 中 `episodes_with_replan = 2`, `successful_recovery_episodes = 2`, `recovery_success_rate = 1.00` |
| task-pool runtime-aligned place | staged formal `0.48`, raw formal `0.08` |

因此，后续判断要区分两个层次：

- staged chain 已经足够作为当前执行主线基线。
- raw task-pool place 仍是扩展到真实 post-pick 状态、多任务泛化和后续认知闭环时的主要风险点。

### 2.2 CausalExplore / Ralph pipeline

当前已有论文可信实验脚手架、metadata-backed smoke comparison、artifact catalog，以及 Ralph Phase1/3 的 MuJoCo simulator-backed 在线 probe 回归。

当前已完成：

- Ralph Phase1：技能库和 CausalExplore probe 默认在 MuJoCo 中执行，64 tests OK。
- Ralph Phase2：结构化 LLM Planner、uncertainty/replan handler 和中文颜色任务解析，88 tests OK。
- Ralph Phase3：comparative/ablation/final demo，65 tests OK。
- simulator-backed 实验结果中保留 `fallback_used`/`error`，失败不得伪装为成功。

当前短板是：

- MuJoCo backend 仍是 kinematic end-effector，不是完整 Franka/Panda dynamics。
- Phase3 静态输出产物已在 2026-05-03 重新运行 comparative、ablation、final demo 并刷新报告；图表生成器仍存在 comparative/ablation 同名图片覆盖问题，后续应拆分输出文件名。
- IsaacLab 训练环境尚未接入 `PickPlaceSimulationProtocol`。

MuJoCo v1 与目标 sim2real backend 的差距：

- 还没有完整 Franka/Panda MJCF robot asset。
- 还没有与 IsaacLab 对齐的 IK/controller/action space。
- contact、friction、gripper dynamics 仍是简化运动学近似。
- 还没有 camera/vision observation 与真实感知闭环。
- 还没有 MuJoCo/IsaacLab 之间的同任务 sim2sim parity benchmark。

### 2.3 总体判断

短期不应把重点放在“大模型自由输出连续动作”或“一次性切换全平台”。更稳妥的路线是：

- 小脑层先保持已验证的 staged chain 能力，并继续解决 raw task-pool place、技能边界和新技能扩展。
- CausalExplore 已完成 MuJoCo simulator-backed v1，下一步是补更真实的对象/机器人模型和正式实验刷新。
- LLM planner 只做任务分解、技能选择和失败恢复决策，不直接输出关节控制或连续轨迹。

## 3. 三条主线

### 3.1 小脑：技能执行层

负责人主线：强化学习运动控制与仿真环境。

短期目标：

- 保持当前 pick/place staged chain 基线不回退。
- 继续推进 task-pool runtime-aligned place，重点提升 raw reset / post-pick 状态下的成功率。
- 将 recovery telemetry 固化为每次链路评估的默认输出。

中期目标：

- 在 pick/place 之外扩展 `press`、`push`、`pull`、`rotate` 等技能。
- 每个技能必须有独立输入参数、成功条件、失败码、formal eval 和最小 smoke。
- 技能接口保持统一，便于 planner 和 CausalExplore 共同调用。

小脑不负责：

- 物体用途推断。
- 高层任务分解。
- 让 LLM 直接替代技能策略输出连续动作。

### 3.2 认知桥梁：CausalExplore

负责人主线：因果认知、物体理解和用途推断。

短期目标：

- 保持 MuJoCo simulator-backed execution 不回退。
- 在同一 object manifest 上继续对比 `Random Explore`、`Curiosity-Driven` 和 `CausalExplore`。
- 输出可被 planner 消费的属性、用途、置信度和推荐探索动作。
- 刷新 Phase3 正式实验 JSON、图表和论文材料，使数值与 MuJoCo 口径一致。

中期目标：

- 对可按压、可推动、可拉动、可旋转、需抓握后移动等 affordance 建立稳定表达。
- 支持不确定性驱动的主动探索：当用途不确定时，先推荐 probe，而不是让 planner 盲目调用技能。
- 将探索轨迹、接触状态、推断结果和最终执行效果关联到实验 artifact。

CausalExplore 不负责：

- 直接完成全局任务规划。
- 直接输出底层连续控制。
- 替代 executor 判定技能成功或失败。

### 3.3 大脑：结构化 LLM Planner

负责人主线：高层规划、系统集成、文档和论文材料。

短期目标：

- 从当前 rule-based planner 过渡到结构化 planner 接口。
- LLM 输出只允许是任务步骤、技能选择、目标对象、前置条件和 fallback，不允许直接输出连续控制。
- 当 CausalExplore 的 `uncertainty_score` 高或缺少 affordance 信息时，planner 必须先触发探索。

中期目标：

- 支持技能失败后的 bounded retry / replan。
- 使用 executor 的 `failure_history` 做重规划输入，而不是只根据自然语言重新猜。
- 形成可展示的自然语言任务闭环：任务输入 -> 探索 -> 推断 -> 规划 -> 执行 -> 反馈。

## 4. 统一接口

### 4.1 CausalExplore 输出

```json
{
  "object_id": "red_block",
  "object_category": "rigid_block",
  "property_belief": {
    "mass": {"label": "light", "confidence": 0.82},
    "friction": {"label": "medium", "confidence": 0.71},
    "joint_type": {"label": "none", "confidence": 0.93}
  },
  "affordance_candidates": [
    {"name": "graspable", "confidence": 0.91},
    {"name": "pushable", "confidence": 0.76}
  ],
  "uncertainty_score": 0.24,
  "recommended_probe": "lateral_push",
  "contact_region": "side_center",
  "skill_constraints": {
    "preferred_skill": "pick",
    "max_force": 12.0,
    "approach_axis": "top_down"
  }
}
```

### 4.2 Planner 输出

```json
{
  "task_id": "move_red_block_to_green_zone",
  "step_index": 1,
  "selected_skill": "pick",
  "target_object": "red_block",
  "skill_args": {
    "target_zone": "green_zone",
    "grasp_region": "top_center"
  },
  "preconditions": [
    "object_visible",
    "affordance.graspable.confidence >= 0.70"
  ],
  "expected_effect": "holding(red_block)",
  "fallback_action": "probe_or_replan"
}
```

### 4.3 Executor 返回

```json
{
  "success": true,
  "reward": 1.0,
  "final_state": {
    "holding": true,
    "object_pose": [0.31, -0.12, 0.76]
  },
  "contact_state": {
    "has_contact": true,
    "contact_region": "top_center"
  },
  "error_code": null,
  "rollout_path": "outputs/evaluations/<run_id>/<episode_id>.jsonl",
  "failure_history": []
}
```

### 4.4 接口约束

- 所有模块必须使用结构化字段传递状态，不依赖自然语言字符串解析作为唯一接口。
- `object_id`、`target_object`、`object_name` 必须可追踪到同一个 scene manifest。
- executor 的 `failure_history` 是重规划和论文 failure analysis 的共同事实来源。
- CausalExplore 的 `uncertainty_score` 必须影响 planner 行为：高不确定性先探索，低不确定性才执行。

## 5. 阶段安排

### P0：冻结当前 baseline 和接口文档

目标：

- 以当前 `embodied_agent` checkpoint 为基线，明确 staged chain、task-pool place、recovery telemetry 的最新状态。
- 新增或更新接口文档，固定 CausalExplore、Planner、Executor 三方字段。
- 保留 2026-04-19 的历史瓶颈说明，但不再把它当作最新结果。

验收：

- `NewPlan.md`、`EMBODIED_AGENT_TEAM_DIVISION.md`、`embodied_agent/EVAL_STANDARD.md` 之间没有关键矛盾。
- 每个模块负责人知道自己下一步指标是什么。
- 不改动旧 `my_practice`，新计划只写入 `my_new_practice`。

### P1：稳住并扩展小脑技能

目标：

- staged chain 继续维持当前接受基线。
- raw task-pool place 从当前低基线继续提升，重点看真实 post-pick 或 raw reset 下的 transport 与 release 失败。
- 准备 `press`、`push`、`pull`、`rotate` 的最小技能接口和 smoke 任务。

验收：

- pick/place formal eval 不回退。
- staged chain 20/50 episode regression 不低于当前接受基线。
- raw task-pool place 的 failure slicing 显示主要错误类型发生可解释变化。
- 每个新增技能至少能被独立 smoke 调用，并返回标准 executor 字段。

### P2：CausalExplore simulator-backed 闭环

目标：

- 已将 CausalExplore 接入 MuJoCo v1 可执行交互环境。
- 在统一 object manifest 上运行 `Random Explore`、`Curiosity-Driven`、`CausalExplore`。
- 输出属性推断、用途推断、不确定性、推荐 probe 和 contact region。

验收：

- 每条探索轨迹都能关联到 object manifest、probe action、接触状态和推断输出。
- 指标至少包括属性推断准确率、用途准确率、平均探索步数和 artifact 路径。
- CausalExplore 输出可以被 planner mock 消费，不需要人工改字段。
- 非默认目标对象如 `blue_block -> yellow_zone`、`yellow_block -> blue_zone` 不得因 scene 初始化缺失而 fallback。

### P3：结构化 LLM Planner 与系统闭环

目标：

- 已接入结构化 planner/mock LLM，不让 LLM 直接输出连续动作。
- 当对象用途不确定时，planner 触发 CausalExplore probe。
- 当 skill post-condition 失败时，executor 记录 `failure_history`，planner 根据失败类型 retry 或 replan。
- 中文颜色任务解析必须保持对象和目标区域分离。

验收：

- 输入自然语言任务后，planner 输出结构化步骤。
- `uncertainty_score` 高时优先探索，而不是直接执行。
- `failure_history` 非空时，planner 能生成有界 retry/replan。
- 最小 demo 完成红/蓝/黄方块多目标放置，以及一个可交互对象用途推断案例。
- Phase1/2/3 tests = 64/88/65 OK。

### P4：论文实验、消融和答辩材料

目标：

- 证明 CausalExplore 信息能改善规划质量或执行成功率。
- 证明 RL / scripted 技能层能稳定承担低层动作执行，不需要 LLM 输出连续控制。
- 形成论文实验表、系统图、演示视频和答辩材料。

验收：

- 对比实验包括无 CausalExplore、metadata-backed CausalExplore、simulator-backed CausalExplore。
- 消融实验至少覆盖探索策略、planner 是否使用 uncertainty、executor 是否启用 recovery。
- 所有主实验有可复现 config、artifact catalog 和 evaluation summary。
- 正式提交前如需最终论文数值，应再次运行 Phase3 实验；2026-05-03 已完成一轮 MuJoCo 口径刷新。

## 6. 验证口径

### 6.1 小脑验证

- 单技能 pick/place formal eval 使用 `embodied_agent/EVAL_STANDARD.md` 中的 canonical config。
- 链路 regression 不使用 5-episode smoke 作为主指标，至少保留 20-episode regression 和 50-episode confirmation。
- 多物体任务必须记录 success rate、failure source、failure action、recovery policy 和 episode-level `failure_history`。
- 如果后续转入 IsaacSim/IsaacLab，所有训练和测试命令必须使用 `--headless`；smoke `num_envs <= 4`，短训练验证 `num_envs <= 16`。
- MuJoCo backend 当前是默认测试后端；PyBullet 只作为 legacy fallback，不作为新实验主口径。

### 6.2 CausalExplore 验证

- `Random Explore`、`Curiosity-Driven`、`CausalExplore` 必须在同一 object manifest 上比较。
- metadata-backed smoke 只能作为早期 sanity check，不能作为最终闭环证据。
- simulator-backed execution 后至少记录：
  - 属性推断准确率。
  - 用途推断准确率。
  - 平均探索步数。
  - 不确定性下降曲线。
  - artifact 路径。

### 6.3 系统验证

- planner 输出必须是结构化 JSON 或等价 schema。
- CausalExplore 高不确定性必须触发 probe。
- executor post-condition 失败必须进入 `failure_history`。
- retry/replan 必须有上限，避免无限循环。
- 最小 demo 必须能从自然语言任务走完整链路。

## 7. 三人协作分工

| 角色 | 主责 | 近期交付 |
| --- | --- | --- |
| 成员 A | 小脑技能执行、仿真、评测 | 保持 staged chain 基线，推进 raw task-pool place，准备新增技能 smoke |
| 成员 B | CausalExplore、对象属性和用途推断 | simulator-backed 探索闭环，输出统一 schema，完成三种探索策略对比 |
| 成员 C | LLM planner、系统集成、论文答辩 | 结构化 planner、失败重规划、系统 demo、实验表和答辩材料 |

协作规则：

- 成员 A 的 executor 字段是系统反馈事实来源。
- 成员 B 的 CausalExplore schema 是 planner 的环境理解输入。
- 成员 C 不直接绕过 A/B 的接口，不让 LLM 直接控制连续动作。
- 每个阶段结束都要留下 config、artifact、summary 和一句话结论。

## 8. 当前优先级

1. 把本文件作为三人团队主计划入口。
2. 将 `embodied_agent` 最新 checkpoint 和 eval standard 作为小脑当前基线，不再用旧 smoke 结果做最新判断。
3. 为 CausalExplore simulator-backed 闭环补第一版接口适配计划。
4. 在 planner 层先做结构化输出和 mock integration，等 CausalExplore 有真实在线输出后再接入完整闭环。
5. 最终 demo 聚焦两类能力：多目标方块放置，以及一个可交互对象的用途推断和技能选择。
