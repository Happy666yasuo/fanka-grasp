# 当前状态

> 2026-05-03 状态提示：本文是 2026-04-19/20 的历史状态快照。当前项目已新增 MuJoCo 默认 simulator backend，Ralph Phase1/2/3 分别为 64/88/65 tests OK，`embodied_agent` 为 112 tests OK。最新状态请以 `../开发日志_20260503.md` 和 `../NewPlan.md` 为准。

## 一句话判断

当前项目已经从“单物体 baseline 跑通”推进到“多物体 benchmark 已接通、失败模式可拆分、place 单技能语义已修正、pick 多物体瓶颈已定位”的阶段。

## 这轮已经完成的关键工作

- 已完成多物体 benchmark 接线：`evaluation -> planner -> executor -> skills -> simulator` 现在都支持 `task_pool`、显式 `object_name/zone_name` 和多实体场景。
- 已完成多物体布局采样改造：candidate slots 取代 rejection sampling，3 个物体和 3 个区域可稳定采样。
- 已完成 failure_history 汇总链路：`failure_analysis.py` 与 `analyze_recovery_failures.py` 能把 recovery、失败来源、失败动作和恢复成功率整理成表。
- 已完成 place release-settling 语义修正：place 不再在任意 `release_event` 上立刻终止，而是等待几步 settling 后按最终落点评分。
- 已完成对应回归测试：`tests/test_rl_envs.py` 已覆盖 settling 行为；相关 simulator/layout 测试也已补齐。
- 已完成 pick 训练覆盖核对：已经确认多物体 pick 失败不是 target routing 漂移，而是运行时起始几何偏出原训练流形。
- 已完成两条新 pick 训练尝试：SAC broad-uniform coverage 和 BC candidate-grid v2 都已经训练并回测。

## 已验证的关键结果

### 单技能结果

- pick formal eval 仍强：fixed `0.94`，random `0.92`。
- place formal eval 在新 settling 语义下仍强：fixed `1.00`，random `1.00`。

### 多物体 smoke 结果

- scripted_multi_object_smoke：`1.00`
- learned_pick_multi_object_smoke：`0.50`
- learned_place_multi_object_smoke：`0.50`
- learned_pick_place_multi_object_smoke：`0.25`
- bc_pick_candidate_grid_multi_object_smoke：`0.25`
- bc_pick_place_multi_object_smoke：`0.083`
- conservative_pick_multi_object_smoke：`0.50`

这组结果说明两件事：

- benchmark 本身是好的，因为 scripted 在多物体里仍然稳定。
- learned pick 和 learned place 在多物体里都掉点，但掉点来源不同，不能混成一个问题看。

## 当前已经确认的诊断

### pick

- 当前主力 pick manifest 在单技能 formal eval 上依然强，但一进入多物体 runtime 就掉到 `0.50`。
- 失败主要卡在 pre-grasp approach / acquisition，而不是 symbolic target 选错。
- 新一轮 coverage 扩展没有超过旧 conservative baseline，说明问题不只是“训练空间不够大”，而是 representation / objective / runtime manifold mismatch 更关键。

### place

- place 的环境语义问题已经修正，但 full-task 多物体里仍只有 `0.50`。
- 这说明 place 当前更像是“训练时 staging-to-zone 假设”和真实 post-pick 状态不一致”，而不是 release 语义本身错误。

## 当前最大的短板

- 还没有一个比旧 conservative SAC pick 更好的 ready-made 多物体 pick manifest。
- learned place 单技能已经过关，但还没有完成和真实 post-pick runtime 的状态对齐。
- learned full hybrid 在多物体里还没有形成稳定闭环，当前最好结果仍只有 `0.25`。

## 目前最真实的阶段判断

更准确的说法不是“系统已经完成”，而是：

- 单技能 pick/place 都有了可信的强基线。
- 多物体 benchmark 和 failure-driven 诊断链已经建立起来。
- 当前主战场已经从“能不能跑”变成“为什么在多物体里掉点，以及该先修 place 还是先改 pick”。
