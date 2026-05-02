# 2026-04-19 完整分析归档

> 2026-05-03 状态提示：本文是 2026-04-19 的完整历史分析，不再代表当前平台状态。当前项目已迁移到 MuJoCo simulator-backed v1，并通过 `embodied_agent 112 OK`、Ralph Phase1/2/3 `64/88/65 OK` 验证。

## 整体判断

这轮工作的真正进展，不是“又训了一批模型”，而是把 `projects/embodied_agent` 推进到了一个新的阶段：

- 多物体 benchmark 已经完整落地。
- failure-driven 分析链已经建立起来。
- place 的环境语义问题已经修正并验证。
- pick 和 place 在多物体里的失败机理已经被拆成两类不同问题。

换句话说，2026-04-19 当时的主线已经不再停留在“单物体 PyBullet baseline 跑通”，而是在做真正有诊断能力的多物体回归；2026-05-02 后平台状态进一步更新为 MuJoCo simulator-backed v1。

## 这轮具体做了什么

### 1. 先把 failure_history 做成可用的失败分析工具

本轮先补齐了失败分析链，而不是继续盲目看 success_rate：

- 扩展 `src/embodied_agent/failure_analysis.py`，统一 pick/place 失败分类。
- 新增 `analyze_recovery_failures.py`，可以把 evaluation run 目录里的 `failure_history` 直接汇总成表。
- 结果中可以拆出：失败发生在哪个动作、失败来源是什么、触发了哪种 recovery policy、恢复是否成功。

这一步非常关键，因为多物体 benchmark 一旦掉点，如果只看一条 success_rate 曲线，几乎看不出是 pick 崩了、place 崩了，还是 recovery 链没起作用。

### 2. 把 runtime 真正扩展到多物体 benchmark

这轮已经把多物体支持接到完整主链：

- `evaluation.py` 支持 `task_pool` 和按任务池推导 active scene entities。
- `planner.py` / `executor.py` / `skills.py` 不再把 pick/place 默认绑死到单一 `red_block / green_zone`。
- `simulator.py` 支持显式 `object_names / zone_names`、多实体 reset，以及 candidate-slot 布局采样。
- 新增 `configs/eval/batch_eval_12_multi_object_smoke.yaml` 与相关单测。

第一批验证已经说明 benchmark 是健康的：

- `scripted_multi_object_smoke = 1.00`

所以现在的 learned 掉点，不能再归咎于“多物体配置本身不稳定”。

### 3. 首轮多物体 hybrid smoke 已经给出清晰问题图谱

首轮多物体 hybrid smoke 结果：

| 实验 | 成功率 |
|------|--------|
| scripted_multi_object_smoke | 1.00 |
| learned_pick_multi_object_smoke | 0.50 |
| learned_place_multi_object_smoke | 0.50 |
| learned_pick_place_multi_object_smoke | 0.25 |

失败分析表明：

- learned pick 主要是重复 `execution_error`，恢复策略 `retry_pick_then_continue` 没有救回来。
- learned place 主要也是重复 `execution_error`，恢复策略 `repick_then_place` 没有救回来。
- full hybrid 里仍然以 pick 失败为主，place 是次要但明确存在的第二类问题。

更细的 episode-level 诊断进一步表明：

- pick 多数失败并不是 target 选错，而是 pre-grasp approach / acquisition 没形成。
- place 多数失败是 `released_outside_zone`，说明 release 发生了，但最终落点不在成功容差内。

### 4. place 的环境语义已经被正确修正

这轮对 `src/embodied_agent/rl_envs.py` 做了关键改动：

- 以前 place 只要发生任意 `release_event` 就立刻结束。
- 现在 place 会进入 `post_release_settle_steps` 窗口。
- 最终 reward、success 和 penalty 由 settled landing state 决定，而不是由松爪瞬间决定。

这一步的意义是把“稍微对准就松手”这种局部最优从训练目标里拿掉。

关键验证结果：

- `tests/test_rl_envs.py` 新增 settling 语义测试并通过。
- `place_formal_eval.yaml` 回归结果：

| 实验 | 成功率 |
|------|--------|
| place_fixed_50 | 1.00 |
| place_random_50 | 1.00 |

因此可以明确下结论：

- 这次 place 语义修正是对的。
- 它没有把当前单技能 learned place 模型打坏。
- full-task 多物体里 learned place 仍然只有 `0.50`，问题更像 runtime mismatch，而不是 reward 语义还没修好。

### 5. pick 的训练分布核对已经完成，结论是否定“简单扩覆盖就行”

这轮专门对 active pick manifest 的训练分布和多物体运行时分布做了核对。

已经确认的事实是：

- 现有 pick manifest 在单技能 formal eval 上依然很强：fixed `0.94`、random `0.92`。
- 但它训练时的 reset 覆盖本质上仍接近单布局。
- 多物体首轮 pick 尝试里，初始 `ee_object_distance`、EE 位姿和 staging 后高度分布已经明显偏出原训练流形。
- 最显著的偏移是 lateral staging pose，训练期 `ee_y` 几乎只在正值窄带，而多物体首轮样本几乎都落到负 `ee_y` 一侧。

这说明 pick 当前的多物体失败，核心不是 symbolic routing 漂了，而是 approach manifold 变了。

### 6. 已经试过两条新的 pick 路线，但都没超过旧 baseline

本轮已经实际试过两条针对多物体覆盖的新路线：

| 训练路线 | 结果 |
|----------|------|
| SAC broad-uniform coverage | `best_success_rate = 0.50`，随后塌到 `0.00` |
| BC candidate-grid v2 | `best_success_rate = 0.45`，随后过拟合回落 |

然后又把 BC pick v2 接回多物体 smoke：

| 实验 | 成功率 |
|------|--------|
| bc_pick_candidate_grid_multi_object_smoke | 0.25 |
| bc_pick_place_multi_object_smoke | 0.083 |

最后再拿旧 conservative SAC pick 做多物体 smoke 对照：

| 实验 | 成功率 |
|------|--------|
| conservative_pick_multi_object_smoke | 0.50 |

因此当前最明确的工程结论是：

- 到目前为止，没有一个新的 pick manifest 超过旧 conservative baseline。
- 只扩布局覆盖，不足以打破当前 `0.50` 上限。
- 下一步 pick 工作不能再停留在“再多训几轮同类随机化”。

## 当前这组结果说明了什么

### 1. 单技能强，不等于多物体 full chain 强

现在最容易误判的地方是：

- pick 单技能 formal eval 很强。
- place 单技能 formal eval 也很强。
- 但 full hybrid 多物体只有 `0.25`。

这说明系统现在的核心问题已经不是“有没有技能”，而是“技能训练时的起始状态和真实运行时状态是否对齐”。

### 2. place 和 pick 不能再混着改

本轮之前，pick 和 place 掉点容易被混成一个泛泛的“learned skill 不稳定”。

现在已经可以把两者拆开：

- place：主要是 runtime mismatch，单技能语义已经健康。
- pick：主要是 representation / objective / manifold mismatch，单纯扩覆盖不够。

### 3. 这轮不应该继续追求“再训一次也许会好”

SAC broad-uniform 和 BC candidate-grid v2 已经说明：

- 同类思路继续堆试验，收益正在快速下降。
- 需要换问题表述，而不是继续重复同一类 sweep。

## 下一步最该做什么

### P0：先把 learned place 的 runtime alignment 打穿

优先顺序应该是：

1. 记录真实 post-pick state。
2. 和当前 `use_staging=true` 的 place 训练起始状态做对比。
3. 增加从 post-pick state 起步的 place 评估配置。
4. 必要时新增一版针对 post-pick state 的 place 训练或微调。

原因很简单：

- place 单技能已经是 `1.00 / 1.00`。
- 所以继续在单技能固定/random formal eval 上追数没有意义。
- 现在最值钱的是把它和真实运行时状态对齐。

### P0：再做 pick 的 representation / objective 改造

建议的方向是：

1. 强化相对量表征，降低对绝对 object/zone 坐标和固定 EE 起始位姿的依赖。
2. 针对当前多物体里最差的 lateral approach 区域补更聚焦的 expert 数据。
3. 优先考虑 BC/SAC 混合或更强的 supervised 目标，而不是继续期待单一 SAC 目标自动穿越流形偏移。

原因也很明确：

- 现在旧 conservative baseline 已经是 `0.50`。
- 新路线如果不能超过这个点，就没有上线价值。

### P1：pick 和 place 都换新结果后，再回跑 full hybrid

等两个方向都换了新 manifest，再回跑：

- `batch_eval_12_multi_object_hybrid_smoke.yaml`
- recovery/failure 汇总分析

如果这一步仍然卡住，才值得再考虑 planner 层或更高层控制问题。

## 当前不建议做什么

- 当时不建议立即切 MuJoCo 或 Isaac 平台；该判断已被 2026-05-02 的 MuJoCo v1 迁移取代。
- 不建议现在把精力转到 LLM planner。
- 不建议继续做更多同类 broad randomization pick sweep。
- 不建议在低层瓶颈还没拆清前，同时动 planner、感知和仿真器。

## 最后一句话总结

当前项目最准确的状态不是“准备换平台”，而是：

- 多物体 benchmark 已经接通。
- failure-driven 分析链已经建立。
- place 的环境语义已经理顺。
- pick 的多物体瓶颈已经从“会不会学”收敛为“该怎么改 representation/objective”。

下一步最合理的路线是先修 place 的 runtime alignment，再改 pick 的 representation/objective，最后再回跑 full hybrid，而不是继续盲目扩随机化或提前切平台。
