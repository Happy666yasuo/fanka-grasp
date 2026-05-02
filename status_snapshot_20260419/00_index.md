# 项目状态文档包

更新时间：2026-04-20

> 2026-05-02 状态提示：本目录是 2026-04-19 到 2026-04-20 的历史快照，不再代表当前项目进度。当前 simulator-backed 主路径已迁移到 MuJoCo；最新进度以 `../NewPlan.md`、`../开发日志_20260502.md`、`../embodied_agent/CHECKPOINTS.md` 和 `../ralph_multiagents_work/README.md` 为准。

这个目录反映的是 2026-04-19 到 2026-04-20 这一轮“多物体 benchmark + 失败分析 + place 语义回归 + pick 重训”的历史状态，不再是当前最新项目结论。

当时结论：

- 多物体 benchmark 已经完整接通，脚本基线在 12-episode smoke 上保持 `1.00`。
- learned place 的单技能语义问题已经修正，并重新通过 formal eval；但 full-task 多物体运行时仍只有 `0.50`，问题更像 post-pick runtime mismatch。
- learned pick 的单技能 formal eval 仍强，现有主力 pick manifest 在单技能上是 `0.94 / 0.92`，但 multi-object smoke 只有 `0.50`；新一轮 SAC/BC 覆盖扩展没有打破这个上限。
- 因此下一阶段不应继续简单加随机化，而应分别处理 place 的 runtime alignment 和 pick 的 representation/objective。

推荐阅读顺序：

1. `../开发日志_20260419.md`：看这轮实际做了什么、得到了什么结论。
2. `01_current_status.md`：看当前真实状态。
3. `02_next_actions.md`：看下一步应该做什么。
4. `05_full_analysis_20260419.md`：看完整分析归档。
5. `04_feishu_alignment.md`：看飞书 AI 大创文档和当前路线的对应关系。
6. `03_github_references.md`：看后续可参考的 GitHub 项目地址。

当时最关键的结果可以直接记成下面这组数：

- pick 单技能 formal eval：fixed `0.94`，random `0.92`
- place 单技能 formal eval：fixed `1.00`，random `1.00`
- scripted 多物体 smoke：`1.00`
- learned pick 多物体 smoke：`0.50`
- learned place 多物体 smoke：`0.50`
- learned full hybrid 多物体 smoke：`0.25`
- BC pick v2 多物体 smoke：`0.25`
- conservative SAC pick 多物体 smoke：`0.50`

Linux 下快速查看：

```bash
cd projects/status_snapshot_20260419
ls -1
sed -n '1,220p' 01_current_status.md
sed -n '1,220p' 02_next_actions.md
sed -n '1,320p' 05_full_analysis_20260419.md
```

如果只想看当时的一句话版状态：

- 当前项目处于“多物体 benchmark 已跑通、place 单技能语义已修正、pick 与 place 的多物体失败机理已拆分，但 learned full hybrid 仍未收敛”的阶段。
