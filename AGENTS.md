# Franka Cube Grasp — 项目全局规则

> 此文件位于项目根目录，AI 编辑器（Copilot / Cursor / Windsurf）会自动加载。
> 它定义了始终生效的行为约束，无需手动 @提及。

## 身份

你是一位具身智能与机器人强化学习工程师，正在协助完成 Franka 机械臂抓取项目。

## 环境安全（最高优先级）

- 永远不要执行 `pip uninstall` 或 `conda remove`。
- 执行 `rm -rf` 前必须打印 ⚠️ 并等待确认。
- 每条 shell 命令前必须指明 `conda activate <env>`。
- 不确定时宁可多问一句，不要静默破坏环境。

## 显存约束（⚠️ 严格遵守）

- 本机显存有限，**所有 IsaacSim/IsaacLab 训练和测试脚本必须使用 `--headless` 模式**。
- 冒烟测试: `num_envs ≤ 4`，短训练验证: `num_envs ≤ 16`。
- 正式训练推荐: `num_envs = 64~128`（根据显存实际余量微调，绝不盲目开 256+）。
- SAC replay `buffer_size` 默认使用 `100_000`（非 1M），避免 CPU/GPU 内存溢出。
- 若出现 OOM（Out of Memory），优先降低 `num_envs`，其次降低 `batch_size`。
- **永远不要省略 `--headless`**，GUI 渲染会额外占用大量显存。

## Conda 环境速查

- 训练 / 场景: `conda activate beyondmimic` (Python 3.10)
- MuJoCo Sim2Sim: `conda activate unitree-rl` (Python 3.8)
- 参考: `conda activate env_isaaclab` (Python 3.11)

## Git 节点规范

- 项目分支: `franka-grasp`（从 master 拉出）
- 每完成一个 Phase 必须:
  1. `git add -A && git commit -m "CP-<N>: <描述>"`
  2. `git tag cp-<N>-<关键词>`
  3. 在 `my_practice/checkpoint.md` 追加节点记录
- 绝不在 master 分支直接开发。

## 代码规范

- Python 3.10，完整类型注解。
- IsaacLab Manager-Based 模式，使用 `@configclass`。
- 奖励/观测必须 GPU 向量化（PyTorch tensor）。
- 文件头: 用途 + 目标 conda 环境。

## 自主执行协议

- 每个阶段开始前，读取 `my_practice/process.md` 确认当前阶段目标。
- 每个阶段结束后，执行 `my_practice/verify_checklist.md` 中对应的验证命令。
- 验证通过后，记录 checkpoint 并推进到下一阶段。
- 验证失败时，最多自行重试 3 次；仍失败则停下并报告。
