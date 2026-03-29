# Checkpoint Log — 版本节点记录

> **用途**: AI 在完成每个关键阶段后，在此文件记录 git tag / commit hash / 状态摘要，
>           方便人类随时回退到任意节点。
> **规则**: 只追加，不删除已有记录。

---

## 节点格式

```
### CP-<编号>: <简短标题>
- **时间**: YYYY-MM-DD HH:MM
- **Git Tag**: cp-<编号>-<关键词>
- **Commit**: <hash 前 8 位>
- **分支**: franka-grasp
- **状态**: ✅ 通过 / ⚠️ 部分通过 / ❌ 失败
- **内容**: <做了什么>
- **验证**: <跑了什么测试，结果如何>
- **回退命令**: `git checkout cp-<编号>-<关键词>`
```

---

## 节点记录

（AI 将在此处追加记录，请勿手动删除）

### CP-0: 环境验证与项目初始化
- **时间**: 2026-03-29 17:15
- **Git Tag**: cp-0-env-init
- **Commit**: 590b576
- **分支**: franka-grasp
- **状态**: ✅ 通过
- **内容**:
  - 验证 `beyondmimic` 环境: IsaacSim 可导入, IsaacLab 0.36.21, SB3 SAC ready, PyTorch 2.5.1+cu124 CUDA:True
  - 验证 `unitree-rl` 环境: MuJoCo 3.2.3
  - GPU 确认: NVIDIA RTX 4060 Laptop 8GB, Driver 580.95.05, CUDA 13.0
  - 从 master 创建 `franka-grasp` 分支
  - 创建 `my_practice/franka_cube_grasp/` 完整目录结构 (16 文件)
  - 编写 `scripts/smoke_test.py` (Phase 0 版本, 仅验证导入)
  - 创建 `.gitignore` 排除 isaacgym 等大目录
- **验证**: V0-1 ~ V0-5 全部通过, smoke_test.py 输出 "ALL CHECKS PASSED ✅"
- **回退命令**: `git checkout cp-0-env-init`
