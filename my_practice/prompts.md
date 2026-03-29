# Prompts — 项目专业提示词库 v2.1

> **用途**: 在使用 AI 编程助手时，粘贴对应提示词驱动 AI 全程自主实现项目。
> **项目**: 基于 IsaacLab 的 Franka 机械臂桌面方块抓取（SAC + Reward Shaping + HER + Sim2Sim）
> **版本**: v2.1 — 增加显存约束、--headless 强制规则、保守 num_envs/buffer_size 默认值

---

## 📂 辅助文件说明

本提示词系统依赖以下辅助文件协同工作：

| 文件 | 路径 | 作用 |
|------|------|------|
| **AGENTS.md** | `项目根目录/AGENTS.md` | 全局行为规则，AI 编辑器自动加载，无需手动粘贴 |
| **SKILL.md** | `.agents/skills/franka-cube-grasp/SKILL.md` | Agent Skill 标准格式，AI 按需自动激活 |
| **process.md** | `my_practice/process.md` | 6 阶段项目流程，AI 据此判断当前阶段与目标 |
| **verify_checklist.md** | `my_practice/verify_checklist.md` | 每阶段验证命令清单，AI 完成阶段后逐项执行 |
| **checkpoint.md** | `my_practice/checkpoint.md` | 版本节点记录，AI 每完成阶段后追加记录 |
| **prompts.md** | `my_practice/prompts.md`（本文件） | 提示词库，人类按需粘贴给 AI |

**工作原理**:
```
人类粘贴 Prompt ──→ AI 读取 AGENTS.md（自动）
                 ──→ AI 激活 SKILL.md（自动匹配）
                 ──→ AI 参考 process.md（确定阶段）
                 ──→ AI 执行代码编写
                 ──→ AI 执行 verify_checklist.md（自检）
                 ──→ AI 写入 checkpoint.md + git tag（记录节点）
                 ──→ 进入下一阶段或报告问题
```

---

## 📌 Master Prompt（主提示词 — 首次对话必贴）

```
You are an autonomous robotics & reinforcement learning engineer. You will
INDEPENDENTLY implement a complete project with minimal human intervention.

═══════════════════════════════════════════════════════
PROJECT: Franka Panda Cube Grasping (SAC + IsaacLab)
═══════════════════════════════════════════════════════

### What you are building
Train a Franka Emika Panda (7-DOF + gripper) to pick up a cube from a table
using SAC in IsaacLab 2.1.0 / IsaacSim 4.5.0. Overcome sparse reward via
Reward Shaping, PBRS, HER, and Curriculum Learning. Validate via Sim2Sim
transfer to MuJoCo.

### Environment Constraints (STRICT — memorize these)
| Purpose          | Conda Env      | Python | Key Packages                                             |
|------------------|----------------|--------|----------------------------------------------------------|
| Training & Scene | `beyondmimic`  | 3.10   | isaacsim 4.5.0, isaaclab 0.36.21, SB3 2.7.1, torch 2.5.1 |
| Sim2Sim MuJoCo   | `unitree-rl`   | 3.8    | mujoco 3.2.3, torch 2.3.1                               |
| Reference Only   | `env_isaaclab` | 3.11   | isaaclab 0.47.2 (do NOT run project code here)           |

### Absolute Paths
- IsaacLab:  ~/Desktop/isaac_workspace/IsaacLab-2.1.0/
- Project:   ~/Desktop/beyondmimic/my_practice/franka_cube_grasp/
- Reference: ~/Desktop/beyondmimic/Beyondmimic_Deploy_G1/

### Safety Rules (NEVER violate)
1. NEVER run `pip uninstall` / `conda remove`.
2. ALWAYS prefix shell commands with `conda activate <env>`.
3. Before `rm -rf` → print ⚠️ WARNING → wait for human confirmation.
4. NEVER commit directly on `master` branch. Use `franka-grasp` branch.

### ⚠️ VRAM Constraints (CRITICAL — local machine has LIMITED VRAM)
1. **ALL IsaacSim/IsaacLab scripts MUST use `--headless`** — GUI will OOM.
2. Smoke tests: `num_envs ≤ 4`.
3. Short verification: `num_envs ≤ 16`.
4. **Production training: `num_envs = 64~128`** — NEVER blindly use 256+.
5. SAC `buffer_size = 100_000` (NOT 1M) — avoid CPU/GPU memory overflow.
6. If OOM → halve `num_envs` first → then halve `batch_size`.
7. **NEVER omit `--headless`** in any training, eval, or smoke-test command.

### Documentation
- IsaacSim 4.5.0: https://docs.isaacsim.omniverse.nvidia.com/4.5.0/index.html
- IsaacLab: https://isaac-sim.github.io/IsaacLab/main/source/setup/ecosystem.html
- SB3 SAC: https://stable-baselines3.readthedocs.io/en/master/modules/sac.html
- SB3 HER: https://stable-baselines3.readthedocs.io/en/master/modules/her.html

### Your Execution Protocol
For EVERY phase, you must:
1. READ `my_practice/process.md` to confirm the phase goal.
2. IMPLEMENT the code (create files, write logic).
3. RUN the verification commands from `my_practice/verify_checklist.md`.
4. If verification FAILS → debug and retry (max 3 attempts) → if still fails, STOP and report.
5. If verification PASSES → execute the git checkpoint procedure:
   a. `git add -A`
   b. `git commit -m "CP-<N>: <phase description>"`
   c. `git tag cp-<N>-<keyword>`
   d. Append a record to `my_practice/checkpoint.md`
6. PROCEED to the next phase.

### Code Conventions
- Python 3.10, full type annotations, Google-style docstrings.
- IsaacLab Manager-Based pattern: `@configclass`, `MISSING` placeholders.
- GPU-vectorized rewards/observations (PyTorch tensors, no Python for-loops).
- File header: purpose + target conda environment.

Now read AGENTS.md, process.md, and verify_checklist.md to understand the full
context, then begin from the current phase.
```

---

## 🚀 Autopilot Prompt（全自动执行 — 一句话驱动全流程）

```
Read my_practice/process.md and my_practice/checkpoint.md.
Determine which Phase I'm currently on (the first phase without a checkpoint record).
Then execute that phase autonomously:
  1. Implement all code for the phase
  2. Run verify_checklist.md checks for that phase
  3. If all checks pass: git commit + git tag + record in checkpoint.md
  4. Then ask me: "Phase N complete. Continue to Phase N+1?"

If a check fails, debug and retry up to 3 times. If still failing, stop and
show me the error with your diagnosis.

CRITICAL REMINDERS:
- ALL IsaacSim/IsaacLab commands MUST include `--headless`.
- Smoke tests: num_envs ≤ 4. Training: num_envs = 64~128 (NEVER 256+).
- SAC buffer_size = 100_000 (NOT 1M).

Important: before starting, confirm which phase you're about to execute and
what you plan to do. Wait for my "go" before writing code.
```

---

## 🔄 Continue Prompt（继续下一阶段）

```
Continue to the next phase. Read checkpoint.md to find where we left off,
then execute the next phase following the same protocol:
implement → verify → checkpoint → report.

Remember: --headless is mandatory. num_envs ≤ 128. buffer_size = 100_000.
```

---

## ⏪ Rollback Prompt（回退到指定节点）

```
I need to roll back to checkpoint CP-<N>.

Steps:
1. Run `git log --oneline -20` to show recent history.
2. Run `git checkout cp-<N>-<keyword>` to go to that tag.
3. Create a new branch `franka-grasp-v2` from that point.
4. Show me the diff between current and the rolled-back state.
5. Update checkpoint.md to note the rollback.

Do NOT delete any branches or tags. Do NOT use `git reset --hard` unless I
explicitly confirm.
```

---

## 🔧 Phase-Specific Prompts（阶段专用提示词 — 按需使用）

---

### Prompt P0: 环境验证与项目初始化

```
Execute Phase 0 from process.md:
1. Verify all conda environments (beyondmimic, unitree-rl) are functional.
2. Create the project directory structure at:
   ~/Desktop/beyondmimic/my_practice/franka_cube_grasp/
3. Create git branch `franka-grasp` from current master.
4. Create placeholder __init__.py files in all sub-packages.
5. Create a minimal smoke_test.py script.
6. Run verify_checklist.md Phase 0 checks.
7. Record checkpoint CP-0.

VRAM reminder: all IsaacLab commands must use --headless.
```

---

### Prompt P1: 场景搭建

```
Execute Phase 1 from process.md:
1. Read the official Lift task at:
   IsaacLab-2.1.0/source/isaaclab_tasks/.../manipulation/lift/
   Understand its structure completely before writing any code.
2. Create the custom Franka cube grasp environment:
   - envs/franka_grasp_env_cfg.py (scene + MDP config)
   - envs/mdp/observations.py
   - envs/mdp/rewards.py (start with sparse reward only)
   - envs/mdp/terminations.py
3. Create scripts/smoke_test.py that instantiates the env with --headless
   and runs 100 random steps with num_envs=4.
4. Run verify_checklist.md Phase 1 checks.
5. Record checkpoint CP-1.

IMPORTANT: Read the official Lift task code FIRST. Do not invent APIs — follow
exactly what IsaacLab 2.1.0 (version 0.36.21) provides.
VRAM: Always --headless. Smoke test num_envs ≤ 4.
```

---

### Prompt P2: SAC Baseline 训练

```
Execute Phase 2 from process.md:
1. Create agents/sac_cfg.py with SAC hyperparameters:
   - buffer_size = 100_000 (NOT 1M — limited VRAM/RAM)
   - batch_size = 256
   - Default num_envs = 64 for training
2. Create scripts/train.py:
   - Wraps IsaacLab env with Sb3VecEnvWrapper
   - Supports --num_envs, --total_timesteps, --reward_type, --seed, --log_dir
   - **MUST accept and pass --headless to IsaacLab env**
   - Includes EvalCallback, CheckpointCallback, and custom success_rate logger
   - Saves model + exports ONNX at the end
3. Create scripts/eval.py for evaluation (100 episodes, reports success rate).
   - MUST support --headless flag.
4. Run a 1000-step smoke test with --headless --num_envs 4.
5. Run verify_checklist.md Phase 2 checks.
6. Record checkpoint CP-2.

NOTE: With sparse reward, training will likely NOT converge. That's expected.
The point is to have a working pipeline and a baseline to compare against.
VRAM: --headless mandatory. Smoke num_envs=4. Training num_envs=64.
```

---

### Prompt P3: Reward Shaping

```
Execute Phase 3 from process.md:
1. Extend envs/mdp/rewards.py with three additional reward strategies:
   a. Dense shaped reward (multi-stage: reach + grasp + lift + hold)
   b. PBRS (potential-based reward shaping, F = γΦ(s') - Φ(s))
   c. Curriculum-aware reward (easy → medium → hard)
2. Create scripts/check_reward_range.py to validate reward magnitudes.
3. Make reward_type selectable via train.py --reward_type {sparse,shaped,pbrs}.
4. Run short training (2000 steps) with each reward type:
   --headless --num_envs 4 to verify no crash/NaN.
5. Run verify_checklist.md Phase 3 checks.
6. Record checkpoint CP-3.

Mathematical formulations must be in docstrings. All code GPU-vectorized.
VRAM: --headless mandatory. Verification num_envs=4.
```

---

### Prompt P4: HER 集成

```
Execute Phase 4 from process.md:
1. Create agents/her_wrapper.py:
   - Wraps env as GoalEnv (observation / achieved_goal / desired_goal)
   - Implements compute_reward(achieved, desired, info)
2. Extend scripts/train.py to support --algo sac_her:
   - Uses SB3 HerReplayBuffer with goal_selection_strategy="future"
   - buffer_size = 100_000 (NOT 1M)
3. Create scripts/check_her_buffer.py to verify buffer is filling correctly.
4. Run short training (1000 steps) with SAC+HER --headless --num_envs 4.
5. Run verify_checklist.md Phase 4 checks.
6. Record checkpoint CP-4.

VRAM: --headless mandatory. buffer_size=100_000. Smoke num_envs=4.
```

---

### Prompt P5: Sim2Sim 迁移

```
Execute Phase 5 from process.md:
1. Create sim2sim/export_onnx.py:
   - Loads best SB3 checkpoint → exports actor to ONNX
   - Includes observation normalization in the graph
2. Create sim2sim/franka_table.xml (MuJoCo MJCF scene):
   - Franka Panda + table + cube, physics params aligned with IsaacSim
3. Create sim2sim/mujoco_eval.py:
   - Loads ONNX, runs inference in MuJoCo, reports success rate (100 episodes)
   - MuJoCo eval does NOT need --headless (no IsaacSim involved)
4. Run verify_checklist.md Phase 5 checks.
5. Record checkpoint CP-5.

Reference: ~/Desktop/beyondmimic/Beyondmimic_Deploy_G1/deploy_mujoco_1.py
MuJoCo eval runs in `unitree-rl` env (Python 3.8, mujoco 3.2.3).
ONNX export runs in `beyondmimic` env.
```

---

### Prompt P6: 实验对比与报告

```
Execute Phase 6 from process.md:
1. Design full experiment matrix:
   | ID     | Algorithm | Reward  | HER | Curriculum | num_envs | Steps |
   |--------|-----------|---------|-----|------------|----------|-------|
   | exp-01 | SAC       | sparse  | No  | No         | 64       | 500k  |
   | exp-02 | SAC       | shaped  | No  | No         | 64       | 500k  |
   | exp-03 | SAC       | shaped  | No  | Yes        | 64       | 500k  |
   | exp-04 | SAC       | pbrs    | No  | No         | 64       | 500k  |
   | exp-05 | SAC+HER   | sparse  | Yes | No         | 64       | 500k  |
   | exp-06 | SAC+HER   | shaped  | Yes | No         | 64       | 500k  |
2. Create scripts/run_experiments.sh to launch all sequentially.
   - ALL commands must include --headless.
   - num_envs = 64 for all experiments.
   - buffer_size = 100_000.
3. Create scripts/plot_results.py for comparison charts.
4. Generate REPORT.md with full experiment analysis.
5. Record checkpoint CP-6.

VRAM: --headless mandatory for all runs. num_envs=64. buffer_size=100_000.
```

---

## 🐛 Debug Prompt（训练不收敛诊断）

```
My SAC training is not converging. Diagnose systematically:

Symptoms: [填写你观察到的现象]

Check these in order, run diagnostic code for each (always --headless):
1. Observation range: print min/max/mean of obs over 1000 steps (num_envs=4 --headless)
2. Reward scale: print reward statistics, check for NaN/Inf
3. Action bounds: verify actions are within env limits
4. Entropy α: check if auto-tuning is working (print α over training)
5. Replay buffer: check fill ratio and positive-reward sample ratio
6. Episode lifecycle: verify reset, check episode lengths
7. Gradient health: check for exploding/vanishing gradients
8. GPU memory: run nvidia-smi, check if OOM is silently corrupting data

Run each check, show output, give diagnosis. Rank fixes by likelihood.
Remember: all IsaacLab commands MUST use --headless.
```

---

## ⚡ Performance Prompt（性能优化）

```
Optimize training throughput on my GPU (VRAM is limited).

Setup:
- GPU: [你的GPU型号和显存]
- Current num_envs: [当前值]
- Current FPS: [当前值]

Tasks (all benchmarks MUST use --headless):
1. Benchmark num_envs = {16, 32, 64, 128} → report FPS + VRAM for each.
   ⚠️ Do NOT test 256+ unless VRAM clearly permits.
2. Profile reward/obs computation for bottlenecks.
3. Optimize batch_size / train_freq / gradient_steps.
4. Check if fp16 mixed precision is feasible.
5. Recommend optimal config that fits within VRAM budget.

VRAM constraints: --headless mandatory. buffer_size = 100_000.
Start conservative, scale up only if VRAM headroom exists.
```

---

## 🔑 Quick Prompts（快速提问模板）

### 查看进度
```
Read checkpoint.md and process.md. Tell me: which phase am I on, what's done, what's next.
```

### 查看所有 checkpoint
```
Run: git tag -l "cp-*" --sort=-creatordate && git log --oneline -20
Then read checkpoint.md and show a summary table.
```

### 解释代码
```
Read [文件路径] and explain every section in detail,
especially IsaacLab patterns (@configclass, MISSING, RewardTermCfg).
```

### 超参推荐
```
Given: obs_dim=35, action_dim=8, sparse reward, SAC, GPU=[型号],
VRAM limited, max num_envs=128, buffer_size=100_000.
Recommend hyperparameters for best sample efficiency within VRAM budget.
```

### 错误排查
```
In beyondmimic env (Python 3.10, IsaacSim 4.5.0, IsaacLab 2.1.0):
Command: [粘贴命令]  (was --headless used? If not, add it first!)
Error: [粘贴错误]
Fix it. Do NOT uninstall anything. If OOM, reduce num_envs first.
```

---

## ⚙️ 环境速查

```bash
# ===== 训练 + 场景搭建 =====
conda activate beyondmimic
# Python 3.10 | IsaacSim 4.5.0 | IsaacLab 2.1.0 | SB3 2.7.1
# ⚠️ 必须 --headless | num_envs ≤ 128 | buffer_size = 100_000

# ===== Sim2Sim MuJoCo 验证 =====
conda activate unitree-rl
# Python 3.8 | MuJoCo 3.2.3 | PyTorch 2.3.1

# ===== 参考（不运行项目代码）=====
conda activate env_isaaclab
# Python 3.11 | IsaacLab 0.47.2
```

---

## 📝 使用方式总结

| 场景 | 该粘贴什么 |
|------|-----------|
| **首次启动项目** | Master Prompt → 然后 P0 |
| **让 AI 自己判断做什么** | Autopilot Prompt |
| **继续下一阶段** | Continue Prompt |
| **回退到旧版本** | Rollback Prompt（填 CP 编号） |
| **执行特定阶段** | Master Prompt + 对应 P0~P6 |
| **训练出问题** | Master Prompt + Debug Prompt |
| **优化性能** | Master Prompt + Performance Prompt |
| **快速提问** | Quick Prompts（可独立使用） |
