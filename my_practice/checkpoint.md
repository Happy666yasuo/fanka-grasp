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

### CP-1: 场景与环境配置
- **时间**: 2026-03-29 18:10
- **Git Tag**: cp-1-scene-setup
- **Commit**: 34ec480
- **分支**: franka-grasp
- **状态**: ✅ 通过
- **内容**:
  - 完整阅读官方 Lift 任务源码 (lift_env_cfg.py, joint_pos_env_cfg.py, franka.py, mdp/*)
  - 编写 `envs/mdp/observations.py`: 3 个自定义观测函数 (object_pos, ee_pos, ee_object_rel)
  - 编写 `envs/mdp/rewards.py`: 3 个奖励函数 (sparse lift, dense reach, goal tracking)
  - 编写 `envs/mdp/terminations.py`: 1 个终止条件 (object_dropped_below_table)
  - 编写 `envs/mdp/__init__.py`: 重导出所有 MDP 函数
  - 编写 `envs/franka_grasp_env_cfg.py`: 完整场景 + MDP 配置
    - FrankaGraspSceneCfg: Franka Panda + table + cube + EE frame + ground + light
    - FrankaGraspEnvCfg: obs(32D) + action(8D) + 5 reward terms + 2 terminations + 2 events
    - 使用 AppLauncher 标准启动模式
  - 编写 `envs/__init__.py`: gymnasium 环境注册 (Isaac-Grasp-Cube-Franka-v0)，使用字符串引用避免提前导入
  - 重写 `scripts/smoke_test.py`: Phase 1 版本，AppLauncher 启动 + 环境实例化 + 随机 rollout
- **验证**:
  - V1-1: 12 个 .py 文件全部 py_compile 通过 ✅
  - V1-2: headless, 2 envs, 50 步随机 rollout ✅
    - obs_space: `Dict('policy': Box(-inf, inf, (2, 32), float32))`
    - act_space: `Box(-inf, inf, (2, 8), float32)`
    - Observation: joint_pos(9) + joint_vel(9) + object_pos(3) + ee_object_rel(3) + actions(8) = 32D
    - Action: arm(7) + gripper(1) = 8D
    - Reward: reaching(1.0) + lifting(15.0) + goal_tracking(10.0) + action_rate(-1e-4) + joint_vel(-1e-4)
    - "ALL CHECKS PASSED ✅"
- **回退命令**: `git checkout cp-1-scene-setup`

### CP-2: SAC Baseline 训练管线
- **时间**: 2026-03-29 18:30
- **Git Tag**: cp-2-sac-baseline
- **Commit**: 712c51d
- **分支**: franka-grasp
- **状态**: ✅ 通过
- **内容**:
  - 编写 `agents/sac_cfg.py`: SAC 超参数配置 (dataclass)
    - buffer_size=100,000, batch_size=256, net_arch=[256,256], gamma=0.99, ent_coef=auto
  - 编写 `scripts/train.py`: 完整 SAC 训练管线
    - AppLauncher 启动 → ManagerBasedRLEnv → Sb3VecEnvWrapper → SAC
    - 支持 --num_envs, --total_timesteps, --reward_type, --seed, --log_dir, --headless
    - CheckpointCallback + SuccessRateCallback (自定义)
    - 训练结束自动保存 latest.zip + 导出 policy.onnx
  - 编写 `scripts/eval.py`: 评估脚本
    - 加载 checkpoint → 确定性推理 → 报告 Mean Reward / Success Rate
    - 支持 --checkpoint, --num_episodes, --num_envs, --headless
  - 更新 `agents/__init__.py`: 导出 SAC_DEFAULT_CFG
- **验证**:
  - V2-1: 3 个新 .py 文件全部 py_compile 通过 ✅
  - V2-2: 1000 步冒烟训练 (headless, 4 envs, sparse) ✅
    - 生成 `logs/test_run/checkpoints/latest.zip` (3.4MB)
    - 生成 `logs/test_run/policy.onnx` (316KB)
    - 生成 TensorBoard events 文件
  - V2-3: checkpoint 文件存在 ✅
  - V2-4: TB 日志存在 ✅
  - V2-5: eval.py 运行 5 episodes ✅
    - Mean Reward: -0.332 ± 0.086
    - Mean Ep Length: 250.0 (全部超时)
    - Success Rate: 0.0% (预期：稀疏奖励 + 仅 1000 步 → 不收敛)
- **回退命令**: `git checkout cp-2-sac-baseline`

### CP-3: Reward Shaping — 三种奖励策略
- **时间**: 2026-03-29 18:50
- **Git Tag**: cp-3-reward-shaping
- **Commit**: ff96763
- **分支**: franka-grasp
- **状态**: ✅ 通过
- **内容**:
  - 扩展 `envs/mdp/rewards.py`: 新增 3 个高级奖励函数
    - `shaped_multi_stage`: 4 阶段密集奖励 (reach → grasp → lift → hold), 加权组合
    - `pbrs_shaping`: Potential-Based Reward Shaping (F = γΦ(s') - Φ(s)), 保留最优策略
    - `curriculum_reward`: 课程学习奖励 (easy/medium/hard 三级, 按 step 数切换)
  - 扩展 `envs/franka_grasp_env_cfg.py`: 新增 3 种 RewardsCfg
    - `SparseRewardsCfg`: 纯稀疏奖励 (消融基线)
    - `ShapedRewardsCfg`: shaped_multi_stage + penalties
    - `PBRSRewardsCfg`: sparse lift + PBRS shaping + penalties
  - 更新 `scripts/train.py`: 实现 --reward_type 奖励配置切换
    - 替换 TODO → reward_map 字典映射 sparse/shaped/pbrs → 对应 RewardsCfg
  - 新建 `scripts/check_reward_range.py`: 奖励范围诊断工具
    - 随机动作跑 N 步, 报告 min/max/mean/std/NaN/Inf
    - 逐 reward term 细分报告 (使用 reward_manager._step_reward)
    - 健康度评估 (范围过大/过小/NaN 警告)
- **验证**:
  - V3-1: 4 个 .py 文件全部 py_compile 通过 ✅
  - V3-2: shaped 训练 1000 步 (4 envs, headless) ✅
    - RewardManager 正确加载: shaped(1.0) + action_rate(-1e-4) + joint_vel(-1e-4)
    - 生成 latest.zip + policy.onnx
  - V3-3: PBRS 训练 1000 步 (4 envs, headless) ✅
    - RewardManager 正确加载: lifting_object(15.0) + pbrs(5.0) + action_rate(-1e-4) + joint_vel(-1e-4)
    - 生成 latest.zip + policy.onnx
  - V3-4: check_reward_range.py (shaped, 200 步, 4 envs) ✅
    - Total: min=-0.000086, max=0.000706, mean=0.000029 — 无 NaN/Inf
    - Per-term: shaped [0, 0.0388], action_rate [-0.0021, -0.0001], joint_vel [-0.0039, -0.0004]
    - 低幅值 (随机动作预期行为) — 正式训练中 shaped 会因策略改善而增大
- **回退命令**: `git checkout cp-3-reward-shaping`

### CP-4: HER 集成 (Hindsight Experience Replay)
- **时间**: 2026-03-30 16:30
- **Git Tag**: cp-4-her
- **Commit**: f6a80fc
- **分支**: franka-grasp
- **状态**: ✅ 通过
- **内容**:
  - 实现 `agents/her_wrapper.py`: HERGoalVecEnvWrapper
    - 将 flat obs (32D) 拆分为 Dict: observation(29D) + achieved_goal(3D) + desired_goal(3D)
    - 实现 compute_reward() — 稀疏距离阈值奖励 (0/-1)
    - 修复 terminal_observation dict 转换（SB3 _store_transition 兼容）
    - 修复 env_method("compute_reward") 路由（HER buffer 调用拦截）
  - 更新 `agents/__init__.py`: 导出 HERGoalVecEnvWrapper
  - 扩展 `scripts/train.py`:
    - 新增 --algo 参数 (sac / sac_her)
    - sac_her 模式: MultiInputPolicy + HerReplayBuffer (future, n=4)
    - 自动计算 learning_starts = max_ep_steps * num_envs (HER 需要至少 1 完整 episode)
  - 新建 `scripts/check_her_buffer.py`: HER buffer 诊断工具
    - 报告 buffer size / fill ratio / goal strategy
    - 采样验证 + reward 分布分析
- **验证**:
  - V4-1: 3 个 .py 文件 py_compile 通过 ✅
  - V4-2: SAC+HER 2000 步冒烟训练 (4 envs, headless, sparse reward) ✅
    - HER wrapper: Dict obs (achieved_goal:3D, desired_goal:3D, observation:29D)
    - learning_starts=1000 (250 steps/ep × 4 envs)
    - 模型保存: latest.zip
  - V4-3: check_her_buffer.py 1500 步 ✅
    - Buffer: HerReplayBuffer, size=375, strategy=FUTURE, n_sampled_goal=4
    - Sampling: obs keys 正确, shapes 正确 (64×29, 64×3, 64×3, 64×8, 64×1)
    - Reward stats: min=-1.0, max=0.0, mean=-0.043 — HER 重标注生效 (292/375 success)
- **回退命令**: `git checkout cp-4-her`

### CP-5: Sim2Sim — IsaacSim → MuJoCo 迁移
- **时间**: 2026-03-30 16:50
- **Git Tag**: cp-5-sim2sim
- **Commit**: 41f3583
- **分支**: franka-grasp
- **状态**: ✅ 通过
- **内容**:
  - 下载 MuJoCo Menagerie Franka Panda 模型 (sparse clone + 本地复制)
  - 创建 `sim2sim/franka_emika_panda/franka_table.xml`: Sim2Sim 场景
    - Franka Panda (include panda.xml) + 桌面 + 方块 (DexCube 近似)
    - 物理对齐: timestep=0.002, gravity=9.81, control 每 10 步 (=0.01s × decimation=2)
    - grasp_home keyframe: IsaacLab 默认关节位置 + cube 初始位置
    - 修复 Menagerie panda.xml keyframe qpos 冲突 (删除原 home keyframe)
  - 创建 `sim2sim/export_onnx.py`: ONNX 导出脚本 (beyondmimic 环境)
    - 加载 SB3 SAC checkpoint → 提取 actor → torch.onnx.export
    - 支持 --checkpoint, --output, --obs_dim 参数
    - 可选 onnxruntime 验证
  - 创建 `sim2sim/mujoco_eval.py`: MuJoCo 推理评估脚本 (unitree-rl 环境)
    - FrankaGraspMuJoCoEnv: 完整 obs 空间对齐 (32D)
      - joint_pos_rel[0:9], joint_vel_rel[9:18], object_pos_b[18:21],
        ee_object_rel[21:24], actions[24:32]
    - ONNXPolicy: onnxruntime InferenceSession wrapper
    - 关节映射: MuJoCo joint1-7 ↔ IsaacLab panda_joint1-7
    - 执行器映射: actuator0-6 位置控制, actuator7 tendon (0-255)
    - 支持 cube 随机化 (匹配 IsaacLab EventCfg)
    - 成功判定: cube z > init_z + 0.06m
- **验证**:
  - V5-1: ONNX 导出成功 ✅ (logs/test_run → policy_franka_grasp.onnx, 310KB)
  - V5-2: ONNX 文件存在 ✅
  - V5-3: MuJoCo 推理 10 episodes 无报错 ✅
    - ONNX 加载: input=obs[batch,32], output=action[batch,8]
    - 策略输出非零动作 (范围 [-0.92, 0.51])
  - V5-4: Success rate: 0.0% (0/10) — 预期结果
    - 原因: checkpoint 仅训练 1000 步, 策略远未收敛
    - 物理差异: IsaacLab PhysX vs MuJoCo implicit integrator
    - 完整训练后的 Sim2Sim 迁移效果待正式实验评估
- **回退命令**: `git checkout cp-5-sim2sim`

### CP-6: 总结报告 + 收尾
- **时间**: 2026-03-30 17:00
- **Git Tag**: cp-6-report
- **Commit**: fcc3026
- **分支**: franka-grasp
- **状态**: ✅ 通过
- **内容**:
  - 编写 `REPORT.md`: 完整实验报告 (~350 行)
    - 问题定义、环境设计 (场景/观测/动作/事件详表)
    - 算法分析: SAC 选择理由 + 超参数
    - 四种奖励策略: Sparse / Shaped / PBRS / Curriculum
    - HER 实现: GoalEnv wrapper + 三个关键 bug 修复
    - Sim2Sim: 物理参数对齐表 + 观测/动作映射 + 差异分析
    - 结论与展望 + 项目文件结构 + 运行命令速查
  - 更新 `README.md`: 所有 Phase 状态更新为 ✅
  - 编写 `daily_log_2026-03-30.md`: Day 2 工作日志
  - GitHub 推送: franka-grasp 分支 + cp-4/5/6 tags
- **验证**: 报告内容完整, Git 推送成功
- **回退命令**: `git checkout cp-6-report`

### CP-6 补全: Phase 6 交付物完善
- **时间**: 2026-03-30 18:30
- **Git Tag**: cp-6-complete
- **Commit**: f1cd302
- **分支**: franka-grasp
- **状态**: ✅ 通过
- **内容**:
  - 新增 `CurriculumRewardsCfg` 到 `franka_grasp_env_cfg.py` (支持 curriculum 奖励方案)
  - 扩展 `train.py`: `--reward_type` 新增 `curriculum` 选项
  - 新建 `scripts/run_experiments.sh`: 6 组实验批量运行器
    - exp-01: SAC + Sparse, exp-02: SAC + Shaped, exp-03: SAC + Curriculum
    - exp-04: SAC + PBRS, exp-05: SAC+HER + Sparse, exp-06: SAC+HER + Shaped
    - 全部 headless, num_envs=64, 500K 步, seed=42
  - 新建 `scripts/plot_results.py`: TensorBoard 日志比较绘图工具
    - 读取 TB event 文件, 绘制 reward/episode_length/success_rate 曲线
    - 支持 EMA 平滑, 生成 summary_bar.png 汇总柱状图
  - 更新 `REPORT.md`: 新增 §5.1 实验矩阵表 (6 行), §5.4 预测分析, 更新文件结构和命令附录
- **验证**:
  - 4 文件 py_compile / bash -n 语法检查全部通过 ✅
  - 5 项交付物全部确认存在 ✅
  - GitHub 推送成功: franka-grasp + main 分支均同步到 f1cd302 ✅
- **回退命令**: `git checkout cp-6-complete`
