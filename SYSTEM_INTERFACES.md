# System Interfaces

> Scope: shared schemas for CausalExplore, Planner, and Executor.
> Rule: these schemas are the contract between the three project tracks. Natural-language descriptions may be logged, but must not be the only machine-readable interface.

## 1. Naming and Identity

All modules must preserve object identity across the full loop:

```text
scene manifest object_id
  -> CausalExplore object_id
  -> Planner target_object
  -> Executor object_name
  -> evaluation episode record
```

Required identity fields:

| Field | Owner | Meaning |
| --- | --- | --- |
| `scene_id` | simulator/evaluation | Unique scene or episode scene identifier |
| `object_id` | CausalExplore | Object identifier from the scene manifest |
| `target_object` | Planner | Object selected for a skill step |
| `object_name` | Executor | Runtime object name used by skill execution |
| `zone_name` | Planner/Executor | Runtime target zone when a skill needs a placement target |

## 1.1 Simulator Backend Contract

Current status as of 2026-05-03:

- Default test backend: MuJoCo kinematic pick/place scene.
- Legacy backend: PyBullet, available only through explicit factory selection.
- Placeholder backend: IsaacLab is registered in the factory but raises a clear not-implemented error until the real training environment is connected.

Required simulator factory shape:

```python
create_pick_place_simulation(backend=None, **kwargs)
```

Backend selection order:

- Explicit `backend` argument wins.
- If `backend is None`, `EMBODIED_SIM_BACKEND` is used.
- If neither is provided, the default is `mujoco`.

Backend values:

| Backend | Status | Intended use |
| --- | --- | --- |
| `mujoco` | default | simulator-backed tests, CausalExplore probe execution, sim2sim/sim2real validation scaffold |
| `pybullet` / `legacy_pybullet` | legacy | historical comparison or fallback only |
| `isaaclab` | registered placeholder | future RL skill training backend; currently raises a clear runtime error |

Required backend-neutral methods:

- `observe_scene(instruction="")`
- `get_object_position(object_name)`
- `get_object_pose(object_name)`
- `get_quaternion_from_euler(euler)`
- `pick_object(object_name)`
- `place_object(zone_name)`
- `apply_skill_action(delta_position, gripper_command, action_steps, object_name)`
- `capture_skill_state(object_name, zone_name)`
- `restore_runtime_state(state, object_name, zone_name)`

Consumer rule:

- Ralph and planner/executor code must not directly depend on `client_id`, PyBullet body ids, or backend-specific pose APIs.
- New simulator-backed tests should instantiate through the factory or through a protocol-typed object, not by hard-coding the legacy class.

IsaacLab adapter contract:

- IsaacLab will own high-throughput RL training and robot dynamics.
- It must implement the same observation/action semantics as MuJoCo before replacing any training path.
- MuJoCo remains the current sim2sim/sim2real validation backend until IsaacLab parity tests exist.

## 2. CausalExplore Output

Minimum schema:

```json
{
  "scene_id": "scene_0001",
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
  },
  "artifact_path": "outputs/causal_explore/run_0001/object_red_block.json"
}
```

Field rules:

| Field | Required | Rule |
| --- | --- | --- |
| `scene_id` | yes | Must match the evaluated scene or object manifest. |
| `object_id` | yes | Must be stable across CausalExplore, Planner, and Executor. |
| `object_category` | yes | Use a compact category such as `rigid_block`, `button`, `drawer`, `hinged_door`, or `rotary_knob`. |
| `property_belief` | yes | Values must include `label` and `confidence`. |
| `affordance_candidates` | yes | Candidate names should map to available or planned skills. |
| `uncertainty_score` | yes | Float in `[0.0, 1.0]`; higher means planner should prefer exploration. |
| `recommended_probe` | yes when uncertainty is high | Use values such as `tap`, `lateral_push`, `pull_handle`, `press_top`, `rotate_axis`. |
| `contact_region` | yes when a probe or skill is recommended | Must be specific enough for skill arguments. |
| `skill_constraints` | yes | Constraints are hints, not direct continuous control. |
| `artifact_path` | yes | Must point to the saved probe/evidence artifact. |

Planner behavior dependency:

- If `uncertainty_score >= 0.50`, planner should prefer a probe step before task execution.
- If an affordance required by a user task has confidence below `0.70`, planner should either probe or select a safer fallback skill.

## 3. Planner Output

Minimum schema:

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

Allowed `selected_skill` values for the current roadmap:

- `observe`
- `probe`
- `pick`
- `place`
- `press`
- `push`
- `pull`
- `rotate`
- `replan`

Planner restrictions:

- Do not output joint positions, motor torques, raw Cartesian trajectories, or dense continuous controls.
- Do not infer physical properties by natural language alone when CausalExplore reports high uncertainty.
- Do not skip `preconditions`; missing preconditions make failure analysis ambiguous.
- Always include `fallback_action`.

Fallback values:

| `fallback_action` | Meaning |
| --- | --- |
| `probe_or_replan` | Explore first if uncertainty is high; otherwise replan from latest state. |
| `retry_same_skill` | Retry the same skill with bounded retry count. |
| `switch_skill` | Select another skill because affordance or failure history makes the original skill weak. |
| `abort_with_reason` | Stop and report a structured failure when the task cannot be safely attempted. |

## 4. Executor Return

Minimum schema:

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
  "rollout_path": "outputs/evaluations/run_0001/episode_0001.jsonl",
  "failure_history": []
}
```

Failure example:

```json
{
  "success": false,
  "reward": 0.0,
  "final_state": {
    "holding": false,
    "object_pose": [0.22, -0.18, 0.74]
  },
  "contact_state": {
    "has_contact": false,
    "contact_region": null
  },
  "error_code": "released_outside_zone",
  "rollout_path": "outputs/evaluations/run_0001/episode_0007.jsonl",
  "failure_history": [
    {
      "step_index": 2,
      "selected_skill": "place",
      "failure_source": "execution_error",
      "reason": "released_outside_zone",
      "replan_attempt": 1,
      "selected_recovery_policy": "repick_then_place"
    }
  ]
}
```

Required executor behavior:

- Every skill call must return `success`, `final_state`, `error_code`, `rollout_path`, and `failure_history`.
- `failure_history` must be append-only within an episode.
- `error_code` must be compact and enumerable.
- `rollout_path` must point to a saved artifact when running evaluations.

Recommended `error_code` values:

| Error code | Meaning |
| --- | --- |
| `precondition_failed` | Planner asked for a skill whose preconditions were not satisfied. |
| `object_not_found` | Target object was missing from scene state. |
| `zone_not_found` | Target zone was missing from scene state. |
| `grasp_failed` | Pick/grasp did not acquire the object. |
| `transport_failed` | Object was held or moved but did not reach the intended region. |
| `released_outside_zone` | Place released but final object pose missed the zone. |
| `release_missing` | Place reached a release-ready state but did not release. |
| `probe_inconclusive` | Probe did not reduce uncertainty enough for planning. |
| `timeout` | Skill exceeded its step budget. |

## 5. Closed-Loop Control Flow

Default flow:

```text
observe scene
  -> CausalExplore emits object understanding
  -> Planner emits structured step
  -> Executor runs selected skill
  -> Executor returns result
  -> if success: continue next step
  -> if failure: planner reads failure_history and emits bounded retry/replan
  -> if uncertainty high: planner emits probe before skill execution
```

Loop limits:

- Maximum skill retries per step: `2`.
- Maximum planner replans per episode: `3`.
- Maximum probe actions per uncertain object before aborting or asking for fallback: `3`.

These limits should be configurable later, but fixed values are enough for the next integration step.

## 6. Artifact Requirements

Each experiment or demo episode should save:

- Scene manifest.
- CausalExplore output.
- Planner step list.
- Executor episode records.
- Evaluation summary.
- Failure analysis summary when any `failure_history` item exists.

Minimum artifact paths should be stored in the corresponding JSON records so paper tables and demos can trace claims back to runs.
