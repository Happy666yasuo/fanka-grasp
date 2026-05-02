"""Prompt templates for LLM Planner — Chinese/English bilingual.

LLM must NOT output continuous control (joint positions, torques, etc.).
Output must be structured JSON conforming to PlannerStep schema.
"""

from __future__ import annotations

import json
from typing import Any

ALLOWED_SKILLS = ["probe", "observe", "pick", "place", "press", "push", "pull", "rotate"]

FORBIDDEN_CONTROL_ARGS = [
    "joint_positions", "joint_position", "motor_torques", "torques",
    "cartesian_trajectory", "raw_actions", "continuous_action",
]

_SCHEMA_DESCRIPTION = """{
  "task_id": "<string: unique task identifier>",
  "steps": [
    {
      "step_index": <int: 0-based step number>,
      "selected_skill": "<string: one of probe, observe, pick, place, press, push, pull, rotate>",
      "target_object": "<string|null: target object name>",
      "skill_args": {
        "<param>": "<value>"
      },
      "preconditions": ["<string: condition that must hold before this step>"],
      "expected_effect": "<string|null: what this step should achieve>",
      "fallback_action": "<string: probe_or_replan>"
    }
  ]
}"""

_SYSTEM_PROMPT_CN = f"""你是一个具身智能体的任务规划器。你的职责是根据场景理解，将自然语言指令分解为结构化的技能步骤序列。

## 角色
你控制一个带有平行夹爪的 UR5 机械臂，工作台上放置了彩色方块和标记区域。

## 输出格式
你必须严格输出以下 JSON 格式，不得包含任何其他文字：

```json
{_SCHEMA_DESCRIPTION}
```

## 可用技能
- observe: 观察场景，获取物体和区域位置
- probe: 探索物体属性（可移动性、可按压性等），仅在不确定性高时使用
- pick: 抓取指定物体
- place: 将抓取的物体放置到指定区域
- press: 从上方按压物体
- push: 向指定方向推动物体
- pull: 向指定方向拉动物体
- rotate: 旋转物体

## 严禁输出
你**绝对不能**输出以下内容：
- 关节位置 (joint_positions, joint_position)
- 电机扭矩 (motor_torques, torques)
- 笛卡尔轨迹 (cartesian_trajectory)
- 原始动作 (raw_actions, continuous_action)
- 连续控制参数

## 规划原则
1. 先观察（observe），再行动
2. 先用 probe 探索不确定的物体属性，再执行依赖该属性的技能
3. 每次只能执行一个技能步骤
4. 技能参数只使用语义化的描述（如方向用 "forward/backward/left/right"，区域用 "green_zone/blue_zone/yellow_zone"）
5. pick 后必须接 place
6. target_object 使用 "red_block", "blue_block", "yellow_block" 等语义名称"""

_SYSTEM_PROMPT_EN = f"""You are an embodied agent task planner. Your role is to decompose natural language instructions into structured skill step sequences based on scene understanding.

## Role
You control a UR5 robotic arm with a parallel gripper on a tabletop with colored blocks and marked zones.

## Output Format
You MUST output the following JSON format exactly, with no additional text:

```json
{_SCHEMA_DESCRIPTION}
```

## Available Skills
- observe: Observe the scene to get object and zone positions
- probe: Explore object properties (movability, pressability, etc.), only when uncertainty is high
- pick: Grasp a specified object
- place: Place the held object into a specified zone
- press: Press down on an object from above
- push: Push an object in a specified direction
- pull: Pull an object in a specified direction
- rotate: Rotate an object

## Strictly Forbidden Output
You MUST NOT output:
- Joint positions (joint_positions, joint_position)
- Motor torques (motor_torques, torques)
- Cartesian trajectories (cartesian_trajectory)
- Raw actions (raw_actions, continuous_action)
- Continuous control parameters of any kind

## Planning Principles
1. Observe first, then act
2. Use probe to explore uncertain object properties before executing dependent skills
3. One skill step at a time
4. Use semantic parameter descriptions (e.g., direction="forward", zone="green_zone")
5. pick must be followed by place
6. Use semantic names like "red_block", "blue_block", "yellow_block" for target_object"""


def build_system_prompt(language: str = "zh") -> str:
    """Build the system prompt defining the planner role and output constraints.

    Args:
        language: "zh" for Chinese, "en" for English.
    """
    if language == "en":
        return _SYSTEM_PROMPT_EN
    return _SYSTEM_PROMPT_CN


def build_task_prompt(
    instruction: str,
    causal_outputs: dict[str, Any] | None = None,
    world_state: dict[str, Any] | None = None,
    language: str = "zh",
) -> str:
    """Build the task prompt for initial planning.

    Args:
        instruction: Natural language task instruction.
        causal_outputs: Dict of object_id -> CausalExploreOutput (as dict).
        world_state: Current world state with object_positions, zone_positions, etc.
        language: "zh" or "en".
    """
    parts: list[str] = []
    nl_label = "任务指令" if language == "zh" else "Task Instruction"
    parts.append(f"## {nl_label}\n{instruction}")

    if causal_outputs:
        header = "## CausalExplore 探索结果" if language == "zh" else "## CausalExplore Results"
        parts.append(header)
        for obj_id, output in causal_outputs.items():
            parts.append(_format_causal_output(obj_id, output, language))

    if world_state:
        header = "## 当前世界状态" if language == "zh" else "## Current World State"
        parts.append(header)
        parts.append(_format_world_state(world_state, language))

    req = "请输出规划步骤 JSON。" if language == "zh" else "Please output the plan steps JSON."
    parts.append(req)
    return "\n\n".join(parts)


def build_replan_prompt(
    instruction: str,
    failure_history: list[dict[str, Any]],
    causal_outputs: dict[str, Any] | None = None,
    state: dict[str, Any] | None = None,
    language: str = "zh",
) -> str:
    """Build the replanning prompt with failure context.

    Args:
        instruction: Original task instruction.
        failure_history: List of failure records (as dicts).
        causal_outputs: Dict of object_id -> CausalExploreOutput (as dict).
        state: Current world state.
        language: "zh" or "en".
    """
    parts: list[str] = []
    nl_label = "原始任务指令" if language == "zh" else "Original Task Instruction"
    parts.append(f"## {nl_label}\n{instruction}")

    if language == "zh":
        parts.append("## 执行失败记录\n之前的步骤执行失败了。请根据失败原因重新规划：")
    else:
        parts.append("## Execution Failure Records\nPrevious steps failed. Re-plan based on failure reasons:")

    for i, failure in enumerate(failure_history):
        parts.append(_format_failure_record(i, failure, language))

    if causal_outputs:
        header = "## CausalExplore 探索结果" if language == "zh" else "## CausalExplore Results"
        parts.append(header)
        for obj_id, output in causal_outputs.items():
            parts.append(_format_causal_output(obj_id, output, language))

    if state:
        header = "## 当前世界状态" if language == "zh" else "## Current World State"
        parts.append(header)
        parts.append(_format_world_state(state, language))

    req = (
        "请根据失败原因调整规划，输出新的步骤 JSON。如果某个技能参数需要调整，请修改参数后重试。"
        if language == "zh"
        else "Please adjust the plan based on failure reasons and output new step JSON. If a skill parameter needs adjustment, modify it and retry."
    )
    parts.append(req)
    return "\n\n".join(parts)


def _format_causal_output(obj_id: str, output: dict[str, Any], language: str) -> str:
    lines = [f"- **{obj_id}**:"]
    uc = output.get("uncertainty_score", 0.0)
    if language == "zh":
        lines.append(f"  - 不确定性分数: {uc:.2f}")
    else:
        lines.append(f"  - Uncertainty score: {uc:.2f}")

    affordances = output.get("affordance_candidates", [])
    if affordances:
        label = "可供性" if language == "zh" else "Affordances"
        lines.append(f"  - {label}:")
        for aff in affordances:
            name = aff.get("name", "unknown")
            conf = aff.get("confidence", 0.0)
            lines.append(f"    - {name}: {conf:.2f}")

    beliefs = output.get("property_belief", {})
    if beliefs:
        label = "属性信念" if language == "zh" else "Property beliefs"
        lines.append(f"  - {label}:")
        for prop_name, prop_val in beliefs.items():
            if isinstance(prop_val, dict):
                conf = prop_val.get("confidence", 0.0)
            else:
                conf = float(prop_val)
            lines.append(f"    - {prop_name}: {conf:.2f}")

    rec_probe = output.get("recommended_probe")
    if rec_probe:
        prefix = "推荐探索" if language == "zh" else "Recommended probe"
        lines.append(f"  - {prefix}: {rec_probe}")

    contact = output.get("contact_region")
    if contact:
        prefix = "接触区域" if language == "zh" else "Contact region"
        lines.append(f"  - {prefix}: {contact}")

    return "\n".join(lines)


def _format_world_state(state: dict[str, Any], language: str) -> str:
    lines = []
    obj_positions = state.get("object_positions", {})
    if obj_positions:
        label = "物体位置" if language == "zh" else "Object positions"
        lines.append(f"- {label}:")
        for name, pos in obj_positions.items():
            lines.append(f"  - {name}: {pos}")

    zone_positions = state.get("zone_positions", {})
    if zone_positions:
        label = "区域位置" if language == "zh" else "Zone positions"
        lines.append(f"- {label}:")
        for name, pos in zone_positions.items():
            lines.append(f"  - {name}: {pos}")

    held = state.get("held_object_name")
    if held:
        label = "当前持有" if language == "zh" else "Currently holding"
        lines.append(f"- {label}: {held}")

    ee_pos = state.get("end_effector_position")
    if ee_pos:
        label = "末端执行器位置" if language == "zh" else "End effector position"
        lines.append(f"- {label}: {ee_pos}")

    return "\n".join(lines) if lines else ""


def _format_failure_record(index: int, failure: dict[str, Any], language: str) -> str:
    skill = failure.get("selected_skill", "unknown")
    reason = failure.get("reason", "unknown")
    source = failure.get("failure_source", "unknown")
    attempt = failure.get("replan_attempt", 0)
    if language == "zh":
        return (
            f"  {index + 1}. 技能 `{skill}` 失败\n"
            f"     - 失败来源: {source}\n"
            f"     - 原因: {reason}\n"
            f"     - 重规划次数: {attempt}"
        )
    return (
        f"  {index + 1}. Skill `{skill}` failed\n"
        f"     - Source: {source}\n"
        f"     - Reason: {reason}\n"
        f"     - Replan attempt: {attempt}"
    )


def parse_llm_json_response(response_text: str) -> list[dict[str, Any]]:
    """Parse LLM JSON response into list of step dicts.

    Handles both raw JSON and JSON wrapped in markdown code fences.
    """
    text = response_text.strip()
    if "```" in text:
        lines = text.split("\n")
        json_lines = []
        in_block = False
        for line in lines:
            if line.startswith("```"):
                if in_block:
                    break
                in_block = True
                continue
            if in_block:
                json_lines.append(line)
        text = "\n".join(json_lines)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            data = json.loads(text[start:end + 1])
        else:
            raise ValueError(f"Failed to parse LLM JSON response: {text[:200]}...")

    if isinstance(data, list):
        steps = data
    elif isinstance(data, dict) and "steps" in data:
        steps = data["steps"]
    else:
        steps = [data]
    return steps


def validate_step_dict(step: dict[str, Any]) -> None:
    """Validate a step dict conforms to constraints. Raises ValueError on violation."""
    skill = step.get("selected_skill", "")
    if skill not in ALLOWED_SKILLS:
        raise ValueError(
            f"Invalid skill '{skill}'. Allowed: {ALLOWED_SKILLS}"
        )

    skill_args = step.get("skill_args", {})
    forbidden = set(FORBIDDEN_CONTROL_ARGS) & set(skill_args.keys())
    if forbidden:
        raise ValueError(
            f"Step contains forbidden continuous control args: {sorted(forbidden)}"
        )

    if "step_index" not in step:
        raise ValueError("Step missing required field: step_index")
