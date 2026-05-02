# 系统架构图描述

## 1. 高层架构概览

```
                          ┌──────────────────────────────────┐
                          │         User / Environment         │
                          │   Natural Language Instruction    │
                          └──────────────┬───────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        Embodied Agent System                         │
│                                                                      │
│  ┌───────────────────┐  ┌───────────────────┐  ┌──────────────────┐ │
│  │   CausalExplore   │  │    LLM Planner    │  │  Skill Executor  │ │
│  │     (Phase 1)     │  │     (Phase 2)     │  │    (Core)        │ │
│  │                   │  │                   │  │                  │ │
│  │ • Probe Actions   │  │ • Prompt Templates│  │ • Pick/Place     │ │
│  │ • Explore Strategy│  │ • LLM Callable    │  │ • Press/Push     │ │
│  │ • Probe Executor  │  │ • Uncertainty H.  │  │ • Pull/Rotate    │ │
│  │ • Eval Runner     │  │ • Replan Handler  │  │ • Recovery       │ │
│  └────────┬──────────┘  └────────┬──────────┘  └────────┬─────────┘ │
│           │                      │                       │           │
│           └──────────────────────┼───────────────────────┘           │
│                                  │                                    │
│                    ┌─────────────┴─────────────┐                     │
│                    │   Contracts & Bridge       │                     │
│                    │ • ContractPlanningBridge   │                     │
│                    │ • ContractPlannerAdapter   │                     │
│                    │ • CausalOutputProvider     │                     │
│                    └─────────────┬─────────────┘                     │
│                                  │                                    │
│                    ┌─────────────┴─────────────┐                     │
│                    │   Simulator (MuJoCo)       │                     │
│                    │ • Pick/Place Simulation    │                     │
│                    │ • Fast Reset               │                     │
│                    │ • Object/Zone State        │                     │
│                    └───────────────────────────┘                     │
└─────────────────────────────────────────────────────────────────────┘
```

## 2. CausalExplore 模块详细架构

```
┌─────────────────────────────────────────────────────┐
│                 CausalExplore Module                 │
│                                                      │
│  ObjectManifest ──────────────────────┐              │
│    • object_id                        │              │
│    • expected_properties              │              │
│    • candidate_affordances            ▼              │
│                                                      │
│  ┌──────────────────────────────────────────────┐   │
│  │         ExploreStrategy (Abstract)            │   │
│  │  select_next(history, probes, objects)        │   │
│  │                                               │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────────┐  │   │
│  │  │ Random   │ │Curiosity │ │ CausalExplore│  │   │
│  │  │ uniform  │ │displace- │ │ uncertainty  │  │   │
│  │  │ sampling │ │ment-based│ │ -based       │  │   │
│  │  └──────────┘ └──────────┘ └──────────────┘  │   │
│  └──────────────────────┬───────────────────────┘   │
│                         │                            │
│  ┌──────────────────────▼───────────────────────┐   │
│  │           ProbeExecutor                      │   │
│  │  execute_probe(name, object) → Result        │   │
│  │  build_causal_output(manifest, results)      │   │
│  │  save_artifact(output, results)              │   │
│  └──────────────────────┬───────────────────────┘   │
│                         │                            │
│  ┌──────────────────────▼───────────────────────┐   │
│  │           Probe Actions                       │   │
│  │  lateral_push | top_press | side_pull        │   │
│  │  surface_tap  | grasp_attempt                │   │
│  └──────────────────────────────────────────────┘   │
│                                                      │
│  Output: CausalExploreOutput                         │
│    • PropertyBelief (confidence per property)        │
│    • AffordanceCandidate (confidence per affordance) │
│    • uncertainty_score                               │
│    • recommended_probe                               │
└─────────────────────────────────────────────────────┘
```

## 3. LLM Planner 模块详细架构

```
┌──────────────────────────────────────────────────────┐
│                  LLM Planner Module                   │
│                                                       │
│  ┌─────────────────────────────────────────────┐     │
│  │         Prompt Templates                      │     │
│  │  build_system_prompt(language)                │     │
│  │  build_task_prompt(instruction, causal, state)│     │
│  │  build_replan_prompt(instruction, failure)    │     │
│  │  parse_llm_json_response(response)            │     │
│  │  validate_step_dict(step)                     │     │
│  └──────────────────┬──────────────────────────┘     │
│                     │                                  │
│  ┌──────────────────▼──────────────────────────┐     │
│  │           LLMPlanner                          │     │
│  │  plan(instruction, state) → list[PlanStep]    │     │
│  │  replan(instruction, state, failure)          │     │
│  │  update_causal_outputs(outputs)               │     │
│  └──────────────────┬──────────────────────────┘     │
│                     │                                  │
│  ┌──────────────────▼──────────────────────────┐     │
│  │         Support Handlers                      │     │
│  │  UncertaintyHandler.evaluate(causal_output)   │     │
│  │  ReplanHandler (MAX_REPLANS, retries)         │     │
│  └──────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────┘
```

## 4. 数据流图（时序视角）

```
  User          LLM Planner      CausalExplore     Simulator      Executor
   │                │                 │                │              │
   │  instruction   │                 │                │              │
   │───────────────▶│                 │                │              │
   │                │                 │                │              │
   │                │  explore scene  │                │              │
   │                │────────────────▶│                │              │
   │                │                 │ execute probes │              │
   │                │                 │───────────────▶│              │
   │                │                 │◀───────────────│              │
   │                │                 │                │              │
   │                │ CausalExploreOutput              │              │
   │                │◀────────────────│                │              │
   │                │                 │                │              │
   │                │ plan(instruction, causal_outputs)│              │
   │                │────────┐        │                │              │
   │                │◀───────┘        │                │              │
   │                │                 │                │              │
   │                │    inject probe if needed        │              │
   │                │────────────────────────────────────────────────▶│
   │                │                 │                │  execute     │
   │                │                 │                │◀─────────────│
   │                │                 │                │──────────────│
   │                │                 │                │              │
   │                │  ExecutorResult │                │              │
   │                │◀────────────────────────────────────────────────│
   │                │                 │                │              │
   │                │  [if failure] replan              │              │
   │                │────────┐        │                │              │
   │                │◀───────┘        │                │              │
   │                │                 │                │              │
   │  task complete │                 │                │              │
   │◀───────────────│                 │                │              │
```

## 5. 实验框架架构

```
┌────────────────────────────────────────────────────────┐
│                  Experiment Framework                    │
│                                                         │
│  ┌─────────────────────┐  ┌──────────────────────────┐ │
│  │ Comparative Exp.     │  │ Ablation Exp.             │ │
│  │                      │  │                           │ │
│  │ Conditions:          │  │ Dimensions:               │ │
│  │ • NO_CAUSAL          │  │ • Strategy (3)            │ │
│  │ • METADATA_BACKED    │  │ • Uncertainty (2)         │ │
│  │ • SIMULATOR_BACKED   │  │ • Recovery (2)            │ │
│  │                      │  │                           │ │
│  │ Metrics:             │  │ Total: 12 conditions      │ │
│  │ • success_rate       │  │                           │ │
│  │ • explore_steps      │  │ Metrics same as left      │ │
│  │ • planning_quality   │  │ + uncertainty_score       │ │
│  │ • replan_count       │  │ + recovery_count          │ │
│  └──────────┬───────────┘  └────────────┬─────────────┘ │
│             │                            │               │
│             └──────────┬─────────────────┘               │
│                        ▼                                 │
│  ┌──────────────────────────────────────────────┐       │
│  │            Reporting Module                    │       │
│  │  ExperimentReporter → Markdown tables         │       │
│  │  ChartGenerator → matplotlib charts          │       │
│  └──────────────────────────────────────────────┘       │
│                                                         │
│  Outputs:                                               │
│  • experiments/*.json (structured results)              │
│  • reports/*.md (Markdown reports)                      │
│  • reports/*.png (charts)                               │
│  • paper/*.md (paper draft + presentation)              │
└────────────────────────────────────────────────────────┘
```
