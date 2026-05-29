# Dataset Schema

## Overview

The dataset is represented as JSONL. Each line in `dataset_index.jsonl` is one processed sample record. The current data is goal-level navigation video data: each successful sample links a spoken goal-level instruction to a trimmed navigation clip and associated metadata.

The current midterm version does not provide ground-truth low-level dx, dy, dyaw control labels. The action interface is retained for future extensions and for clearly marked pseudo-label experiments.

## Success Sample Schema

```json
{
  "sample_id": "...",
  "original_video": "...",
  "navigation_clip": "...",
  "instruction": "...",
  "instruction_segments": [],
  "speech_end_time": 0.0,
  "visual_start_time": 0.0,
  "navigation_start_time": 0.0,
  "qa_status": "success",
  "quality_score": {},
  "instruction_analysis": {},
  "action_format": "dx_dy_dyaw",
  "actions": [],
  "action_source": "empty",
  "action_note": "...",
  "motion_type": "unknown",
  "source": "self_collected"
}
```

Important fields:

| Field | Description |
| --- | --- |
| `sample_id` | Stable sample identifier, usually derived from the original video filename. |
| `original_video` | Path to the raw or converted source video. |
| `navigation_clip` | Path to the trimmed navigation clip after speech-motion alignment. |
| `instruction` | Whisper-transcribed spoken instruction. |
| `instruction_segments` | Whisper segment metadata used for speech timing analysis. |
| `speech_end_time` | Estimated end time of the spoken instruction. |
| `visual_start_time` | Estimated start time of visible motion from frame-difference QA. |
| `navigation_start_time` | Start time used for trimming, currently the later of speech end and visual motion start. |
| `qa_status` | Processing status, usually `success` or `failed`. |
| `quality_score` | Visual QA metrics and thresholds. |
| `instruction_analysis` | Goal-level versus low-level instruction analysis. |
| `action_format` | Reserved action representation name, currently `dx_dy_dyaw`. |
| `actions` | Empty or pseudo-action list in the current prototype; not ground-truth action labels. |
| `action_source` | Source of action values, such as `empty` or a pseudo-label method. |
| `action_note` | Explanation of action-label limitations. |
| `motion_type` | Legacy coarse motion type field, currently not the main supervision target. |
| `source` | Data source label, currently `self_collected`. |

## Failure Sample Schema

```json
{
  "sample_id": "...",
  "original_video": "...",
  "qa_status": "failed",
  "failure_reason": "...",
  "debug_info": {}
}
```

Failure records are written when a sample cannot be processed successfully. The `failure_reason` field should be concise and machine-readable where possible. The `debug_info` field can store additional diagnostic context.

## Instruction Analysis Schema

```json
{
  "instruction_type": "goal_level",
  "instruction_actions": [],
  "goal_keywords": [],
  "goal_description": "...",
  "analysis_note": "..."
}
```

Fields:

| Field | Description |
| --- | --- |
| `instruction_type` | Coarse instruction category, such as `goal_level`, `low_level_action`, `goal_level_with_action_hint`, or `unknown`. |
| `instruction_actions` | Parsed low-level action hints when present. These are language-derived and should not be treated as measured control signals. |
| `goal_keywords` | Goal or task keywords detected in the instruction. |
| `goal_description` | Normalized or copied goal-level instruction description. |
| `analysis_note` | Short explanation of the classification result. |

## Quality Score Schema

```json
{
  "baseline_score": 0.0,
  "dynamic_motion_threshold": 0.0,
  "max_motion_score": 0.0,
  "shake_threshold": 55.0
}
```

Fields:

| Field | Description |
| --- | --- |
| `baseline_score` | Baseline frame-difference score estimated from the initial stationary window. |
| `dynamic_motion_threshold` | Motion threshold derived from the baseline score and configured offset. |
| `max_motion_score` | Maximum frame-difference score observed during visual QA. |
| `shake_threshold` | Threshold used to flag severe camera shake. |

## Action Interface Note

The `action_format` and `actions` fields are included to keep the schema extensible. They are not ground-truth low-level control labels in the current midterm version.

Future action labels would require stronger motion sources, such as visual odometry, SLAM, synchronized sensor logs, or UAV telemetry. Any pseudo-actions generated before those sources are available must be clearly documented as non-ground-truth.
