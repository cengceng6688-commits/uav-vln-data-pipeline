# AGENTS.md

This repository is a research prototype for constructing a real-world vision-language navigation dataset from videos.

Each raw video contains:
1. A spoken navigation/action instruction at the beginning.
2. A corresponding movement segment after the instruction.

The goal is to process raw videos into structured VLN-style dataset samples with:
- navigation video clip
- transcribed instruction
- visual QA status
- speech-motion alignment timestamps
- quality scores
- action sequence in dx, dy, dyaw format
- metadata for research analysis

Important constraints:
1. Do not exaggerate the accuracy of pseudo-actions.
2. dx, dy, dyaw actions are pseudo-labels unless ground-truth odometry is available.
3. The current pipeline is for dataset construction and midterm research demonstration.
4. UAV fine-tuning or pi0 fine-tuning is optional downstream work, not the core current task.
5. Keep the code modular and easy to run on small local datasets.

Coding requirements:
- Use Python.
- Prefer clear, maintainable code over complex abstractions.
- Use argparse for scripts.
- Write JSONL dataset indexes.
- Use pathlib for paths.
- Add type hints where reasonable.
- Add docstrings for public functions.
- Handle errors gracefully and write failure records.
- Do not hard-code absolute paths.
- Keep heavy model loading, such as Whisper, outside tight loops when possible.
- Use matplotlib for plots.
- Do not use seaborn.

Success sample schema:
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
  "quality_score": {
    "baseline_score": 0.0,
    "dynamic_motion_threshold": 0.0,
    "max_motion_score": 0.0,
    "shake_threshold": 55.0
  },
  "action_format": "dx_dy_dyaw",
  "actions": [],
  "motion_type": "unknown",
  "source": "self_collected"
}

Failed sample schema:
{
  "sample_id": "...",
  "original_video": "...",
  "qa_status": "failed",
  "failure_reason": "...",
  "debug_info": {}
}