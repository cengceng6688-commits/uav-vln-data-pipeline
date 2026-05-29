from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import cv2

from instruction_analysis import analyze_instruction
from report import _get_pyplot, _to_float, load_jsonl


def select_sample(
    records: list[dict[str, Any]],
    sample_id: str | None = None,
    index: int = 0,
) -> dict[str, Any]:
    """Select one sample by sample_id or zero-based index."""
    if not records:
        raise ValueError("dataset index contains no records")

    if sample_id:
        for record in records:
            if record.get("sample_id") == sample_id:
                return record
        raise ValueError(f"sample_id not found: {sample_id}")

    if index < 0 or index >= len(records):
        raise IndexError(f"index out of range: {index}; dataset has {len(records)} records")

    return records[index]


def _resolve_path(path_value: Any, dataset_index: Path) -> Path | None:
    if not path_value:
        return None

    path = Path(str(path_value))
    if path.exists():
        return path

    candidate = dataset_index.parent / path
    if candidate.exists():
        return candidate

    return path


def build_case_summary(record: dict[str, Any], dataset_index: Path) -> dict[str, Any]:
    """Build a compact case summary for JSON and Markdown outputs."""
    quality_score = record.get("quality_score")
    if not isinstance(quality_score, dict):
        quality_score = {}

    navigation_clip = _resolve_path(record.get("navigation_clip"), dataset_index)
    original_video = _resolve_path(record.get("original_video"), dataset_index)
    instruction_analysis = record.get("instruction_analysis")
    if not isinstance(instruction_analysis, dict):
        instruction_analysis = analyze_instruction(str(record.get("instruction") or ""))

    return {
        "sample_id": record.get("sample_id"),
        "qa_status": record.get("qa_status"),
        "failure_reason": record.get("failure_reason"),
        "original_video": str(original_video) if original_video else None,
        "navigation_clip": str(navigation_clip) if navigation_clip else None,
        "instruction": record.get("instruction"),
        "speech_end_time": record.get("speech_end_time"),
        "visual_start_time": record.get("visual_start_time"),
        "navigation_start_time": record.get("navigation_start_time"),
        "quality_score": {
            "baseline_score": quality_score.get("baseline_score"),
            "dynamic_motion_threshold": quality_score.get("dynamic_motion_threshold"),
            "max_motion_score": quality_score.get("max_motion_score"),
            "shake_threshold": quality_score.get("shake_threshold"),
        },
        "action_format": record.get("action_format"),
        "action_source": record.get("action_source"),
        "action_note": record.get("action_note"),
        "instruction_analysis": instruction_analysis,
        "instruction_type": instruction_analysis.get("instruction_type"),
        "goal_keywords": instruction_analysis.get("goal_keywords", []),
        "goal_description": instruction_analysis.get("goal_description"),
        "analysis_note": instruction_analysis.get("analysis_note"),
        "instruction_actions": instruction_analysis.get(
            "instruction_actions", record.get("instruction_actions", [])
        ),
        "actions_count": len(record.get("actions", [])) if isinstance(record.get("actions"), list) else None,
        "motion_type": record.get("motion_type"),
        "source": record.get("source"),
        "debug_info": record.get("debug_info", {}),
    }


def write_case_summary_json(summary: dict[str, Any], output_path: str | Path) -> None:
    """Write case summary JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def write_case_summary_md(summary: dict[str, Any], output_path: str | Path) -> None:
    """Write a Markdown case report for presentation review."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    quality_score = summary.get("quality_score") or {}
    debug_info = summary.get("debug_info") or {}
    instruction_actions = summary.get("instruction_actions")
    if not isinstance(instruction_actions, list):
        instruction_actions = []
    goal_keywords = summary.get("goal_keywords")
    if not isinstance(goal_keywords, list):
        goal_keywords = []

    lines = [
        f"# Case Report: {summary.get('sample_id')}",
        "",
        "## Status",
        "",
        f"- qa_status: {summary.get('qa_status')}",
        f"- failure_reason: {summary.get('failure_reason')}",
        "",
        "## Instruction",
        "",
        summary.get("instruction") or "",
        "",
        "## Instruction Analysis",
        "",
        f"- instruction_type: {summary.get('instruction_type')}",
        f"- goal_keywords: {', '.join(str(keyword) for keyword in goal_keywords) if goal_keywords else None}",
        f"- goal_description: {summary.get('goal_description')}",
        f"- analysis_note: {summary.get('analysis_note')}",
        "",
        "## Timing",
        "",
        f"- speech_end_time: {summary.get('speech_end_time')}",
        f"- visual_start_time: {summary.get('visual_start_time')}",
        f"- navigation_start_time: {summary.get('navigation_start_time')}",
        "",
        "## Files",
        "",
        f"- original_video: {summary.get('original_video')}",
        f"- navigation_clip: {summary.get('navigation_clip')}",
        "",
        "## Quality Score",
        "",
        f"- baseline_score: {quality_score.get('baseline_score')}",
        f"- dynamic_motion_threshold: {quality_score.get('dynamic_motion_threshold')}",
        f"- max_motion_score: {quality_score.get('max_motion_score')}",
        f"- shake_threshold: {quality_score.get('shake_threshold')}",
        "",
        "## Actions",
        "",
        f"- action_format: {summary.get('action_format')}",
        f"- action_source: {summary.get('action_source')}",
        f"- action_note: {summary.get('action_note')}",
        f"- actions_count: {summary.get('actions_count')}",
        f"- motion_type: {summary.get('motion_type')}",
        "",
        "## Instruction Actions",
        "",
    ]

    if instruction_actions:
        lines.extend(
            [
                "```json",
                json.dumps(instruction_actions, ensure_ascii=False, indent=2),
                "```",
                "",
            ]
        )
    else:
        lines.extend(["No instruction actions were parsed.", ""])

    if summary.get("action_source") in {
        "heuristic_pseudo_label",
        "optical_flow_pseudo_label",
        "instruction_derived_pseudo_action",
    }:
        lines.extend(
            [
                "The action curve is generated from pseudo-actions. It is not ground-truth odometry.",
                "",
            ]
        )

    if debug_info:
        lines.extend(
            [
                "## Debug Info",
                "",
                "```json",
                json.dumps(debug_info, ensure_ascii=False, indent=2),
                "```",
                "",
            ]
        )

    path.write_text("\n".join(lines), encoding="utf-8")


def append_empty_action_note(summary_path: str | Path) -> None:
    """Append a short note when a sample has no actions."""
    path = Path(summary_path)
    with path.open("a", encoding="utf-8") as f:
        f.write("\nActions are empty for this sample.\n")


def _write_placeholder_image(output_path: Path, message: str) -> None:
    plt = _get_pyplot()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 3))
    plt.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_sample_keyframes(
    record: dict[str, Any],
    dataset_index: str | Path,
    output_path: str | Path,
    num_frames: int = 4,
) -> None:
    """Extract evenly spaced keyframes from the navigation clip and save a grid image."""
    output = Path(output_path)
    index_path = Path(dataset_index)
    clip_path = _resolve_path(record.get("navigation_clip"), index_path)

    if not clip_path or not clip_path.exists():
        _write_placeholder_image(output, "Navigation clip not available for this sample.")
        return

    cap = cv2.VideoCapture(str(clip_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if total_frames <= 0:
        cap.release()
        _write_placeholder_image(output, "Could not read frames from navigation clip.")
        return

    if num_frames == 1:
        frame_indices = [total_frames // 2]
    else:
        frame_indices = [
            round(i * (total_frames - 1) / (num_frames - 1)) for i in range(num_frames)
        ]

    frames: list[tuple[int, Any]] = []
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append((frame_idx, frame_rgb))

    cap.release()

    if not frames:
        _write_placeholder_image(output, "Could not extract keyframes from navigation clip.")
        return

    plt = _get_pyplot()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(frames), figsize=(4 * len(frames), 3))
    if len(frames) == 1:
        axes = [axes]

    for axis, (frame_idx, frame_rgb) in zip(axes, frames):
        axis.imshow(frame_rgb)
        timestamp = frame_idx / fps if fps else 0.0
        axis.set_title(f"{timestamp:.2f}s")
        axis.axis("off")

    fig.suptitle(f"Keyframes: {record.get('sample_id')}")
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def plot_timing_case(record: dict[str, Any], output_path: str | Path) -> bool:
    """Plot speech, visual, and navigation timestamps for one sample if available."""
    timing_values = [
        ("speech_end_time", _to_float(record.get("speech_end_time"))),
        ("visual_start_time", _to_float(record.get("visual_start_time"))),
        ("navigation_start_time", _to_float(record.get("navigation_start_time"))),
    ]
    timing_values = [(label, value) for label, value in timing_values if value is not None]

    if not timing_values:
        return False

    plt = _get_pyplot()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    labels = [label for label, _ in timing_values]
    values = [value for _, value in timing_values]

    plt.figure(figsize=(8, 3))
    plt.barh(labels, values)
    plt.title(f"Timing: {record.get('sample_id')}")
    plt.xlabel("Time (seconds)")
    plt.tight_layout()
    plt.savefig(output)
    plt.close()
    return True


def plot_action_curve(record: dict[str, Any], output_path: str | Path) -> bool:
    """Plot dx, dy, and dyaw over time when pseudo-actions are available."""
    actions = record.get("actions")
    if not isinstance(actions, list) or not actions:
        return False

    rows: list[tuple[float, float, float, float]] = []
    for action in actions:
        if not isinstance(action, dict):
            continue
        t = _to_float(action.get("t"))
        dx = _to_float(action.get("dx"))
        dy = _to_float(action.get("dy"))
        dyaw = _to_float(action.get("dyaw"))
        if t is None or dx is None or dy is None or dyaw is None:
            continue
        rows.append((t, dx, dy, dyaw))

    if not rows:
        return False

    rows.sort(key=lambda item: item[0])
    times = [row[0] for row in rows]
    dx_values = [row[1] for row in rows]
    dy_values = [row[2] for row in rows]
    dyaw_values = [row[3] for row in rows]

    plt = _get_pyplot()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 4))
    plt.plot(times, dx_values, marker="o", label="dx")
    plt.plot(times, dy_values, marker="o", label="dy")
    plt.plot(times, dyaw_values, marker="o", label="dyaw")
    plt.title(f"Pseudo-Action Curve: {record.get('sample_id')}")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Pseudo-action value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output)
    plt.close()
    return True


def generate_case_report(
    dataset_index: str | Path,
    output_dir: str | Path,
    sample_id: str | None = None,
    index: int = 0,
) -> dict[str, Any]:
    """Generate JSON, Markdown, keyframes, and optional timing plot for one sample."""
    index_path = Path(dataset_index)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    records = load_jsonl(index_path)
    record = select_sample(records, sample_id=sample_id, index=index)
    summary = build_case_summary(record, index_path)

    write_case_summary_json(summary, output_path / "case_summary.json")
    write_case_summary_md(summary, output_path / "case_summary.md")
    plot_sample_keyframes(record, index_path, output_path / "sample_keyframes.png")

    timing_plot_created = plot_timing_case(record, output_path / "timing_plot.png")
    action_curve_created = plot_action_curve(record, output_path / "action_curve.png")
    if not action_curve_created:
        append_empty_action_note(output_path / "case_summary.md")

    summary["generated_files"] = {
        "case_summary_json": str(output_path / "case_summary.json"),
        "case_summary_md": str(output_path / "case_summary.md"),
        "sample_keyframes": str(output_path / "sample_keyframes.png"),
        "timing_plot": str(output_path / "timing_plot.png") if timing_plot_created else None,
        "action_curve": str(output_path / "action_curve.png") if action_curve_created else None,
    }
    write_case_summary_json(summary, output_path / "case_summary.json")

    return summary
