from __future__ import annotations

from pathlib import Path
from typing import Any

from action_estimation import (
    INSTRUCTION_ACTION_NOTE,
    estimate_actions,
    estimate_instruction_derived_actions,
)
from align_and_trim import compute_navigation_start_time, get_video_duration, trim_navigation_clip
from instruction_analysis import analyze_instruction
from motion_type import load_motion_type_csv, resolve_motion_type
from speech_transcribe import transcribe_video_instruction
from visual_qa import detect_visual_motion


def build_failure_record(
    sample_id: str,
    original_video: str | Path,
    failure_reason: str,
    debug_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a failed sample record matching the project schema."""
    return {
        "sample_id": sample_id,
        "original_video": str(original_video),
        "qa_status": "failed",
        "failure_reason": failure_reason,
        "debug_info": debug_info or {},
    }


def process_video(
    video_path: str | Path,
    output_dir: str | Path,
    model: Any,
    baseline_duration: float = 3.0,
    motion_offset: float = 2.0,
    shake_threshold: float = 55.0,
    min_motion_frames: int = 5,
    min_shake_frames: int = 5,
    min_clip_duration: float = 0.0,
    action_mode: str = "empty",
    motion_type_csv: str | Path | None = None,
    motion_type_mapping: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Process one video into a success or failure dataset record."""
    path = Path(video_path)
    sample_id = path.stem
    output_path = Path(output_dir)
    clips_dir = output_path / "clips"
    temp_dir = output_path / "tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        visual_result = detect_visual_motion(
            path,
            baseline_duration=baseline_duration,
            motion_offset=motion_offset,
            shake_threshold=shake_threshold,
            min_motion_frames=min_motion_frames,
            min_shake_frames=min_shake_frames,
        )

        if visual_result.qa_status != "success":
            return build_failure_record(
                sample_id=sample_id,
                original_video=path,
                failure_reason=visual_result.failure_reason or "visual_qa_failed",
                debug_info=visual_result.debug_info,
            )

        temp_audio_path = temp_dir / f"temp_{sample_id}.wav"
        speech_result = transcribe_video_instruction(
            video_path=path,
            temp_audio_path=temp_audio_path,
            model=model,
        )

        navigation_start_time = compute_navigation_start_time(
            visual_start_time=visual_result.visual_start_time,
            speech_end_time=speech_result.speech_end_time,
        )

        clip_path = clips_dir / f"{sample_id}_clip.mp4"
        trim_navigation_clip(
            video_path=path,
            output_clip_path=clip_path,
            navigation_start_time=navigation_start_time,
        )

        clip_duration = get_video_duration(clip_path)
        if min_clip_duration > 0:
            if clip_duration < min_clip_duration:
                return build_failure_record(
                    sample_id=sample_id,
                    original_video=path,
                    failure_reason="clip_too_short",
                    debug_info={
                        "navigation_clip": str(clip_path),
                        "clip_duration": round(clip_duration, 4),
                        "min_clip_duration": min_clip_duration,
                    },
                )

        mapping = motion_type_mapping if motion_type_mapping is not None else load_motion_type_csv(motion_type_csv)
        motion_type = resolve_motion_type(path, mapping)
        instruction_analysis = analyze_instruction(speech_result.instruction)
        instruction_actions = instruction_analysis["instruction_actions"]

        if action_mode == "instruction_derived":
            action_source = "instruction_derived_pseudo_action"
            if instruction_analysis["instruction_type"] == "low_level_action":
                actions = estimate_instruction_derived_actions(instruction_actions)
                action_note = INSTRUCTION_ACTION_NOTE
            else:
                actions = []
                action_note = "No low-level action labels are generated for goal-level instructions."
        else:
            actions, action_source, action_note = estimate_actions(
                action_mode=action_mode,
                motion_type=motion_type,
                clip_duration=clip_duration,
                navigation_clip=clip_path,
            )

        return {
            "sample_id": sample_id,
            "original_video": str(path),
            "navigation_clip": str(clip_path),
            "instruction": speech_result.instruction,
            "instruction_segments": speech_result.instruction_segments,
            "speech_end_time": round(speech_result.speech_end_time, 2),
            "visual_start_time": round(visual_result.visual_start_time, 2),
            "navigation_start_time": round(navigation_start_time, 2),
            "qa_status": "success",
            "quality_score": {
                "baseline_score": visual_result.baseline_score,
                "dynamic_motion_threshold": visual_result.dynamic_motion_threshold,
                "max_motion_score": visual_result.max_motion_score,
                "shake_threshold": shake_threshold,
            },
            "action_format": "dx_dy_dyaw",
            "actions": actions,
            "instruction_analysis": instruction_analysis,
            "instruction_actions": instruction_actions,
            "action_source": action_source,
            "action_note": action_note,
            "motion_type": motion_type,
            "source": "self_collected",
        }
    except Exception as exc:
        return build_failure_record(
            sample_id=sample_id,
            original_video=path,
            failure_reason=str(exc),
            debug_info={"exception_type": type(exc).__name__},
        )
