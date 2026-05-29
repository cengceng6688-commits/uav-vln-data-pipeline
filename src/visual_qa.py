from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass
class VisualQAResult:
    """Result from frame-difference based visual QA."""

    qa_status: str
    failure_reason: str | None
    visual_start_time: float
    baseline_score: float
    dynamic_motion_threshold: float
    max_motion_score: float
    debug_info: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""
        return asdict(self)


def detect_visual_motion(
    video_path: str | Path,
    baseline_duration: float = 3.0,
    motion_offset: float = 2.0,
    shake_threshold: float = 55.0,
    min_motion_frames: int = 5,
    min_shake_frames: int = 5,
) -> VisualQAResult:
    """Detect motion start and visual QA status using the original frame-difference logic."""
    path = Path(video_path)
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)

    debug_info: dict[str, Any] = {
        "video_path": str(path),
        "baseline_duration": baseline_duration,
        "motion_offset": motion_offset,
        "shake_threshold": shake_threshold,
        "min_motion_frames": min_motion_frames,
        "min_shake_frames": min_shake_frames,
    }

    if fps == 0 or np.isnan(fps):
        cap.release()
        debug_info["fps"] = fps
        return VisualQAResult(
            qa_status="failed",
            failure_reason="error",
            visual_start_time=0.0,
            baseline_score=0.0,
            dynamic_motion_threshold=0.0,
            max_motion_score=0.0,
            debug_info=debug_info,
        )

    ret, frame1 = cap.read()
    if not ret:
        cap.release()
        debug_info["fps"] = fps
        return VisualQAResult(
            qa_status="failed",
            failure_reason="error",
            visual_start_time=0.0,
            baseline_score=0.0,
            dynamic_motion_threshold=0.0,
            max_motion_score=0.0,
            debug_info=debug_info,
        )

    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    start_frame = -1
    motion_streak = 0
    shake_streak = 0
    frame_idx = 1
    status = "no_motion"

    baseline_scores: list[float] = []
    baseline_score = 0.0
    dynamic_motion_threshold: float | None = None
    max_motion_score = 0.0
    baseline_frame_count = int(fps * baseline_duration)

    debug_info.update(
        {
            "fps": float(fps),
            "baseline_frame_count": baseline_frame_count,
        }
    )

    while True:
        ret, frame2 = cap.read()
        if not ret:
            break

        next_frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(next_frame, prvs)
        score = float(np.mean(diff))
        max_motion_score = max(max_motion_score, score)

        if frame_idx <= baseline_frame_count:
            baseline_scores.append(score)
            if frame_idx == baseline_frame_count:
                baseline_score = float(np.percentile(baseline_scores, 25))
                dynamic_motion_threshold = baseline_score + motion_offset
        else:
            if score > shake_threshold:
                shake_streak += 1
                if shake_streak >= min_shake_frames:
                    cap.release()
                    debug_info.update(
                        {
                            "last_frame_idx": frame_idx,
                            "shake_streak": shake_streak,
                            "motion_streak": motion_streak,
                        }
                    )
                    return VisualQAResult(
                        qa_status="failed",
                        failure_reason="shaking",
                        visual_start_time=0.0,
                        baseline_score=round(baseline_score, 4),
                        dynamic_motion_threshold=round(dynamic_motion_threshold or 0.0, 4),
                        max_motion_score=round(max_motion_score, 4),
                        debug_info=debug_info,
                    )
            else:
                shake_streak = 0

            if start_frame == -1 and dynamic_motion_threshold is not None:
                if score > dynamic_motion_threshold:
                    motion_streak += 1
                    if motion_streak >= min_motion_frames:
                        start_frame = frame_idx - min_motion_frames + 1
                        status = "success"
                else:
                    motion_streak = 0

        prvs = next_frame
        frame_idx += 1

    cap.release()
    debug_info.update(
        {
            "last_frame_idx": frame_idx,
            "motion_streak": motion_streak,
            "shake_streak": shake_streak,
        }
    )

    if status == "success":
        return VisualQAResult(
            qa_status="success",
            failure_reason=None,
            visual_start_time=round(start_frame / fps, 4),
            baseline_score=round(baseline_score, 4),
            dynamic_motion_threshold=round(dynamic_motion_threshold or 0.0, 4),
            max_motion_score=round(max_motion_score, 4),
            debug_info=debug_info,
        )

    return VisualQAResult(
        qa_status="failed",
        failure_reason="no_motion",
        visual_start_time=0.0,
        baseline_score=round(baseline_score, 4),
        dynamic_motion_threshold=round(dynamic_motion_threshold or 0.0, 4),
        max_motion_score=round(max_motion_score, 4),
        debug_info=debug_info,
    )
