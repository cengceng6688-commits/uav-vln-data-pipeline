from __future__ import annotations

from pathlib import Path
import subprocess

import cv2
import numpy as np


def compute_navigation_start_time(visual_start_time: float, speech_end_time: float) -> float:
    """Align speech and motion by starting at the later timestamp."""
    return max(visual_start_time, speech_end_time)


def get_video_duration(video_path: str | Path) -> float:
    """Read video duration in seconds with OpenCV."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()

    if fps == 0 or np.isnan(fps):
        return 0.0
    return float(frame_count / fps)


def trim_navigation_clip(
    video_path: str | Path,
    output_clip_path: str | Path,
    navigation_start_time: float,
) -> Path:
    """Trim a navigation clip with ffmpeg using the same encoding settings as the prototype."""
    output_path = Path(output_clip_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-nostdin",
        "-i",
        str(video_path),
        "-ss",
        str(navigation_start_time),
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        "-map",
        "0:v:0?",
        "-map",
        "0:a:0?",
        str(output_path),
        "-y",
    ]
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0 or not output_path.exists():
        error_tail = result.stderr.strip()[-500:]
        raise RuntimeError(f"ffmpeg trim failed: {error_tail}")

    return output_path
