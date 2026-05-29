from __future__ import annotations

from pathlib import Path
import csv


SUPPORTED_MOTION_TYPES = {
    "forward",
    "left_turn",
    "right_turn",
    "forward_left",
    "forward_right",
    "stop",
    "mixed",
    "unknown",
}


def normalize_motion_type(value: str | None) -> str:
    """Normalize a motion type string to the supported label set."""
    if not value:
        return "unknown"
    normalized = value.strip().lower()
    return normalized if normalized in SUPPORTED_MOTION_TYPES else "unknown"


def infer_motion_type_from_filename(filename: str | Path) -> str:
    """Infer a coarse motion type from a video filename."""
    name = Path(filename).name.lower()

    if "forward_left" in name:
        return "forward_left"
    if "forward_right" in name:
        return "forward_right"
    if "left_turn" in name:
        return "left_turn"
    if "right_turn" in name:
        return "right_turn"
    if "left" in name:
        return "left_turn"
    if "right" in name:
        return "right_turn"
    if "forward" in name:
        return "forward"
    if "stop" in name:
        return "stop"
    if "mixed" in name:
        return "mixed"

    return "unknown"


def load_motion_type_csv(csv_path: str | Path | None) -> dict[str, str]:
    """Load a video_name to motion_type mapping from CSV."""
    if csv_path is None:
        return {}

    path = Path(csv_path)
    if not path.exists():
        return {}

    mapping: dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_name = (row.get("video_name") or "").strip()
            if not video_name:
                continue
            motion_type = normalize_motion_type(row.get("motion_type"))
            mapping[video_name] = motion_type
            mapping[Path(video_name).stem] = motion_type

    return mapping


def resolve_motion_type(video_path: str | Path, motion_type_mapping: dict[str, str] | None = None) -> str:
    """Resolve motion type from CSV mapping first, then filename inference."""
    path = Path(video_path)
    mapping = motion_type_mapping or {}

    for key in (path.name, path.stem):
        if key in mapping:
            return normalize_motion_type(mapping[key])

    return infer_motion_type_from_filename(path.name)
