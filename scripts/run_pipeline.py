from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from build_dataset import process_video
from motion_type import load_motion_type_csv
from speech_transcribe import load_whisper_model


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Build a VLN-style dataset index from raw navigation videos."
    )
    parser.add_argument("--input_dir", type=Path, required=True, help="Directory containing MP4 videos.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory for clips and dataset index.")
    parser.add_argument("--whisper_model", default="base", help="Whisper model name to load once.")
    parser.add_argument("--motion_offset", type=float, default=2.0, help="Offset added to baseline motion score.")
    parser.add_argument("--shake_threshold", type=float, default=55.0, help="Frame-difference shake threshold.")
    parser.add_argument(
        "--min_clip_duration",
        type=float,
        default=0.0,
        help="Optional minimum clip duration in seconds. Default 0 keeps prototype behavior.",
    )
    parser.add_argument("--baseline_duration", type=float, default=3.0, help="Baseline window in seconds.")
    parser.add_argument("--min_motion_frames", type=int, default=5, help="Consecutive frames for motion start.")
    parser.add_argument("--min_shake_frames", type=int, default=5, help="Consecutive frames for shake failure.")
    parser.add_argument("--max_workers", type=int, default=1, help="Number of videos to process concurrently.")
    parser.add_argument(
        "--action_mode",
        choices=["empty", "heuristic_from_motion_type", "instruction_derived"],
        default="empty",
        help="Action generation mode. Generated actions are pseudo-labels, not odometry.",
    )
    parser.add_argument(
        "--motion_type_csv",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def write_jsonl(records: list[dict], output_path: Path) -> None:
    """Write dataset records as JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    """Run the dataset construction pipeline."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.input_dir.exists():
        record = {
            "sample_id": "input_dir",
            "original_video": str(args.input_dir),
            "qa_status": "failed",
            "failure_reason": "input_dir_not_found",
            "debug_info": {},
        }
        write_jsonl([record], args.output_dir / "dataset_index.jsonl")
        print(f"[ERROR] Input directory not found: {args.input_dir}")
        print(f"[INFO] Failure index written to: {args.output_dir / 'dataset_index.jsonl'}")
        return

    videos = sorted(
        path
        for path in args.input_dir.iterdir()
        if path.suffix.lower() == ".mp4" and not path.name.endswith("_clip.mp4")
    )

    if not videos:
        record = {
            "sample_id": "no_videos",
            "original_video": str(args.input_dir),
            "qa_status": "failed",
            "failure_reason": "no_mp4_videos_found",
            "debug_info": {},
        }
        write_jsonl([record], args.output_dir / "dataset_index.jsonl")
        print(f"[WARNING] No valid .mp4 videos found in: {args.input_dir}")
        print(f"[INFO] Failure index written to: {args.output_dir / 'dataset_index.jsonl'}")
        return

    print(f"[INFO] Found {len(videos)} videos.")
    print(f"[INFO] Loading Whisper model: {args.whisper_model}")
    model = load_whisper_model(args.whisper_model)
    motion_type_mapping = load_motion_type_csv(args.motion_type_csv)
    if args.motion_type_csv:
        print(f"[INFO] Loaded {len(motion_type_mapping)} motion type keys from: {args.motion_type_csv}")

    records: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(
                process_video,
                video,
                args.output_dir,
                model,
                args.baseline_duration,
                args.motion_offset,
                args.shake_threshold,
                args.min_motion_frames,
                args.min_shake_frames,
                args.min_clip_duration,
                args.action_mode,
                args.motion_type_csv,
                motion_type_mapping,
            ): video
            for video in videos
        }

        for future in as_completed(futures):
            video = futures[future]
            record = future.result()
            records.append(record)
            print(f"[INFO] {video.name}: {record['qa_status']}")

    records.sort(key=lambda item: item["sample_id"])
    index_path = args.output_dir / "dataset_index.jsonl"
    write_jsonl(records, index_path)
    print(f"[INFO] Dataset index written to: {index_path}")


if __name__ == "__main__":
    main()
