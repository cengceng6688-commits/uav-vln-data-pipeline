from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from asr_eval import evaluate_asr, write_asr_eval_outputs


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate Whisper transcription quality with sampled ground-truth transcripts."
    )
    parser.add_argument(
        "--dataset_index",
        type=Path,
        required=True,
        help="Path to dataset_index.jsonl.",
    )
    parser.add_argument(
        "--ground_truth_csv",
        type=Path,
        required=True,
        help="CSV with columns sample_id,ground_truth_text.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory for ASR evaluation outputs.",
    )
    return parser.parse_args()


def main() -> None:
    """Run ASR evaluation."""
    args = parse_args()
    summary, details = evaluate_asr(args.dataset_index, args.ground_truth_csv)
    summary_path, details_path = write_asr_eval_outputs(summary, details, args.output_dir)

    print(f"[INFO] ASR summary written to: {summary_path}")
    print(f"[INFO] ASR details written to: {details_path}")
    print(f"[INFO] num_evaluated: {summary['num_evaluated']}")
    print(f"[INFO] average_cer: {summary['average_cer']}")


if __name__ == "__main__":
    main()
