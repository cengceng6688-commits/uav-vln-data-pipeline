from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from visualize_sample import generate_case_report


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate a single-sample case visualization from a dataset JSONL index."
    )
    parser.add_argument(
        "--dataset_index",
        type=Path,
        required=True,
        help="Path to dataset_index.jsonl.",
    )
    parser.add_argument(
        "--sample_id",
        default=None,
        help="Optional sample_id to visualize. Takes priority over --index.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Zero-based sample index to visualize when --sample_id is not provided.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory for case report files and plots.",
    )
    return parser.parse_args()


def main() -> None:
    """Generate a case visualization."""
    args = parse_args()
    summary = generate_case_report(
        dataset_index=args.dataset_index,
        output_dir=args.output_dir,
        sample_id=args.sample_id,
        index=args.index,
    )

    print(f"[INFO] Case report written to: {args.output_dir}")
    print(f"[INFO] sample_id: {summary.get('sample_id')}")
    print(f"[INFO] qa_status: {summary.get('qa_status')}")


if __name__ == "__main__":
    main()
