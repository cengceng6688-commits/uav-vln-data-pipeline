from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from report import generate_report


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate summary statistics and plots for a dataset JSONL index."
    )
    parser.add_argument(
        "--dataset_index",
        type=Path,
        required=True,
        help="Path to dataset_index.jsonl.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory for summary JSON/CSV files and plots.",
    )
    return parser.parse_args()


def main() -> None:
    """Generate the dataset report."""
    args = parse_args()
    summary = generate_report(args.dataset_index, args.output_dir)

    print(f"[INFO] Report written to: {args.output_dir}")
    print(f"[INFO] Total samples: {summary['total_samples']}")
    print(f"[INFO] Success samples: {summary['success_samples']}")
    print(f"[INFO] Failed samples: {summary['failed_samples']}")
    print(f"[INFO] Success rate: {summary['success_rate']}")


if __name__ == "__main__":
    main()
