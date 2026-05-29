from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any
import csv
import json
import os
import tempfile


def load_jsonl(dataset_index: str | Path) -> list[dict[str, Any]]:
    """Load dataset records from a JSONL index, skipping invalid blank lines."""
    records: list[dict[str, Any]] = []
    path = Path(dataset_index)

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError:
                records.append(
                    {
                        "sample_id": f"invalid_line_{line_number}",
                        "qa_status": "failed",
                        "failure_reason": "invalid_jsonl_record",
                        "debug_info": {"line_number": line_number},
                    }
                )
                continue
            if isinstance(record, dict):
                records.append(record)

    return records


def _get_pyplot() -> Any:
    cache_dir = Path(tempfile.gettempdir()) / "uav_vln_matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def _numeric_field(records: list[dict[str, Any]], field_name: str) -> list[float]:
    values: list[float] = []
    for record in records:
        value = _to_float(record.get(field_name))
        if value is not None:
            values.append(value)
    return values


def _quality_field(records: list[dict[str, Any]], field_name: str) -> list[float]:
    values: list[float] = []
    for record in records:
        quality_score = record.get("quality_score")
        if not isinstance(quality_score, dict):
            continue
        value = _to_float(quality_score.get(field_name))
        if value is not None:
            values.append(value)
    return values


def _instruction_type_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in records:
        instruction_analysis = record.get("instruction_analysis")
        if isinstance(instruction_analysis, dict):
            instruction_type = instruction_analysis.get("instruction_type")
        else:
            instruction_type = record.get("instruction_type")
        counts[str(instruction_type or "unknown")] += 1
    return dict(counts)


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute dataset summary statistics with missing-field tolerance."""
    total_samples = len(records)
    success_records = [record for record in records if record.get("qa_status") == "success"]
    failed_records = [record for record in records if record.get("qa_status") == "failed"]
    success_samples = len(success_records)
    failed_samples = len(failed_records)
    success_rate = round(success_samples / total_samples, 4) if total_samples else 0.0

    failure_reason_counts = Counter(
        str(record.get("failure_reason") or "unknown") for record in failed_records
    )

    speech_end_times = _numeric_field(success_records, "speech_end_time")
    visual_start_times = _numeric_field(success_records, "visual_start_time")
    navigation_start_times = _numeric_field(success_records, "navigation_start_time")
    max_motion_scores = _quality_field(success_records, "max_motion_score")
    baseline_scores = _quality_field(success_records, "baseline_score")

    return {
        "total_samples": total_samples,
        "success_samples": success_samples,
        "failed_samples": failed_samples,
        "success_rate": success_rate,
        "failure_reason_counts": dict(failure_reason_counts),
        "instruction_type_counts": _instruction_type_counts(success_records),
        "average_speech_end_time": _mean(speech_end_times),
        "average_visual_start_time": _mean(visual_start_times),
        "average_navigation_start_time": _mean(navigation_start_times),
        "average_max_motion_score": _mean(max_motion_scores),
        "average_baseline_score": _mean(baseline_scores),
    }


def write_summary_json(summary: dict[str, Any], output_path: str | Path) -> None:
    """Write summary statistics to JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def write_summary_csv(summary: dict[str, Any], output_path: str | Path) -> None:
    """Write summary statistics to a two-column CSV."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key, value in summary.items():
            if isinstance(value, dict):
                writer.writerow([key, json.dumps(value, ensure_ascii=False)])
            else:
                writer.writerow([key, value])


def plot_status_distribution(records: list[dict[str, Any]], output_path: str | Path) -> None:
    """Plot success/failed sample counts."""
    plt = _get_pyplot()
    counts = Counter(str(record.get("qa_status") or "unknown") for record in records)
    labels = list(counts.keys()) or ["none"]
    values = [counts[label] for label in labels] or [0]

    plt.figure(figsize=(6, 4))
    plt.bar(labels, values)
    plt.title("Status Distribution")
    plt.xlabel("QA Status")
    plt.ylabel("Samples")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_failure_reason_distribution(records: list[dict[str, Any]], output_path: str | Path) -> None:
    """Plot failure reason counts."""
    plt = _get_pyplot()
    failed_records = [record for record in records if record.get("qa_status") == "failed"]
    counts = Counter(str(record.get("failure_reason") or "unknown") for record in failed_records)
    labels = list(counts.keys()) or ["no_failures"]
    values = [counts[label] for label in labels] or [0]

    plt.figure(figsize=(8, 4))
    plt.bar(labels, values)
    plt.title("Failure Reason Distribution")
    plt.xlabel("Failure Reason")
    plt.ylabel("Samples")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_motion_score_distribution(records: list[dict[str, Any]], output_path: str | Path) -> None:
    """Plot distributions of max motion score and baseline score."""
    plt = _get_pyplot()
    max_motion_scores = _quality_field(records, "max_motion_score")
    baseline_scores = _quality_field(records, "baseline_score")

    plt.figure(figsize=(8, 4))
    plotted = False
    if max_motion_scores:
        plt.hist(max_motion_scores, bins=min(10, max(1, len(max_motion_scores))), alpha=0.7, label="max_motion_score")
        plotted = True
    if baseline_scores:
        plt.hist(baseline_scores, bins=min(10, max(1, len(baseline_scores))), alpha=0.7, label="baseline_score")
        plotted = True
    if not plotted:
        plt.text(0.5, 0.5, "No motion score data", ha="center", va="center")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
    else:
        plt.legend()
    plt.title("Motion Score Distribution")
    plt.xlabel("Score")
    plt.ylabel("Samples")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_timing_distribution(records: list[dict[str, Any]], output_path: str | Path) -> None:
    """Plot distributions of speech, visual, and navigation timing fields."""
    plt = _get_pyplot()
    speech_end_times = _numeric_field(records, "speech_end_time")
    visual_start_times = _numeric_field(records, "visual_start_time")
    navigation_start_times = _numeric_field(records, "navigation_start_time")

    plt.figure(figsize=(8, 4))
    plotted = False
    for label, values in [
        ("speech_end_time", speech_end_times),
        ("visual_start_time", visual_start_times),
        ("navigation_start_time", navigation_start_times),
    ]:
        if values:
            plt.hist(values, bins=min(10, max(1, len(values))), alpha=0.6, label=label)
            plotted = True

    if not plotted:
        plt.text(0.5, 0.5, "No timing data", ha="center", va="center")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
    else:
        plt.legend()
    plt.title("Timing Distribution")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Samples")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def generate_report(dataset_index: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Generate summary files and plots for a dataset index."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    records = load_jsonl(dataset_index)
    summary = summarize_records(records)

    write_summary_json(summary, output_path / "dataset_summary.json")
    write_summary_csv(summary, output_path / "dataset_summary.csv")
    plot_status_distribution(records, output_path / "status_distribution.png")
    plot_failure_reason_distribution(records, output_path / "failure_reason_distribution.png")
    plot_motion_score_distribution(records, output_path / "motion_score_distribution.png")
    plot_timing_distribution(records, output_path / "timing_distribution.png")

    return summary
