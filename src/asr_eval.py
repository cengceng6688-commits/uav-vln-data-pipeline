from __future__ import annotations

from pathlib import Path
from typing import Any
import csv
import json
import statistics
import string


EXTRA_PUNCTUATION = "，。！？；：、“”‘’（）【】《》〈〉…—·"
PUNCTUATION_TABLE = str.maketrans("", "", string.punctuation + EXTRA_PUNCTUATION)


def normalize_text(text: str | None) -> str:
    """Normalize ASR text for character error rate evaluation."""
    if text is None:
        return ""
    normalized = str(text).lower()
    normalized = normalized.translate(PUNCTUATION_TABLE)
    normalized = "".join(normalized.split())
    return normalized


def character_error_rate(pred: str | None, ref: str | None) -> float:
    """Compute character error rate with Levenshtein edit distance."""
    pred_norm = normalize_text(pred)
    ref_norm = normalize_text(ref)

    if not ref_norm:
        return 0.0 if not pred_norm else 1.0

    distance = _edit_distance(pred_norm, ref_norm)
    return round(distance / len(ref_norm), 4)


def evaluate_asr(
    dataset_index: str | Path,
    ground_truth_csv: str | Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate Whisper instructions against sampled manual ground truth transcripts."""
    records = _load_jsonl(dataset_index)
    ground_truth = _load_ground_truth_csv(ground_truth_csv)

    details: list[dict[str, Any]] = []
    for record in records:
        sample_id = str(record.get("sample_id", ""))
        if sample_id not in ground_truth:
            continue

        whisper_text = str(record.get("instruction") or "")
        ground_truth_text = ground_truth[sample_id]
        cer = character_error_rate(whisper_text, ground_truth_text)
        details.append(
            {
                "sample_id": sample_id,
                "whisper_text": whisper_text,
                "ground_truth_text": ground_truth_text,
                "cer": cer,
            }
        )

    cer_values = [float(row["cer"]) for row in details]
    summary = {
        "num_evaluated": len(details),
        "average_cer": round(sum(cer_values) / len(cer_values), 4) if cer_values else None,
        "median_cer": round(statistics.median(cer_values), 4) if cer_values else None,
        "max_cer": round(max(cer_values), 4) if cer_values else None,
        "min_cer": round(min(cer_values), 4) if cer_values else None,
    }
    return summary, details


def write_asr_eval_outputs(
    summary: dict[str, Any],
    details: list[dict[str, Any]],
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Write ASR evaluation summary JSON and details CSV."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    summary_path = output_path / "asr_eval_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    details_path = output_path / "asr_eval_details.csv"
    with details_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["sample_id", "whisper_text", "ground_truth_text", "cer"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in details:
            writer.writerow(row)

    return summary_path, details_path


def _load_jsonl(dataset_index: str | Path) -> list[dict[str, Any]]:
    path = Path(dataset_index)
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                records.append(record)
    return records


def _load_ground_truth_csv(ground_truth_csv: str | Path) -> dict[str, str]:
    path = Path(ground_truth_csv)
    ground_truth: dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample_id = str(row.get("sample_id") or "").strip()
            if not sample_id:
                continue
            ground_truth[sample_id] = str(row.get("ground_truth_text") or "")
    return ground_truth


def _edit_distance(source: str, target: str) -> int:
    if source == target:
        return 0
    if not source:
        return len(target)
    if not target:
        return len(source)

    previous = list(range(len(target) + 1))
    for i, source_char in enumerate(source, start=1):
        current = [i]
        for j, target_char in enumerate(target, start=1):
            insertion = current[j - 1] + 1
            deletion = previous[j] + 1
            substitution = previous[j - 1] + (source_char != target_char)
            current.append(min(insertion, deletion, substitution))
        previous = current
    return previous[-1]
