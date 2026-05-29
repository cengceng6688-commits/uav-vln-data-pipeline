# UAV VLN Data Pipeline

## Project Overview

This repository is a pilot research prototype for constructing a real-world goal-level navigation video dataset from raw videos. Each input video is assumed to contain a spoken goal-level instruction at the beginning, followed by the corresponding navigation behavior.

The current midterm version focuses on dataset construction rather than model training. It produces aligned navigation clips, Whisper transcriptions, instruction analysis, quality metadata, summary reports, case visualizations, and ASR evaluation outputs.

Current spoken instructions are mostly goal-level task descriptions rather than low-level action commands. The current midterm version does not provide ground-truth dx, dy, dyaw action labels. The dx, dy, dyaw interface is reserved for future extensions such as visual odometry, SLAM, sensor logs, UAV telemetry, or optional π0 fine-tuning.


## Current Pipeline

```text
raw video
  -> visual QA
  -> Whisper transcription
  -> speech-motion alignment
  -> navigation clip trimming
  -> instruction analysis
  -> dataset_index.jsonl generation
  -> summary report generation
  -> case visualization
  -> ASR evaluation
```

The current implementation supports visual QA, Whisper transcription, speech-motion alignment, navigation clip trimming, instruction analysis, JSONL dataset indexing, summary reporting, case visualization, and ASR evaluation with a manually annotated transcript subset.

## Dataset Schema

The main dataset artifact is `dataset_index.jsonl`. Each line is one sample record. A detailed schema is provided in [docs/dataset_schema.md](docs/dataset_schema.md).

Important fields:

| Field | Explanation |
| --- | --- |
| `sample_id` | Stable sample identifier. |
| `original_video` | Path to the source video. |
| `navigation_clip` | Path to the trimmed navigation clip. |
| `instruction` | Whisper-transcribed spoken instruction. |
| `instruction_analysis` | Goal-level versus low-level instruction analysis. |
| `speech_end_time` | Estimated end time of the spoken instruction. |
| `visual_start_time` | Estimated time when visual motion begins. |
| `navigation_start_time` | Timestamp used to trim the navigation clip. |
| `qa_status` | Processing status, such as `success` or `failed`. |
| `quality_score` | Visual QA statistics and thresholds. |
| `action_format` | Reserved action format, currently `dx_dy_dyaw`. |
| `actions` | Empty or reserved in the current midterm version; not ground-truth labels. |
| `action_note` | Explanation of action-label status and limitations. |

The recommended midterm setting keeps `actions` empty. Any generated action values are pseudo-labels and must not be reported as ground-truth trajectories or control labels.

## Output Structure

Expected midterm output folder:

```text
outputs/
  midterm_demo/
    dataset_index.jsonl
    clips/
    report/
    case_0/
    case_1/
    asr_eval/
```

Contents:

- `dataset_index.jsonl`: JSONL dataset index with one record per processed sample.
- `clips/`: trimmed navigation clips.
- `report/`: dataset summary JSON, CSV, and plots.
- `case_0/`, `case_1/`: selected case visualizations for presentation.
- `asr_eval/`: ASR evaluation summary and per-sample CER details.

## Running the Pipeline

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the recommended midterm pipeline:

```bash
python3 scripts/run_pipeline.py \
  --input_dir data/raw_videos \
  --output_dir outputs/midterm_demo \
  --whisper_model base \
  --action_mode empty
```

The recommended midterm action mode is `empty`, because the current dataset is goal-level and does not include ground-truth low-level action labels.

## Pilot Results

Pilot statistics can be generated from the current processed dataset using the reporting script.

Generate dataset summary reports:

```bash
python3 scripts/generate_report.py \
  --dataset_index outputs/midterm_demo/dataset_index.jsonl \
  --output_dir outputs/midterm_demo/report
```

The report includes sample counts, success rate, failure reasons, timing statistics, motion quality statistics, and instruction type distribution when available.


## ASR Evaluation

ASR evaluation requires a small manually annotated ground-truth transcript subset:

```text
data/annotations/asr_ground_truth.csv
```

CSV format:

```csv
sample_id,ground_truth_text
example_sample_id,manual transcript text
```

For the midterm version, annotating 20-50 representative samples is sufficient for a preliminary ASR quality estimate. For Chinese-language ASR, character error rate is the primary metric because word segmentation can be ambiguous.

Run ASR evaluation:

```bash
python3 scripts/evaluate_asr.py \
  --dataset_index outputs/midterm_demo/dataset_index.jsonl \
  --ground_truth_csv data/annotations/asr_ground_truth.csv \
  --output_dir outputs/midterm_demo/asr_eval
```

This writes `asr_eval_summary.json` and `asr_eval_details.csv`.

## Case Visualization

Generate selected case visualizations:

```bash
python3 scripts/visualize_sample.py \
  --dataset_index outputs/midterm_demo/dataset_index.jsonl \
  --index 0 \
  --output_dir outputs/midterm_demo/case_0

python3 scripts/visualize_sample.py \
  --dataset_index outputs/midterm_demo/dataset_index.jsonl \
  --index 1 \
  --output_dir outputs/midterm_demo/case_1
```

Case visualization outputs include `case_summary.json`, `case_summary.md`, `sample_keyframes.png`, and `timing_plot.png` when timing fields are available. `action_curve.png` is generated only when actions are available.

## Current Limitations

- The current pilot dataset has a small sample size and is intended for midterm demonstration and pipeline validation.
- Current instructions are mostly goal-level task descriptions, not complete low-level navigation commands.
- The current midterm version does not include ground-truth dx, dy, dyaw action labels.
- The current collection does not include UAV telemetry or synchronized sensor logs.
- Whisper transcription quality needs manual validation through the ASR subset.
- Future scaling will require more data sources, stronger annotation procedures, and more reliable motion-label sources.
- Optional pseudo-actions, if generated for experiments, should not be reported as real trajectories or ground-truth control labels.

## Release Roadmap

- `midterm_demo`: current pilot version for pipeline validation and presentation.
- `v0.1`: target 30-50 samples with clearer documentation and ASR validation.
- `v1.0`: target 500-2,000 samples with stronger subset organization and broader statistics.
- `large-scale`: future target at 200K-scale samples, not a current result.

The staged release plan is documented in [docs/dataset_release_plan.md](docs/dataset_release_plan.md).

## Next Steps

- Expand the pilot dataset to 30-50 real-world videos for a more reliable midterm evaluation.
- Report dataset-level statistics, including success rate, timing distribution, quality-score distribution, and instruction-type distribution.
- Conduct a small-scale ASR evaluation using manually annotated transcripts and character error rate.
- Select representative case studies to demonstrate speech-motion alignment, navigation clip trimming, and instruction analysis.
- Explore additional data sources and annotation strategies for scaling toward larger dataset releases.
- Investigate future motion-label sources, such as visual odometry, SLAM, sensor logs, UAV telemetry, or optional π0-style VLA fine-tuning.