from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import subprocess
import threading

import whisper


_TRANSCRIBE_LOCK = threading.Lock()


@dataclass
class SpeechTranscriptionResult:
    """Whisper transcription output for one video."""

    instruction: str
    instruction_segments: list[dict[str, Any]]
    speech_end_time: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""
        return asdict(self)


def load_whisper_model(model_name: str = "base") -> Any:
    """Load a Whisper model once, outside the per-video processing loop."""
    return whisper.load_model(model_name)


def extract_audio(video_path: str | Path, audio_path: str | Path) -> bool:
    """Extract mono 16 kHz WAV audio from a video with ffmpeg."""
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        "-map",
        "0:a:0?",
        str(audio_path),
        "-y",
    ]
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return result.returncode == 0 and Path(audio_path).exists()


def transcribe_video_instruction(
    video_path: str | Path,
    temp_audio_path: str | Path,
    model: Any,
    language: str = "zh",
    max_segment_gap: float = 2.5,
    end_padding: float = 0.3,
) -> SpeechTranscriptionResult:
    """Transcribe video audio and estimate instruction end time with the original segment-gap logic."""
    audio_path = Path(temp_audio_path)
    instruction = ""
    speech_end_time = 0.0
    instruction_segments: list[dict[str, Any]] = []

    try:
        if not extract_audio(video_path, audio_path):
            return SpeechTranscriptionResult(
                instruction=instruction,
                instruction_segments=instruction_segments,
                speech_end_time=speech_end_time,
            )

        with _TRANSCRIBE_LOCK:
            result = model.transcribe(str(audio_path), language=language)

        segments = result.get("segments", [])
        instruction_segments = [
            {
                "start": float(segment.get("start", 0.0)),
                "end": float(segment.get("end", 0.0)),
                "text": str(segment.get("text", "")),
            }
            for segment in segments
        ]

        if segments:
            speech_end_time = float(segments[0]["end"])
            for i in range(1, len(segments)):
                if segments[i]["start"] - segments[i - 1]["end"] > max_segment_gap:
                    break
                speech_end_time = float(segments[i]["end"])

            speech_end_time += end_padding

        instruction = "".join(segment["text"] for segment in segments).strip()
        return SpeechTranscriptionResult(
            instruction=instruction,
            instruction_segments=instruction_segments,
            speech_end_time=round(speech_end_time, 4),
        )
    finally:
        if audio_path.exists():
            audio_path.unlink()
