"""Command-line utility for transcribing audio files with Whisper."""
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Dict

import whisper

DEFAULT_AUDIO = "lesson_recording.mp3"
DEFAULT_MODEL = "base"


def parse_arguments() -> Namespace:
    """Parse command-line options for the transcription script."""
    parser = ArgumentParser(description="Transcribe an audio file with Whisper.")
    parser.add_argument(
        "audio_path",
        nargs="?",
        default=DEFAULT_AUDIO,
        help="Path to the audio file to transcribe (default: %(default)s).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Name of the Whisper model to load (default: %(default)s).",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Enable fp16 inference when supported by your hardware.",
    )
    return parser.parse_args()


def load_audio_path(raw_path: str) -> Path:
    """Resolve and validate the provided audio path."""
    audio_path = Path(raw_path).expanduser().resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    return audio_path


def transcribe_audio(audio_path: Path, model_name: str, use_fp16: bool) -> Dict[str, Any]:
    """Load the requested Whisper model and transcribe the given audio file."""
    model = whisper.load_model(model_name)
    return model.transcribe(str(audio_path), fp16=use_fp16)


def main() -> None:
    """Entrypoint for the transcription CLI."""
    args = parse_arguments()
    audio_path = load_audio_path(args.audio_path)
    result = transcribe_audio(audio_path, args.model, args.fp16)
    print("Transcript:", result["text"])


if __name__ == "__main__":
    main()
