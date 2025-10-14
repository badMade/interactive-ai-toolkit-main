"""Command-line utility for transcribing audio files with OpenAI Whisper.

The module exposes small helper functions so other scripts can reuse the
argument parsing and transcription logic programmatically. The default audio
file and model are configured for local experimentation but can be overridden
on the command line.
"""
import sys
from argparse import ArgumentParser, Namespace
from functools import lru_cache
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Any, Dict


MISSING_WHISPER_MESSAGE = (
    "OpenAI Whisper is not installed. "
    "Install it with 'pip install openai-whisper' "
    "or run setup_env.py to configure the environment."
)


# Exposed for test instrumentation; patched in unit tests without requiring Whisper.
whisper: ModuleType | None = None


@lru_cache(maxsize=1)
def _import_whisper() -> ModuleType:
    try:
        return import_module("whisper")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(MISSING_WHISPER_MESSAGE) from exc


def load_whisper_module() -> ModuleType:
    """Import the Whisper module with a helpful error if it is unavailable."""
    global whisper
    if whisper is not None:
        return whisper
    whisper = _import_whisper()
    return whisper


DEFAULT_AUDIO = "lesson_recording.mp3"
DEFAULT_MODEL = "base"


def parse_arguments() -> Namespace:
    """Parse command-line options for the transcription script.

    Returns:
        Namespace: Parsed arguments containing the ``audio_path`` of the file
        to transcribe, the ``model`` name to load, and the ``fp16`` flag that
        indicates whether half-precision inference should be attempted.
    """
    parser = ArgumentParser(
        description="Transcribe an audio file with Whisper.")
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
    """Resolve and validate the audio source path provided by the caller.

    Args:
        raw_path: Raw path value supplied via the CLI or a calling function.

    Returns:
        Path: Absolute path pointing to the audio file on disk.

    Raises:
        FileNotFoundError: If ``raw_path`` does not reference an existing file.
    """
    audio_path = Path(raw_path).expanduser().resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    return audio_path


def transcribe_audio(audio_path: Path,
                     model_name: str,
                     use_fp16: bool) -> Dict[str, Any]:
    """Load the requested Whisper model and transcribe the given audio file.

    Args:
        audio_path: Absolute path to the audio sample to be transcribed.
        model_name: Name of the Whisper checkpoint to load.
        use_fp16: ``True`` to request half-precision inference when supported.

    Returns:
        Dict[str, Any]: Full transcription result emitted by Whisper, including
        ``text`` and segment metadata.
    """
    module = load_whisper_module()
    model = module.load_model(model_name)
    transcription_kwargs: Dict[str, Any] = {}
    if not use_fp16:
        transcription_kwargs["fp16"] = False
    result: Dict[str, Any] = model.transcribe(
        str(audio_path),
        **transcription_kwargs,
    )
    return result


def _missing_whisper_message(exc: ModuleNotFoundError) -> str | None:
    """Return a helpful message when Whisper is not installed."""
    current: BaseException | None = exc
    while current is not None:
        if (
            isinstance(current, ModuleNotFoundError)
            and getattr(current, "name", None) == "whisper"
        ):
            return MISSING_WHISPER_MESSAGE
        current = current.__cause__
    return None


def main() -> None:
    """Execute the transcription workflow for command-line usage.

    The function parses user-provided options, validates the audio source, and
    prints the transcript text returned by Whisper.
    """
    args = parse_arguments()
    audio_path = load_audio_path(args.audio_path)
    try:
        result = transcribe_audio(audio_path, args.model, args.fp16)
    except ModuleNotFoundError as exc:
        whisper_message = _missing_whisper_message(exc)
        if whisper_message is None:
            raise
        print(whisper_message, file=sys.stderr)
        raise SystemExit(1) from None
    print("Transcript:", result["text"])


if __name__ == "__main__":
    main()
