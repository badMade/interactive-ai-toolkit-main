"""Command-line utility for transcribing audio files with OpenAI Whisper.

The module exposes small helper functions so other scripts can reuse the
argument parsing and transcription logic programmatically. The default audio
file and model are configured for local experimentation but can be overridden
on the command line.
"""
import os
import sys
from argparse import ArgumentParser, Namespace
from functools import lru_cache
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Sequence

from compatibility import ensure_numpy_compatible, NumpyCompatibilityError
from ffmpeg_support import (
    FFMPEG_INSTALL_MESSAGE,
    FFmpegInstallationError,
    ensure_ffmpeg_available as ensure_system_ffmpeg_available,
)
from shared_messages import MISSING_WHISPER_MESSAGE


# Exposed for test instrumentation;
# patched in unit tests without requiring Whisper.
whisper: ModuleType | None = None


@lru_cache(maxsize=1)
def _import_whisper() -> ModuleType:
    try:
        return import_module("whisper")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(MISSING_WHISPER_MESSAGE) from exc


def load_whisper_module() -> ModuleType:
    """Import the Whisper module with a helpful error if it is unavailable."""
    cached = whisper
    if cached is not None:
        return cached
    module = _import_whisper()
    globals()["whisper"] = module
    return module


DEFAULT_AUDIO = "lesson_recording.mp3"
DEFAULT_MODEL = "base"
AVAILABLE_MODELS: tuple[str, ...] = (
    "tiny.en",
    "tiny",
    "base.en",
    "base",
    "small.en",
    "small",
    "medium.en",
    "medium",
    "large-v1",
    "large-v2",
    "large-v3",
    "large",
    "large-v3-turbo",
    "turbo",
)

def parse_arguments(argv: Sequence[str] | None = None) -> Namespace:
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
        choices=AVAILABLE_MODELS,
        help=(
            "Name of the Whisper model to load. "
            "Select from available Whisper checkpoints "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Enable fp16 inference when supported by your hardware.",
    )
    parser.add_argument(
        "--ca-bundle",
        default=None,
        help=(
            "Path to a PEM file containing additional certificate "
            "authorities. The bundle is added to REQUESTS_CA_BUNDLE, "
            "CURL_CA_BUNDLE, and SSL_CERT_FILE to support TLS inspection "
            "proxies without disabling verification."
        ),
    )
    return parser.parse_args(argv)


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


def ensure_ffmpeg_available() -> None:
    """Verify that the ffmpeg executable is present on the system path."""

    try:
        ensure_system_ffmpeg_available(allow_auto_install=False)
    except FFmpegInstallationError as exc:
        details = str(exc).strip()
        if details and details != FFMPEG_INSTALL_MESSAGE:
            message = f"{details}\n{FFMPEG_INSTALL_MESSAGE}"
        else:
            message = FFMPEG_INSTALL_MESSAGE
        raise SystemExit(message) from exc


def configure_certificate_bundle(raw_bundle_path: str | None) -> None:
    """Update TLS trust configuration to include a custom certificate bundle.

    Args:
        raw_bundle_path: Optional string path to a PEM-encoded bundle supplied
            by the caller.

    Raises:
        FileNotFoundError: If ``raw_bundle_path`` is provided but does not
            reference an existing file.
    """

    if raw_bundle_path is None:
        # Keep the interpreter's default TLS verification settings intact.
        return

    bundle_path = Path(raw_bundle_path).expanduser().resolve()
    if not bundle_path.is_file():
        raise FileNotFoundError(f"CA bundle not found: {bundle_path}")

    bundle_str = str(bundle_path)
    for env_var in ("REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "SSL_CERT_FILE"):
        # Use standard environment hooks so HTTP clients continue verifying
        # certificates while trusting the additional proxy root.
        os.environ[env_var] = bundle_str


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
    ensure_numpy_compatible()
    module = load_whisper_module()
    try:
        model = module.load_model(model_name)
    except RuntimeError as exc:
        available_models: tuple[str, ...] = AVAILABLE_MODELS
        if hasattr(module, "available_models"):
            try:
                available_models = tuple(module.available_models())
            except Exception:  # pragma: no cover - defensive fallback
                available_models = AVAILABLE_MODELS
        message = str(exc).lower()
        is_unknown_model_error = "not found" in message or "unknown" in message
        if is_unknown_model_error:
            formatted_models = ", ".join(sorted(available_models))
            raise ValueError(
                f"Unknown Whisper model '{model_name}'. Choose from: {formatted_models}."
            ) from exc
        raise
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


def main(argv: Sequence[str] | None = None) -> None:
    """Execute the transcription workflow for command-line usage.

    The function parses user-provided options, validates the audio source, and
    prints the transcript text returned by Whisper.
    """
    args = parse_arguments(argv)
    configure_certificate_bundle(args.ca_bundle)
    audio_path = load_audio_path(args.audio_path)
    try:
        ensure_ffmpeg_available()
        result = transcribe_audio(audio_path, args.model, args.fp16)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1) from None
    except ModuleNotFoundError as exc:
        whisper_message = _missing_whisper_message(exc)
        if whisper_message is None:
            raise
        print(whisper_message, file=sys.stderr)
        raise SystemExit(1) from None
    except NumpyCompatibilityError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1) from None
    print("Transcript:", result["text"])


if __name__ == "__main__":
    main()
