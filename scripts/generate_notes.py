"""Utility helpers to (re)generate the repository ``notes.txt`` file.

The script exposes pure helper functions so they can be imported by tests or
other automation without relying on side effects. Running the module directly
updates ``notes.txt`` in the project root when the canonical contents change.
"""
from __future__ import annotations

from pathlib import Path
from textwrap import dedent


def build_notes() -> str:
    """Construct the canonical ``notes.txt`` contents.

    Returns:
        str: Fully formatted notes document including newline termination.
    """
    return dedent(
        """
        Purpose of the Program
        ----------------------
        - Provide accessible speech tools by combining Whisper-based speech-to-text and Hugging Face SpeechT5 text-to-speech utilities.
        - Help educators and learners quickly transcribe lessons and produce spoken audio without relying on paid cloud services.
        - Demonstrate both offline and optional cloud-powered transcription approaches for inclusive education scenarios.

        Installation Instructions
        -------------------------
        1. Create and activate a Python virtual environment (Python 3.10+ recommended).
           - Windows (PowerShell): ``py -3.12 -m venv .venv`` then ``.\\.venv\\Scripts\\Activate``
           - macOS/Linux (bash/zsh): ``python3 -m venv .venv`` then ``source .venv/bin/activate``
        2. Upgrade pip: ``python -m pip install --upgrade pip``
        3. Install dependencies: ``python -m pip install -r requirements.txt``
        4. Install FFmpeg (required for Whisper audio decoding).
           - Windows: download from https://www.gyan.dev/ffmpeg/builds/ and add ``ffmpeg.exe`` to PATH or copy beside ``transcribe.py``
           - macOS: ``brew install ffmpeg``
           - Linux: ``sudo apt install ffmpeg``
        5. (Optional) Install the OpenAI Python client for API-based transcription: ``python -m pip install openai``

        How to Use It
        -------------
        1. Place an audio file such as ``lesson_recording.mp3`` in the project root.
        2. Run offline speech-to-text with ``python transcribe.py`` (or ``python3`` on macOS/Linux).
           - Add ``--model tiny`` or ``--model small`` for quicker local inference if the script supports it.
        3. Run text-to-speech with ``python tts.py`` to generate ``output.wav`` using SpeechT5.
        4. (Optional) Create ``transcribe_api.py`` as described in ``README.md`` and run it after setting ``OPENAI_API_KEY`` to use the OpenAI Whisper API.
        5. Inspect terminal output for transcripts and confirm audio artefacts (``output.wav``) appear in the project directory.

        Internal Architecture Overview
        ------------------------------
        - ``transcribe.py`` loads a Whisper model, decodes audio via FFmpeg, and prints the recognized transcript.
        - ``tts.py`` loads the SpeechT5 model and vocoder from Hugging Face, converts text input to speech, and writes ``output.wav``.
        - ``requirements.txt`` centralises Python dependencies for reproducible environments.
        - ``README.md`` documents optional extensions such as the API transcription example and environment preparation tips.

        Troubleshooting Tips
        --------------------
        - Whisper "FileNotFoundError" or FFmpeg errors: ensure FFmpeg is installed and accessible via PATH. Verify with ``ffmpeg -version``.
        - ``ImportError`` for ``sentencepiece`` or ``soundfile`` when running ``tts.py``: reinstall dependencies via ``python -m pip install -r requirements.txt``.
        - PyTorch installation issues on Windows: use the CPU wheel ``python -m pip install torch --index-url https://download.pytorch.org/whl/cpu``.
        - Slow performance on CPUs: switch to smaller Whisper models (``tiny`` or ``small``) and keep demo text short for SpeechT5.
        - Missing ``OPENAI_API_KEY`` environment variable when running the optional API script: export the key before execution (PowerShell ``$env:OPENAI_API_KEY="sk-..."``; bash ``export OPENAI_API_KEY="sk-..."``).
        - Logs and debug output: inspect terminal logs; Hugging Face model downloads are cached under the user's home cache directory (``~/.cache/huggingface``).
        """
    ).strip() + "\n"


def write_notes(project_root: Path) -> None:
    """Write ``notes.txt`` with canonical content when changes are detected.

    Args:
        project_root: Root directory that should contain ``notes.txt``.
    """
    notes_path = project_root / "notes.txt"
    content = build_notes()

    if notes_path.exists() and notes_path.read_text(encoding="utf-8") == content:
        return

    notes_path.write_text(content, encoding="utf-8")


def main() -> None:
    """Synchronize ``notes.txt`` with the generated canonical content."""

    write_notes(Path(__file__).resolve().parents[1])


if __name__ == "__main__":
    main()
