"""Utility helpers to (re)generate the repository ``notes.txt`` file.

The script exposes pure helper functions so they can be imported by tests or
other automation without relying on side effects. Running the module directly
updates ``notes.txt`` in the project root when the canonical contents change.
"""
from __future__ import annotations

from pathlib import Path
import textwrap


def build_notes() -> str:
    """Construct the canonical ``notes.txt`` contents.

    Returns:
        str: Fully formatted notes document including newline termination.
    """
    return f"""{textwrap.dedent(
      """
    Purpose of the Program
    ----------------------
    - Provide accessible speech tools by combining Whisper-based speech-to-text
      and Hugging Face SpeechT5 text-to-speech utilities.
    - Help educators and learners prototype inclusive lessons without relying
      on proprietary cloud services.
    - Demonstrate both offline workflows and optional hosted API integrations
      for transcription and synthesis.

    Prerequisites
    -------------
    - Python 3.10 or newer (use virtual environments to isolate dependencies).
    - FFmpeg installed and available on PATH so Whisper can decode audio
      (verify with ``ffmpeg -version``).
    - Internet access the first time Whisper and SpeechT5 model weights
      download from PyPI and Hugging Face.
    - Optional: the ``openai`` Python client when you plan to call the hosted
      Whisper API.

    Installation Instructions
    -------------------------
    1. Create and activate a Python virtual environment.
       - Windows (PowerShell): ``py -3.12 -m venv .venv`` then
       ``.\\.venv\\Scripts\\Activate``
       - macOS/Linux (bash or zsh): ``python3 -m venv .venv`` then
       ``source .venv/bin/activate``
    2. Upgrade pip: ``python -m pip install --upgrade pip``
    3. Install dependencies: ``python -m pip install -r requirements.txt``
    4. Install FFmpeg (required for Whisper audio decoding).
       - Windows: download from https://www.gyan.dev/ffmpeg/builds/ and add
         ``ffmpeg.exe`` to PATH or copy it beside ``transcribe.py``
       - macOS: ``brew install ffmpeg``
       - Linux: ``sudo apt install ffmpeg``
    5. (Optional) Install the OpenAI Python client:
        ``python -m pip install openai``

    Why Some Files Are Not Tracked
    ------------------------------
    - ``lesson_recording.mp3`` is intentionally excluded so the repository
      stays lightweight—add your own audio when following the walkthrough.
    - Hugging Face caches for SpeechT5 live under ``~/.cache/huggingface`` and
      do not belong in source control.

    Internal Architecture Overview
    -------------------------------
    - ``transcribe.py`` wraps Whisper to load audio, perform transcription, and
      print the resulting text.
    - ``tts.py`` loads SpeechT5 and its vocoder to convert text into
      ``output.wav`` using a deterministic speaker embedding.
    - ``requirements.txt`` centralises the Python dependencies required by both
      command-line tools.

    Hands-on Walkthrough
    --------------------
    1. Create and activate a virtual environment.
       - Windows (PowerShell): ``py -3.12 -m venv .venv`` then
       ``.\\.venv\\Scripts\\Activate``
       - macOS/Linux (bash or zsh): ``python3 -m venv .venv`` then
       ``source .venv/bin/activate``
    2. Upgrade packaging tooling and install dependencies:
       ``python -m pip install --upgrade pip`` followed by
       ``python -m pip install -r requirements.txt``.
    3. Install FFmpeg (Windows builds from Gyan.dev, ``brew install ffmpeg`` on
       macOS, or ``sudo apt install ffmpeg`` on Debian/Ubuntu) and confirm the
       command works with ``ffmpeg -version``.
    4. Copy or record an audio sample (for example ``lesson_recording.mp3``)
       into the project root.
    5. Run offline transcription: ``python transcribe.py [audio_path]
       [--model MODEL_NAME]`` and review the printed transcript.
    6. Generate speech from text: ``python tts.py`` to create ``output.wav``
       using the deterministic SpeechT5 voice.
    7. Import or adapt the scripts inside other Python programs to automate
       inclusive lesson preparation.

    Optional API Integration
    ------------------------
    1. Install the OpenAI client if needed: ``python -m pip install openai``.
    2. Configure ``OPENAI_API_KEY`` for repeatable use.
       - Windows (PowerShell): ``setx OPENAI_API_KEY "sk-..."`` then start a
       new shell; for the current session use
       ``$env:OPENAI_API_KEY = "sk-..."``.
       - macOS/Linux (bash or zsh):
         ``echo 'export OPENAI_API_KEY="sk-..."' >> ~/.bashrc`` then
         ``source ~/.bashrc``; to scope it to one session run
         ``export OPENAI_API_KEY="sk-..."``.
    3. Follow the README example to call the hosted Whisper API with
       ``OpenAI().audio.transcriptions.create`` and print the transcript.
    4. Use the hosted option when managed infrastructure or faster turnaround
       is preferable to local hardware.

    Quick Setup Cheatsheet
    ----------------------
    - ``python -m venv .venv`` → activate it for your platform.
    - ``python -m pip install --upgrade pip``
    - ``python -m pip install -r requirements.txt``
    - Ensure ``ffmpeg -version`` succeeds, then run ``python transcribe.py`` or
      ``python tts.py`` as needed.

    Troubleshooting Tips
    --------------------
    - Whisper ``FileNotFoundError`` or FFmpeg errors: ensure FFmpeg is
      installed and accessible via PATH; verify with ``ffmpeg -version``.
    - ``ImportError`` for ``sentencepiece`` or ``soundfile`` when running
      ``tts.py``: reinstall dependencies via
      ``python -m pip install -r requirements.txt``.
    - PyTorch installation issues on Windows: install the CPU wheel
      ``python -m pip install torch
      --index-url https://download.pytorch.org/whl/cpu``.
    - Slow performance on CPUs: choose smaller Whisper checkpoints (``tiny`` or
      ``small``) and keep demo text short for SpeechT5.
    - Missing ``OPENAI_API_KEY`` environment variable when running the optional
      API script: export the key before execution (PowerShell
      ``$env:OPENAI_API_KEY="sk-..."``;
      bash ``export OPENAI_API_KEY="sk-..."``).
- Logs and debug output: inspect terminal logs; Hugging Face downloads are
  cached automatically under the user's home directory.
"""
    ).strip()}
"""


def write_notes(project_root: Path) -> None:
    """Write ``notes.txt`` with canonical content when changes are detected.

    Args:
        project_root: Root directory that should contain ``notes.txt``.
    """
    notes_path = project_root / "notes.txt"
    content = build_notes()

    if notes_path.exists():
        existing_content = notes_path.read_text(encoding="utf-8")
        if existing_content == content:
            return

    notes_path.write_text(content, encoding="utf-8")


def main() -> None:
    """Synchronize ``notes.txt`` with the generated canonical content."""

    write_notes(Path(__file__).resolve().parents[1])


if __name__ == "__main__":
    main()
