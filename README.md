# AI in Inclusive Education: Speech-to-Text + Text-to-Speech (Python)

A tiny, practical toolkit that demonstrates:
- **Speech-to-Text** with local Whisper (free, no API key)
- **Text-to-Speech** with Hugging Face SpeechT5 (works on CPU)
- **Optional:** Speech-to-Text with the OpenAI Whisper API (cloud-based)

This repo supports your article “AI in Inclusive Education: Build Your First Accessibility Tools with Python”.

---

## Demo screenshots
- `images/ffmpeg-version.png` — PowerShell showing `ffmpeg -version`.
- `images/whisper-transcript.png` — Terminal transcript output from Whisper.
- `images/tts-saved-ok.png` — Terminal showing “✅ Audio saved as output.wav”.
- `images/output-wav-explorer.png` — Explorer showing `output.wav` created.

*(Screenshots are optional—add yours or delete this section.)*

---

## 1) Setup

### Windows (PowerShell)
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate
pip install --upgrade pip
pip install -r requirements.txt
```

### macOS / Linux (bash/zsh)
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## Quick Setup Cheatsheet 📝  

| Task                   | Windows (PowerShell)                           | macOS / Linux (Terminal)                |
|------------------------|------------------------------------------------|-----------------------------------------|
| **Create venv**        | `py -3.12 -m venv .venv`                       | `python3 -m venv .venv`                  |
| **Activate venv**      | `.\.venv\Scripts\Activate`                   | `source .venv/bin/activate`              |
| **Install Whisper**    | `pip install openai-whisper`                   | `pip install openai-whisper`             |
| **Install FFmpeg**     | [Download build](https://www.gyan.dev/ffmpeg/builds/) → unzip → add to PATH or copy `ffmpeg.exe` | `brew install ffmpeg` (macOS) <br> `sudo apt install ffmpeg` (Linux) |
| **Run STT script**     | `python transcribe.py`                         | `python3 transcribe.py`                  |
| **Install TTS deps**   | `pip install transformers torch soundfile sentencepiece` | `pip install transformers torch soundfile sentencepiece` |
| **Run TTS script**     | `python tts.py`                                | `python3 tts.py`                         |
| **Install OpenAI client (API)** | `pip install openai`                  | `pip install openai`                     |
| **Run API script**     | `python transcribe_api.py`                     | `python3 transcribe_api.py`              |

💡 **Pro tip for macOS M1/M2 users:** You may need a special PyTorch build for Metal GPU acceleration. Check the [PyTorch install guide](https://pytorch.org/get-started/locally/) for the right wheel.

---

## 2) FFmpeg (required for Whisper)

Whisper needs `ffmpeg` to read audio files.

**Option A (quick workaround):**
- Download FFmpeg (Windows builds: https://www.gyan.dev/ffmpeg/builds/)
- Extract and copy `ffmpeg.exe` into the project root (same folder as `transcribe.py`)

**Option B (system-wide PATH):**
- Add `C:\ffmpeg\bin` to your PATH (Windows), or install via package manager (macOS/Linux).

**Verify:**
```powershell
ffmpeg -version
```

---

## 3) Speech-to-Text (Local Whisper)

Record a short audio clip (e.g., “Welcome to inclusive education with AI.”) and save as `lesson_recording.mp3` in the project root.

Run:
```powershell
python transcribe.py
```

Example output:
```
Transcript:  Welcome to inclusive education with AI.
```

> Tip: Use smaller models for speed: `tiny` or `small`.

---

## 4) Text-to-Speech (SpeechT5)

Run:
```powershell
python tts.py
```

Result:
- Terminal prints `✅ Audio saved as output.wav`
- `output.wav` appears in the project folder

Double-click `output.wav` to play.

> First run downloads model files (hundreds of MB). This is normal.

---

## 5) Optional: Whisper via OpenAI API

### How to get an API key:
1. Go to [OpenAI’s API Keys page](https://platform.openai.com/account/api-keys).
2. Log in (or create an account).
3. Click **“Create new secret key”**.
4. Copy it (starts with `sk-...`). Keep it private!

### Set your API key (PowerShell, session only):
```powershell
$env:OPENAI_API_KEY="your_api_key_here"
```

### Install the OpenAI Python client
```powershell
pip install openai
```

### Create `transcribe_api.py`
```python
from openai import OpenAI
client = OpenAI()

with open("lesson_recording.mp3", "rb") as f:
    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=f
    )

print("Transcript:", transcript.text)
```

### Run it:
```powershell
python transcribe_api.py
```

Expected output:
```
Transcript: Welcome to inclusive education with AI.
```

---

## Local Whisper vs API Whisper — Which Should You Use?

| Feature                | Local Whisper (on your machine)       | OpenAI Whisper API (cloud)         |
|-------------------------|---------------------------------------|-------------------------------------|
| **Setup**              | Needs Python packages + FFmpeg        | Just install `openai` client + set API key |
| **Hardware**           | Runs on your CPU (slower) or GPU (faster) | Runs on OpenAI’s servers (no local compute needed) |
| **Cost**               | ✅ Free after initial download         | 💳 Pay per minute of audio (after free trial quota) |
| **Internet required**  | ❌ No (fully offline once installed)   | ✅ Yes (uploads audio to OpenAI servers) |
| **Accuracy**           | Very good; depends on model size (tiny → large) | Consistently strong; optimized by OpenAI |
| **Speed**              | Slower on CPU, faster with GPU        | Fast (uses OpenAI’s infrastructure) |
| **Privacy**            | Audio never leaves your machine       | Audio is sent to OpenAI (data handling per policy) |

**Rule of thumb:**  
- Use **Local Whisper** if you want free, offline transcription or you’re working with sensitive data.  
- Use the **API Whisper** if you prefer convenience, don’t mind usage billing, and want speed without local setup.

---

## Troubleshooting

**Whisper `FileNotFoundError`**
- Cause: FFmpeg missing / not on PATH
- Fix: Install FFmpeg; either put `ffmpeg.exe` beside `transcribe.py` or add to PATH
- Check: `ffmpeg -version`

**`FileNotFoundError: Audio file not found` when running `transcribe.py`**
- Cause: The audio path supplied to the script does not exist.
- Fix: Place your recording in the project directory (default: `lesson_recording.mp3`) or pass the full path, e.g. `python transcribe.py path/to/audio.mp3`.
- Tip: Run `python transcribe.py --help` to confirm which file will be used.

**`ImportError: sentencepiece not found` (SpeechT5)**
```powershell
pip install sentencepiece
```

**Torch install issues on Windows**
```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**Hugging Face “symlink” cache warning on Windows**
- Safe to ignore. To remove, enable Windows Developer Mode or run Python as admin.

**Slow on CPU**
- Whisper: use `tiny` / `small`
- Keep TTS text short for demos

---

## File structure

```
.
├─ images/                      # (optional) screenshots for your article
├─ requirements.txt
├─ transcribe.py                # Whisper (speech -> text)
├─ tts.py                       # SpeechT5 (text -> speech)
└─ lesson_recording.mp3         # your short test audio (not required in repo)
```

---

## License
MIT — see [LICENSE](./LICENSE).
