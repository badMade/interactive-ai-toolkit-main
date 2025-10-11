"""Speech synthesis utilities built on top of Hugging Face SpeechT5.

The module exposes composable helper functions so that speech generation can be
triggered from the command line or imported into other Python code. The default
behaviour mirrors the original demo: convert a short welcome message into a WAV
file using deterministic speaker embeddings for reproducible output.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple
import wave

import numpy as np

import torch
from transformers import (
    SpeechT5ForTextToSpeech,
    SpeechT5HifiGan,
    SpeechT5Processor
)

DEFAULT_TEXT = "Welcome to inclusive education with AI."
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_MODEL_ID = "microsoft/speecht5_tts"
DEFAULT_VOCODER_ID = "microsoft/speecht5_hifigan"


def load_speecht5_components(
    model_id: str = DEFAULT_MODEL_ID,
    vocoder_id: str = DEFAULT_VOCODER_ID,
) -> Tuple[SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan]:
    """Load the processor, acoustic model, and vocoder for SpeechT5 synthesis.

    Args:
        model_id: Hugging Face identifier for the SpeechT5 acoustic model and
            processor checkpoints.
        vocoder_id: Hugging Face identifier for the compatible HiFi-GAN vocoder
            checkpoint.

    Returns:
        Tuple[SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan]:
        Fully initialized SpeechT5 processor, acoustic model, and vocoder.
    """

    processor = SpeechT5Processor.from_pretrained(model_id)
    model = SpeechT5ForTextToSpeech.from_pretrained(model_id)
    vocoder = SpeechT5HifiGan.from_pretrained(vocoder_id)
    return processor, model, vocoder


def create_default_speaker_embedding(seed: int = 42) -> torch.Tensor:
    """Create a deterministic speaker embedding tensor for reproducible audio.

    Args:
        seed: Random seed used to initialize the PyTorch generator.

    Returns:
        torch.Tensor: Speaker embedding with shape ``(1, 512)`` suitable for
        SpeechT5 generation.
    """

    generator = torch.Generator().manual_seed(seed)
    return torch.randn((1, 512), generator=generator)


def synthesize_speech(
    text: str,
    processor: SpeechT5Processor,
    model: SpeechT5ForTextToSpeech,
    vocoder: SpeechT5HifiGan,
    speaker_embedding: torch.Tensor,
) -> np.ndarray:
    """Generate a speech waveform for the provided text prompt.

    Args:
        text: Plain-text prompt to convert into spoken audio.
        processor: Tokenizer/feature extractor associated with the SpeechT5
            model.
        model: Acoustic model that generates spectrogram tokens.
        vocoder: HiFi-GAN vocoder used to reconstruct a waveform from model
            outputs.
        speaker_embedding: Embedding tensor that controls the synthetic voice
            characteristics.

    Returns:
        numpy.ndarray: Generated waveform in mono format with values in the
        range ``[-1.0, 1.0]``.
    """

    inputs = processor(text=text, return_tensors="pt")
    with torch.no_grad():
        waveform = model.generate_speech(
            inputs["input_ids"], speaker_embedding, vocoder=vocoder
        )
    return waveform.numpy()


def save_waveform(waveform: np.ndarray, sample_rate: int, output_path: Path) -> None:
    """Persist a generated waveform to disk as a WAV file.

    Args:
        waveform: Mono waveform array produced by :func:`synthesize_speech`.
        sample_rate: Sampling rate, in Hertz, used to encode the waveform.
        output_path: Destination path for the WAV file.
    """

    # Ensure 1-D mono float array in [-1, 1]
    arr = np.asarray(waveform)
    if arr.ndim > 1:
        arr = np.squeeze(arr)
    arr = np.clip(arr, -1.0, 1.0)

    # Convert to 16-bit PCM and write with the standard library
    pcm16 = (arr * 32767.0).astype(np.int16)
    with wave.open(str(output_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm16.tobytes())


def main(text: str = DEFAULT_TEXT, output_filename: str = "output.wav") -> None:
    """Generate a speech sample and save it to ``output_filename``.

    Args:
        text: Text prompt to convert into audio. Defaults to
            ``"Welcome to inclusive education with AI."``.
        output_filename: File name or path where the audio will be stored.
    """

    processor, model, vocoder = load_speecht5_components()
    speaker_embedding = create_default_speaker_embedding()
    waveform = synthesize_speech
    (text, processor, model, vocoder, speaker_embedding)
    save_waveform(waveform, DEFAULT_SAMPLE_RATE, Path(output_filename))
    print(f"✅ Audio saved as {output_filename}")


if __name__ == "__main__":
    main()
