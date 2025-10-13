"""Unit tests for the tts module.

This module contains test cases for text-to-speech functionality,
including SpeechT5 model loading, speaker embeddings, and audio synthesis.
"""
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
import wave

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from tts import (
    create_default_speaker_embedding,
    save_waveform,
    load_speecht5_components,
    synthesize_speech,
)


class TestTTS(unittest.TestCase):
    """Test suite for text-to-speech synthesis functions."""

    def setUp(self):
        self.output_path = Path("test_output.wav")

    def tearDown(self):
        if self.output_path.exists():
            self.output_path.unlink()

    def test_create_default_speaker_embedding(self):
        """
        Tests that the speaker embedding is deterministic
        and has the correct shape.
        """
        embedding1 = create_default_speaker_embedding(seed=42)
        embedding2 = create_default_speaker_embedding(seed=42)
        self.assertTrue(torch.equal(embedding1, embedding2))
        self.assertEqual(embedding1.shape, (1, 512))

    def test_save_waveform(self):
        """
        Tests that the waveform is saved correctly as a WAV file.
        """
        sample_rate = 16000
        waveform = np.sin(np.linspace(0, 440 * 2 * np.pi, sample_rate))

        save_waveform(waveform, sample_rate, self.output_path)

        self.assertTrue(self.output_path.exists())

        with wave.open(str(self.output_path), "rb") as wf:
            self.assertEqual(wf.getnchannels(), 1)
            self.assertEqual(wf.getsampwidth(), 2)
            self.assertEqual(wf.getframerate(), sample_rate)

    @patch("tts.SpeechT5Processor")
    @patch("tts.SpeechT5ForTextToSpeech")
    @patch("tts.SpeechT5HifiGan")
    def test_load_speecht5_components(
        self, mock_vocoder, mock_model, mock_processor
    ):
        """
        Tests that the SpeechT5 components are loaded with
        the correct identifiers.
        """
        load_speecht5_components("test_model_id", "test_vocoder_id")
        mock_processor.from_pretrained.assert_called_with(
            "test_model_id"
        )
        mock_model.from_pretrained.assert_called_with("test_model_id")
        mock_vocoder.from_pretrained.assert_called_with(
            "test_vocoder_id"
        )

    def test_synthesize_speech(self):
        """
        Tests that the speech synthesis function calls
        the model correctly.
        """
        mock_processor = MagicMock()
        mock_model = MagicMock()
        mock_vocoder = MagicMock()
        mock_speaker_embedding = torch.randn((1, 512))

        mock_processor.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        mock_model.generate_speech.return_value = torch.tensor([0.1, 0.2, 0.3])

        waveform = synthesize_speech(
            "test text",
            mock_processor,
            mock_model,
            mock_vocoder,
            mock_speaker_embedding,
        )

        mock_processor.assert_called_with(
            text="test text", return_tensors="pt"
        )
        mock_model.generate_speech.assert_called_once()
        self.assertIsInstance(waveform, np.ndarray)


if __name__ == "__main__":
    unittest.main()
