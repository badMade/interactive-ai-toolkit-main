import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import os

from transcribe import load_audio_path, transcribe_audio

class TestTranscribe(unittest.TestCase):

    def setUp(self):
        # Create a dummy file for testing
        self.test_file = Path("test_audio.mp3")
        self.test_file.touch()

    def tearDown(self):
        # Clean up the dummy file
        if self.test_file.exists():
            self.test_file.unlink()

    def test_load_audio_path_valid(self):
        """Tests that load_audio_path returns a valid Path object for an existing file."""
        path = load_audio_path(str(self.test_file))
        self.assertIsInstance(path, Path)
        self.assertTrue(path.exists())

    def test_load_audio_path_not_found(self):
        """Tests that load_audio_path raises FileNotFoundError for a non-existent file."""
        with self.assertRaises(FileNotFoundError):
            load_audio_path("non_existent_file.mp3")

    def test_load_audio_path_is_not_a_file(self):
        """Tests that load_audio_path raises FileNotFoundError for a directory."""
        with self.assertRaises(FileNotFoundError):
            load_audio_path(".") # Current directory

    @patch("transcribe.whisper")
    def test_transcribe_audio(self, mock_whisper):
        """Tests that transcribe_audio calls the whisper library with the correct arguments."""
        # Configure the mock
        mock_model = MagicMock()
        mock_whisper.load_model.return_value = mock_model
        mock_model.transcribe.return_value = {"text": "This is a test transcript."}

        # Call the function
        result = transcribe_audio(self.test_file, "base", False)

        # Assert that the mock was called correctly
        mock_whisper.load_model.assert_called_with("base")
        mock_model.transcribe.assert_called_with(str(self.test_file), fp16=False)

        # Assert the result
        self.assertEqual(result["text"], "This is a test transcript.")

if __name__ == "__main__":
    unittest.main()
