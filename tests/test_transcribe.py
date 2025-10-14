"""Unit tests for the transcribe module.

This module contains test cases for audio transcription functionality,
including file loading and Whisper model integration.
"""
import io
import os
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

# Provide a lightweight stub so importing :mod:`transcribe` does not require
# the heavy Whisper dependency during tests.
sys.modules.setdefault("whisper", MagicMock())

import transcribe
from compatibility import NumpyCompatibilityError
from transcribe import load_audio_path, transcribe_audio


class TestTranscribe(unittest.TestCase):
    """Test suite for audio transcription functions."""

    def setUp(self):
        # Create a dummy file for testing
        self.test_file = Path("test_audio.mp3")
        self.test_file.touch()

    def tearDown(self):
        # Clean up the dummy file
        if self.test_file.exists():
            self.test_file.unlink()

    def test_load_audio_path_valid(self):
        """
        Tests that load_audio_path returns a valid
        Path object for an existing file.
        """
        path = load_audio_path(str(self.test_file))
        self.assertIsInstance(path, Path)
        self.assertTrue(path.exists())

    def test_load_audio_path_not_found(self):
        """
        Tests that load_audio_path raises
        FileNotFoundError for a non-existent file.
        """
        with self.assertRaises(FileNotFoundError):
            load_audio_path("non_existent_file.mp3")

    def test_load_audio_path_is_not_a_file(self):
        """
        Tests that load_audio_path raises
        FileNotFoundError for a directory.
        """
        with self.assertRaises(FileNotFoundError):
            load_audio_path(".")  # Current directory

    @patch("transcribe.ensure_numpy_compatible")
    @patch("transcribe.whisper")
    def test_transcribe_audio(self, mock_whisper, mock_numpy_check):
        """
        Tests that transcribe_audio calls the
        whisper library with the correct arguments.
        """
        # Configure the mock
        mock_model = MagicMock()
        mock_whisper.load_model.return_value = mock_model
        mock_model.transcribe.return_value = {
            "text": "This is a test transcript."
        }

        # Call the function
        result = transcribe_audio(self.test_file, "base", False)

        # Assert that the mock was called correctly
        mock_whisper.load_model.assert_called_with("base")
        mock_model.transcribe.assert_called_with(
            str(self.test_file), fp16=False
        )

        mock_numpy_check.assert_called_once_with()

        # Assert the result
        self.assertEqual(result["text"], "This is a test transcript.")

    @patch("transcribe.import_module", side_effect=ModuleNotFoundError(
        "No module named 'whisper'"))
    def test_load_whisper_module_missing_dependency(self, mock_import_module):
        """Provides clearer guidance when Whisper is unavailable."""

        with self.assertRaisesRegex(ModuleNotFoundError, "openai-whisper"):
            transcribe.load_whisper_module()

        mock_import_module.assert_called_once_with("whisper")

    @patch("transcribe.load_audio_path")
    @patch("transcribe.parse_arguments")
    def test_main_reports_missing_whisper(self,
                                          mock_parse_arguments,
                                          mock_load_audio_path):
        """Ensures the CLI exits cleanly when Whisper is unavailable."""

        args = Namespace(
            audio_path="audio.mp3",
            model="base",
            fp16=False,
            ca_bundle=None,
        )
        mock_parse_arguments.return_value = args
        mock_load_audio_path.return_value = Path("audio.mp3")

        error = ModuleNotFoundError(transcribe.MISSING_WHISPER_MESSAGE,
                                    name="whisper")

        with patch("transcribe.transcribe_audio", side_effect=error):
            with patch("sys.stderr", new_callable=io.StringIO) as stderr:
                with self.assertRaises(SystemExit) as exit_info:
                    transcribe.main()

        self.assertEqual(exit_info.exception.code, 1)
        self.assertEqual(
            stderr.getvalue().strip(),
            transcribe.MISSING_WHISPER_MESSAGE,
        )

    @patch("transcribe.parse_arguments")
    @patch("transcribe.load_audio_path")
    def test_main_reports_incompatible_numpy(
        self,
        mock_load_audio_path,
        mock_parse_arguments,
    ):
        """The CLI should exit cleanly when NumPy is incompatible."""

        args = Namespace(
            audio_path="audio.mp3",
            model="base",
            fp16=False,
            ca_bundle=None,
        )
        mock_parse_arguments.return_value = args
        mock_load_audio_path.return_value = Path("audio.mp3")

        with patch(
            "transcribe.transcribe_audio",
            side_effect=NumpyCompatibilityError("numpy<2 required"),
        ):
            with patch("sys.stderr", new_callable=io.StringIO) as stderr:
                with self.assertRaises(SystemExit) as exit_info:
                    transcribe.main()

        assert exit_info.exception.code == 1
        assert "numpy<2" in stderr.getvalue()


    def test_configure_certificate_bundle_sets_environment(self):
        """A valid CA bundle should configure TLS environment variables."""

        with tempfile.NamedTemporaryFile("w", delete=False) as handle:
            handle.write("test certificate")
            bundle_path = handle.name

        previous_values = {
            key: os.environ.get(key)
            for key in ("REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "SSL_CERT_FILE")
        }

        try:
            transcribe.configure_certificate_bundle(bundle_path)
            expected = str(Path(bundle_path).resolve())
            assert os.environ["REQUESTS_CA_BUNDLE"] == expected
            assert os.environ["CURL_CA_BUNDLE"] == expected
            assert os.environ["SSL_CERT_FILE"] == expected
        finally:
            Path(bundle_path).unlink(missing_ok=True)
            for key, value in previous_values.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_configure_certificate_bundle_missing_file(self):
        """The helper should raise when the bundle path is invalid."""

        with self.assertRaises(FileNotFoundError):
            transcribe.configure_certificate_bundle("missing.pem")


if __name__ == "__main__":
    unittest.main()
